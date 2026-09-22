#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/HyperConnection.h"
#include "metal/abi/PerLayerEmbedding.h"
#include "model/WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <bit>
#include <cstdlib>
#include <dispatch/dispatch.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>

namespace splash::model {
namespace {

// Matches kMoeSkippedRoute in the MoE kernels: a route with no expert.
constexpr uint32_t kMoeSkippedRoute = 0xFFFFFFFFu;

// qwen4exp gates its linear-attention output with a sigmoid (the checkpoint's
// config sets output_gate_type), where Qwen3.8 uses silu. Everything else about
// the GDN shape is shared, so only this flag differs.
[[nodiscard]] ops::GdnShape gdnShape(const QwenTargetGeometry &geometry) {
  ops::GdnShape shape = geometry.gdnShape();
  shape.sigmoidGate = true;
  return shape;
}

// ---------------------------------------------------------------------------
// Per-layer embedding.
//
// Added to the residual entering layer geometry.pleLayer:
//
//   ids      = hashed n-grams of each token and its two predecessors
//   emb      = one gathered row per head, concatenated        rows x 2560
//   keys     = emb @ key projection                           rows x 10240
//   values   = emb @ value projection                         rows x 2560
//   gated    = gate(keys, values, residual)                   rows x 10240
//   residual += gated + silu(dilated conv over normalized gated)
//
// The n-gram hash reads two tokens of history and the convolution nine
// normalized rows, both kept in the GDN cell's auxiliary region.
// ---------------------------------------------------------------------------

PerLayerEmbeddingStateParams pleStateParams(const QwenTargetGeometry &geometry,
                                            uint32_t rows) {
  return {rows, geometry.residualWidth(), geometry.pleHistoryRows,
          geometry.pleEndToken, rows, 0, 0, 0};
}

// One sequence's shifted token rows and gathered n-gram embedding rows.
void gatherNgramRowsOnCpu(const Qwen4ExpWeights &weights,
                          const QwenTargetGeometry &geometry,
                          const uint32_t *tokens, const uint8_t *state,
                          uint16_t *output, uint32_t rows, uint32_t groupRows,
                          uint32_t liveRows);

// `onCpu` requires every earlier submission to have completed, so the token
// ids and stored history can be read; otherwise the lookup is a GPU kernel,
// which is correct but pays for keeping the whole table resident.
void addPleGather(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
                  metal::CommandGraph &graph, const QwenTargetPleBuffers &ple,
                  const QwenTargetGeometry &geometry, metal::MetalBuffer state,
                  uint32_t rowBegin, uint32_t rows, bool onCpu,
                  uint32_t liveRows) {
  if (onCpu) {
    gatherNgramRowsOnCpu(
        weights, geometry,
        static_cast<const uint32_t *>(ple.tokens.contents()) + rowBegin,
        static_cast<const uint8_t *>(state.contents()),
        static_cast<uint16_t *>(ple.embedding.contents()) +
            uint64_t{rowBegin} * geometry.pleEmbeddingSize,
        rows, rows, liveRows);
    return;
  }
  const Qwen4ExpLayout &layout = weights.layout;
  const auto &table = weights.perLayerEmbedding;
  const PerLayerEmbeddingStateParams params = pleStateParams(geometry, rows);
  metal::MetalBuffer tokens =
      backend.view(ple.tokens, uint64_t{rowBegin} * 4, uint64_t{rows} * 4);
  metal::MetalBuffer shifted = backend.view(
      ple.shifted, uint64_t{rowBegin} * 3 * 4, uint64_t{rows} * 3 * 4);
  graph.add("per_layer_embedding_shift", {tokens, std::move(state), shifted},
            params, {(rows + 63) / 64, 1, 1}, {64, 1, 1});
  const NgramEmbeddingParams gather{rows, layout.ngramHeads(),
                                    layout.ngramHeadDimension(),
                                    layout.ngramSize, layout.ngramHeadsPerOrder,
                                    kQ4FineGroupElements};
  const uint64_t embeddingRow = uint64_t{geometry.pleEmbeddingSize} * 2;
  graph.add("ngram_embedding_gather",
            {std::move(shifted), table.layerMultipliers,
             table.headVocabularySizes, table.headOffsets, table.table.weights,
             table.table.scales, table.table.biases,
             backend.view(ple.embedding, rowBegin * embeddingRow,
                          rows * embeddingRow)},
            gather, {rows, 1, 1}, {128, 1, 1});
}

// The n-gram table lookup, on the CPU. The table is 27 GB and a GPU kernel
// that binds it forces all of it resident; a row per head per token is 80
// bytes read from the mapped file. Rows at or past `liveRows` of each group
// of `groupRows` are zero-filled: their output is discarded.
//
// Must run only once the token ids and the stored history are final, which
// the staged execution guarantees at the point layer pleLayer is encoded.
void gatherNgramRowsOnCpu(const Qwen4ExpWeights &weights,
                          const QwenTargetGeometry &geometry,
                          const uint32_t *tokens, const uint8_t *state,
                          uint16_t *output, uint32_t rows, uint32_t groupRows,
                          uint32_t liveRows) {
  const Qwen4ExpLayout &layout = weights.layout;
  const auto &ple = weights.perLayerEmbedding;
  const uint32_t heads = layout.ngramHeads();
  const uint32_t dimension = layout.ngramHeadDimension();
  const uint32_t groups = dimension / kQ4FineGroupElements;
  const auto *offsets = static_cast<const int64_t *>(ple.headOffsets.contents());
  const auto *vocabulary =
      static_cast<const int64_t *>(ple.headVocabularySizes.contents());
  const auto *multipliers =
      static_cast<const int64_t *>(ple.layerMultipliers.contents());
  const auto *packed = static_cast<const uint8_t *>(ple.table.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(ple.table.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(ple.table.biases.contents());
  const uint32_t eos = geometry.pleEndToken;
  const auto *header = reinterpret_cast<const uint32_t *>(state);
  const bool valid = header[0] != 0;
  const uint32_t older = valid ? header[1] : eos;
  const uint32_t newer = valid ? header[2] : eos;
  auto widen = [](uint16_t bits) {
    return std::bit_cast<float>(uint32_t{bits} << 16);
  };
  auto narrow = [](float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
  };
  const uint64_t width = uint64_t{heads} * dimension;
  dispatch_apply(rows, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t index) {
    const uint32_t row = static_cast<uint32_t>(index);
    uint16_t *destination = output + row * width;
    if (row % groupRows >= liveRows) {
      std::fill_n(destination, width, uint16_t{0});
      return;
    }
    // Same window as per_layer_embedding_shift: tokens restart after an
    // end-of-sequence token, and the stored pair precedes the first row.
    const uint32_t base = row - row % groupRows;
    const uint32_t local = row - base;
    const uint32_t current = tokens[row];
    const uint32_t oneBack = local >= 1 ? tokens[row - 1] : newer;
    const uint32_t twoBackRaw =
        local >= 2 ? tokens[row - 2] : (local == 1 ? newer : older);
    const uint32_t twoBack = oneBack == eos ? eos : twoBackRaw;
    const uint32_t shifted[3] = {current, oneBack, twoBack};
    for (uint32_t head = 0; head < heads; ++head) {
      const uint32_t order = head / layout.ngramHeadsPerOrder + 2;
      // Wrapping 64-bit arithmetic, as the reference's int64 tensors wrap.
      uint64_t mixed = uint64_t{shifted[0]} * uint64_t(multipliers[0]);
      for (uint32_t position = 1; position < order; ++position)
        mixed ^= uint64_t{shifted[position]} * uint64_t(multipliers[position]);
      int64_t remainder = static_cast<int64_t>(mixed) % vocabulary[head];
      if (remainder < 0)
        remainder += vocabulary[head];
      const uint64_t entry = uint64_t(remainder + offsets[head]);
      const uint8_t *codes = packed + entry * dimension / 2;
      for (uint32_t i = 0; i < dimension; ++i) {
        const float code = float((codes[i / 2] >> ((i & 1) * 4)) & 0xF);
        const uint64_t parameter = entry * groups + i / kQ4FineGroupElements;
        destination[head * dimension + i] =
            narrow(code * widen(scales[parameter]) + widen(biases[parameter]));
      }
    }
  });
}


// SPLASH_ROUTE_LOG=path appends every routed expert choice, one line per
// (phase, layer, row): "P|D layer e0 e1 ... e9". Measurement only; it lets
// cache sizes and policies be replayed offline against real routing.
void logRoutes(char phase, uint32_t layer, const uint32_t *selected,
               uint32_t rows, uint32_t routesPerRow, uint32_t perToken,
               uint32_t liveRows, uint32_t groupRows) {
  static FILE *file = [] {
    const char *path = std::getenv("SPLASH_ROUTE_LOG");
    return path ? std::fopen(path, "a") : nullptr;
  }();
  if (!file)
    return;
  for (uint32_t row = 0; row < rows; ++row) {
    if (row % groupRows >= liveRows)
      continue;
    std::fprintf(file, "%c %u", phase, layer);
    for (uint32_t k = 0; k < perToken; ++k)
      std::fprintf(file, " %u", selected[row * routesPerRow + k]);
    std::fputc('\n', file);
  }
}

// The hyper-connection mix in front of a block: normalize the streams, reduce
// them (and compute the injection gates), then mix them into the block input.
// Weight-stationary kernels: each weight is read once for all rows.
void addHyperConnection(metal::CommandGraph &graph,
                        const QwenTargetGeometry &geometry,
                        metal::MetalBuffer input,
                        const Qwen4ExpHyperConnection &weights,
                        metal::MetalBuffer normalized,
                        metal::MetalBuffer reduced, metal::MetalBuffer mixed,
                        metal::MetalBuffer injection, uint32_t rows) {
  const bool withInject = weights.blockInject.has_value();
  const HyperConnectionParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, withInject ? 1u : 0u};
  graph.add("hyper_connection_rms", {std::move(input), weights.norm, normalized},
            params, {rows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
  const uint32_t outputs = geometry.hyperConnectionLowRank +
                           (withInject ? geometry.hyperConnectionCount : 0);
  graph.add("hyper_connection_down",
            {normalized, weights.mixDown,
             withInject ? *weights.blockInject : weights.mixDown, reduced,
             withInject ? std::move(injection) : reduced},
            params, {(outputs + 7) / 8, 1, 1}, {256, 1, 1});
  graph.add("hyper_connection_up_mix",
            {std::move(normalized), std::move(reduced), weights.mixUp,
             std::move(mixed)},
            params, {geometry.hiddenSize / 8, 1, 1}, {256, 1, 1});
}

// Reads missed experts straight from the layer file into their cache slots:
// three reads per expert (gate, up, down), all in flight together. The SSD
// reaches ~15 GB/s this way; faulting pages in one at a time reached ~1.5.
template <class Miss>
void readMissedExperts(const Qwen4ExpExpertSource &source, const Miss *misses,
                       size_t count, uint64_t stride, char *gate, char *up,
                       char *down) {
  if (!count)
    return;
  dispatch_apply(count * 3,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t index) {
    const Miss &miss = misses[index / 3];
    const uint32_t matrix = static_cast<uint32_t>(index % 3);
    char *slot = (matrix == 0 ? gate : matrix == 1 ? up : down) +
                 uint64_t{miss.slot} * stride;
    const uint64_t offset =
        (matrix == 0 ? source.gate : matrix == 1 ? source.up : source.down) +
        uint64_t{miss.expert} * stride;
    uint64_t done = 0;
    while (done < stride) {
      const ssize_t got = ::pread(source.fd, slot + done, stride - done,
                                  static_cast<off_t>(offset + done));
      if (got <= 0)
        break; // Leaves the slot short; the file was validated at load.
      done += static_cast<uint64_t>(got);
    }
  });
}

// One sequence's gate, convolution and residual update. `normalized` holds
// this sequence's history rows followed by its current rows.
void addPleApply(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
                 metal::CommandGraph &graph, const QwenTargetPleBuffers &ple,
                 const QwenTargetGeometry &geometry, metal::MetalBuffer state,
                 metal::MetalBuffer normalized, metal::MetalBuffer residual,
                 uint32_t rowBegin, uint32_t rows) {
  const auto &table = weights.perLayerEmbedding;
  const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
  const uint64_t hidden = uint64_t{geometry.hiddenSize} * 2;
  const PerLayerEmbeddingStateParams stateParams =
      pleStateParams(geometry, rows);
  const PerLayerEmbeddingParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      weights.layout.pleConvolutionTaps, weights.layout.ngramSize, 1e-6f};
  metal::MetalBuffer gated =
      backend.view(ple.gated, rowBegin * width, rows * width);
  graph.add("per_layer_embedding_history", {std::move(state), normalized},
            stateParams, {64, 1, 1}, {256, 1, 1});
  graph.add("per_layer_embedding_gate",
            {backend.view(ple.keys, rowBegin * width, rows * width),
             backend.view(ple.values, rowBegin * hidden, rows * hidden),
             residual, table.keyNorm, table.queryNorm, table.convolutionNorm,
             gated, normalized},
            params, {rows, 1, 1}, {256, 1, 1});
  graph.add("per_layer_embedding_convolve",
            {normalized, table.convolutionWeights, gated}, params, {64, 1, 1},
            {256, 1, 1});
  graph.add("per_layer_embedding_add", {gated, std::move(residual)},
            stateParams, {64, 1, 1}, {256, 1, 1});
}

metal::MetalBuffer auxiliaryView(metal::MetalBackend &backend,
                                 const QwenTargetGeometry &geometry,
                                 const metal::MetalBuffer &cell) {
  return backend.view(cell, geometry.stateLayout.auxiliaryOffset(),
                      geometry.stateLayout.auxiliaryBytes);
}

} // namespace

void Qwen4ExpTarget::addPerLayerEmbeddingCommit(
    const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
    metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers,
    uint32_t lanes) {
  constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
  const uint64_t laneNormalized =
      (uint64_t{geometry.pleHistoryRows} + laneRows) *
      geometry.residualWidth() * 2;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    PerLayerEmbeddingStateParams params = pleStateParams(geometry, laneRows);
    params.retained_from_buffer = 1;
    params.lane = lane;
    graph.add("per_layer_embedding_commit",
              {backend.view(buffers.pleTokens, uint64_t{lane} * laneRows * 4,
                            uint64_t{laneRows} * 4),
               backend.view(buffers.pleNormalized, lane * laneNormalized,
                            laneNormalized),
               auxiliaryView(backend, geometry, buffers.currentStates[lane]),
               auxiliaryView(backend, geometry, buffers.nextStates[lane]),
               buffers.retainedCounts},
              params, {64, 1, 1}, {256, 1, 1});
  }
}

void Qwen4ExpTarget::addEmbedding(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::CommandGraph &graph,
    metal::MetalBuffer tokens,
    metal::MetalBuffer hidden,
    metal::MetalBuffer scratch,
    uint32_t rows) {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget) {
    throw std::invalid_argument("invalid embedding row count");
  }
  // 1. Gather tokens into [rows, hiddenSize] scratch buffer
  ops::Embedding::add(graph, std::move(tokens), weights.tokenEmbedding, scratch,
                      rows);

  // 2. Broadcast across hyperConnectionCount streams
  const HyperConnectionParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, 1};
  const uint32_t total = rows * geometry.residualWidth();
  const uint32_t groups = (total + 255) / 256;
  graph.add("hyper_connection_broadcast",
            {std::move(scratch), std::move(hidden)}, params,
            {std::min(groups, 64U), 1, 1}, {256, 1, 1});
}

void Qwen4ExpTarget::addHead(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    metal::MetalBuffer hidden,
    metal::MetalBuffer finalHidden,
    metal::MetalBuffer logits,
    metal::MetalBuffer headNormalized,
    metal::MetalBuffer headReduced,
    uint32_t normalizedRows) {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  // Collapse the streams: the final mixer has no injection gates.
  addHyperConnection(graph, geometry, std::move(hidden),
                     weights.hyperConnectionMixer, headNormalized, headReduced,
                     finalHidden, {}, normalizedRows);

  // 3. Project finalHidden -> logits
  const ops::LinearMatrix head{geometry.vocabularySize, geometry.hiddenSize};
  operators.linear().addDecode(graph, std::move(finalHidden),
                               weights.logitsProjection, std::move(logits),
                               head);
}

void Qwen4ExpTarget::addPrefill(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::MetalBackend &backend,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences,
    uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }

  const ops::LinearMatrix gdnInput{geometry.packedGdnWidth,
                                   geometry.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth,
                                         geometry.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry.hiddenSize,
                                      geometry.attentionWidth};
  const ops::MoePlan moePlan = operators.moePrefill(geometry.moe, rows);

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  const HyperConnectionParams hcParams{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, 1};

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  auto encodePerLayerEmbedding = [&](metal::CommandGraph &g,
                                     uint32_t layerIndex, bool onCpu) {
    const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
    const uint32_t history = geometry.pleHistoryRows;
    for (const QwenTargetPrefillSequence &sequence : sequences)
      addPleGather(weights, backend, g, buffers.ple, geometry,
                   sequence.auxiliaryIn, sequence.rowBegin, sequence.rows,
                   onCpu, sequence.rows);
    const ops::LinearMatrix keys{geometry.residualWidth(),
                                 geometry.pleEmbeddingSize};
    const ops::LinearMatrix values{geometry.hiddenSize,
                                   geometry.pleEmbeddingSize};
    // The sums depend only on the input rows, so one pass serves both.
    operators.linear().addPrefillSums(g, buffers.ple.embedding,
                                      buffers.projectionSums, keys, rows);
    operators.linear().addPrefill(g, buffers.ple.embedding,
                                  weights.perLayerEmbedding.keyProjection,
                                  buffers.ple.keys, buffers.projectionSums,
                                  keys, rows);
    operators.linear().addPrefill(g, buffers.ple.embedding,
                                  weights.perLayerEmbedding.valueProjection,
                                  buffers.ple.values, buffers.projectionSums,
                                  values, rows);
    for (uint32_t index = 0; index < sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = sequences[index];
      // Each sequence's history sits ahead of its own rows.
      metal::MetalBuffer normalized = backend.view(
          buffers.ple.normalized,
          (uint64_t{sequence.rowBegin} + uint64_t{index} * history) * width,
          (uint64_t{history} + sequence.rows) * width);
      addPleApply(weights, backend, g, buffers.ple, geometry,
                  sequence.auxiliaryIn, normalized,
                  u16(buffers.hidden[layerIndex & 1], sequence.rowBegin,
                      sequence.rows, geometry.residualWidth()),
                  sequence.rowBegin, sequence.rows);
      // A prefill keeps every row it processes.
      const PerLayerEmbeddingStateParams commit =
          pleStateParams(geometry, sequence.rows);
      metal::MetalBuffer tokens =
          backend.view(buffers.ple.tokens, uint64_t{sequence.rowBegin} * 4,
                       uint64_t{sequence.rows} * 4);
      g.add("per_layer_embedding_commit",
            {tokens, normalized, sequence.auxiliaryIn, sequence.auxiliaryOut,
             tokens},
            commit, {64, 1, 1}, {256, 1, 1});
    }
  };

  // priorWorkComplete: every earlier submission of this step has finished,
  // so host-side reads of GPU-written inputs are safe.
  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               bool priorWorkComplete = false) {
    if (geometry.hasPerLayerEmbedding() && layerIndex == geometry.pleLayer)
      encodePerLayerEmbedding(g, layerIndex, priorWorkComplete);
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows);
  };

  auto encodeMixer = [&](metal::CommandGraph &g, uint32_t layerIndex,
                         uint32_t &gdnIdx, uint32_t &attnIdx) -> metal::MetalBuffer {
    const auto &layer = weights.layers[layerIndex];
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addPrefillSums(g, buffers.hyperMixed,
                                        buffers.projectionSums, gdnInput, rows);
      operators.linear().addPrefill(g, buffers.hyperMixed,
                                    mixer.inputProjection, buffers.gdnPacked,
                                    buffers.projectionSums, gdnInput, rows);
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        ops::GDN::addPrefill(
            g,
            {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                 geometry.packedGdnWidth),
             mixer.convolutionWeights, sequence.convolutionIn[gdnIdx],
             sequence.convolutionOut[gdnIdx],
             u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                 geometry.gdnKeyWidth()),
             u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                 geometry.gdnKeyWidth()),
             u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth),
             mixer.decay, mixer.timeBias,
             f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                 geometry.gdnValueHeads),
             u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                 geometry.gdnValueHeads),
             sequence.recurrentIn[gdnIdx],
             sequence.recurrentOut[gdnIdx],
             u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth),
             mixer.mixerNorm,
             u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth)},
            gdnShape(geometry), sequence.rows);
      }
      operators.linear().addPrefillSums(g, buffers.gdnHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(g, buffers.gdnHidden,
                                    mixer.outputProjection, buffers.gdnOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      ++gdnIdx;
      return buffers.gdnOutput;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      operators.linear().addPrefillSums(g, buffers.hyperMixed,
                                        buffers.projectionSums, attentionInput,
                                        rows);
      operators.linear().addPrefill(g, buffers.hyperMixed,
                                    mixer.inputProjection, buffers.fullPacked,
                                    buffers.projectionSums, attentionInput,
                                    rows);
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        const uint64_t queryBytes =
            uint64_t{geometry.attentionQueryHeads} *
            sequence.attentionStride * geometry.attentionHeadDimension *
            sizeof(uint16_t);
        const uint64_t kvBytes =
            uint64_t{geometry.attentionKvHeads} *
            sequence.attentionStride * geometry.attentionHeadDimension *
            sizeof(uint16_t);
        metal::MetalBuffer queries = backend.view(
            buffers.fullQueries, sequence.queryOffset, queryBytes);
        metal::MetalBuffer attentionRows = backend.view(
            buffers.fullAttention, sequence.queryOffset, queryBytes);
        metal::MetalBuffer keys = backend.view(
            buffers.chunkKeys, sequence.kvOffset, kvBytes);
        metal::MetalBuffer values = backend.view(
            buffers.chunkValues, sequence.kvOffset, kvBytes);
        ops::PagedAttention::addPrefillProjection(
            g,
            u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                geometry.packedAttentionWidth),
            mixer.queryNorm, mixer.keyNorm,
            f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                geometry.rotaryPairs),
            f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                geometry.rotaryPairs),
            queries, keys, values, sequence.rows,
            sequence.attentionStride, sequence.attentionStride,
            geometry.attentionQueryHeads, geometry.kvLayout);
        ops::PagedAttention::addPrefillStore(
            g, kvLayers[attnIdx], keys, values,
            sequence.pageTable, sequence.q8, geometry.kvLayout);
        ops::PagedAttention::addPrefill(
            g, kvLayers[attnIdx], queries, attentionRows,
            buffers.attentionPartials, buffers.attentionStatistics,
            sequence.pageTable, sequence.q8,
            operators.prefillAttention(
                sequence.rows, geometry.attentionQueryHeads,
                geometry.kvLayout, sequence.q8.committed_tokens));
        ops::PagedAttention::addPrefillGate(
            g,
            u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                geometry.packedAttentionWidth),
            attentionRows,
            u16(buffers.attentionHidden, sequence.rowBegin, sequence.rows,
                geometry.attentionWidth),
            sequence.rows, sequence.attentionStride, sequence.attentionStride,
            geometry.attentionQueryHeads, geometry.kvLayout);
      }
      operators.linear().addPrefillSums(g, buffers.attentionHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(g, buffers.attentionHidden,
                                    mixer.outputProjection,
                                    buffers.attentionOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      ++attnIdx;
      return buffers.attentionOutput;
    }
  };

  auto encodeMixerUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer mixerOutBuffer) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_update",
          {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
          {64, 1, 1}, {256, 1, 1});
  };

  auto encodeMlpHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.mlpHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows);
  };

  auto encodeMoERoute = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    ops::MoE::addRoute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan);
  };

  auto encodeMoEExecute = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse) {
    ops::MoE::addExecute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, moePlan, /*addResidual=*/false);
  };

  auto encodeMlpUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    g.add("hyper_connection_update_out",
          {input, output, buffers.gdnOutput, buffers.hyperInjection},
          hcParams, {64, 1, 1}, {256, 1, 1});
  };

  auto encodeCapture = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto captureLayers = geometry.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      const uint32_t slot =
          static_cast<uint32_t>(captured - captureLayers.begin());
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        for (uint32_t index = 0; index < sequence.captureCount; ++index) {
          const QwenTargetPrefillCapture &capture = sequence.captures[index];
          ops::DraftAttention::captureTargetHidden(
              g, buffers.gdnOutput, buffers.captured, capture.rows, slot,
              capture.sourceStart, capture.destinationStart,
              geometry.hiddenSize, geometry.capturedHiddenSize());
        }
      }
    }
  };

  const uint32_t R = std::min(weights.residentLayers, geometry.layers);
  const bool useStreamingCache = (R < geometry.layers && weights.streamingCacheGate);

  if (!useStreamingCache) {
    for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
      encodeAttentionHC(graph, layerIndex);
      metal::MetalBuffer mixerOut = encodeMixer(graph, layerIndex, gdnIndex, attentionIndex);
      encodeMixerUpdate(graph, layerIndex, mixerOut);
      encodeMlpHC(graph, layerIndex);
      ops::MoE::add(
          graph,
          {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
           buffers.selectedExperts, buffers.routingWeights,
           buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
           buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
           buffers.expertOutput},
          weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
      encodeMlpUpdate(graph, layerIndex);
      encodeCapture(graph, layerIndex);
    }
    if (gdnIndex != geometry.stateLayout.layers ||
        attentionIndex != kvLayers.size()) {
      throw std::logic_error("Qwen target layer partition mismatch");
    }
    return;
  }

  // =========================================================================
  // Staged Streaming Execution Path with Active Expert Caching for Prefill
  // =========================================================================

  uint32_t totalMisses = 0;

  auto stageActiveExperts = [&](uint32_t layerIndex) -> bool {
    auto &cache = weights.layers[layerIndex].expertCache;
    auto *selPtr = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t expertsPerToken = weights.layout.expertsPerToken;
    const uint32_t totalExperts = weights.layout.experts;
    constexpr uint32_t kMaxExperts = 512;
    logRoutes('P', layerIndex, selPtr, rows, routesPerRow, expertsPerToken,
              rows, rows);
    int16_t stepExpertSeen[kMaxExperts];
    std::fill_n(stepExpertSeen, kMaxExperts, -1);
    std::vector<uint32_t> uniqueExperts;
    uniqueExperts.reserve(32);

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && stepExpertSeen[exp] == -1) {
          stepExpertSeen[exp] = 1;
          uniqueExperts.push_back(exp);
        }
      }
    }

    if (layerIndex < weights.lastSelectedExperts.size()) {
      weights.lastSelectedExperts[layerIndex] = uniqueExperts;
    }

    if (uniqueExperts.size() > cache.capacity) {
      // Prompt requires more unique experts than cache capacity in this layer;
      // fall back to monolithic layer.ffn
      return false;
    }

    struct Miss {
      uint32_t expert;
      uint32_t slot;
    };
    std::vector<Miss> misses;
    misses.reserve(uniqueExperts.size());
    ++cache.clock;

    int16_t expertToAssignedSlot[kMaxExperts];
    std::fill_n(expertToAssignedSlot, kMaxExperts, -1);

    for (uint32_t exp : uniqueExperts) {
      int16_t slot = cache.expertToSlot[exp];
      if (slot != -1) {
        // Cache hit: refresh LRU timestamp
        cache.lruTime[slot] = cache.clock;
        expertToAssignedSlot[exp] = slot;
      } else {
        // Cache miss: find a slot
        uint32_t assignSlot = 0;
        if (cache.numCached < cache.capacity) {
          assignSlot = cache.numCached++;
        } else {
          // Evict LRU slot not in active step
          uint32_t oldest = UINT32_MAX;
          uint32_t best = 0;
          for (uint32_t s = 0; s < cache.capacity; ++s) {
            if (cache.lruTime[s] == cache.clock) continue;
            if (cache.lruTime[s] < oldest) {
              oldest = cache.lruTime[s];
              best = s;
            }
          }
          assignSlot = best;
          int16_t evicted = cache.slotToExpert[assignSlot];
          if (evicted != -1) {
            cache.expertToSlot[evicted] = -1;
          }
        }
        cache.slotToExpert[assignSlot] = exp;
        cache.expertToSlot[exp] = assignSlot;
        cache.lruTime[assignSlot] = cache.clock;
        expertToAssignedSlot[exp] = assignSlot;
        misses.push_back({exp, assignSlot});
      }
    }

    totalMisses += static_cast<uint32_t>(misses.size());

    if (!misses.empty()) {
      const auto &layer = weights.layers[layerIndex];
      const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
      char *cg = static_cast<char *>(cache.cacheGate.contents());
      char *cu = static_cast<char *>(cache.cacheUp.contents());
      char *cd = static_cast<char *>(cache.cacheDown.contents());

      // The expert regions are mapped for random access, so copying a missed
      // expert faults it in one 16 KB page at a time, each its own disk read.
      // Asking for every missed range first lets the reads go out together
      // and in large pieces; the copies below then find the pages in memory.
      readMissedExperts(layer.expertSource, misses.data(), misses.size(),
                        stride, cg, cu, cd);
    }

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts) {
          selPtr[r * routesPerRow + k] = expertToAssignedSlot[exp];
        }
      }
      selPtr[r * routesPerRow + expertsPerToken] = 512;
    }

    return true;
  };

  auto makeCacheWeights = [&](uint32_t layerIndex) -> ops::MoeWeights {
    const auto &layer = weights.layers[layerIndex];
    const auto &cache = layer.expertCache;
    const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
    return ops::MoeWeights{
        .router = layer.ffn.router,
        .expertGate = {cache.cacheGate, cache.capacity,
                       weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertUp = {cache.cacheUp, cache.capacity,
                     weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertDown = {cache.cacheDown, cache.capacity,
                       weights.layout.hiddenSize, weights.layout.expertIntermediateSize, stride},
        .sharedGate = layer.ffn.sharedGate,
        .sharedUp = layer.ffn.sharedUp,
        .sharedDown = layer.ffn.sharedDown,
        .sharedExpertGate = layer.ffn.sharedExpertGate,
    };
  };

  metal::CommandGraph residentGraph = std::move(graph);
  for (uint32_t layerIndex = 0; layerIndex < R; ++layerIndex) {
    encodeAttentionHC(residentGraph, layerIndex);
    metal::MetalBuffer mixerOut = encodeMixer(residentGraph, layerIndex, gdnIndex, attentionIndex);
    encodeMixerUpdate(residentGraph, layerIndex, mixerOut);
    encodeMlpHC(residentGraph, layerIndex);
    ops::MoE::add(
        residentGraph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
    encodeMlpUpdate(residentGraph, layerIndex);
    encodeCapture(residentGraph, layerIndex);
  }

  // Layer R base up to router
  encodeAttentionHC(residentGraph, R);
  metal::MetalBuffer mixerOutR = encodeMixer(residentGraph, R, gdnIndex, attentionIndex);
  encodeMixerUpdate(residentGraph, R, mixerOutR);
  encodeMlpHC(residentGraph, R);
  encodeMoERoute(residentGraph, R);

  // Submit residentGraph and prefetch streaming experts in background
  auto t0 = std::chrono::steady_clock::now();
  metal::CommandTicket residentTicket = backend.submitCommandAsync(residentGraph.dispatches());
  // No blanket read-ahead of the previous step's experts: after a long
  // prompt that is nearly every expert, ~68 GB the OS then reads in the
  // background for minutes, stalling decode. Misses are fetched on demand.
  (void)residentTicket.wait();
  auto t1 = std::chrono::steady_clock::now();
  double residentMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double totalStageMs = 0.0;
  double totalGpuMs = 0.0;

  // Loop through streaming layers R .. geometry.layers - 2
  for (uint32_t L = R; L < geometry.layers - 1; ++L) {
    auto ts0 = std::chrono::steady_clock::now();
    bool staged = stageActiveExperts(L);
    auto ts1 = std::chrono::steady_clock::now();
    totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    ops::MoeWeights moeW = staged ? makeCacheWeights(L) : weights.layers[L].ffn;

    metal::CommandGraph stepGraph;
    encodeMoEExecute(stepGraph, moeW);
    encodeMlpUpdate(stepGraph, L);
    encodeCapture(stepGraph, L);

    // Layer L + 1 base up to router
    encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
    metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
    encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
    encodeMlpHC(stepGraph, L + 1);
    encodeMoERoute(stepGraph, L + 1);

    auto tg0 = std::chrono::steady_clock::now();
    (void)backend.submitCommandAsync(stepGraph.dispatches()).wait();
    auto tg1 = std::chrono::steady_clock::now();
    totalGpuMs += std::chrono::duration<double, std::milli>(tg1 - tg0).count();
  }

  const uint32_t lastL = geometry.layers - 1;
  auto ts0 = std::chrono::steady_clock::now();
  bool stagedLast = stageActiveExperts(lastL);
  auto ts1 = std::chrono::steady_clock::now();
  totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  ops::MoeWeights moeWLast = stagedLast ? makeCacheWeights(lastL) : weights.layers[lastL].ffn;

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE, MLP update, and capture into caller's graph
  encodeMoEExecute(graph, moeWLast);
  encodeMlpUpdate(graph, lastL);
  encodeCapture(graph, lastL);

  std::cerr << "[Prefill Timing] Resident 0.." << R << ": " << residentMs << " ms | Streaming Staging: "
            << totalStageMs << " ms (misses: " << totalMisses << ") | Streaming GPU: " << totalGpuMs << " ms\n";

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

void Qwen4ExpTarget::addVerify(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::MetalBackend &backend,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify,
    uint32_t lanes,
    ops::Q4DispatchStats &stats) {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators.verifyAttention(
      lanes, geometry.attentionQueryHeads, geometry.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry.packedGdnWidth,
                                   geometry.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth,
                                         geometry.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry.hiddenSize,
                                      geometry.attentionWidth};
  const ops::MoePlan moePlan = operators.moeDecode(geometry.moe, lanes);

  const HyperConnectionParams hcParams{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, 1};

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  auto encodePerLayerEmbedding = [&](metal::CommandGraph &g,
                                     uint32_t layerIndex, bool onCpu) {
    constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
    const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
    const uint64_t laneNormalized =
        (uint64_t{geometry.pleHistoryRows} + laneRows) * width;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      addPleGather(weights, backend, g, buffers.ple, geometry,
                   auxiliaryView(backend, geometry,
                                 buffers.currentGdnStates[lane]),
                   lane * laneRows, laneRows, onCpu, buffers.liveRowsPerLane);
    operators.linear().addDecodeBatch(
        g, buffers.ple.embedding, weights.perLayerEmbedding.keyProjection,
        buffers.ple.keys,
        {geometry.residualWidth(), geometry.pleEmbeddingSize}, lanes, stats);
    operators.linear().addDecodeBatch(
        g, buffers.ple.embedding, weights.perLayerEmbedding.valueProjection,
        buffers.ple.values, {geometry.hiddenSize, geometry.pleEmbeddingSize},
        lanes, stats);
    // Which rows to keep is not known until the verifier decides, so the
    // history is written by the state commit, not here.
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      addPleApply(
          weights, backend, g, buffers.ple, geometry,
          auxiliaryView(backend, geometry, buffers.currentGdnStates[lane]),
          backend.view(buffers.ple.normalized, lane * laneNormalized,
                       laneNormalized),
          backend.view(buffers.hidden[layerIndex & 1],
                       uint64_t{lane} * laneRows * width, laneRows * width),
          lane * laneRows, laneRows);
    }
  };

  // priorWorkComplete: every earlier submission of this step has finished,
  // so host-side reads of GPU-written inputs are safe.
  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               bool priorWorkComplete = false) {
    if (geometry.hasPerLayerEmbedding() && layerIndex == geometry.pleLayer)
      encodePerLayerEmbedding(g, layerIndex, priorWorkComplete);
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows);
  };

  auto encodeMixer = [&](metal::CommandGraph &g, uint32_t layerIndex,
                         uint32_t &gdnIdx, uint32_t &attnIdx) -> metal::MetalBuffer {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer mixerOutBuffer;
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addDecodeBatch(
          g, buffers.hyperMixed, mixer.inputProjection,
          buffers.gdnPacked[gdnIdx], gdnInput, lanes, stats);
      ops::GDN::addDecode(
          g,
          {buffers.gdnPacked[gdnIdx], mixer.convolutionWeights,
           buffers.currentGdnStates, buffers.nextGdnStates,
           buffers.gdnMixed[gdnIdx], mixer.decay, mixer.timeBias,
           buffers.gdnDecay[gdnIdx], buffers.gdnBeta[gdnIdx],
           buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
           buffers.arrived, buffers.generation},
          gdnShape(geometry), lanes, gdnIdx,
          {geometry.stateLayout.convolutionLayerBytes(),
           geometry.stateLayout.recurrentLayerBytes(),
           geometry.stateLayout.convolutionBytes()});
      operators.linear().addDecodeBatch(g, buffers.gdnHidden,
                                        mixer.outputProjection,
                                        buffers.gdnOutput, mixerOutput, lanes,
                                        stats);
      mixerOutBuffer = buffers.gdnOutput;
      ++gdnIdx;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      const uint32_t tileRows = kv::kPageTokens;
      operators.linear().addDecodeBatch(
          g, buffers.hyperMixed, mixer.inputProjection, buffers.fullPacked,
          attentionInput, lanes, stats);
      ops::PagedAttention::addVerifyProjection(
          g, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
          buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
          buffers.chunkKeys[attnIdx], buffers.chunkValues[attnIdx],
          ExecutionLimits::targetVerifyRows, tileRows, tileRows,
          geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      ops::PagedAttention::addVerify(
          g, kvLayers[attnIdx],
          {buffers.chunkKeys[attnIdx], buffers.chunkValues[attnIdx],
           buffers.fullQueries, buffers.attentionPartials,
           buffers.attentionStatistics, buffers.fullAttention,
           buffers.pageTables},
          q8, verify, attentionPlan);
      ops::PagedAttention::addVerifyGate(
          g, buffers.fullPacked, buffers.fullAttention,
          buffers.attentionHidden, ExecutionLimits::targetVerifyRows, tileRows,
          tileRows, geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      operators.linear().addDecodeBatch(g, buffers.attentionHidden,
                                        mixer.outputProjection,
                                        buffers.attentionOutput, mixerOutput,
                                        lanes, stats);
      mixerOutBuffer = buffers.attentionOutput;
      ++attnIdx;
    }
    return mixerOutBuffer;
  };

  auto encodeMixerUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer mixerOutBuffer) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_update",
          {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
          {64, 1, 1}, {256, 1, 1});
  };

  auto encodeMlpHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.mlpHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows);
  };

  auto encodeMoERoute = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    ops::MoE::addRoute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan);
  };

  auto encodeMoEExecute = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse) {
    ops::MoE::addExecute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, moePlan, /*addResidual=*/false);
  };

  auto encodeMlpUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    g.add("hyper_connection_update_out",
          {input, output, buffers.gdnOutput, buffers.hyperInjection},
          hcParams, {64, 1, 1}, {256, 1, 1});
  };

  auto encodeCapture = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto captureLayers = geometry.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      ops::DraftAttention::captureTargetHidden(
          g, buffers.gdnOutput, buffers.capturedTargetHidden, rows,
          static_cast<uint32_t>(captured - captureLayers.begin()), 0, 0,
          geometry.hiddenSize, geometry.capturedHiddenSize());
    }
  };

  auto encodeHead = [&](metal::CommandGraph &g) {
    addHyperConnection(g, geometry, buffers.hidden[geometry.layers & 1],
                       weights.hyperConnectionMixer, buffers.normalized,
                       buffers.hyperReduced, buffers.finalHidden, {}, rows);
    const ops::LinearMatrix head{geometry.vocabularySize, geometry.hiddenSize};
    operators.linear().addDecodeBatch(g, buffers.finalHidden,
                                      weights.logitsProjection, buffers.logits,
                                      head, lanes, stats);
  };

  const uint32_t R = std::min(weights.residentLayers, geometry.layers);
  const bool useStreamingCache = (R < geometry.layers && weights.streamingCacheGate);

  if (!useStreamingCache) {
    for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
      encodeAttentionHC(graph, layerIndex);
      metal::MetalBuffer mixerOut = encodeMixer(graph, layerIndex, gdnIndex, attentionIndex);
      encodeMixerUpdate(graph, layerIndex, mixerOut);
      encodeMlpHC(graph, layerIndex);
      ops::MoE::add(
          graph,
          {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
           buffers.selectedExperts, buffers.routingWeights,
           buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
           buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
           buffers.expertOutput},
          weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
      encodeMlpUpdate(graph, layerIndex);
      encodeCapture(graph, layerIndex);
    }
    if (gdnIndex != geometry.stateLayout.layers ||
        attentionIndex != kvLayers.size()) {
      throw std::logic_error("Qwen target layer partition mismatch");
    }
    encodeHead(graph);
    return;
  }

  // =========================================================================
  // Staged Streaming Execution Path with Active Expert Caching & Speculation
  // =========================================================================

  uint32_t totalMisses = 0;

  auto stageActiveExperts = [&](uint32_t layerIndex) -> bool {
    auto &cache = weights.layers[layerIndex].expertCache;
    auto *selPtr = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t expertsPerToken = weights.layout.expertsPerToken;
    const uint32_t totalExperts = weights.layout.experts;
    constexpr uint32_t kMaxExperts = 512;
    logRoutes('D', layerIndex, selPtr, rows, routesPerRow, expertsPerToken,
              buffers.liveRowsPerLane, ExecutionLimits::targetVerifyRows);
    int16_t stepExpertSeen[kMaxExperts];
    std::fill_n(stepExpertSeen, kMaxExperts, -1);
    std::vector<uint32_t> uniqueExperts;
    uniqueExperts.reserve(32);

    for (uint32_t r = 0; r < rows; ++r) {
      // Rows that cannot be kept select no expert at all, so they neither
      // cost expert work nor evict experts the kept rows will need.
      if (r % ExecutionLimits::targetVerifyRows >= buffers.liveRowsPerLane) {
        for (uint32_t k = 0; k < expertsPerToken; ++k)
          selPtr[r * routesPerRow + k] = kMoeSkippedRoute;
        continue;
      }
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && stepExpertSeen[exp] == -1) {
          stepExpertSeen[exp] = 1;
          uniqueExperts.push_back(exp);
        }
      }
    }

    if (layerIndex < weights.lastSelectedExperts.size()) {
      weights.lastSelectedExperts[layerIndex] = uniqueExperts;
    }

    if (uniqueExperts.size() > cache.capacity) {
      return false;
    }

    struct Miss {
      uint32_t expert;
      uint32_t slot;
    };
    std::vector<Miss> misses;
    misses.reserve(uniqueExperts.size());
    ++cache.clock;

    int16_t expertToAssignedSlot[kMaxExperts];
    std::fill_n(expertToAssignedSlot, kMaxExperts, -1);

    for (uint32_t exp : uniqueExperts) {
      int16_t slot = cache.expertToSlot[exp];
      if (slot != -1) {
        // Cache hit: refresh LRU timestamp
        cache.lruTime[slot] = cache.clock;
        expertToAssignedSlot[exp] = slot;
      } else {
        // Cache miss: find a slot
        uint32_t assignSlot = 0;
        if (cache.numCached < cache.capacity) {
          assignSlot = cache.numCached++;
        } else {
          // Evict LRU slot not in active step
          uint32_t oldest = UINT32_MAX;
          uint32_t best = 0;
          for (uint32_t s = 0; s < cache.capacity; ++s) {
            if (cache.lruTime[s] == cache.clock) continue;
            if (cache.lruTime[s] < oldest) {
              oldest = cache.lruTime[s];
              best = s;
            }
          }
          assignSlot = best;
          int16_t evicted = cache.slotToExpert[assignSlot];
          if (evicted != -1) {
            cache.expertToSlot[evicted] = -1;
          }
        }
        cache.slotToExpert[assignSlot] = exp;
        cache.expertToSlot[exp] = assignSlot;
        cache.lruTime[assignSlot] = cache.clock;
        expertToAssignedSlot[exp] = assignSlot;
        misses.push_back({exp, assignSlot});
      }
    }

    totalMisses += static_cast<uint32_t>(misses.size());

    if (!misses.empty()) {
      const auto &layer = weights.layers[layerIndex];
      const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
      char *cg = static_cast<char *>(cache.cacheGate.contents());
      char *cu = static_cast<char *>(cache.cacheUp.contents());
      char *cd = static_cast<char *>(cache.cacheDown.contents());

      // The expert regions are mapped for random access, so copying a missed
      // expert faults it in one 16 KB page at a time, each its own disk read.
      // Asking for every missed range first lets the reads go out together
      // and in large pieces; the copies below then find the pages in memory.
      readMissedExperts(layer.expertSource, misses.data(), misses.size(),
                        stride, cg, cu, cd);
    }

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && expertToAssignedSlot[exp] >= 0) {
          selPtr[r * routesPerRow + k] = expertToAssignedSlot[exp];
        }
      }
      selPtr[r * routesPerRow + expertsPerToken] = 512;
    }
    return true;
  };

  auto makeCacheWeights = [&](uint32_t layerIndex) -> ops::MoeWeights {
    const auto &layer = weights.layers[layerIndex];
    const auto &cache = layer.expertCache;
    const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
    return ops::MoeWeights{
        .router = layer.ffn.router,
        .expertGate = {cache.cacheGate, cache.capacity,
                       weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertUp = {cache.cacheUp, cache.capacity,
                     weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertDown = {cache.cacheDown, cache.capacity,
                       weights.layout.hiddenSize, weights.layout.expertIntermediateSize, stride},
        .sharedGate = layer.ffn.sharedGate,
        .sharedUp = layer.ffn.sharedUp,
        .sharedDown = layer.ffn.sharedDown,
        .sharedExpertGate = layer.ffn.sharedExpertGate,
    };
  };

  metal::CommandGraph residentGraph = std::move(graph);
  for (uint32_t layerIndex = 0; layerIndex < R; ++layerIndex) {
    encodeAttentionHC(residentGraph, layerIndex);
    metal::MetalBuffer mixerOut = encodeMixer(residentGraph, layerIndex, gdnIndex, attentionIndex);
    encodeMixerUpdate(residentGraph, layerIndex, mixerOut);
    encodeMlpHC(residentGraph, layerIndex);
    ops::MoE::add(
        residentGraph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
    encodeMlpUpdate(residentGraph, layerIndex);
    encodeCapture(residentGraph, layerIndex);
  }

  // Layer R base up to router
  encodeAttentionHC(residentGraph, R);
  metal::MetalBuffer mixerOutR = encodeMixer(residentGraph, R, gdnIndex, attentionIndex);
  encodeMixerUpdate(residentGraph, R, mixerOutR);
  encodeMlpHC(residentGraph, R);
  encodeMoERoute(residentGraph, R);

  // Submit residentGraph and prefetch streaming experts in background
  auto t0 = std::chrono::steady_clock::now();
  metal::CommandTicket residentTicket = backend.submitCommandAsync(residentGraph.dispatches());
  // No blanket read-ahead of the previous step's experts: after a long
  // prompt that is nearly every expert, ~68 GB the OS then reads in the
  // background for minutes, stalling decode. Misses are fetched on demand.
  (void)residentTicket.wait();
  auto t1 = std::chrono::steady_clock::now();
  double residentMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double totalStageMs = 0.0;
  double totalGpuMs = 0.0;
  double totalPureGpuMs = 0.0;

  // Loop through streaming layers R .. geometry.layers - 2
  for (uint32_t L = R; L < geometry.layers - 1; ++L) {
    auto ts0 = std::chrono::steady_clock::now();
    bool staged = stageActiveExperts(L);
    auto ts1 = std::chrono::steady_clock::now();
    totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

    ops::MoeWeights cacheW = staged ? makeCacheWeights(L) : weights.layers[L].ffn;

    metal::CommandGraph stepGraph;
    encodeMoEExecute(stepGraph, cacheW);
    encodeMlpUpdate(stepGraph, L);
    encodeCapture(stepGraph, L);

    // Layer L + 1 base up to router
    encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
    metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
    encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
    encodeMlpHC(stepGraph, L + 1);
    encodeMoERoute(stepGraph, L + 1);

    auto tg0 = std::chrono::steady_clock::now();
    metal::CommandTiming timing = backend.submitCommandAsync(stepGraph.dispatches()).wait();
    auto tg1 = std::chrono::steady_clock::now();
    totalGpuMs += std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    totalPureGpuMs += timing.gpuSeconds * 1000.0;
  }

  const uint32_t lastL = geometry.layers - 1;
  auto ts0 = std::chrono::steady_clock::now();
  bool stagedLast = stageActiveExperts(lastL);
  auto ts1 = std::chrono::steady_clock::now();
  totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  ops::MoeWeights cacheWLast = stagedLast ? makeCacheWeights(lastL) : weights.layers[lastL].ffn;

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE, MLP update, capture, and head into the caller's graph
  encodeMoEExecute(graph, cacheWLast);
  encodeMlpUpdate(graph, lastL);
  encodeCapture(graph, lastL);
  encodeHead(graph);

  static uint32_t verifyStepCount = 0;
  if (++verifyStepCount <= 10) {
    std::cerr << "[Verify Timing] Resident 0.." << R << ": " << residentMs << " ms | Staging: "
              << totalStageMs << " ms (misses: " << totalMisses << ") | GPU Wall: " << totalGpuMs << " ms (pure GPU: " << totalPureGpuMs << " ms)\n";
  }
}

} // namespace splash::model

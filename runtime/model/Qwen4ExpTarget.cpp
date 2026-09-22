#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/HyperConnection.h"
#include "metal/abi/MoE.h"
#include "metal/abi/PerLayerEmbedding.h"
#include "model/WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <random>
#include <cmath>
#include <map>
#include <deque>
#include <functional>
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
                  uint32_t liveRows,
                  std::vector<std::function<void()>> *deferred = nullptr) {
  if (onCpu) {
    auto gather = [&weights, &geometry, tokens = ple.tokens, state,
                   embedding = ple.embedding, rowBegin, rows, liveRows] {
      gatherNgramRowsOnCpu(
          weights, geometry,
          static_cast<const uint32_t *>(tokens.contents()) + rowBegin,
          static_cast<const uint8_t *>(state.contents()),
          static_cast<uint16_t *>(embedding.contents()) +
              uint64_t{rowBegin} * geometry.pleEmbeddingSize,
          rows, rows, liveRows);
    };
    // A pipelined step encodes every layer before its inputs exist; the
    // gather then runs between stages, once they do.
    if (deferred)
      deferred->push_back(std::move(gather));
    else
      gather();
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
                        metal::MetalBuffer injection, uint32_t rows,
                        uint32_t rowStep = 1) {
  const bool withInject = weights.blockInject.has_value();
  const HyperConnectionParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, withInject ? 1u : 0u, rowStep, 0};
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
  // One read per matrix. Splitting each into 4 smaller reads in flight
  // together was measured slower (staging 30 -> 39 ms a step).
  constexpr uint32_t kPieces = 1;
  const uint64_t piece = (stride + kPieces - 1) / kPieces;
  dispatch_apply(count * 3 * kPieces,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t index) {
    const Miss &miss = misses[index / (3 * kPieces)];
    const uint32_t matrix = static_cast<uint32_t>(index / kPieces % 3);
    const uint64_t begin = uint64_t(index % kPieces) * piece;
    const uint64_t length = std::min(piece, stride - std::min(stride, begin));
    if (!length)
      return;
    char *slot = (matrix == 0 ? gate : matrix == 1 ? up : down) +
                 uint64_t{miss.slot} * stride + begin;
    const uint64_t offset =
        (matrix == 0 ? source.gate : matrix == 1 ? source.up : source.down) +
        uint64_t{miss.expert} * stride + begin;
    uint64_t done = 0;
    while (done < length) {
      const ssize_t got = ::pread(source.fd, slot + done, length - done,
                                  static_cast<off_t>(offset + done));
      if (got <= 0)
        break; // Leaves the slot short; the file was validated at load.
      done += static_cast<uint64_t>(got);
    }
  });
}

// Top-k of one row of the router's bf16 scores, as moe_route_select picks
// them: highest first, ties to the lower expert id; weights are the softmax
// over just the k chosen scores.
void topExperts(const uint16_t *scores, uint32_t experts, uint32_t k,
                uint32_t *ids, float *weights) {
  auto widen = [](uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); };
  float chosen[16];
  for (uint32_t rank = 0; rank < k; ++rank) {
    float best = -INFINITY;
    uint32_t bestId = 0;
    for (uint32_t e = 0; e < experts; ++e) {
      bool taken = false;
      for (uint32_t r = 0; r < rank; ++r) taken |= ids[r] == e;
      const float v = widen(scores[e]);
      if (!taken && v > best) { best = v; bestId = e; }
    }
    ids[rank] = bestId;
    chosen[rank] = best;
  }
  if (!weights) return;
  float total = 0.0f;
  for (uint32_t r = 0; r < k; ++r) total += std::exp(chosen[r] - chosen[0]);
  for (uint32_t r = 0; r < k; ++r) weights[r] = std::exp(chosen[r] - chosen[0]) / total;
}

// sigmoid of the shared expert's scalar gate: a Q8 projection padded to 256
// outputs, of which output 0 is used (see moe_route_select_impl).
float sharedExpertGate(const ops::Q8Projection &gate, const uint16_t *input,
                       uint32_t size) {
  constexpr uint32_t kStorageN = 256;
  const auto *w = static_cast<const uint8_t *>(gate.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(gate.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(gate.biases.contents());
  auto widen = [](uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); };
  float total = 0.0f;
  for (uint32_t d = 0; d < size; ++d) {
    const uint32_t group = d / 64;
    const float value = float(w[uint64_t{group} * kStorageN * 64 + d % 64]) *
                            widen(scales[uint64_t{group} * kStorageN]) +
                        widen(biases[uint64_t{group} * kStorageN]);
    total += widen(input[d]) * value;
  }
  return 1.0f / (1.0f + std::exp(-total));
}

uint16_t toBf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

constexpr uint32_t kFrequencyHalfLife = 512;

// Counts a use of `expert`, halving every count each kFrequencyHalfLife steps.
void countExpertUse(Qwen4ExpLayerExpertCache &cache, uint32_t expert) {
  if (cache.frequency.empty())
    cache.frequency.assign(cache.expertToSlot.size(), 0);
  if (cache.clock >= cache.nextHalving) {
    for (auto &count : cache.frequency) count >>= 1;
    cache.nextHalving = cache.clock + kFrequencyHalfLife;
  }
  cache.frequency[expert] += 16;
}

// The slot to evict: least used, then least recent, never one this step uses.
int32_t pickVictim(const Qwen4ExpLayerExpertCache &cache) {
  int32_t best = -1;
  uint64_t bestKey = UINT64_MAX;
  for (uint32_t s = 0; s < cache.capacity; ++s) {
    if (cache.lruTime[s] == cache.clock) continue;
    const int16_t expert = cache.slotToExpert[s];
    const uint64_t count = expert >= 0 && !cache.frequency.empty()
                               ? cache.frequency[expert] : 0;
    const uint64_t key = (count << 32) | cache.lruTime[s];
    if (key < bestKey) { bestKey = key; best = static_cast<int32_t>(s); }
  }
  return best;
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
      geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};
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
      geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};

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
        attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
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
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE, MLP update, and capture into caller's graph
  encodeMoEExecute(graph, moeWLast);
  encodeMlpUpdate(graph, lastL);
  encodeCapture(graph, lastL);

  // MTP head: fill its attention cache for this chunk's positions. Row r
  // pairs the trunk's final residual at r with the prompt token at r + 1, so
  // every row but the chunk's last has its input (the last prompt row is
  // covered by the first decode step). Only the keys and values are needed:
  // combiner, attention mix, projection, store - no attention, experts or head.
  if (weights.mtpLayer && weights.mtpCombiner && rows > 1 &&
      !std::getenv("SPLASH_NO_MTP")) {
    const auto &head = *weights.mtpLayer;
    const auto &combiner = *weights.mtpCombiner;
    const uint32_t kvIndex = geometry.kvLayout.attentionLayers - 1;
    const uint32_t hidden = geometry.hiddenSize;
    const uint32_t count = geometry.hyperConnectionCount;
    const uint32_t m = rows - 1;
    const metal::MetalBuffer residual = buffers.hidden[geometry.layers & 1];
    const metal::MetalBuffer X = buffers.hidden[(geometry.layers & 1) ^ 1];
    if (!weights.mtpPrefillOnes) {
      weights.mtpPrefillOnes = backend.allocateBuffer(
          uint64_t{ExecutionLimits::prefillTokenBudget} * count * 2,
          metal::BufferStorage::Shared, "mtp-prefill-ones");
      std::fill_n(static_cast<uint16_t *>(weights.mtpPrefillOnes.contents()),
                  ExecutionLimits::prefillTokenBudget * count, uint16_t{0x3F80});
    }
    const ops::LinearMatrix square{hidden, hidden};
    // Prefill linear kernels read and write whole 32-row tiles, so views are
    // padded to one; the padding rows are scratch.
    auto padded = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                      uint32_t count, uint32_t width) {
      return u16(buffer, begin, (count + 31) / 32 * 32, width);
    };
    const HyperConnectionParams single{m, hidden, 1, geometry.hyperConnectionLowRank,
                                       1e-6f, 0, 1, 0};
    const HyperConnectionParams all{m, hidden, count, geometry.hyperConnectionLowRank,
                                    1e-6f, 1, 1, 0};
    // Tokens shifted by one row.
    metal::MetalBuffer next = backend.view(buffers.ple.tokens, 4, uint64_t{m} * 4);
    metal::MetalBuffer embedded = padded(buffers.recurrent, 0, m, hidden);
    metal::MetalBuffer normalizedE = padded(buffers.gdnHidden, 0, m, hidden);
    metal::MetalBuffer e = padded(buffers.attentionHidden, 0, m, hidden);
    ops::Embedding::add(graph, next, weights.tokenEmbedding, embedded, m);
    graph.add("hyper_connection_rms", {embedded, combiner.embeddingNorm, normalizedE},
              single, {m, 1, 1}, {256, 1, 1});
    operators.linear().addPrefillSums(graph, normalizedE, buffers.projectionSums, square, m);
    operators.linear().addPrefill(graph, normalizedE, combiner.fcEmbedding, e,
                                  buffers.projectionSums, square, m);
    graph.add("hyper_connection_rms", {residual, combiner.hiddenNorm, buffers.normalized},
              all, {m, count, 1}, {256, 1, 1});
    // fc_hidden on each stream: m rows of 4 streams, in prefill-sized pieces.
    const uint32_t streamRows = m * count;
    for (uint32_t first = 0; first < streamRows;
         first += ExecutionLimits::prefillTokenBudget) {
      const uint32_t piece = std::min(ExecutionLimits::prefillTokenBudget, streamRows - first);
      metal::MetalBuffer in = padded(buffers.normalized, first, piece, hidden);
      metal::MetalBuffer out = padded(X, first, piece, hidden);
      operators.linear().addPrefillSums(graph, in, buffers.projectionSums, square, piece);
      operators.linear().addPrefill(graph, in, combiner.fcHidden, out,
                                    buffers.projectionSums, square, piece);
    }
    graph.add("hyper_connection_update", {X, e, weights.mtpPrefillOnes}, all,
              {64, 1, 1}, {256, 1, 1});
    addHyperConnection(graph, geometry, X, head.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced, buffers.hyperMixed,
                       buffers.hyperInjection, m, 1);
    const auto &mixer = std::get<QwenAttentionWeights>(head.mixer);
    const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth, hidden};
    operators.linear().addPrefillSums(graph, buffers.hyperMixed, buffers.projectionSums,
                                      attentionInput, m);
    operators.linear().addPrefill(graph, buffers.hyperMixed, mixer.inputProjection,
                                  buffers.fullPacked, buffers.projectionSums,
                                  attentionInput, m);
    for (const QwenTargetPrefillSequence &sequence : sequences) {
      if (sequence.rows < 2)
        continue;
      const uint32_t kept = sequence.rows - 1;
      const uint64_t queryBytes = uint64_t{geometry.attentionQueryHeads} *
                                  sequence.attentionStride *
                                  geometry.attentionHeadDimension * 2;
      const uint64_t kvBytes = uint64_t{geometry.attentionKvHeads} *
                               sequence.attentionStride *
                               geometry.attentionHeadDimension * 2;
      metal::MetalBuffer queries = backend.view(buffers.fullQueries, sequence.queryOffset, queryBytes);
      metal::MetalBuffer keys = backend.view(buffers.chunkKeys, sequence.kvOffset, kvBytes);
      metal::MetalBuffer values = backend.view(buffers.chunkValues, sequence.kvOffset, kvBytes);
      ops::PagedAttention::addPrefillProjection(
          graph, u16(buffers.fullPacked, sequence.rowBegin, kept, geometry.packedAttentionWidth),
          mixer.queryNorm, mixer.keyNorm,
          f32(buffers.ropeCos, sequence.rowBegin, kept, geometry.rotaryPairs),
          f32(buffers.ropeSin, sequence.rowBegin, kept, geometry.rotaryPairs),
          queries, keys, values, kept, sequence.attentionStride,
          sequence.attentionStride, geometry.attentionQueryHeads, geometry.kvLayout);
      kv::Q8ChunkedPrefillParams store = sequence.q8;
      store.chunk_tokens = kept;
      ops::PagedAttention::addPrefillStore(graph, kvLayers[kvIndex], keys, values,
                                           sequence.pageTable, store, geometry.kvLayout);
    }
  }

  std::cerr << "[Prefill Timing] Resident 0.." << R << ": " << residentMs << " ms | Streaming Staging: "
            << totalStageMs << " ms (misses: " << totalMisses << ") | Streaming GPU: " << totalGpuMs << " ms\n";

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
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

  // With one live row per lane, hyper-connections run on those rows only:
  // lane rows 0, 8, 16, ... The other rows' results are never kept.
  // One lane: its live rows are rows 0..live-1. Several lanes with one live
  // row each: rows 0, 8, 16, ...
  // Recomputed after the MTP head drafts, since it may guess fewer rows.
  uint32_t hcRows = 0, hcStep = 1;
  HyperConnectionParams hcParams{};
  auto setLiveRows = [&] {
    const uint32_t live = buffers.liveRowsPerLane;
    hcRows = lanes == 1 ? live : (live == 1 ? lanes : rows);
    hcStep = lanes != 1 && live == 1 ? ExecutionLimits::targetVerifyRows : 1;
    hcParams = {hcRows, geometry.hiddenSize, geometry.hyperConnectionCount,
                geometry.hyperConnectionLowRank, 1e-6f, 1, hcStep, 0};
  };
  setLiveRows();

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  // Set while a pipelined step is being encoded; see the staged loop.
  std::vector<std::function<void()>> *pipelineDeferred = nullptr;
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
                   lane * laneRows, laneRows, onCpu, buffers.liveRowsPerLane,
                   pipelineDeferred);
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
                       buffers.hyperMixed, buffers.hyperInjection, hcRows,
                       hcStep);
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
                       buffers.hyperMixed, buffers.hyperInjection, hcRows,
                       hcStep);
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

  // Lookahead: layer `target`'s router applied to the input the previous
  // layer's router just read. Only a guess, used to start reads early.
  auto encodePredictRoute = [&](metal::CommandGraph &g, uint32_t target) {
    if (!weights.predictSelected) {
      weights.predictSelected = backend.allocateBuffer(
          buffers.selectedExperts.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-selected");
      weights.predictWeights = backend.allocateBuffer(
          buffers.routingWeights.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-weights");
      weights.predictScratch = backend.allocateBuffer(
          buffers.groupedInput.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-scratch");
    }
    // Scores only; the host picks the likely experts from them.
    ops::MoE::addRouteScores(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         weights.predictSelected, weights.predictWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, weights.predictScratch, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[target].ffn, moePlan);
  };

  auto encodeMoEExecute = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse,
                              bool hostGrouped = false) {
    ops::MoE::addExecute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, moePlan, /*addResidual=*/false, hostGrouped);
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
                       buffers.hyperReduced, buffers.finalHidden, {}, hcRows,
                       hcStep);
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
        attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
      throw std::logic_error("Qwen target layer partition mismatch");
    }
    encodeHead(graph);
    return;
  }

  // =========================================================================
  // Staged Streaming Execution Path with Active Expert Caching & Speculation
  // =========================================================================

  uint32_t totalMisses = 0;

  // Layer `geometry.layers` is the MTP head's decoder layer.
  auto layerRef = [&](uint32_t index) -> const Qwen4ExpLayerWeights & {
    return index < geometry.layers ? weights.layers[index] : *weights.mtpLayer;
  };
  auto stageActiveExperts = [&](uint32_t layerIndex) -> bool {
    auto &cache = layerRef(layerIndex).expertCache;
    // Predicted experts may still be loading; their slots are already claimed.
    if (cache.pending) {
      dispatch_group_wait(cache.inflight, DISPATCH_TIME_FOREVER);
      cache.pending = false;
    }
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

    for (uint32_t exp : uniqueExperts)
      countExpertUse(cache, exp);
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
          // Evict the least-used slot this step does not need.
          const int32_t victim = pickVictim(cache);
          assignSlot = victim >= 0 ? static_cast<uint32_t>(victim) : 0;
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
      const auto &layer = layerRef(layerIndex);
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

  // Start background reads for the experts lookahead predicts layer `layer`
  // will want. Claims least-recently-used slots of that layer only; nothing
  // on the GPU reads that layer's cache until its own stage.
  auto prefetchPredicted = [&](uint32_t layer) {
    if (!weights.predictSelected || layer >= geometry.layers)
      return;
    auto &cache = weights.layers[layer].expertCache;
    const auto *predictedScores =
        static_cast<const uint16_t *>(weights.predictScratch.contents());
    const uint32_t perToken = weights.layout.expertsPerToken;
    const uint32_t width = moePlan.shape().routerWidth();
    struct Miss { uint32_t expert; uint32_t slot; };
    std::vector<Miss> misses;
    ++cache.clock;
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % ExecutionLimits::targetVerifyRows >= buffers.liveRowsPerLane)
        continue;
      // The router's own top-k. Prefetching 14 instead cut misses 23 -> 20
      // a step but raised staging 15.6 -> 19.8 ms: the extra reads compete
      // with the ones that matter. SPLASH_LOOKAHEAD_EXPERTS overrides.
      static const uint32_t widened = [] {
        const char *value = std::getenv("SPLASH_LOOKAHEAD_EXPERTS");
        return value ? static_cast<uint32_t>(std::atoi(value)) : 10u;
      }();
      const uint32_t guesses = std::clamp(widened, perToken, 16u);
      uint32_t ids[16];
      topExperts(predictedScores + uint64_t{r} * width, weights.layout.experts,
                 guesses, ids, nullptr);
      for (uint32_t k = 0; k < guesses; ++k) {
        const uint32_t expert = ids[k];
        if (expert >= weights.layout.experts)
          continue;
        const int16_t existing = cache.expertToSlot[expert];
        if (existing >= 0) {
          cache.lruTime[existing] = cache.clock;
          continue;
        }
        uint32_t slot = 0;
        if (cache.numCached < cache.capacity) {
          slot = cache.numCached++;
        } else {
          const int32_t victim = pickVictim(cache);
          if (victim < 0)
            break;
          slot = static_cast<uint32_t>(victim);
          if (cache.slotToExpert[slot] >= 0)
            cache.expertToSlot[cache.slotToExpert[slot]] = -1;
        }
        cache.slotToExpert[slot] = static_cast<int16_t>(expert);
        cache.expertToSlot[expert] = static_cast<int16_t>(slot);
        cache.lruTime[slot] = cache.clock;
        misses.push_back({expert, slot});
      }
    }
    if (misses.empty())
      return;
    weights.predictIssued += misses.size();
    if (!cache.inflight)
      cache.inflight = dispatch_group_create();
    const auto &source = weights.layers[layer].expertSource;
    const uint64_t stride = weights.layers[layer].ffn.expertGate.expertStrideBytes;
    char *gate = static_cast<char *>(cache.cacheGate.contents());
    char *up = static_cast<char *>(cache.cacheUp.contents());
    char *down = static_cast<char *>(cache.cacheDown.contents());
    auto owned = std::make_shared<std::vector<Miss>>(std::move(misses));
    dispatch_group_async(cache.inflight,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      readMissedExperts(source, owned->data(), owned->size(), stride, gate, up, down);
    });
    cache.pending = true;
  };

  // Host routing for a pipelined step. The GPU stage ends at the router's
  // scores; the host, already paused there to stage experts, selects and
  // groups. Two single-threadgroup kernels that each cost ~0.1-0.2 ms of
  // latency per layer, a third of decode GPU time, become microseconds here.
  auto encodeRouteScores = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer scores) {
    ops::MoE::addRouteScores(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, std::move(scores), buffers.expertIntermediate,
         buffers.expertOutput},
        layerRef(layerIndex).ffn, moePlan);
  };
  const uint32_t hostRoutesPerRow = moePlan.shape().routesPerToken();
  auto isLive = [&](uint32_t row) {
    return row % ExecutionLimits::targetVerifyRows < buffers.liveRowsPerLane;
  };
  auto hostSelect = [&](uint32_t layerIndex) {
    const auto *scores = static_cast<const uint16_t *>(buffers.groupedInput.contents());
    const auto *input = static_cast<const uint16_t *>(buffers.hyperMixed.contents());
    auto *selected = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    auto *routing = static_cast<uint16_t *>(buffers.routingWeights.contents());
    const uint32_t k = weights.layout.expertsPerToken;
    const uint32_t width = moePlan.shape().routerWidth();
    for (uint32_t r = 0; r < rows; ++r) {
      if (!isLive(r)) continue;
      uint32_t ids[16];
      float w[16];
      topExperts(scores + uint64_t{r} * width, weights.layout.experts, k, ids, w);
      for (uint32_t j = 0; j < k; ++j) {
        selected[r * hostRoutesPerRow + j] = ids[j];
        routing[r * hostRoutesPerRow + j] = toBf16(w[j]);
      }
      selected[r * hostRoutesPerRow + k] = weights.layout.experts;
      routing[r * hostRoutesPerRow + k] = toBf16(sharedExpertGate(
          layerRef(layerIndex).ffn.sharedExpertGate,
          input + uint64_t{r} * geometry.hiddenSize, geometry.hiddenSize));
    }
  };
  // As moe_group_routes lays it out: routed tiles in ascending slot order,
  // tile_rows each, padding marked ~0; then the shared expert's tiles. Rows
  // that cannot be kept get no routes at all.
  auto hostGroup = [&]() {
    const auto *selected = static_cast<const uint32_t *>(buffers.selectedExperts.contents());
    auto *tiles = static_cast<MoeTileDescriptor *>(buffers.tileDescriptors.contents());
    auto *tileCount = static_cast<uint32_t *>(buffers.tileCount.contents());
    auto *grouped = static_cast<uint32_t *>(buffers.groupedRoutes.contents());
    auto *routeRows = static_cast<uint32_t *>(buffers.routeRows.contents());
    const uint32_t k = weights.layout.expertsPerToken;
    const uint32_t tileRows = moePlan.tileRows();
    const uint32_t experts = moePlan.shape().experts;
    std::vector<std::vector<uint32_t>> byExpert(experts);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t j = 0; j <= k; ++j) {
        const uint32_t route = r * hostRoutesPerRow + j;
        routeRows[route] = 0xFFFFFFFFu;
        if (!isLive(r) || j == k) continue;
        const uint32_t e = selected[route];
        if (e < experts) byExpert[e].push_back(route);
      }
    uint32_t tile = 0;
    for (uint32_t e = 0; e < experts; ++e) {
      const auto &routes = byExpert[e];
      for (uint32_t first = 0; first < routes.size(); first += tileRows) {
        const uint32_t count = std::min<uint32_t>(tileRows, routes.size() - first);
        tiles[tile] = MoeTileDescriptor{e, count};
        for (uint32_t i = 0; i < tileRows; ++i) {
          const uint32_t row = tile * tileRows + i;
          if (i < count) {
            grouped[row] = routes[first + i];
            routeRows[routes[first + i]] = row;
          } else {
            grouped[row] = 0xFFFFFFFFu;
          }
        }
        ++tile;
      }
    }
    std::vector<uint32_t> shared;
    for (uint32_t r = 0; r < rows; ++r)
      if (isLive(r)) shared.push_back(r * hostRoutesPerRow + k);
    for (uint32_t first = 0; first < shared.size(); first += tileRows) {
      const uint32_t count = std::min<uint32_t>(tileRows, shared.size() - first);
      tiles[tile] = MoeTileDescriptor{experts, count};
      for (uint32_t i = 0; i < tileRows; ++i) {
        const uint32_t row = tile * tileRows + i;
        if (i < count) {
          grouped[row] = shared[first + i];
          routeRows[shared[first + i]] = row;
        } else {
          grouped[row] = 0xFFFFFFFFu;
        }
      }
      ++tile;
    }
    *tileCount = tile;
  };

  auto makeCacheWeights = [&](uint32_t layerIndex) -> ops::MoeWeights {
    const auto &layer = layerRef(layerIndex);
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


  // -------------------------------------------------------------------------
  // MTP draft head (lane 0). Runs before this step's verify: its inputs are
  // the previous step's final residual rows, still in Hidden0. Each draft
  // step is two GPU stages around a host pause for the head's experts:
  //   A  combiner, the head's attention layer (13th KV layer), MLP mix, router
  //   B  experts, update, the head's own mixer, the shared output head
  // In shadow mode its guesses are only scored against what the target then
  // produces; nothing is proposed.
  // -------------------------------------------------------------------------
  uint32_t drafted = 0;
  std::array<std::array<uint32_t, 16>, 3> draftCandidates{};
  std::array<std::array<float, 16>, 3> draftProbabilities{};
  auto runMtpDraft = [&]() -> std::array<uint32_t, 3> {
    std::array<uint32_t, 3> drafts{};
    const QwenMtpLane &mtp = buffers.mtp[0];
    const auto &head = *weights.mtpLayer;
    const auto &combiner = *weights.mtpCombiner;
    const uint32_t mtpIndex = geometry.layers;
    const uint32_t kvIndex = geometry.kvLayout.attentionLayers - 1;
    constexpr uint32_t kRows = ExecutionLimits::targetVerifyRows;
    const uint32_t hidden = geometry.hiddenSize, width = geometry.residualWidth();
    const uint32_t vocabulary = geometry.vocabularySize;
    auto shared = [&](metal::MetalBuffer &b, uint64_t bytes, const char *label) {
      if (!b) b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    };
    shared(weights.mtpTokens, kRows * 4, "mtp-tokens");
    shared(weights.mtpEmbed, kRows * hidden * 2, "mtp-embed");
    shared(weights.mtpNorm, kRows * hidden * 2, "mtp-norm");
    shared(weights.mtpE, kRows * hidden * 2, "mtp-e");
    shared(weights.mtpOnes, kRows * 4 * 2, "mtp-ones");
    shared(weights.mtpCos, kRows * geometry.rotaryPairs * 4, "mtp-cos");
    shared(weights.mtpSin, kRows * geometry.rotaryPairs * 4, "mtp-sin");
    shared(weights.mtpHin, kRows * uint64_t{width} * 2, "mtp-h-in");
    std::fill_n(static_cast<uint16_t *>(weights.mtpOnes.contents()), kRows * 4,
                uint16_t{0x3F80});  // bf16 1.0
    auto *hIn = static_cast<uint16_t *>(weights.mtpHin.contents());
    const auto *hidden0 = static_cast<const uint16_t *>(buffers.hidden[0].contents());
    metal::MetalBuffer X = buffers.hidden[1], Y = buffers.hidden[0];
    const HyperConnectionParams all{kRows, hidden, geometry.hyperConnectionCount,
                                    geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};
    const HyperConnectionParams single{kRows, hidden, 1,
                                       geometry.hyperConnectionLowRank, 1e-6f, 0, 1, 0};
    const ops::LinearMatrix square{hidden, hidden};
    const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth, hidden};
    const ops::LinearMatrix mixerOutput{hidden, geometry.attentionWidth};

    // One draft step over `live` rows at positions position..position+live-1,
    // whose inputs are rows 0..live-1 of mtpHin and `tokens`. Returns the
    // head's top token after the last row.
    float lastConfidence = 1.0f;
    std::array<uint32_t, 16> lastCandidates{};
    std::array<float, 16> lastProbabilities{};
    auto step = [&](uint64_t position, uint32_t live,
                    const uint32_t *tokens) -> uint32_t {
      auto *tokenOut = static_cast<uint32_t *>(weights.mtpTokens.contents());
      auto *cosines = static_cast<float *>(weights.mtpCos.contents());
      auto *sines = static_cast<float *>(weights.mtpSin.contents());
      for (uint32_t r = 0; r < kRows; ++r) {
        tokenOut[r] = tokens[std::min(r, live - 1)];
        for (uint32_t d = 0; d < geometry.rotaryPairs; ++d) {
          const float frequency = std::pow(geometry.rotaryTheta,
                                           -float(d) / float(geometry.rotaryPairs));
          const float angle = float(position + r) * frequency;
          cosines[r * geometry.rotaryPairs + d] = std::cos(angle);
          sines[r * geometry.rotaryPairs + d] = std::sin(angle);
        }
      }
      std::array<kv::Q8ChunkedPrefillParams, ExecutionLimits::maximumBatchWidth> q8m{};
      std::array<kv::Q8VerifyAttentionParams, ExecutionLimits::maximumBatchWidth> vm{};
      std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
      for (uint32_t lane = 0; lane < ExecutionLimits::maximumBatchWidth; ++lane) {
        q8m[lane] = ops::PagedAttention::prefillParams(
            position, kRows, kv::kPageTokens, mtp.pageTable, buffers.kvPageCount);
        vm[lane] = kv::q8VerifyAttentionParams(
            q8m[lane].committed_tokens, q8m[lane].chunk_tokens,
            q8m[lane].chunk_stride, q8m[lane].page_table_entries,
            q8m[lane].physical_page_count);
        histories[lane] = q8m[lane].committed_tokens;
      }
      const auto plan = operators.verifyAttention(
          lanes, geometry.attentionQueryHeads, geometry.kvLayout, histories);

      metal::CommandGraph a;
      ops::Embedding::add(a, weights.mtpTokens, weights.tokenEmbedding,
                          weights.mtpEmbed, kRows);
      a.add("hyper_connection_rms", {weights.mtpEmbed, combiner.embeddingNorm,
                                     weights.mtpNorm},
            single, {kRows, 1, 1}, {256, 1, 1});
      operators.linear().addDecodeBatch(a, weights.mtpNorm, combiner.fcEmbedding,
                                        weights.mtpE, square, lanes, stats);
      a.add("hyper_connection_rms", {weights.mtpHin, combiner.hiddenNorm,
                                     buffers.normalized},
            all, {kRows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
      // fc_hidden on each stream: 8 rows of 4 streams are 32 rows of hidden.
      operators.linear().addDecodeBatch(a, buffers.normalized, combiner.fcHidden,
                                        X, square, 4 * lanes, stats);
      a.add("hyper_connection_update", {X, weights.mtpE, weights.mtpOnes}, all,
            {64, 1, 1}, {256, 1, 1});
      addHyperConnection(a, geometry, X, head.attentionHyperConnection,
                         buffers.normalized, buffers.hyperReduced,
                         buffers.hyperMixed, buffers.hyperInjection, kRows, 1);
      const auto &mixer = std::get<QwenAttentionWeights>(head.mixer);
      operators.linear().addDecodeBatch(a, buffers.hyperMixed, mixer.inputProjection,
                                        buffers.fullPacked, attentionInput, lanes, stats);
      ops::PagedAttention::addVerifyProjection(
          a, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm, weights.mtpCos,
          weights.mtpSin, buffers.fullQueries, buffers.chunkKeys[kvIndex],
          buffers.chunkValues[kvIndex], kRows, kv::kPageTokens, kv::kPageTokens,
          geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      ops::PagedAttention::addVerify(
          a, kvLayers[kvIndex],
          {buffers.chunkKeys[kvIndex], buffers.chunkValues[kvIndex],
           buffers.fullQueries, buffers.attentionPartials,
           buffers.attentionStatistics, buffers.fullAttention, buffers.pageTables},
          q8m, vm, plan);
      ops::PagedAttention::addVerifyGate(
          a, buffers.fullPacked, buffers.fullAttention, buffers.attentionHidden,
          kRows, kv::kPageTokens, kv::kPageTokens, geometry.attentionQueryHeads,
          geometry.kvLayout, lanes);
      operators.linear().addDecodeBatch(a, buffers.attentionHidden,
                                        mixer.outputProjection,
                                        buffers.attentionOutput, mixerOutput, lanes, stats);
      a.add("hyper_connection_update", {X, buffers.attentionOutput,
                                        buffers.hyperInjection},
            all, {64, 1, 1}, {256, 1, 1});
      addHyperConnection(a, geometry, X, head.mlpHyperConnection,
                         buffers.normalized, buffers.hyperReduced,
                         buffers.hyperMixed, buffers.hyperInjection, kRows, 1);
      encodeRouteScores(a, mtpIndex, buffers.groupedInput);
      (void)backend.submitCommand(a.dispatches());

      const uint32_t saved = buffers.liveRowsPerLane;
      buffers.liveRowsPerLane = live;
      hostSelect(mtpIndex);
      if (!stageActiveExperts(mtpIndex))
        throw std::logic_error("MTP head needs more experts than its cache holds");
      hostGroup();
      buffers.liveRowsPerLane = saved;

      metal::CommandGraph b;
      encodeMoEExecute(b, makeCacheWeights(mtpIndex), /*hostGrouped=*/true);
      b.add("hyper_connection_update_out", {X, Y, buffers.gdnOutput,
                                            buffers.hyperInjection},
            all, {64, 1, 1}, {256, 1, 1});
      addHyperConnection(b, geometry, Y, combiner.mixer, buffers.normalized,
                         buffers.hyperReduced, buffers.finalHidden, {}, kRows, 1);
      operators.linear().addDecodeBatch(b, buffers.finalHidden,
                                        weights.logitsProjection, buffers.logits,
                                        {vocabulary, hidden}, lanes, stats);
      (void)backend.submitCommand(b.dispatches());

      const auto *logits = static_cast<const uint16_t *>(buffers.logits.contents()) +
                           uint64_t{live - 1} * vocabulary;
      auto logit = [&](uint32_t v) {
        return std::bit_cast<float>(uint32_t{logits[v]} << 16);
      };
      // Greedy: the top token, reported as certain. Sampled: the request's
      // own sampling applied to the head's logits - top-k (at most 16, the
      // acceptance's candidate width), temperature, top-p - and the guess
      // drawn from it; speculative sampling then keeps it with probability
      // min(1, p/q), so the output is still exactly the target's.
      const uint32_t width16 = mtp.temperature > 0.0f
          ? std::min<uint32_t>(mtp.topK ? mtp.topK : 16, 16) : 1;
      std::array<uint32_t, 16> ids{};
      std::array<float, 16> values{};
      uint32_t filled = 0;
      for (uint32_t v = 0; v < vocabulary; ++v) {
        const float value = logit(v);
        if (filled < width16) {
          uint32_t at = filled++;
          while (at > 0 && values[at - 1] < value) {
            ids[at] = ids[at - 1]; values[at] = values[at - 1]; --at;
          }
          ids[at] = v; values[at] = value;
        } else if (value > values[width16 - 1]) {
          uint32_t at = width16 - 1;
          while (at > 0 && values[at - 1] < value) {
            ids[at] = ids[at - 1]; values[at] = values[at - 1]; --at;
          }
          ids[at] = v; values[at] = value;
        }
      }
      // Confidence: the head's own probability for its top token.
      float total = 0.0f;
      for (uint32_t v = 0; v < vocabulary; ++v) {
        const float value = logit(v);
        if (value > values[0] - 16.0f) total += std::exp(value - values[0]);
      }
      lastConfidence = 1.0f / total;
      lastCandidates.fill(UINT32_MAX);
      lastProbabilities.fill(0.0f);
      if (mtp.temperature <= 0.0f) {
        lastCandidates[0] = ids[0];
        lastProbabilities[0] = 1.0f;
        return ids[0];
      }
      std::array<float, 16> q{};
      float sum = 0.0f;
      for (uint32_t i = 0; i < width16; ++i) {
        q[i] = std::exp((values[i] - values[0]) / mtp.temperature);
        sum += q[i];
      }
      uint32_t kept = width16;
      float cumulative = 0.0f;
      for (uint32_t i = 0; i < width16; ++i) {
        cumulative += q[i] / sum;
        if (cumulative >= mtp.topP) { kept = i + 1; break; }
      }
      float keptSum = 0.0f;
      for (uint32_t i = 0; i < kept; ++i) keptSum += q[i];
      static std::mt19937 random(20260922);
      for (uint32_t i = 0; i < kept; ++i) {
        lastCandidates[i] = ids[i];
        lastProbabilities[i] = q[i] / keptSum;
      }
      float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(random);
      for (uint32_t i = 0; i + 1 < kept; ++i) {
        if (draw < lastProbabilities[i]) return ids[i];
        draw -= lastProbabilities[i];
      }
      return ids[kept - 1];
    };

    // Step 1: refresh the accepted rows with the target's own residuals, and
    // guess the token after the anchor.
    for (uint32_t r = 0; r < mtp.rows; ++r)
      std::memcpy(hIn + uint64_t{r} * width,
                  hidden0 + uint64_t{mtp.firstRow + r} * width, width * 2);
    // Guessing stops once the head is unsure (as llama.cpp's
    // --spec-draft-p-min): an unlikely guess is usually rejected, and every
    // guessed row costs the verifier its own experts.
    static const float confident = [] {
      const char *value = std::getenv("SPLASH_MTP_P_MIN");
      return value ? static_cast<float>(std::atof(value)) : 0.3f;
    }();
    drafted = 0;
    drafts[0] = step(mtp.firstPosition, mtp.rows, mtp.tokens.data());
    draftCandidates[0] = lastCandidates;
    draftProbabilities[0] = lastProbabilities;
    if (lastConfidence < confident)
      return drafts;
    drafted = 1;
    // Steps 2 and 3 chain on the head's own residual. Each re-runs the rows
    // before it (same inputs, same keys) so its attention sees them.
    const uint64_t anchorPosition = mtp.firstPosition + mtp.rows;
    std::vector<uint16_t> chain(3 * uint64_t{width});
    const auto *y = static_cast<const uint16_t *>(Y.contents());
    std::memcpy(chain.data(), y + uint64_t{mtp.rows - 1} * width, width * 2);
    for (uint32_t k = 1; k < 3; ++k) {
      for (uint32_t r = 0; r < k; ++r)
        std::memcpy(hIn + uint64_t{r} * width, chain.data() + uint64_t{r} * width, width * 2);
      drafts[k] = step(anchorPosition - 1 + 1, k, drafts.data());
      draftCandidates[k] = lastCandidates;
      draftProbabilities[k] = lastProbabilities;
      if (lastConfidence < confident)
        return drafts;
      drafted = k + 1;
      std::memcpy(chain.data() + uint64_t{k} * width, y + uint64_t{k - 1} * width, width * 2);
    }
    return drafts;
  };

  double mtpMs = 0.0;
  if (buffers.mtpEnabled && weights.mtpLayer && lanes == 1 && buffers.mtp[0].rows) {
    const auto mtpStart = std::chrono::steady_clock::now();
    const auto drafts = runMtpDraft();
    mtpMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mtpStart).count();
    if (!buffers.mtpShadow) {
      buffers.liveRowsPerLane = 1 + drafted;
      setLiveRows();
      if (buffers.mtpProposedOut)
        *buffers.mtpProposedOut = drafted;
      // The verify rows are the anchor then these proposals. Past the third
      // the proposals repeat it: the retained-row cap keeps them from counting.
      auto *proposed = static_cast<uint32_t *>(buffers.proposedTokens.contents());
      auto *candidates = static_cast<uint32_t *>(buffers.proposalCandidates.contents());
      auto *probabilities = static_cast<float *>(buffers.proposalProbabilities.contents());
      for (uint32_t k = 0; k < ExecutionLimits::draftProposalTokens; ++k) {
        const uint32_t source = std::min<uint32_t>(k, 2);
        proposed[k] = drafts[source];
        // The distribution each guess was drawn from (one point if greedy).
        for (uint32_t c = 0; c < 16; ++c) {
          candidates[k * 16 + c] = draftCandidates[source][c];
          probabilities[k * 16 + c] = draftProbabilities[source][c];
        }
      }
    }
    // Score each guess when the token it guessed is decided. The anchor of
    // this step is the target's token at position firstPosition + rows.
    static std::map<uint64_t, std::array<uint32_t, 3>> guesses;
    static std::array<uint64_t, 3> hits{}, total{};
    const QwenMtpLane &mtp = buffers.mtp[0];
    const uint64_t anchorPosition = mtp.firstPosition + mtp.rows;
    const uint32_t anchor = mtp.tokens[mtp.rows - 1];
    if (auto found = guesses.find(anchorPosition); found != guesses.end()) {
      for (uint32_t k = 0; k < 3; ++k)
        if (found->second[k] != UINT32_MAX) {
          ++total[k];
          hits[k] += found->second[k] == anchor;
        }
      guesses.erase(guesses.begin(), std::next(found));
    }
    for (uint32_t k = 0; k < 3; ++k) {
      auto &slot = guesses[anchorPosition + 1 + k];
      if (slot[0] == 0 && slot[1] == 0 && slot[2] == 0) slot = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
      slot[k] = drafts[k];
    }
    if (total[0] && total[0] % 16 == 0)
      std::cerr << "[MTP shadow] guess 1: " << hits[0] << "/" << total[0]
                << "  guess 2: " << hits[1] << "/" << total[1]
                << "  guess 3: " << hits[2] << "/" << total[2] << "\n";
  }

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

  const bool pipelined = !backend.dispatchProfiling() &&
                         !std::getenv("SPLASH_NO_PIPELINE") &&
                         weights.layers[R].expertCache.capacity >=
                             rows * weights.layout.expertsPerToken;
  if (pipelined) {
    const bool lookahead = !std::getenv("SPLASH_NO_LOOKAHEAD");
    // Every layer is encoded and committed up front. Stage k runs layer
    // R+k-1's experts and layer R+k up to its router, then raises the
    // pipeline event; the host stages layer R+k's experts (and any deferred
    // work) and signals the GPU on. Nothing waits for a submission; the GPU
    // waits only for experts that were actually missing.
    std::vector<metal::ComputeDispatch> all;
    std::vector<size_t> starts;
    std::vector<std::vector<std::function<void()>>> deferredPerStage;
    // Dispatches point into their graph's parameter storage, so every stage
    // graph must outlive the submission.
    std::deque<metal::CommandGraph> stageGraphs;
    auto append = [&](metal::CommandGraph &g) {
      starts.push_back(all.size());
      for (auto &d : g.dispatches()) all.push_back(d);
    };
    // Stage 0 has already been submitted and completed above (resident
    // graph), so stages here start at layer R's experts.
    for (uint32_t L = R; L < geometry.layers - 1; ++L) {
      deferredPerStage.emplace_back();
      pipelineDeferred = &deferredPerStage.back();
      metal::CommandGraph &stepGraph = stageGraphs.emplace_back();
      encodeMoEExecute(stepGraph, makeCacheWeights(L), /*hostGrouped=*/true);
      encodeMlpUpdate(stepGraph, L);
      encodeCapture(stepGraph, L);
      encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
      metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
      encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
      encodeMlpHC(stepGraph, L + 1);
      encodeRouteScores(stepGraph, L + 1, buffers.groupedInput);
      if (lookahead && L + 2 < geometry.layers)
        encodePredictRoute(stepGraph, L + 2);
      append(stepGraph);
      pipelineDeferred = nullptr;
    }
    // The first stage's host work (layer R's experts) happens before commit:
    // its router result is already in memory. Shift so stage 0 needs none.
    auto ts0 = std::chrono::steady_clock::now();
    if (!stageActiveExperts(R))
      throw std::logic_error("pipelined decode found more experts than the cache holds");
    hostGroup();
    for (auto &work : deferredPerStage[0]) work();
    totalStageMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ts0).count();
    const uint32_t stages = static_cast<uint32_t>(starts.size());
    const uint64_t base = backend.reservePipelineEvents(stages);
    auto tg0 = std::chrono::steady_clock::now();
    metal::CommandTicket ticket = backend.submitPipelineAsync(all, starts, base);
    for (uint32_t k = 1; k < stages; ++k) {
      if (!backend.waitPipelineEvent(base + 2 * k - 1, 60000))
        throw std::runtime_error("pipelined decode stage timed out");
      auto ts = std::chrono::steady_clock::now();
      const uint32_t layer = R + k;
      hostSelect(layer);
      if (!stageActiveExperts(layer))
        throw std::logic_error("pipelined decode found more experts than the cache holds");
      hostGroup();
      // Stage k-1 also predicted layer + 1; start those reads now so they
      // land while stage k runs.
      if (lookahead)
        prefetchPredicted(layer + 1);
      for (auto &work : deferredPerStage[k]) work();
      totalStageMs += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - ts).count();
      backend.signalPipelineEvent(base + 2 * k);
    }
    metal::CommandTiming timing = ticket.wait();
    totalGpuMs += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - tg0).count();
    totalPureGpuMs += timing.gpuSeconds * 1000.0;
  } else {
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
  }

  const uint32_t lastL = geometry.layers - 1;
  auto ts0 = std::chrono::steady_clock::now();
  // The pipeline's last stage stopped at the last layer's router scores.
  if (pipelined)
    hostSelect(lastL);
  bool stagedLast = stageActiveExperts(lastL);
  if (pipelined)
    hostGroup();
  auto ts1 = std::chrono::steady_clock::now();
  totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  ops::MoeWeights cacheWLast = stagedLast ? makeCacheWeights(lastL) : weights.layers[lastL].ffn;

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE, MLP update, capture, and head into the caller's graph
  encodeMoEExecute(graph, cacheWLast, /*hostGrouped=*/pipelined);
  encodeMlpUpdate(graph, lastL);
  encodeCapture(graph, lastL);
  encodeHead(graph);

  static uint32_t verifyStepCount = 0;
  static const bool everyStep = std::getenv("SPLASH_STEP_TIMING") != nullptr;
  if (++verifyStepCount <= 10 || everyStep) {
    std::cerr << "[Verify Timing] Resident 0.." << R << ": " << residentMs << " ms | Staging: "
              << totalStageMs << " ms (misses: " << totalMisses << ") | GPU Wall: " << totalGpuMs << " ms (pure GPU: " << totalPureGpuMs << " ms) | MTP: " << mtpMs << " ms\n";
  }
}

} // namespace splash::model

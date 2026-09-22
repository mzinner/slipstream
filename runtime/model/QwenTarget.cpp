#include "model/QwenTarget.hpp"

#include "model/Qwen4Exp.hpp"
#include "model/Qwen4ExpTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  QwenTargetGeometry result;
  result.maximumContextTokens = layout.maximumContextTokens;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedAttentionWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.kvLayout = layout.q8Layout();
  result.stateLayout = layout.gdnStateLayout();
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.ffnKind = QwenFfnKind::Dense;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen4ExpLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.moe = {layout.hiddenSize, layout.experts, layout.expertsPerToken,
                layout.expertIntermediateSize, layout.expertStorageN};
  result.ffnKind = QwenFfnKind::SparseMoe;
  result.hyperConnectionCount = layout.hyperConnectionCount;
  result.hyperConnectionLowRank = layout.hyperConnectionLowRank;
  result.pleLayer = layout.ngramLayer;
  result.pleEmbeddingSize = layout.ngramEmbeddingSize;
  result.pleHistoryRows = layout.pleConvolutionState();
  // The reference's end-of-sequence token, which both fills a fresh n-gram
  // history and restarts the window: config eos_token_id, the first stop.
  result.pleEndToken = layout.stopTokens[0];
  result.extraKvLayers = layout.mtpLayers;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.moe = {layout.hiddenSize, layout.experts, layout.expertsPerToken,
                layout.expertIntermediateSize};
  result.ffnKind = QwenFfnKind::SparseMoe;
  return result;
}

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers + geometry.extraKvLayers !=
          geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

template <class Layer>
constexpr bool hasDenseFfn = requires(const Layer &layer) {
  layer.gateProjection;
  layer.upProjection;
  layer.downProjection;
};

template <class Mixer>
constexpr bool isGdnMixer =
    std::is_same_v<std::remove_cvref_t<Mixer>, QwenGdnWeights>;

} // namespace

QwenMixerWeights readQwenMixer(WeightFile &file, metal::MetalBackend &backend,
                               const QwenMixerGeometry &geometry,
                               bool fullAttention, bool eightBit) {
  constexpr uint64_t kFloat32Bytes = 4;
  auto projection = [&](auto &q4, auto &q8, uint32_t out, uint32_t in,
                        const char *label) {
    if (eightBit)
      q8 = readQ8Projection(file, backend, out, in, label);
    else
      q4 = readQ4Projection(file, backend, out, in, label);
  };
  if (fullAttention) {
    QwenAttentionWeights attention;
    projection(attention.inputProjection, attention.inputProjectionQ8,
               geometry.packedAttentionWidth, geometry.hiddenSize,
               "attention-input");
    const uint64_t headNormBytes = checkedWeightMultiply(
        geometry.attentionHeadDimension, kBFloat16Bytes, "head norm bytes");
    attention.queryNorm = file.section(headNormBytes, "query-norm");
    attention.keyNorm = file.section(headNormBytes, "key-norm");
    projection(attention.outputProjection, attention.outputProjectionQ8,
               geometry.hiddenSize, geometry.attentionWidth,
               "attention-output");
    return attention;
  }
  QwenGdnWeights gdn;
  projection(gdn.inputProjection, gdn.inputProjectionQ8, geometry.packedGdnWidth,
             geometry.hiddenSize, "gdn-input");
  gdn.convolutionWeights = file.section(
      checkedWeightMultiply(
          checkedWeightMultiply(geometry.convolutionDimension, kGdnConvolutionTaps,
                                "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(checkedWeightMultiply(geometry.gdnValueHeads,
                                                 kFloat32Bytes,
                                                 "GDN decay bytes"),
                           "gdn-decay");
  gdn.timeBias = file.section(
      checkedWeightMultiply(geometry.gdnValueHeads, kBFloat16Bytes,
                            "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = file.section(
      checkedWeightMultiply(geometry.gdnHeadDimension, kBFloat16Bytes,
                            "GDN norm bytes"),
      "gdn-norm");
  projection(gdn.outputProjection, gdn.outputProjectionQ8, geometry.hiddenSize,
             geometry.attentionWidth, "gdn-output");
  return gdn;
}

QwenTarget::QwenTarget(const Qwen3_8Weights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(const Qwen3_6MoeWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(const Qwen4ExpWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
  embeddingScratch_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::prefillTokenBudget} * geometry_.hiddenSize * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-embedding-scratch");
  headNormalized_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::targetVerifyRows} * geometry_.residualWidth() * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-head-normalized");
  headReduced_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::targetVerifyRows} * geometry_.hyperConnectionLowRank * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-head-reduced");
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &weights) {
  return geometryFor(weights.layout);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &weights) {
  return geometryFor(weights.layout);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights) {
  return geometryFor(weights.layout);
}

const ops::Q4Projection *QwenTarget::vocabularyProjection() const noexcept {
  return std::visit([](const auto *weights) -> const ops::Q4Projection * {
    if constexpr (std::is_same_v<decltype(weights->logitsProjection),
                                 const ops::Q4Projection>)
      return &weights->logitsProjection;
    else
      return nullptr;  // an 8-bit head; only placeholder drafts pair with it
  }, weights_);
}

void QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) const {
  std::visit(
      [&](const auto *weights) {
        using W = std::remove_cvref_t<decltype(*weights)>;
        if constexpr (std::is_same_v<W, Qwen4ExpWeights>) {
          Qwen4ExpTarget::addPrefill(*weights, geometry_, backend_, operators_,
                                     graph, std::move(buffers), sequences, rows,
                                     kvLayers);
        } else {
          addPrefillImpl(*weights, graph, std::move(buffers), sequences, rows,
                         kvLayers);
        }
      },
      weights_);
}

template <class Weights>
void QwenTarget::addPrefillImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth,
                                     geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth,
                                           geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize,
                                        geometry_.attentionWidth};
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moePrefill(geometry_.moe, rows);
    return std::nullopt;
  }();

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRmsWithQ4Sums(
        graph, input, layer.inputNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            operators_.linear().addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.gdnPacked,
                           buffers.projectionSums, gdnInput, rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              ops::GDN::addPrefill(
                  graph,
                  {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                       geometry_.packedGdnWidth),
                   mixer.convolutionWeights, sequence.convolutionIn[gdnIndex],
                   sequence.convolutionOut[gdnIndex],
                   u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.decay, mixer.timeBias,
                   f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnIndex],
                   sequence.recurrentOut[gdnIndex],
                   u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.mixerNorm,
                   u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows);
            }
            operators_.linear().addPrefillSums(graph, buffers.gdnHidden,
                               buffers.projectionSums, mixerOutput, rows);
            operators_.linear().addPrefillResidual(
                graph, buffers.gdnHidden, mixer.outputProjection, input,
                buffers.gdnOutput, buffers.projectionSums, mixerOutput,
                rows);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            operators_.linear().addPrefill(graph, buffers.normalized,
                           mixer.inputProjection, buffers.fullPacked,
                           buffers.projectionSums, attentionInput, rows);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              const uint64_t queryBytes =
                  uint64_t{geometry_.attentionQueryHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              const uint64_t kvBytes =
                  uint64_t{geometry_.attentionKvHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              metal::MetalBuffer queries = backend_.view(
                  buffers.fullQueries, sequence.queryOffset, queryBytes);
              metal::MetalBuffer attentionRows = backend_.view(
                  buffers.fullAttention, sequence.queryOffset, queryBytes);
              metal::MetalBuffer keys = backend_.view(
                  buffers.chunkKeys, sequence.kvOffset, kvBytes);
              metal::MetalBuffer values = backend_.view(
                  buffers.chunkValues, sequence.kvOffset, kvBytes);
              ops::PagedAttention::addPrefillProjection(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  mixer.queryNorm, mixer.keyNorm,
                  f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  queries, keys, values, sequence.rows,
                  sequence.attentionStride, sequence.attentionStride,
                  geometry_.attentionQueryHeads, geometry_.kvLayout);
              ops::PagedAttention::addPrefillStore(
                  graph, kvLayers[attentionIndex], keys, values,
                  sequence.pageTable, sequence.q8, geometry_.kvLayout);
              ops::PagedAttention::addPrefill(
                  graph, kvLayers[attentionIndex], queries, attentionRows,
                  buffers.attentionPartials, buffers.attentionStatistics,
                  sequence.pageTable, sequence.q8,
                  operators_.prefillAttention(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.kvLayout, sequence.q8.committed_tokens));
              ops::PagedAttention::addPrefillGate(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  attentionRows,
                  u16(buffers.attentionHidden, sequence.rowBegin,
                      sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride,
                  sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
            }
            operators_.linear().addPrefillSums(graph, buffers.attentionHidden,
                               buffers.projectionSums, mixerOutput, rows);
            operators_.linear().addPrefillResidual(
                graph, buffers.attentionHidden, mixer.outputProjection, input,
                buffers.attentionOutput, buffers.projectionSums, mixerOutput,
                rows);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRmsWithQ4Sums(
        graph, residual, layer.postAttentionNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);
    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize,
                                   geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize,
                                     geometry_.denseIntermediateSize};
      operators_.linear().addPrefill(graph, buffers.normalized, layer.gateProjection,
                     buffers.denseGateScratch, buffers.projectionSums, up,
                     rows);
      operators_.linear().addPrefillUpWithGate(
          graph, buffers.normalized, layer.upProjection,
          buffers.denseGateScratch, buffers.denseIntermediate,
          buffers.projectionSums, buffers.downProjectionSums, up, rows);
      operators_.linear().addPrefillResidual(
          graph, buffers.denseIntermediate, layer.downProjection, residual,
          output, buffers.downProjectionSums, down, rows);
    } else {
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      const uint32_t slot =
          static_cast<uint32_t>(captured - captureLayers.begin());
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        for (uint32_t index = 0; index < sequence.captureCount; ++index) {
          const QwenTargetPrefillCapture &capture = sequence.captures[index];
          ops::DraftAttention::captureTargetHidden(
              graph, output, buffers.captured, capture.rows, slot,
              capture.sourceStart, capture.destinationStart,
              geometry_.hiddenSize, geometry_.capturedHiddenSize());
        }
      }
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  std::visit(
      [&](const auto *weights) {
        using W = std::remove_cvref_t<decltype(*weights)>;
        if constexpr (std::is_same_v<W, Qwen4ExpWeights>) {
          Qwen4ExpTarget::addVerify(*weights, geometry_, backend_, operators_,
                                    graph, std::move(buffers), kvLayers, q8,
                                    verify, lanes, stats);
        } else {
          addVerifyImpl(*weights, graph, std::move(buffers), kvLayers, q8,
                        verify, lanes, stats);
        }
      },
      weights_);
}

template <class Weights>
void QwenTarget::addVerifyImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators_.verifyAttention(
      lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth, geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth, geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize, geometry_.attentionWidth};
  const auto moePlan = [&]() -> std::optional<ops::MoePlan> {
    if constexpr (!hasDenseFfn<typename std::remove_cvref_t<decltype(weights.layers)>::value_type>)
      return operators_.moeDecode(geometry_.moe, lanes);
    return std::nullopt;
  }();
  constexpr uint32_t tileRows = kv::kPageTokens;

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRms(graph, input, layer.inputNorm,
                               buffers.normalized, geometry_.hiddenSize, rows);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            operators_.linear().addDecodeBatch(graph,
                               buffers.normalized, mixer.inputProjection,
                               buffers.gdnPacked[gdnIndex], gdnInput, lanes,
                               stats);
            ops::GDN::addDecode(
                graph,
                {buffers.gdnPacked[gdnIndex], mixer.convolutionWeights,
                 buffers.currentGdnStates, buffers.nextGdnStates,
                 buffers.gdnMixed[gdnIndex], mixer.decay, mixer.timeBias,
                 buffers.gdnDecay[gdnIndex], buffers.gdnBeta[gdnIndex],
                 buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
                 buffers.arrived, buffers.generation},
                geometry_.gdnShape(), lanes, gdnIndex,
                {geometry_.stateLayout.convolutionLayerBytes(),
                 geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()});
            operators_.linear().addResidualBatch(
                graph, buffers.gdnHidden,
                mixer.outputProjection, input, buffers.gdnOutput, mixerOutput,
                lanes, stats);
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            operators_.linear().addDecodeBatch(graph,
                               buffers.normalized, mixer.inputProjection,
                               buffers.fullPacked, attentionInput, lanes,
                               stats);
            ops::PagedAttention::addVerifyProjection(
                graph, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
                buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
                buffers.chunkKeys[attentionIndex],
                buffers.chunkValues[attentionIndex],
                ExecutionLimits::targetVerifyRows, tileRows, tileRows,
                geometry_.attentionQueryHeads, geometry_.kvLayout, lanes);
            ops::PagedAttention::addVerify(
                graph, kvLayers[attentionIndex],
                {buffers.chunkKeys[attentionIndex],
                 buffers.chunkValues[attentionIndex], buffers.fullQueries,
                 buffers.attentionPartials, buffers.attentionStatistics,
                 buffers.fullAttention, buffers.pageTables},
                q8, verify, attentionPlan);
            ops::PagedAttention::addVerifyGate(
                graph, buffers.fullPacked, buffers.fullAttention,
                buffers.attentionHidden, ExecutionLimits::targetVerifyRows,
                tileRows, tileRows, geometry_.attentionQueryHeads,
                geometry_.kvLayout, lanes);
            operators_.linear().addResidualBatch(
                graph, buffers.attentionHidden,
                mixer.outputProjection, input, buffers.attentionOutput,
                mixerOutput, lanes, stats);
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRms(graph, residual, layer.postAttentionNorm,
                               buffers.normalized, geometry_.hiddenSize, rows);
    if constexpr (hasDenseFfn<std::remove_cvref_t<decltype(layer)>>) {
      const ops::LinearMatrix up{geometry_.denseIntermediateSize, geometry_.hiddenSize};
      const ops::LinearMatrix down{geometry_.hiddenSize, geometry_.denseIntermediateSize};
      operators_.linear().addGateUpBatch(graph, buffers.normalized, layer.gateProjection,
                         layer.upProjection, buffers.denseGateScratch,
                         buffers.denseIntermediate, up, lanes, stats);
      operators_.linear().addResidualBatch(
          graph, buffers.denseIntermediate,
          layer.downProjection, residual, output, down, lanes, stats);
    } else {
      ops::MoE::add(
          graph,
          {buffers.normalized, residual, output, buffers.selectedExperts,
           buffers.routingWeights, buffers.tileDescriptors, buffers.tileCount,
           buffers.groupedRoutes, buffers.routeRows, buffers.groupedInput,
           buffers.expertIntermediate, buffers.expertOutput},
          layer.ffn, *moePlan);
    }

    const auto captureLayers = geometry_.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      ops::DraftAttention::captureTargetHidden(
          graph, output, buffers.capturedTargetHidden, rows,
          static_cast<uint32_t>(captured - captureLayers.begin()), 0, 0,
          geometry_.hiddenSize, geometry_.capturedHiddenSize());
    }
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  ops::Normalization::addRms(graph, buffers.hidden[geometry_.layers & 1],
                             std::visit([](const auto *value) {
                               return value->finalNorm;
                             }, weights_),
                             buffers.finalHidden, geometry_.hiddenSize, rows);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  operators_.linear().addDecodeBatch(graph, buffers.finalHidden,
                     *vocabularyProjection(), buffers.logits, head, lanes,
                     stats);
}

void QwenTarget::addHead(metal::CommandGraph &graph,
                         metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden,
                         metal::MetalBuffer logits,
                         uint32_t normalizedRows) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  if (std::holds_alternative<const Qwen4ExpWeights *>(weights_)) {
    const auto *exp = std::get<const Qwen4ExpWeights *>(weights_);
    Qwen4ExpTarget::addHead(*exp, geometry_, operators_, graph,
                            std::move(hidden), std::move(finalHidden),
                            std::move(logits), headNormalized_, headReduced_,
                            normalizedRows);
    return;
  }
  const metal::MetalBuffer norm = std::visit(
      [](const auto *weights) { return weights->finalNorm; }, weights_);
  ops::Normalization::addRms(graph, std::move(hidden), norm, finalHidden,
                             geometry_.hiddenSize, normalizedRows);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  operators_.linear().addDecode(graph,
                std::move(finalHidden), *vocabularyProjection(),
                std::move(logits), head);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  if (std::holds_alternative<const Qwen4ExpWeights *>(weights_)) {
    const auto *exp = std::get<const Qwen4ExpWeights *>(weights_);
    Qwen4ExpTarget::addEmbedding(*exp, geometry_, graph, std::move(tokens),
                                 std::move(hidden), embeddingScratch_, rows);
    return;
  }
  std::visit(
      [&](const auto *weights) {
        ops::Embedding::add(graph, std::move(tokens), weights->tokenEmbedding,
                            std::move(hidden), rows);
      },
      weights_);
}

void QwenTarget::addDraftEmbedding(metal::CommandGraph &graph,
                                   metal::MetalBuffer tokens,
                                   metal::MetalBuffer hidden,
                                   uint32_t rows) const {
  if (std::holds_alternative<const Qwen4ExpWeights *>(weights_)) {
    const auto *exp = std::get<const Qwen4ExpWeights *>(weights_);
    ops::Embedding::add(graph, std::move(tokens), exp->tokenEmbedding,
                        std::move(hidden), rows);
    return;
  }
  addEmbedding(graph, std::move(tokens), std::move(hidden), rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (geometry_.hasPerLayerEmbedding())
    Qwen4ExpTarget::addPerLayerEmbeddingCommit(geometry_, backend_, graph,
                                               buffers, lanes);
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

} // namespace splash::model

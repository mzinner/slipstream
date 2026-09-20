#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/HyperConnection.h"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::model {

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
      geometry.hyperConnectionLowRank, 1e-6f};
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
  const HyperConnectionParams params{
      normalizedRows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f};

  // 1. Normalize residual -> normalized, reduced
  graph.add("hyper_connection_normalize",
            {std::move(hidden), weights.hyperConnectionMixer.norm,
             weights.hyperConnectionMixer.mixDown, headNormalized, headReduced},
            params, {normalizedRows, 1, 1}, {256, 1, 1});

  // 2. Mix without injection -> finalHidden
  graph.add("hyper_connection_mix_no_inject",
            {std::move(headNormalized), std::move(headReduced),
             weights.hyperConnectionMixer.mixUp, finalHidden},
            params, {normalizedRows, 1, 1}, {256, 1, 1});

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
      geometry.hyperConnectionLowRank, 1e-6f};

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];

    // 1. Attention Hyper-connection
    graph.add("hyper_connection_normalize",
              {input, layer.attentionHyperConnection.norm,
               layer.attentionHyperConnection.mixDown, buffers.normalized,
               buffers.hyperReduced},
              hcParams, {rows, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_mix",
              {buffers.normalized, buffers.hyperReduced,
               layer.attentionHyperConnection.mixUp,
               *layer.attentionHyperConnection.blockInject,
               buffers.hyperMixed, buffers.hyperInjection},
              hcParams, {rows, 1, 1}, {256, 1, 1});

    // 2. Mixer block (GDN or Full Attention)
    metal::MetalBuffer mixerOutBuffer;
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addPrefillSums(graph, buffers.hyperMixed,
                                        buffers.projectionSums, gdnInput, rows);
      operators.linear().addPrefill(graph, buffers.hyperMixed,
                                    mixer.inputProjection, buffers.gdnPacked,
                                    buffers.projectionSums, gdnInput, rows);
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        ops::GDN::addPrefill(
            graph,
            {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                 geometry.packedGdnWidth),
             mixer.convolutionWeights, sequence.convolutionIn[gdnIndex],
             sequence.convolutionOut[gdnIndex],
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
             sequence.recurrentIn[gdnIndex],
             sequence.recurrentOut[gdnIndex],
             u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth),
             mixer.mixerNorm,
             u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth)},
            geometry.gdnShape(), sequence.rows);
      }
      operators.linear().addPrefillSums(graph, buffers.gdnHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(graph, buffers.gdnHidden,
                                    mixer.outputProjection, buffers.gdnOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      mixerOutBuffer = buffers.gdnOutput;
      ++gdnIndex;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      operators.linear().addPrefillSums(graph, buffers.hyperMixed,
                                        buffers.projectionSums, attentionInput,
                                        rows);
      operators.linear().addPrefill(graph, buffers.hyperMixed,
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
            graph,
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
            graph, kvLayers[attentionIndex], keys, values,
            sequence.pageTable, sequence.q8, geometry.kvLayout);
        ops::PagedAttention::addPrefill(
            graph, kvLayers[attentionIndex], queries, attentionRows,
            buffers.attentionPartials, buffers.attentionStatistics,
            sequence.pageTable, sequence.q8,
            operators.prefillAttention(
                sequence.rows, geometry.attentionQueryHeads,
                geometry.kvLayout, sequence.q8.committed_tokens));
        ops::PagedAttention::addPrefillGate(
            graph,
            u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                geometry.packedAttentionWidth),
            attentionRows,
            u16(buffers.attentionHidden, sequence.rowBegin, sequence.rows,
                geometry.attentionWidth),
            sequence.rows, sequence.attentionStride, sequence.attentionStride,
            geometry.attentionQueryHeads, geometry.kvLayout);
      }
      operators.linear().addPrefillSums(graph, buffers.attentionHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(graph, buffers.attentionHidden,
                                    mixer.outputProjection,
                                    buffers.attentionOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      mixerOutBuffer = buffers.attentionOutput;
      ++attentionIndex;
    }

    // 3. Residual update after mixer: input += injection * mixerOutBuffer
    graph.add("hyper_connection_update",
              {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
              {64, 1, 1}, {256, 1, 1});

    // 4. MLP Hyper-connection
    graph.add("hyper_connection_normalize",
              {input, layer.mlpHyperConnection.norm,
               layer.mlpHyperConnection.mixDown, buffers.normalized,
               buffers.hyperReduced},
              hcParams, {rows, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_mix",
              {buffers.normalized, buffers.hyperReduced,
               layer.mlpHyperConnection.mixUp,
               *layer.mlpHyperConnection.blockInject,
               buffers.hyperMixed, buffers.hyperInjection},
              hcParams, {rows, 1, 1}, {256, 1, 1});

    // 5. MoE block: input is buffers.hyperMixed, output is buffers.gdnOutput, addResidual=false
    ops::MoE::add(
        graph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan, /*addResidual=*/false);

    // 6. Residual update after MLP: output = input + injection * moeOutput
    graph.add("hyper_connection_update_out",
              {input, output, buffers.gdnOutput, buffers.hyperInjection},
              hcParams, {64, 1, 1}, {256, 1, 1});

    // 7. Target hidden capture (for draft, if configured)
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
              graph, buffers.gdnOutput, buffers.captured, capture.rows, slot,
              capture.sourceStart, capture.destinationStart,
              geometry.hiddenSize, geometry.capturedHiddenSize());
        }
      }
    }
  }

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
  static_cast<void>(backend);
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
      geometry.hyperConnectionLowRank, 1e-6f};

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];

    // 1. Attention Hyper-connection
    graph.add("hyper_connection_normalize",
              {input, layer.attentionHyperConnection.norm,
               layer.attentionHyperConnection.mixDown, buffers.normalized,
               buffers.hyperReduced},
              hcParams, {rows, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_mix",
              {buffers.normalized, buffers.hyperReduced,
               layer.attentionHyperConnection.mixUp,
               *layer.attentionHyperConnection.blockInject,
               buffers.hyperMixed, buffers.hyperInjection},
              hcParams, {rows, 1, 1}, {256, 1, 1});

    // 2. Mixer
    metal::MetalBuffer mixerOutBuffer;
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addDecodeBatch(
          graph, buffers.hyperMixed, mixer.inputProjection,
          buffers.gdnPacked[gdnIndex], gdnInput, lanes, stats);
      ops::GDN::addDecode(
          graph,
          {buffers.gdnPacked[gdnIndex], mixer.convolutionWeights,
           buffers.currentGdnStates, buffers.nextGdnStates,
           buffers.gdnMixed[gdnIndex], mixer.decay, mixer.timeBias,
           buffers.gdnDecay[gdnIndex], buffers.gdnBeta[gdnIndex],
           buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
           buffers.arrived, buffers.generation},
          geometry.gdnShape(), lanes, gdnIndex,
          {geometry.stateLayout.convolutionLayerBytes(),
           geometry.stateLayout.recurrentLayerBytes(),
           geometry.stateLayout.convolutionBytes()});
      operators.linear().addDecodeBatch(graph, buffers.gdnHidden,
                                        mixer.outputProjection,
                                        buffers.gdnOutput, mixerOutput, lanes,
                                        stats);
      mixerOutBuffer = buffers.gdnOutput;
      ++gdnIndex;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      const uint32_t tileRows = kv::kPageTokens;
      operators.linear().addDecodeBatch(
          graph, buffers.hyperMixed, mixer.inputProjection, buffers.fullPacked,
          attentionInput, lanes, stats);
      ops::PagedAttention::addVerifyProjection(
          graph, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
          buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
          buffers.chunkKeys[attentionIndex],
          buffers.chunkValues[attentionIndex],
          ExecutionLimits::targetVerifyRows, tileRows, tileRows,
          geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      ops::PagedAttention::addVerify(
          graph, kvLayers[attentionIndex],
          {buffers.chunkKeys[attentionIndex],
           buffers.chunkValues[attentionIndex], buffers.fullQueries,
           buffers.attentionPartials, buffers.attentionStatistics,
           buffers.fullAttention, buffers.pageTables},
          q8, verify, attentionPlan);
      ops::PagedAttention::addVerifyGate(
          graph, buffers.fullPacked, buffers.fullAttention,
          buffers.attentionHidden, ExecutionLimits::targetVerifyRows, tileRows,
          tileRows, geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      operators.linear().addDecodeBatch(graph, buffers.attentionHidden,
                                        mixer.outputProjection,
                                        buffers.attentionOutput, mixerOutput,
                                        lanes, stats);
      mixerOutBuffer = buffers.attentionOutput;
      ++attentionIndex;
    }

    // 3. Residual update after mixer: input += injection * mixerOutBuffer
    graph.add("hyper_connection_update",
              {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
              {64, 1, 1}, {256, 1, 1});

    // 4. MLP Hyper-connection
    graph.add("hyper_connection_normalize",
              {input, layer.mlpHyperConnection.norm,
               layer.mlpHyperConnection.mixDown, buffers.normalized,
               buffers.hyperReduced},
              hcParams, {rows, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_mix",
              {buffers.normalized, buffers.hyperReduced,
               layer.mlpHyperConnection.mixUp,
               *layer.mlpHyperConnection.blockInject,
               buffers.hyperMixed, buffers.hyperInjection},
              hcParams, {rows, 1, 1}, {256, 1, 1});

    // 5. MoE block: input is buffers.hyperMixed, output is buffers.gdnOutput, addResidual=false
    ops::MoE::add(
        graph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan, /*addResidual=*/false);

    // 6. Residual update after MLP: output = input + injection * moeOutput
    graph.add("hyper_connection_update_out",
              {input, output, buffers.gdnOutput, buffers.hyperInjection},
              hcParams, {64, 1, 1}, {256, 1, 1});

    // 7. Target hidden capture
    const auto captureLayers = geometry.captureLayers();
    const auto captured =
        std::find(captureLayers.begin(), captureLayers.end(), layerIndex);
    if (captured != captureLayers.end()) {
      ops::DraftAttention::captureTargetHidden(
          graph, buffers.gdnOutput, buffers.capturedTargetHidden, rows,
          static_cast<uint32_t>(captured - captureLayers.begin()), 0, 0,
          geometry.hiddenSize, geometry.capturedHiddenSize());
    }
  }

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Head at the end of verify
  // buffers.hidden[geometry.layers & 1] holds the final 10240 residual!
  graph.add("hyper_connection_normalize",
            {buffers.hidden[geometry.layers & 1],
             weights.hyperConnectionMixer.norm,
             weights.hyperConnectionMixer.mixDown, buffers.normalized,
             buffers.hyperReduced},
            hcParams, {rows, 1, 1}, {256, 1, 1});
  graph.add("hyper_connection_mix_no_inject",
            {buffers.normalized, buffers.hyperReduced,
             weights.hyperConnectionMixer.mixUp, buffers.finalHidden},
            hcParams, {rows, 1, 1}, {256, 1, 1});

  const ops::LinearMatrix head{geometry.vocabularySize, geometry.hiddenSize};
  operators.linear().addDecodeBatch(graph, buffers.finalHidden,
                                    weights.logitsProjection, buffers.logits,
                                    head, lanes, stats);
}

} // namespace splash::model

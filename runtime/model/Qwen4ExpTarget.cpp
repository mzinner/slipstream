#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/HyperConnection.h"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <dispatch/dispatch.h>
#include <iostream>
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

  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_normalize",
          {input, layer.attentionHyperConnection.norm,
           layer.attentionHyperConnection.mixDown, buffers.normalized,
           buffers.hyperReduced},
          hcParams, {rows, 1, 1}, {256, 1, 1});
    g.add("hyper_connection_mix",
          {buffers.normalized, buffers.hyperReduced,
           layer.attentionHyperConnection.mixUp,
           *layer.attentionHyperConnection.blockInject,
           buffers.hyperMixed, buffers.hyperInjection},
          hcParams, {rows, 1, 1}, {256, 1, 1});
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
            geometry.gdnShape(), sequence.rows);
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
    g.add("hyper_connection_normalize",
          {input, layer.mlpHyperConnection.norm,
           layer.mlpHyperConnection.mixDown, buffers.normalized,
           buffers.hyperReduced},
          hcParams, {rows, 1, 1}, {256, 1, 1});
    g.add("hyper_connection_mix",
          {buffers.normalized, buffers.hyperReduced,
           layer.mlpHyperConnection.mixUp,
           *layer.mlpHyperConnection.blockInject,
           buffers.hyperMixed, buffers.hyperInjection},
          hcParams, {rows, 1, 1}, {256, 1, 1});
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
      const char *fg = static_cast<const char *>(layer.ffn.expertGate.packed.contents());
      const char *fu = static_cast<const char *>(layer.ffn.expertUp.packed.contents());
      const char *fd = static_cast<const char *>(layer.ffn.expertDown.packed.contents());

      if (misses.size() <= 2) {
        for (const auto &m : misses) {
          uint32_t exp = m.expert;
          uint32_t slot = m.slot;
          std::memcpy(cg + uint64_t{slot} * stride, fg + uint64_t{exp} * stride, stride);
          std::memcpy(cu + uint64_t{slot} * stride, fu + uint64_t{exp} * stride, stride);
          std::memcpy(cd + uint64_t{slot} * stride, fd + uint64_t{exp} * stride, stride);
        }
      } else {
        dispatch_apply(misses.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
          uint32_t exp = misses[i].expert;
          uint32_t slot = misses[i].slot;
          std::memcpy(cg + uint64_t{slot} * stride, fg + uint64_t{exp} * stride, stride);
          std::memcpy(cu + uint64_t{slot} * stride, fu + uint64_t{exp} * stride, stride);
          std::memcpy(cd + uint64_t{slot} * stride, fd + uint64_t{exp} * stride, stride);
        });
      }
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
  weights.prefetchStreamingExperts();
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
    encodeAttentionHC(stepGraph, L + 1);
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

  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_normalize",
          {input, layer.attentionHyperConnection.norm,
           layer.attentionHyperConnection.mixDown, buffers.normalized,
           buffers.hyperReduced},
          hcParams, {rows, 1, 1}, {256, 1, 1});
    g.add("hyper_connection_mix",
          {buffers.normalized, buffers.hyperReduced,
           layer.attentionHyperConnection.mixUp,
           *layer.attentionHyperConnection.blockInject,
           buffers.hyperMixed, buffers.hyperInjection},
          hcParams, {rows, 1, 1}, {256, 1, 1});
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
          geometry.gdnShape(), lanes, gdnIdx,
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
    g.add("hyper_connection_normalize",
          {input, layer.mlpHyperConnection.norm,
           layer.mlpHyperConnection.mixDown, buffers.normalized,
           buffers.hyperReduced},
          hcParams, {rows, 1, 1}, {256, 1, 1});
    g.add("hyper_connection_mix",
          {buffers.normalized, buffers.hyperReduced,
           layer.mlpHyperConnection.mixUp,
           *layer.mlpHyperConnection.blockInject,
           buffers.hyperMixed, buffers.hyperInjection},
          hcParams, {rows, 1, 1}, {256, 1, 1});
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
    g.add("hyper_connection_normalize",
          {buffers.hidden[geometry.layers & 1],
           weights.hyperConnectionMixer.norm,
           weights.hyperConnectionMixer.mixDown, buffers.normalized,
           buffers.hyperReduced},
          hcParams, {rows, 1, 1}, {256, 1, 1});
    g.add("hyper_connection_mix_no_inject",
          {buffers.normalized, buffers.hyperReduced,
           weights.hyperConnectionMixer.mixUp, buffers.finalHidden},
          hcParams, {rows, 1, 1}, {256, 1, 1});
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
      const char *fg = static_cast<const char *>(layer.ffn.expertGate.packed.contents());
      const char *fu = static_cast<const char *>(layer.ffn.expertUp.packed.contents());
      const char *fd = static_cast<const char *>(layer.ffn.expertDown.packed.contents());

      if (misses.size() <= 2) {
        for (const auto &m : misses) {
          uint32_t exp = m.expert;
          uint32_t slot = m.slot;
          std::memcpy(cg + uint64_t{slot} * stride, fg + uint64_t{exp} * stride, stride);
          std::memcpy(cu + uint64_t{slot} * stride, fu + uint64_t{exp} * stride, stride);
          std::memcpy(cd + uint64_t{slot} * stride, fd + uint64_t{exp} * stride, stride);
        }
      } else {
        dispatch_apply(misses.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
          uint32_t exp = misses[i].expert;
          uint32_t slot = misses[i].slot;
          std::memcpy(cg + uint64_t{slot} * stride, fg + uint64_t{exp} * stride, stride);
          std::memcpy(cu + uint64_t{slot} * stride, fu + uint64_t{exp} * stride, stride);
          std::memcpy(cd + uint64_t{slot} * stride, fd + uint64_t{exp} * stride, stride);
        });
      }
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
  weights.prefetchStreamingExperts();
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
    encodeAttentionHC(stepGraph, L + 1);
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

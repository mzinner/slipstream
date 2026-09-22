#pragma once

#include "model/Qwen4Exp.hpp"
#include "model/QwenTarget.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <span>

namespace splash::model {

// How many tokens the MTP head guesses per step at most (SPLASH_MTP_DRAFTS,
// 1..7; default 3).
[[nodiscard]] uint32_t mtpDraftLimit() noexcept;

class Qwen4ExpTarget final {
public:
  static void addPrefill(
      const Qwen4ExpWeights &weights,
      const QwenTargetGeometry &geometry,
      metal::MetalBackend &backend,
      const ops::ExecutionPlans &operators,
      metal::CommandGraph &graph,
      QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences,
      uint32_t rows,
      std::span<const kv::Q8LayerStorage> kvLayers);

  static void addVerify(
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
      ops::Q4DispatchStats &stats);

  static void addHead(
      const Qwen4ExpWeights &weights,
      const QwenTargetGeometry &geometry,
      const ops::ExecutionPlans &operators,
      metal::CommandGraph &graph,
      metal::MetalBuffer hidden,
      metal::MetalBuffer finalHidden,
      metal::MetalBuffer logits,
      metal::MetalBuffer headNormalized,
      metal::MetalBuffer headReduced,
      uint32_t normalizedRows);

  // Writes the per-layer embedding history for the rows the verifier kept.
  static void addPerLayerEmbeddingCommit(
      const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
      metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers,
      uint32_t lanes);

  static void addEmbedding(
      const Qwen4ExpWeights &weights,
      const QwenTargetGeometry &geometry,
      metal::CommandGraph &graph,
      metal::MetalBuffer tokens,
      metal::MetalBuffer hidden,
      metal::MetalBuffer scratch,
      uint32_t rows);
};

} // namespace splash::model

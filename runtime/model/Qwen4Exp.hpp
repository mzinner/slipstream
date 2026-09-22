#pragma once

#include "QwenTarget.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// Qwen3.8-Flash-Next, upstream architecture name `qwen4exp`. Its linear
// attention is dimensionally identical to the Qwen3.8 GDN block, so three
// layers in four reuse the existing mixer geometry; the fourth is sparse
// attention with a routing indexer, which has no operator yet.
//
// Numbers below come from the published config.json. The fields under
// "no Splash equivalent" describe modules this engine cannot execute today;
// they are recorded here so the package validator can check them and so the
// gaps are visible in one place. See kUnsupported* below.
struct Qwen4ExpLayout final {
  static constexpr std::string_view layerMagic = "MDFN0001";

  // Provisional: the draft is deferred, so these five indices are evenly
  // spaced rather than chosen against a trained DFlash 2 draft.
  static constexpr std::array<uint32_t, 5> hiddenCaptureLayers{3, 13, 23, 33, 43};

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 48;
  uint32_t hiddenSize = 2560;
  uint32_t vocabularySize = 248320;
  uint32_t packedGdnWidth = 16640;
  uint32_t packedFullWidth = 13312;
  uint32_t convolutionDimension = 10240;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t attentionWidth = 6144;
  uint32_t attentionQueryHeads = 24;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 4;
  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  // 640 is not a multiple of 256 but is a multiple of 128, so the expert
  // projections tile narrower than everything else rather than padding to
  // 768. See kQ4ExpertStorageN.
  uint32_t expertStorageN = kQ4ExpertStorageN;

  // TODO(qwen4exp): confirm against the shipped tokenizer; the vocabulary
  // matches Qwen3.8 exactly, so these are carried over as placeholders.
  uint32_t maskToken = 248070;
  std::array<uint32_t, 2> stopTokens{248044, 248046};

  // ---- no operator yet; the weights are still described and mapped ------
  uint32_t hyperConnectionCount = 4;
  uint32_t hyperConnectionLowRank = 320;
  uint32_t indexerHeads = 4;
  uint32_t indexerKvHeads = 1;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerBudget = 2048;
  // The n-gram table is a Q4 lookup table on one layer, not a per-layer cost:
  // 20,000,000 rows of 2560 at nine bytes per sixteen weights is 26.8 GiB,
  // which is what this model measures on disk.
  uint32_t ngramSize = 3;
  uint32_t ngramLayer = 1;
  // The table is one row per hashed n-gram per head, not one per token. Each
  // head gets its own prime-sized vocabulary just above ngramVocabularyBase,
  // and the padded total is what the packed file holds. Sixteen heads of 160
  // multiply out to the same element count as eight of 2560, which is why a
  // wrong shape can still produce the right total.
  uint32_t ngramVocabularyBase = 20'000'000;
  uint32_t ngramVocabularySize = 320'001'536;
  uint32_t ngramEmbeddingSize = 2560;
  uint32_t ngramHeadsPerOrder = 8;
  uint32_t ngramShards = 128;
  uint32_t pleConvolutionTaps = 4;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return fullAttentionPeriod && (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return fullAttentionPeriod ? layers / fullAttentionPeriod : 0;
  }
  // Packed projections are concatenated then padded up to StorageN.
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return convolutionDimension + attentionWidth + 2 * gdnValueHeads;
  }
  // Query width is doubled for the output gate, as in Qwen3.8.
  [[nodiscard]] constexpr uint32_t actualFullWidth() const noexcept {
    return 2 * attentionQueryHeads * attentionHeadDimension +
           2 * attentionKvHeads * attentionHeadDimension;
  }
  [[nodiscard]] constexpr kv::Q8Layout q8Layout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads, attentionHeadDimension};
  }
  // The per-layer embedding carries two pieces of history between steps: the
  // last ngramSize - 1 tokens, for the n-gram hash, and the last
  // pleConvolutionState() normalized rows, for its dilated convolution. They
  // live in the GDN state cell's auxiliary region:
  //
  //   uint32 valid, older token, newer token, padding     16 bytes
  //   bf16   [pleConvolutionState()][hyperConnectionWidth()]
  //
  // A zeroed region (valid = 0) is a fresh sequence, which the reference
  // treats as two end-of-sequence tokens and a zero convolution history.
  static constexpr uint32_t pleHistoryHeaderBytes = 16;
  [[nodiscard]] constexpr uint64_t pleAuxiliaryBytes() const noexcept {
    return pleHistoryHeaderBytes + uint64_t{pleConvolutionState()} *
                                       hyperConnectionWidth() * 2;
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), kGdnConvolutionTaps - 1,
            convolutionDimension, gdnValueHeads, gdnHeadDimension,
            gdnHeadDimension, pleAuxiliaryBytes()};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  // Four residual streams, so every hyper-connection tensor is this wide.
  [[nodiscard]] constexpr uint32_t hyperConnectionWidth() const noexcept {
    return hiddenSize * hyperConnectionCount;
  }
  // Query and key-value share one indexer projection, as in the reference.
  [[nodiscard]] constexpr uint32_t indexerProjectionWidth() const noexcept {
    return (indexerHeads + indexerKvHeads) * indexerHeadDimension;
  }
  // The PLE convolution is dilated by the n-gram order, so its state spans
  // more positions than it has taps.
  [[nodiscard]] constexpr uint32_t pleConvolutionState() const noexcept {
    return (pleConvolutionTaps - 1) * ngramSize;
  }
  // One head per context position per hash, so the orders below the full
  // n-gram each contribute a set.
  [[nodiscard]] constexpr uint32_t ngramHeads() const noexcept {
    return (ngramSize - 1) * ngramHeadsPerOrder;
  }
  // The heads partition the embedding width between them.
  [[nodiscard]] constexpr uint32_t ngramHeadDimension() const noexcept {
    return ngramHeads() ? ngramEmbeddingSize / ngramHeads() : 0;
  }

  // Weights this model can be planned without holding all of: the routed
  // experts, which are touched ten at a time out of five hundred and twelve,
  // and the per-layer embedding table, which is one gathered row per head.
  // Everything else - attention, the mixers, the shared expert, the norms -
  // is read for every token and stays resident.
  [[nodiscard]] constexpr uint64_t streamableWeightBytes() const noexcept {
    const uint64_t perExpert =
        (2ULL * expertIntermediateSize * hiddenSize +
         uint64_t{hiddenSize} * expertIntermediateSize) *
        9 / 16;
    const uint64_t experts_ = perExpert * experts * layers;
    const uint64_t table = uint64_t{ngramVocabularySize} *
                           ngramHeadDimension() * 9 / 16;
    return experts_ + table;
  }

  bool operator==(const Qwen4ExpLayout &) const = default;
};

// Hyper-connections replace the usual input and post-attention norms. The
// residual is four parallel streams carried as one row of hc*hidden, and each
// block reads a mix of them and writes back a gated injection.
//
// Over a row x of width H = hc*hidden, with d = hidden (from the reference
// implementation, Qwen4ExpTextGatedResidual):
//
//   xn    = hc_norm(x)                  RMS per stream, gain over all of H
//   w     = silu(down(xn) / hc)         H -> lowRank
//   w     = sigmoid(up(w))              lowRank -> H
//   input = mean over hc of             the d-wide block input
//             reshape(w,[hc,d]) * reshape(xn,[hc,d])
//   inj   = 2 * sigmoid(block_inject(xn) / hc)      H -> hc
//
// and after the block produces y of width d, the residual update is an outer
// product against the *unnormalized* input:
//
//   x = x + outer(inj, y)               [hc,d], flattened back to H
//
// The mix projections are stored bf16, not Q4: the down projection is 320
// wide, which is not a multiple of either tile width.
struct Qwen4ExpHyperConnection final {
  metal::MetalBuffer norm;         // hyperConnectionWidth
  metal::MetalBuffer mixDown;      // lowRank x hyperConnectionWidth
  metal::MetalBuffer mixUp;        // hyperConnectionWidth x lowRank
  // Absent on the final mixer, which collapses the streams before the head
  // and never injects a block result back (use_combine=False upstream).
  std::optional<metal::MetalBuffer> blockInject;  // count x width
};

// The per-layer embedding on layer ngramLayer. It gathers a hashed n-gram
// embedding, projects it to one value and one key per residual stream, gates
// the value by the key against the normalized stream, and adds a dilated
// depthwise convolution of the result. It writes the full hc*hidden width, so
// it feeds every stream.
//
// The embedding table itself is sharded, with a vocabulary size and offset per
// head, which is why the head tables travel with it.
struct Qwen4ExpPerLayerEmbedding final {
  // Rows are one head wide, and quantized in finer groups than everything
  // else because 160 is not a whole number of 64-element groups.
  ops::Q4Projection table;            // ngram vocabulary x head dimension
  metal::MetalBuffer headOffsets;     // ngramHeads
  metal::MetalBuffer headVocabularySizes;
  metal::MetalBuffer layerMultipliers;
  ops::Q4Projection keyProjection;    // hc width x embedding size
  ops::Q4Projection valueProjection;  // hidden x embedding size
  metal::MetalBuffer keyNorm;         // hc width
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer convolutionNorm;
  metal::MetalBuffer convolutionWeights;  // hc width x taps
};

// Sparse attention keeps an indexer that scores which keys to read.
struct Qwen4ExpIndexer final {
  ops::Q4Projection queryKeyProjection;  // indexerProjectionWidth x hidden
  metal::MetalBuffer queryNorm;          // indexerHeadDimension
  metal::MetalBuffer keyNorm;            // indexerHeadDimension
};

struct Qwen4ExpLayerExpertCache final {
  metal::MetalBuffer cacheGate;
  metal::MetalBuffer cacheUp;
  metal::MetalBuffer cacheDown;
  uint32_t capacity = 0;
  uint32_t numCached = 0;
  uint32_t clock = 0;
  std::vector<int16_t> expertToSlot;
  std::vector<int16_t> slotToExpert;
  std::vector<uint32_t> lruTime;
};

// Section order per layer file, which the packer must follow exactly:
//
//   attention hyper-connection   norm, inject, mix down, mix up
//   mixer                        GDN or sparse attention, as Qwen3.8
//   indexer                      attention layers only
//   mlp hyper-connection         norm, inject, mix down, mix up
//   experts                      router, gate, up, down, shared x3, gate
// Where a streaming layer's routed experts sit in its file, for reading
// cache misses directly (file cache off) instead of faulting them in.
struct Qwen4ExpExpertSource final {
  int fd = -1;
  uint64_t gate = 0; // file offsets of expert 0 of each matrix
  uint64_t up = 0;
  uint64_t down = 0;
};

struct Qwen4ExpLayerWeights final {
  Qwen4ExpHyperConnection attentionHyperConnection;
  QwenMixerWeights mixer;
  std::optional<Qwen4ExpIndexer> indexer;
  Qwen4ExpHyperConnection mlpHyperConnection;
  ops::MoeWeights ffn;
  mutable Qwen4ExpLayerExpertCache expertCache;
  Qwen4ExpExpertSource expertSource;
};

struct Qwen4ExpWeights final {
  Qwen4ExpLayout layout;
  std::vector<Qwen4ExpLayerWeights> layers;
  Qwen4ExpHyperConnection hyperConnectionMixer;
  metal::MetalBuffer finalNorm;
  ops::Q4Projection logitsProjection;
  ops::Q4Projection tokenEmbedding;
  // Its own file: the embedding table alone is 26.8 GiB, which does not
  // belong inside a layer file.
  Qwen4ExpPerLayerEmbedding perLayerEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
  uint32_t residentLayers = 0;
  metal::MetalBuffer streamingCacheGate;
  metal::MetalBuffer streamingCacheUp;
  metal::MetalBuffer streamingCacheDown;
  uint32_t streamingCacheCapacity = 0;
  mutable std::vector<std::vector<uint32_t>> lastSelectedExperts;

  void prefetchStreamingExperts() const noexcept;
  void evictStreamingExperts() const noexcept;

  [[nodiscard]] uint64_t expertCacheActualAllocatedBytes() const noexcept {
    uint64_t total = 0;
    for (const auto &layer : layers) {
      total += layer.expertCache.cacheGate.sizeBytes();
      total += layer.expertCache.cacheUp.sizeBytes();
      total += layer.expertCache.cacheDown.sizeBytes();
    }
    return total;
  }
};

// Defined in QwenTarget.cpp beside the other two, so all three share
// commonGeometry.
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen4ExpWeights &weights);

[[nodiscard]] Qwen4ExpWeights
loadQwen4ExpWeights(metal::MetalBackend &backend,
                    const std::filesystem::path &directory,
                    Qwen4ExpLayout layout = {});

// Both of this architecture's MoE limits are now lifted. The expert width is
// expressible at kQ4ExpertStorageN = 128, the packed tiles are instantiated at
// that width, and ops::MoE routes at a padded router width rather than a fixed
// 256. What remains for this block is not a width problem.
inline constexpr uint32_t kQwen4ExpExpertCount = 512;

} // namespace splash::model

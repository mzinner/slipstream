#include "models/qwen4exp/Qwen4Exp.hpp"
#include "model/WeightStore.hpp"
#include "ops/GDN.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {
using namespace splash;
using namespace splash::model;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Mirrors validateQ4Layout, which lives in WeightStore.cpp and would pull
// Metal into what is otherwise a pure arithmetic test.
constexpr bool q4LayoutIsValid(uint32_t outputSize, uint32_t inputSize,
                               uint32_t storageN = kQ4StorageN) {
  return outputSize && inputSize && inputSize % kQ4GroupElements == 0 &&
         outputSize % storageN == 0;
}

// The packed widths are derived, not transcribed: a concatenation padded up
// to StorageN. Checking them here catches a bad constant before any weight
// file is written against it.
void packedWidthsFollowFromTheConcatenation() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.actualGdnWidth() == 16480,
                "qkv + z + b + a must be 16480 rows");
  static_assert(layout.packedGdnWidth == 16640,
                "16480 padded up to StorageN 256 is 16640");
  static_assert(layout.packedGdnWidth % kQ4StorageN == 0);
  static_assert(layout.packedGdnWidth - layout.actualGdnWidth() == 160,
                "the last GDN tile carries 160 zero rows");

  static_assert(layout.actualFullWidth() == 13312,
                "2*q + k + v with a gated query is 13312 rows");
  static_assert(layout.packedFullWidth == layout.actualFullWidth(),
                "the attention concatenation needs no padding");
  static_assert(layout.packedFullWidth % kQ4StorageN == 0);

  static_assert(layout.attentionWidth ==
                    layout.gdnValueHeads * layout.gdnHeadDimension,
                "attention width is value heads by head dimension");
  static_assert(layout.convolutionDimension ==
                    2 * layout.gdnKeyHeads * layout.gdnHeadDimension +
                        layout.gdnValueHeads * layout.gdnHeadDimension,
                "conv width is 2*k + v head widths");
}

// Three layers in four are linear attention (the shared GDN kernels), one in
// four full attention.
void linearAttentionLayers() {
  constexpr Qwen4ExpLayout next;

  // 48 layers at period 4 gives 12 attention layers and 36 linear ones.
  static_assert(next.attentionLayerCount() == 12);
  static_assert(next.layers - next.attentionLayerCount() == 36);
  static_assert(!next.isFullAttentionLayer(0));
  static_assert(next.isFullAttentionLayer(3));
  static_assert(next.isFullAttentionLayer(47));
}

// The linear-attention block needs no new kernel. ops::GDN already compiles
// for this exact shape - see kernelShape() in ops/GDN.cpp - because Qwen3.8
// has the same head geometry, and the GDN kernels are templated on head
// counts rather than hidden size.
void gdnShapeIsAlreadyCompiled() {
  constexpr Qwen4ExpLayout l;
  constexpr ops::GdnShape shape{l.gdnKeyHeads, l.gdnValueHeads,
                                l.gdnHeadDimension, l.convolutionDimension,
                                l.packedGdnWidth};
  static_assert(shape.valid(), "the qwen4exp GDN shape must be well formed");
  static_assert(shape == ops::GdnShape{16, 48, 128, 10240, 16640},
                "qwen4exp must match a GDN shape the engine already compiles");
}

// Every projection the engine can already express must satisfy the Q4 rules.
void supportedProjectionsAreQ4Aligned() {
  constexpr Qwen4ExpLayout layout;
  static_assert(q4LayoutIsValid(layout.packedGdnWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.packedFullWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.hiddenSize, layout.attentionWidth));
  static_assert(q4LayoutIsValid(layout.vocabularySize, layout.hiddenSize));
  require(layout.hiddenSize % kQ4GroupElements == 0,
          "hidden size must be group-aligned as a projection input");
  require(layout.hiddenSize % kQ4StorageN == 0,
          "hidden size must be StorageN-aligned as a projection output");
}

// Expert projections tile narrower than everything else. This is the whole
// reason kQ4ExpertStorageN exists, so pin both directions.
void expertProjectionsTileAt128() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.expertStorageN == kQ4ExpertStorageN);

  // 640 is not expressible at the usual width, which is why it tiles narrower.
  static_assert(!q4LayoutIsValid(layout.expertIntermediateSize,
                                 layout.hiddenSize, kQ4StorageN),
                "640 must not be expressible at StorageN 256");
  static_assert(q4LayoutIsValid(layout.expertIntermediateSize,
                                layout.hiddenSize, kQ4ExpertStorageN),
                "640 must be expressible at StorageN 128");
  static_assert(q4LayoutIsValid(layout.hiddenSize,
                                layout.expertIntermediateSize,
                                kQ4ExpertStorageN),
                "the down projection must tile at 128 as well");

  // The narrower tile must still be a whole number of 32-wide matmul slices.
  static_assert(kQ4ExpertStorageN % 32 == 0);
  static_assert(kQ4StorageN % kQ4ExpertStorageN == 0);

  // What padding to the usual width would have cost, for the record: three
  // projections per expert, 512 experts, 48 layers, at 9 bytes per 16 weights.
  constexpr uint64_t stored =
      uint64_t(layout.expertIntermediateSize) * layout.hiddenSize * 3 *
      layout.experts * layout.layers * 9 / 16;
  constexpr uint64_t padded = uint64_t(768) * layout.hiddenSize * 3 *
                              layout.experts * layout.layers * 9 / 16;
  require(padded - stored > 12ULL * 1024 * 1024 * 1024,
          "padding to 768 would have cost more than 12 GiB");
}

// The MoE shape this layout produces must be one ops::MoE accepts. This used
// to assert the opposite - that the expert count was over the ceiling - and
// tightening the limit is what brought it here.
void moeShapeIsAccepted() {
  constexpr Qwen4ExpLayout layout;
  constexpr ops::MoeShape shape{layout.hiddenSize, layout.experts,
                                layout.expertsPerToken,
                                layout.expertIntermediateSize,
                                layout.expertStorageN};
  static_assert(layout.experts == kQwen4ExpExpertCount);
  static_assert(shape.valid(), "ops::MoE must accept the qwen4exp MoE shape");
  // Ten of five hundred and twelve, in 128-wide tiles, and the router pads to
  // whole 256-wide tiles.
  static_assert(shape.routerWidth() == 512);
  static_assert(shape.expertIntermediateSize % shape.storageN == 0);
}

// The per-layer embedding's shape was wrong once in a way the totals hid:
// sixteen heads of 160 and eight of 2560 have the same element count, so the
// footprint agreed while the structure did not. Pin the structure.
void ngramGeometryIsPinned() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.ngramHeads() == 16,
                "one head per context position per hash");
  static_assert(layout.ngramHeadDimension() == 160,
                "the heads partition the embedding width");
  static_assert(layout.ngramHeads() * layout.ngramHeadDimension() ==
                    layout.ngramEmbeddingSize,
                "the heads must tile the embedding exactly");

  // 160 is not a whole number of ordinary groups, which is the whole reason
  // this table is quantized in finer ones.
  static_assert(layout.ngramHeadDimension() % kQ4GroupElements != 0);
  static_assert(layout.ngramHeadDimension() % kQ4FineGroupElements == 0);
  static_assert(layout.ngramVocabularySize % layout.ngramShards == 0,
                "the table must divide evenly into its shards");

  // Rows are prime-sized per head, above the base, then padded; the padded
  // total must still cover every head.
  static_assert(layout.ngramVocabularySize >
                    uint64_t(layout.ngramHeads()) * layout.ngramVocabularyBase,
                "the padded vocabulary must hold every head's table");
}

void stateLayoutsAreConsistent() {
  constexpr Qwen4ExpLayout layout;
  const auto kv = layout.q8Layout();
  // The trunk's 12 attention layers, and one for the MTP draft head.
  require(kv.attentionLayers == 12 + layout.mtpLayers && layout.mtpLayers == 1,
          "KV cache covers the attention layers and the MTP head");
  require(kv.kvHeads == layout.attentionKvHeads, "KV heads mismatch");
  require(kv.headDimension == layout.attentionHeadDimension,
          "KV head dimension mismatch");

  const auto gdn = layout.gdnStateLayout();
  require(gdn.layers == 36, "GDN state covers only the linear layers");
  require(gdn.convolutionChannels == layout.convolutionDimension,
          "GDN conv channel count mismatch");
  require(gdn.convolutionHistory == kGdnConvolutionTaps - 1,
          "GDN conv history must be taps minus one");
}

void neonHostRoutingMatchesScalarReference() {
  auto widen = [](uint16_t bits) {
    return std::bit_cast<float>(uint32_t{bits} << 16);
  };
  auto toBf16 = [](float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
  };

  auto scalarTopExperts = [&](const uint16_t *scores, uint32_t experts,
                              uint32_t k, uint32_t *ids, float *weights) {
    float chosen[16];
    uint32_t filled = 0;
    for (uint32_t e = 0; e < experts; ++e) {
      const float v = widen(scores[e]);
      if (filled == k && !(v > chosen[k - 1])) continue;
      uint32_t at = filled < k ? filled++ : k - 1;
      while (at > 0 && chosen[at - 1] < v) {
        chosen[at] = chosen[at - 1];
        ids[at] = ids[at - 1];
        --at;
      }
      chosen[at] = v;
      ids[at] = e;
    }
    float total = 0.0f;
    for (uint32_t r = 0; r < k; ++r) total += std::exp(chosen[r] - chosen[0]);
    for (uint32_t r = 0; r < k; ++r)
      weights[r] = std::exp(chosen[r] - chosen[0]) / total;
  };

  auto neonTopExperts = [&](const uint16_t *scores, uint32_t experts,
                            uint32_t k, uint32_t *ids, float *weights) {
    float chosen[16];
    uint32_t filled = 0;
#if defined(__ARM_NEON)
    uint32_t e = 0;
    for (; e + 8 <= experts; e += 8) {
      const uint16x8_t raw = vld1q_u16(scores + e);
      if (filled == k) {
        const float32x4_t lo =
            vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
        const float32x4_t hi =
            vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw), 16));
        const float blockMax = vmaxvq_f32(vmaxq_f32(lo, hi));
        if (!(blockMax > chosen[k - 1])) continue;
      }
      for (uint32_t j = 0; j < 8; ++j) {
        const float v = widen(scores[e + j]);
        if (filled == k && !(v > chosen[k - 1])) continue;
        uint32_t at = filled < k ? filled++ : k - 1;
        while (at > 0 && chosen[at - 1] < v) {
          chosen[at] = chosen[at - 1];
          ids[at] = ids[at - 1];
          --at;
        }
        chosen[at] = v;
        ids[at] = e + j;
      }
    }
    for (; e < experts; ++e) {
#else
    for (uint32_t e = 0; e < experts; ++e) {
#endif
      const float v = widen(scores[e]);
      if (filled == k && !(v > chosen[k - 1])) continue;
      uint32_t at = filled < k ? filled++ : k - 1;
      while (at > 0 && chosen[at - 1] < v) {
        chosen[at] = chosen[at - 1];
        ids[at] = ids[at - 1];
        --at;
      }
      chosen[at] = v;
      ids[at] = e;
    }
    float total = 0.0f;
    for (uint32_t r = 0; r < k; ++r) total += std::exp(chosen[r] - chosen[0]);
    for (uint32_t r = 0; r < k; ++r)
      weights[r] = std::exp(chosen[r] - chosen[0]) / total;
  };

  std::mt19937 rng(20261015);
  std::normal_distribution<float> dist(0.0f, 2.0f);
  for (uint32_t trial = 0; trial < 32; ++trial) {
    std::vector<uint16_t> scores(512);
    for (uint32_t i = 0; i < 512; ++i) {
      scores[i] = (trial == 0 && i % 17 == 0) ? toBf16(1.5f)
                                              : toBf16(dist(rng));
    }
    uint32_t idsRef[10]{}, idsNeon[10]{};
    float wRef[10]{}, wNeon[10]{};
    scalarTopExperts(scores.data(), 512, 10, idsRef, wRef);
    neonTopExperts(scores.data(), 512, 10, idsNeon, wNeon);
    for (uint32_t k = 0; k < 10; ++k) {
      require(idsRef[k] == idsNeon[k],
              "NEON topExperts selected different expert id");
      require(toBf16(wRef[k]) == toBf16(wNeon[k]),
              "NEON topExperts produced different routing weight");
    }
  }

  constexpr uint32_t kSize = 2560;
  constexpr uint32_t kStorageN = 256;
  constexpr uint32_t kGroups = kSize / 64;
  std::vector<uint8_t> w(kGroups * kStorageN * 64);
  std::vector<uint16_t> scales(kGroups * kStorageN);
  std::vector<uint16_t> biases(kGroups * kStorageN);
  std::vector<uint16_t> input(kSize);
  for (auto &b : w) b = static_cast<uint8_t>(rng() & 0xFF);
  for (auto &s : scales) s = toBf16(dist(rng) * 0.01f);
  for (auto &b : biases) b = toBf16(dist(rng) * 0.01f);
  for (auto &x : input) x = toBf16(dist(rng) * 0.5f);

  float scalarTotal = 0.0f;
  for (uint32_t d = 0; d < kSize; ++d) {
    const uint32_t group = d / 64;
    const float value =
        float(w[uint64_t{group} * kStorageN * 64 + d % 64]) *
            widen(scales[uint64_t{group} * kStorageN]) +
        widen(biases[uint64_t{group} * kStorageN]);
    scalarTotal += widen(input[d]) * value;
  }
  const float scalarGate = 1.0f / (1.0f + std::exp(-scalarTotal));

  float neonTotal = 0.0f;
#if defined(__ARM_NEON)
  for (uint32_t group = 0; group < kGroups; ++group) {
    const float scale = widen(scales[uint64_t{group} * kStorageN]);
    const float bias = widen(biases[uint64_t{group} * kStorageN]);
    const uint8_t *gw = w.data() + uint64_t{group} * kStorageN * 64;
    const uint16_t *gx = input.data() + uint64_t{group} * 64;
    float32x4_t wSum0 = vdupq_n_f32(0.0f);
    float32x4_t wSum1 = vdupq_n_f32(0.0f);
    float32x4_t xSum0 = vdupq_n_f32(0.0f);
    float32x4_t xSum1 = vdupq_n_f32(0.0f);
    for (uint32_t i = 0; i < 64; i += 8) {
      const uint8x8_t w8 = vld1_u8(gw + i);
      const uint16x8_t w16 = vmovl_u8(w8);
      const float32x4_t wf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16)));
      const float32x4_t wf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
      const uint16x8_t x16 = vld1q_u16(gx + i);
      const float32x4_t xf0 =
          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x16), 16));
      const float32x4_t xf1 =
          vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x16), 16));
      wSum0 = vfmaq_f32(wSum0, xf0, wf0);
      wSum1 = vfmaq_f32(wSum1, xf1, wf1);
      xSum0 = vaddq_f32(xSum0, xf0);
      xSum1 = vaddq_f32(xSum1, xf1);
    }
    const float groupW = vaddvq_f32(vaddq_f32(wSum0, wSum1));
    const float groupX = vaddvq_f32(vaddq_f32(xSum0, xSum1));
    neonTotal += groupW * scale + groupX * bias;
  }
#else
  neonTotal = scalarTotal;
#endif
  const float neonGate = 1.0f / (1.0f + std::exp(-neonTotal));
  require(std::fabs(scalarGate - neonGate) < 1e-6f,
          "NEON sharedExpertGate diverges from scalar reference");
  require(toBf16(scalarGate) == toBf16(neonGate),
          "NEON sharedExpertGate bf16 result diverges from scalar reference");
}

} // namespace

int main() {
  try {
    packedWidthsFollowFromTheConcatenation();
    linearAttentionLayers();
    supportedProjectionsAreQ4Aligned();
    gdnShapeIsAlreadyCompiled();
    expertProjectionsTileAt128();
    moeShapeIsAccepted();
    ngramGeometryIsPinned();
    stateLayoutsAreConsistent();
    neonHostRoutingMatchesScalarReference();
  } catch (const std::exception &error) {
    std::cerr << "qwen4exp layout test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "qwen4exp layout test passed\n";
  return 0;
}

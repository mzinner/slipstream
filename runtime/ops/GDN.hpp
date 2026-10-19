#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Tensor geometry mapped to a compiled Metal variant during graph construction.
struct GdnShape final {
  uint32_t keyHeads = 0;
  uint32_t valueHeads = 0;
  uint32_t headDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;
  // Output gate activation: silu when false, sigmoid when true. Not part of
  // the compiled geometry; see GDNPreparePrefillParams.
  bool sigmoidGate = false;
  uint32_t treeParents = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return keyHeads && valueHeads && valueHeads % keyHeads == 0 &&
           headDimension && convolutionDimension && packedWidth &&
           convolutionDimension ==
               (uint64_t{2} * keyHeads + valueHeads) * headDimension &&
           packedWidth >= convolutionDimension +
                              uint64_t{valueHeads} * headDimension +
                              uint64_t{2} * valueHeads;
  }

  bool operator==(const GdnShape &) const = default;
};

struct GdnStateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes;
  }
};

struct GdnPrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  metal::MetalBuffer recurrentRows;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
};

// A lane's decode tape (metal/abi/GDN.h): each step leaves its rows'
// convolution inputs, k, v and gates there, and the lane's next step folds
// the rows it retained into the state before its own, so the state cell a
// step writes holds the state before the step's rows. A slot holds one step
// of every layer, gdnTapeLayerBytes(shape) apart; the offsets are in
// GdnDecodeBuffers::tape.
struct GdnTapeLane final {
  uint64_t pendingSlot = 0; // the lane's previous step's
  uint64_t stepSlot = 0;    // this step's
  // The previous step's rows the lane retained, which the state does not
  // hold yet: 0 after a prefill or a restore.
  uint32_t pendingRows = 0;
};

[[nodiscard]] uint64_t gdnTapeLayerBytes(const GdnShape &shape) noexcept;

struct GdnDecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer tape;
  std::span<const GdnTapeLane> tapeLanes;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
};

class GDN final {
public:
  static void addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                         GdnShape shape, uint32_t tokens);
  static void addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                        GdnShape shape, uint32_t lanes, uint32_t layer,
                        GdnStateStrides state);
};

} // namespace splash::ops

#include "ops/GDN.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/LaneBindings.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 24);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  GdnShape geometry = shape;
  geometry.sigmoidGate = false;
  geometry.treeParents = 0;
  if (geometry == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (geometry == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const GDNPreparePrefillParams prepare{tokens, shape.packedWidth,
                                        shape.sigmoidGate ? 1u : 0u};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            prepare, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            GDNPrefillParams{tokens},
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 SPLASH_GDN_SCAN_STATE_ROWS,
             1, 1},
            {SPLASH_GDN_SCAN_THREADS, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_gate",
                       "prefill_gdn_gate_vh32"),
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm,
             buffers.hidden},
            prepare, {uint64_t{tokens} * shape.valueHeads, 1, 1},
            {128, 1, 1});
}

void GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                    GdnShape shape, uint32_t lanes, uint32_t layer,
                    GdnStateStrides state) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid() ||
      buffers.tapeLanes.size() < lanes)
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  GDNDecodeBatchParams params{shape.sigmoidGate ? 1u : 0u,
                              shape.packedWidth,
                              lanes,
                              layer,
                              shape.treeParents,
                              0u,
                              state.convolutionLayerBytes,
                              state.recurrentLayerBytes,
                              state.convolutionStateBytes,
                              {},
                              {},
                              {}};
  // Each lane reads its pending rows' tape of this layer and writes its
  // step's; no step's tape may overlap a tape another lane, or the step's own
  // lane, reads or writes.
  const uint64_t tapeBytes = gdnTapeLayerBytes(shape);
  const auto overlaps = [&](uint64_t a, uint64_t b) {
    return a < b + tapeBytes && b < a + tapeBytes;
  };
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const GdnTapeLane &tape = buffers.tapeLanes[lane];
    params.pending_tape[lane] = tape.pendingSlot + uint64_t{layer} * tapeBytes;
    params.step_tape[lane] = tape.stepSlot + uint64_t{layer} * tapeBytes;
    params.pending_rows[lane] = tape.pendingRows;
    if (tape.pendingRows > SPLASH_TARGET_VERIFY_ROWS ||
        params.pending_tape[lane] % 16 || params.step_tape[lane] % 16)
      throw std::invalid_argument("invalid GDN decode tape");
    for (uint32_t other = 0; other <= lane; ++other)
      if (overlaps(params.step_tape[lane], params.pending_tape[other]) ||
          overlaps(params.step_tape[other], params.pending_tape[lane]) ||
          (other != lane &&
           overlaps(params.step_tape[lane], params.step_tape[other])))
        throw std::invalid_argument("invalid GDN decode tape");
  }
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  bindings.reserve(18);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.tape, buffers.decayWeights, buffers.timeBias,
                   buffers.recurrent, buffers.mixerNorm, buffers.hidden,
                   buffers.arrived, buffers.generation});
  graph.add(kernelName(kernel, "verify_gdn_fused", "verify_gdn_fused_vh32"),
            std::move(bindings), params, {shape.valueHeads, lanes, 1});
}

uint64_t gdnTapeLayerBytes(const GdnShape &shape) noexcept {
  return gdn_tape_layer_bytes(
      shape.convolutionDimension,
      (shape.keyHeads + shape.valueHeads) * shape.headDimension,
      shape.valueHeads);
}

} // namespace splash::ops

// Checks expert selection at the qwen4exp router width.
//
// Splash routes at a padded router width of 256; Qwen3.8-Flash-Next needs
// 512. The selection kernel is templated on that width, and the semantics it
// already implements - take the top_k highest logits, then softmax over just
// those - match the reference router exactly, because softmax is monotonic
// and renormalizing the top_k probabilities is softmax over the selected
// logits. That equivalence was checked against Qwen4ExpTextTopKRouter under
// torch, where the indices were identical and the weights agreed to 3e-08.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBuffer;
using splash::metal::MetalBackend;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

struct RouteParams {
  uint32_t rows;
  uint32_t inputSize;
  uint32_t experts;
  uint32_t topK;
};

// The shared expert's scalar gate stays padded to this, whatever the router
// width is; the two widths used to share one constant in the kernel.
constexpr uint32_t kGateStorageN = 256;

uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + ((bits >> 16) & 1) + 0x7FFF) >> 16);
}
float fromBf16(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

MetalBuffer upload(MetalBackend &backend, const void *data, size_t bytes,
                   const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: moe-route-512 <metallib>");
    MetalBackend backend(argv[1]);

    const uint32_t routerWidth = 512;
    const RouteParams params{6, 2560, 512, 10};
    const uint32_t quantGroups = params.inputSize / 64;

    std::mt19937 engine(20260919);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<uint16_t> scores(size_t(params.rows) * routerWidth);
    for (uint16_t &value : scores) value = toBf16(normal(engine) * 2.0f);
    std::vector<uint16_t> input(size_t(params.rows) * params.inputSize);
    for (uint16_t &value : input) value = toBf16(normal(engine) * 0.4f);

    // The gate's Q8 slab is [quant group][row][64] with the row dimension
    // padded to kGateStorageN; only row zero carries the scalar gate.
    std::vector<uint8_t> gateWeights(size_t(quantGroups) * kGateStorageN * 64);
    for (uint8_t &value : gateWeights)
      value = uint8_t(engine() & 0xFF);
    std::vector<uint16_t> gateScales(size_t(quantGroups) * kGateStorageN);
    std::vector<uint16_t> gateBiases(gateScales.size());
    for (size_t i = 0; i < gateScales.size(); ++i) {
      gateScales[i] = toBf16(normal(engine) * 0.002f);
      gateBiases[i] = toBf16(normal(engine) * 0.05f);
    }

    const size_t routeSlots = size_t(params.rows) * (params.topK + 1);
    MetalBuffer scoreBuffer =
        upload(backend, scores.data(), scores.size() * 2, "route-scores");
    MetalBuffer inputBuffer =
        upload(backend, input.data(), input.size() * 2, "route-input");
    MetalBuffer weightBuffer = upload(backend, gateWeights.data(),
                                      gateWeights.size(), "gate-weights");
    MetalBuffer scaleBuffer = upload(backend, gateScales.data(),
                                     gateScales.size() * 2, "gate-scales");
    MetalBuffer biasBuffer = upload(backend, gateBiases.data(),
                                    gateBiases.size() * 2, "gate-biases");
    MetalBuffer selectedBuffer = backend.allocateBuffer(
        routeSlots * 4, BufferStorage::Shared, "route-selected");
    MetalBuffer weightsOut = backend.allocateBuffer(
        routeSlots * 2, BufferStorage::Shared, "route-weights");

    CommandGraph graph;
    graph.add("moe_route_select_q8_n512",
              {scoreBuffer, inputBuffer, weightBuffer, scaleBuffer, biasBuffer,
               selectedBuffer, weightsOut},
              params, {params.rows, 1, 1}, {routerWidth, 1, 1});
    (void)backend.submitCommand(graph.dispatches());

    const uint32_t *selected =
        static_cast<const uint32_t *>(selectedBuffer.contents());
    const uint16_t *routing =
        static_cast<const uint16_t *>(weightsOut.contents());

    double worstWeight = 0.0, worstGate = 0.0;
    for (uint32_t row = 0; row < params.rows; ++row) {
      // Descending score, ascending expert id on a tie - the kernel's
      // documented contract.
      std::vector<uint32_t> order(params.experts);
      std::iota(order.begin(), order.end(), 0u);
      const uint16_t *rowScores = scores.data() + size_t(row) * routerWidth;
      std::stable_sort(order.begin(), order.end(),
                       [&](uint32_t a, uint32_t b) {
                         const float sa = fromBf16(rowScores[a]);
                         const float sb = fromBf16(rowScores[b]);
                         return sa != sb ? sa > sb : a < b;
                       });
      const size_t base = size_t(row) * (params.topK + 1);
      for (uint32_t rank = 0; rank < params.topK; ++rank)
        require(selected[base + rank] == order[rank],
                "selected expert " + std::to_string(rank) + " of row " +
                    std::to_string(row) + " is " +
                    std::to_string(selected[base + rank]) + ", expected " +
                    std::to_string(order[rank]));

      // softmax over the selected logits, shifted by the highest
      const float top = fromBf16(rowScores[order[0]]);
      double denominator = 0.0;
      for (uint32_t rank = 0; rank < params.topK; ++rank)
        denominator += std::exp(double(fromBf16(rowScores[order[rank]]) - top));
      for (uint32_t rank = 0; rank < params.topK; ++rank) {
        const double want =
            std::exp(double(fromBf16(rowScores[order[rank]]) - top)) /
            denominator;
        const double got = fromBf16(routing[base + rank]);
        worstWeight = std::max(worstWeight, std::abs(got - want) /
                                                std::max(want, 1e-3));
      }

      require(selected[base + params.topK] == params.experts,
              "the shared expert must occupy the last route slot");
      double gate = 0.0;
      for (uint32_t dimension = 0; dimension < params.inputSize; ++dimension) {
        const uint32_t quantGroup = dimension / 64;
        const uint32_t within = dimension % 64;
        const size_t index =
            size_t(quantGroup) * kGateStorageN * 64 + within;
        const size_t parameter = size_t(quantGroup) * kGateStorageN;
        const double value = double(gateWeights[index]) *
                                 fromBf16(gateScales[parameter]) +
                             fromBf16(gateBiases[parameter]);
        gate += double(fromBf16(input[size_t(row) * params.inputSize +
                                      dimension])) *
                value;
      }
      const double wantGate = 1.0 / (1.0 + std::exp(-gate));
      const double gotGate = fromBf16(routing[base + params.topK]);
      worstGate = std::max(worstGate,
                           std::abs(gotGate - wantGate) /
                               std::max(std::abs(wantGate), 1e-3));
    }

    std::cout << "moe routing at width " << routerWidth << ", top "
              << params.topK << " of " << params.experts << '\n'
              << "  selected experts        exact for all "
              << params.rows << " rows\n"
              << "  routing weight error    " << worstWeight << '\n'
              << "  shared gate error       " << worstGate << '\n';
    require(worstWeight < 0.02, "routing weights do not match");
    require(worstGate < 0.05, "shared expert gate does not match");

    std::cout << "moe route 512 metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "moe route 512 metal test failed: " << error.what() << '\n';
    return 1;
  }
}

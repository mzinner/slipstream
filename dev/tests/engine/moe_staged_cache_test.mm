#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace splash;
using namespace splash::metal;
using namespace splash::ops;

namespace {

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

MetalBuffer uploadShared(MetalBackend &backend, const void *data, size_t bytes,
                         const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: moe-staged-cache-test METALLIB\n";
    return 2;
  }

  MetalBackend backend(argv[1]);

  MoeShape shape{
      .hiddenSize = 2560,
      .experts = 512,
      .expertsPerToken = 10,
      .expertIntermediateSize = 640,
      .storageN = 128,
  };
  const uint32_t lanes = 1;
  const MoePlan plan = MoE::decodePlan(shape, lanes, {MoeExpertTile::M8});

  const uint64_t expertStride = 640ULL * 2560 * 9 / 16; // 1,024,000 bytes
  const uint64_t fullExpertBytes = 512ULL * expertStride;

  std::mt19937 rng(42);
  std::uniform_int_distribution<uint32_t> byteDist(0, 255);

  // 1. Router weights
  const uint64_t routerBytes = 512ULL * 2560;
  std::vector<uint8_t> routerW(routerBytes);
  for (auto &b : routerW) b = byteDist(rng);
  std::vector<uint16_t> routerS(512 * 40, toBf16(0.01f));
  std::vector<uint16_t> routerB(512 * 40, toBf16(0.001f));

  Q8Projection router{
      .weights = uploadShared(backend, routerW.data(), routerBytes, "router-w"),
      .scales = uploadShared(backend, routerS.data(), routerS.size() * 2, "router-s"),
      .biases = uploadShared(backend, routerB.data(), routerB.size() * 2, "router-b"),
      .outputSize = 512,
      .inputSize = 2560,
  };

  // Shared expert gate
  std::vector<uint8_t> sharedGW(256 * 2560, 0);
  std::vector<uint16_t> sharedGS(256 * 40, toBf16(0.01f));
  std::vector<uint16_t> sharedGB(256 * 40, toBf16(0.0f));
  Q8Projection sharedExpertGate{
      .weights = uploadShared(backend, sharedGW.data(), sharedGW.size(), "shared-gw"),
      .scales = uploadShared(backend, sharedGS.data(), sharedGS.size() * 2, "shared-gs"),
      .biases = uploadShared(backend, sharedGB.data(), sharedGB.size() * 2, "shared-gb"),
      .outputSize = 256,
      .inputSize = 2560,
  };

  // 2. Full expert weights
  std::vector<uint8_t> fullGate(fullExpertBytes);
  std::vector<uint8_t> fullUp(fullExpertBytes);
  std::vector<uint8_t> fullDown(fullExpertBytes);
  for (size_t i = 0; i < fullExpertBytes; ++i) {
    fullGate[i] = uint8_t((i * 17) & 0xFF);
    fullUp[i] = uint8_t((i * 31) & 0xFF);
    fullDown[i] = uint8_t((i * 53) & 0xFF);
  }

  ExpertQ4Projection expertGate{
      .packed = uploadShared(backend, fullGate.data(), fullExpertBytes, "full-gate"),
      .experts = 512,
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection expertUp{
      .packed = uploadShared(backend, fullUp.data(), fullExpertBytes, "full-up"),
      .experts = 512,
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection expertDown{
      .packed = uploadShared(backend, fullDown.data(), fullExpertBytes, "full-down"),
      .experts = 512,
      .outputSize = 2560,
      .inputSize = 640,
      .expertStrideBytes = expertStride,
  };

  // Shared expert
  std::vector<uint8_t> sharedExpGate(expertStride, 1);
  std::vector<uint8_t> sharedExpUp(expertStride, 2);
  std::vector<uint8_t> sharedExpDown(expertStride, 3);
  ExpertQ4Projection sharedGate{
      .packed = uploadShared(backend, sharedExpGate.data(), expertStride, "shared-gate"),
      .experts = 1,
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection sharedUp{
      .packed = uploadShared(backend, sharedExpUp.data(), expertStride, "shared-up"),
      .experts = 1,
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection sharedDown{
      .packed = uploadShared(backend, sharedExpDown.data(), expertStride, "shared-down"),
      .experts = 1,
      .outputSize = 2560,
      .inputSize = 640,
      .expertStrideBytes = expertStride,
  };

  MoeWeights fullWeights{
      .router = router,
      .expertGate = expertGate,
      .expertUp = expertUp,
      .expertDown = expertDown,
      .sharedGate = sharedGate,
      .sharedUp = sharedUp,
      .sharedDown = sharedDown,
      .sharedExpertGate = sharedExpertGate,
  };

  // 3. Input buffer
  const uint32_t rows = plan.rows();
  const uint64_t rowBytes = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  std::vector<uint16_t> inputH(rows * shape.hiddenSize);
  for (size_t i = 0; i < inputH.size(); ++i) {
    inputH[i] = toBf16(std::sin(float(i) * 0.05f) * 0.5f);
  }
  MetalBuffer inputBuf =
      uploadShared(backend, inputH.data(), rowBytes, "input");

  // Helper to create workspace buffers
  auto makeBuffers = [&](const char *tag) {
    const auto &ws = plan.workspace();
    MoeBuffers b;
    b.input = inputBuf;
    b.residual = backend.allocateBuffer(rowBytes, BufferStorage::Shared, "res");
    b.output = backend.allocateBuffer(rowBytes, BufferStorage::Shared, (std::string(tag) + "-out").c_str());
    b.selectedExperts = backend.allocateBuffer(ws.selectedExpertsBytes, BufferStorage::Shared, "sel");
    b.routingWeights = backend.allocateBuffer(ws.routingWeightsBytes, BufferStorage::Shared, "rw");
    b.tileDescriptors = backend.allocateBuffer(ws.tileDescriptorsBytes, BufferStorage::Shared, "td");
    b.tileCount = backend.allocateBuffer(ws.tileCountBytes, BufferStorage::Shared, "tc");
    b.groupedRoutes = backend.allocateBuffer(ws.groupedRoutesBytes, BufferStorage::Shared, "gr");
    b.routeRows = backend.allocateBuffer(ws.routeRowsBytes, BufferStorage::Shared, "rr");
    b.groupedInput = backend.allocateBuffer(ws.groupedInputBytes, BufferStorage::Shared, "gi");
    b.expertIntermediate = backend.allocateBuffer(ws.expertIntermediateBytes, BufferStorage::Shared, "ei");
    b.expertOutput = backend.allocateBuffer(ws.expertOutputBytes, BufferStorage::Shared, "eo");
    return b;
  };

  // ==========================================
  // RUN 1: Full Reference Path
  // ==========================================
  MoeBuffers refBuffers = makeBuffers("ref");
  CommandGraph graphRef;
  MoE::add(graphRef, refBuffers, fullWeights, plan, /*addResidual=*/false);
  (void)backend.submitCommandAsync(graphRef.dispatches()).wait();

  const auto *refOutPtr = static_cast<const uint16_t *>(refBuffers.output.contents());
  const auto *refSelPtr = static_cast<const uint32_t *>(refBuffers.selectedExperts.contents());
  const auto *refWeightsPtr = static_cast<const uint16_t *>(refBuffers.routingWeights.contents());

  std::cout << "[Reference] Selected experts (row 0): ";
  for (uint32_t i = 0; i < shape.expertsPerToken; ++i) {
    std::cout << refSelPtr[i] << " (w=" << fromBf16(refWeightsPtr[i]) << ") ";
  }
  std::cout << "\n";

  // ==========================================
  // RUN 2: Staged Streaming Cache Path
  // ==========================================
  MoeBuffers stagedBuffers = makeBuffers("staged");

  // Step 2a: Run route only
  CommandGraph graphRoute;
  MoE::addRoute(graphRoute, stagedBuffers, fullWeights, plan);
  (void)backend.submitCommandAsync(graphRoute.dispatches()).wait();

  auto *stagedSelPtr = static_cast<uint32_t *>(stagedBuffers.selectedExperts.contents());

  // Step 2b: Find unique experts across all rows
  std::vector<uint32_t> uniqueExperts;
  std::vector<int32_t> expertToSlot(shape.experts, -1);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t k = 0; k < shape.expertsPerToken; ++k) {
      uint32_t exp = stagedSelPtr[r * shape.routesPerToken() + k];
      if (expertToSlot[exp] == -1) {
        expertToSlot[exp] = static_cast<int32_t>(uniqueExperts.size());
        uniqueExperts.push_back(exp);
      }
    }
  }

  std::cout << "[Staged] GPU Route unique experts count across " << rows
            << " rows: " << uniqueExperts.size() << "\n";

  // Verify that router selected identical experts
  for (uint32_t i = 0; i < rows * shape.routesPerToken(); ++i) {
    if (stagedSelPtr[i] != refSelPtr[i]) {
      std::cerr << "FAIL: router mismatch at index " << i << "\n";
      return 1;
    }
  }

  // Step 2c: Stage the active unique experts into cache buffer!
  const uint64_t cacheBytes = std::max<uint64_t>(1, uniqueExperts.size()) * expertStride;
  MetalBuffer cacheGate = backend.allocateBuffer(cacheBytes, BufferStorage::Shared, "cache-gate");
  MetalBuffer cacheUp = backend.allocateBuffer(cacheBytes, BufferStorage::Shared, "cache-up");
  MetalBuffer cacheDown = backend.allocateBuffer(cacheBytes, BufferStorage::Shared, "cache-down");

  char *cg = static_cast<char *>(cacheGate.contents());
  char *cu = static_cast<char *>(cacheUp.contents());
  char *cd = static_cast<char *>(cacheDown.contents());
  const char *fg = reinterpret_cast<const char *>(fullGate.data());
  const char *fu = reinterpret_cast<const char *>(fullUp.data());
  const char *fd = reinterpret_cast<const char *>(fullDown.data());

  for (size_t slot = 0; slot < uniqueExperts.size(); ++slot) {
    uint32_t exp = uniqueExperts[slot];
    std::memcpy(cg + slot * expertStride, fg + exp * expertStride, expertStride);
    std::memcpy(cu + slot * expertStride, fu + exp * expertStride, expertStride);
    std::memcpy(cd + slot * expertStride, fd + exp * expertStride, expertStride);
  }

  // Overwrite selected with the slot index!
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t k = 0; k < shape.expertsPerToken; ++k) {
      uint32_t exp = stagedSelPtr[r * shape.routesPerToken() + k];
      stagedSelPtr[r * shape.routesPerToken() + k] = expertToSlot[exp];
    }
    // Shared expert index is 512
    stagedSelPtr[r * shape.routesPerToken() + shape.expertsPerToken] = 512;
  }

  // Build cache weights:
  ExpertQ4Projection cGateProj{
      .packed = cacheGate,
      .experts = static_cast<uint32_t>(uniqueExperts.size()),
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection cUpProj{
      .packed = cacheUp,
      .experts = static_cast<uint32_t>(uniqueExperts.size()),
      .outputSize = 640,
      .inputSize = 2560,
      .expertStrideBytes = expertStride,
  };
  ExpertQ4Projection cDownProj{
      .packed = cacheDown,
      .experts = static_cast<uint32_t>(uniqueExperts.size()),
      .outputSize = 2560,
      .inputSize = 640,
      .expertStrideBytes = expertStride,
  };

  MoeWeights cacheWeights{
      .router = router,
      .expertGate = cGateProj,
      .expertUp = cUpProj,
      .expertDown = cDownProj,
      .sharedGate = sharedGate,
      .sharedUp = sharedUp,
      .sharedDown = sharedDown,
      .sharedExpertGate = sharedExpertGate,
  };

  // Step 2d: Run MoE execution using the staged cache!
  CommandGraph graphExecute;
  MoE::addExecute(graphExecute, stagedBuffers, cacheWeights, plan, /*addResidual=*/false);
  (void)backend.submitCommandAsync(graphExecute.dispatches()).wait();

  const auto *stagedOutPtr = static_cast<const uint16_t *>(stagedBuffers.output.contents());

  // Compare outputs across all rows
  float maxDiff = 0.0f;
  for (size_t i = 0; i < rows * shape.hiddenSize; ++i) {
    float r = fromBf16(refOutPtr[i]);
    float s = fromBf16(stagedOutPtr[i]);
    float diff = std::abs(r - s);
    if (diff > maxDiff) maxDiff = diff;
  }

  std::cout << "Max absolute difference between Reference and Staged Cache: "
            << maxDiff << "\n";

  if (maxDiff != 0.0f) {
    std::cerr << "FAIL: outputs do not match exactly!\n";
    return 1;
  }

  std::cout << "PASS: Staged streaming expert cache matches reference with ZERO difference!\n";
  return 0;
}

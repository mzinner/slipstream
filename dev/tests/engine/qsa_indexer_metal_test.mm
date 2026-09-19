// Checks the qwen4exp sparse-attention indexer's block scoring.
//
// The CPU reference transcribes Qwen4ExpTextQSAIndexer from the upstream
// modelling code (transformers 5.16.1). That transcription was checked once
// against the real module run under torch, where the selected-token mask it
// produced matched exactly, so what remains to test here is the kernel
// against the transcription.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

struct Params {
  uint32_t blocks;
  uint32_t heads;
  uint32_t headDimension;
  uint32_t compressRatio;
  uint32_t rotaryDimension;
  float epsilon;
};

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

// Transcribed from Qwen4ExpTextQSAIndexer.forward.
std::vector<float> cpuScores(const Params &p,
                             const std::vector<uint16_t> &queries,
                             const std::vector<uint16_t> &keys,
                             const std::vector<uint32_t> &starts,
                             const std::vector<uint16_t> &gain,
                             const std::vector<uint16_t> &cosTable,
                             const std::vector<uint16_t> &sinTable) {
  const uint32_t d = p.headDimension;
  const uint32_t rotaryHalf = p.rotaryDimension / 2;
  std::vector<float> scores(p.blocks, 0.0f);
  std::vector<float> pooled(d), rotated(d);

  for (uint32_t block = 0; block < p.blocks; ++block) {
    const uint32_t start = starts[block];
    for (uint32_t i = 0; i < d; ++i) {
      float sum = 0.0f;
      for (uint32_t token = 0; token < p.compressRatio; ++token)
        sum += fromBf16(keys[size_t(start + token) * d + i]);
      pooled[i] = sum / float(p.compressRatio);
    }
    float square = 0.0f;
    for (uint32_t i = 0; i < d; ++i) square += pooled[i] * pooled[i];
    const float scale = 1.0f / std::sqrt(square / float(d) + p.epsilon);
    for (uint32_t i = 0; i < d; ++i)
      rotated[i] = pooled[i] * scale * (1.0f + fromBf16(gain[i]));

    std::vector<float> source(rotated);
    for (uint32_t i = 0; i < p.rotaryDimension; ++i) {
      const float c =
          fromBf16(cosTable[size_t(start) * p.rotaryDimension + i]);
      const float s =
          fromBf16(sinTable[size_t(start) * p.rotaryDimension + i]);
      const float partner =
          i < rotaryHalf ? -source[i + rotaryHalf] : source[i - rotaryHalf];
      rotated[i] = source[i] * c + partner * s;
    }
    float total = 0.0f;
    for (uint32_t head = 0; head < p.heads; ++head) {
      float dot = 0.0f;
      for (uint32_t i = 0; i < d; ++i)
        dot += fromBf16(queries[size_t(head) * d + i]) * rotated[i];
      total += std::max(dot, 0.0f);
    }
    scores[block] = total / std::sqrt(float(d));
  }
  return scores;
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: qsa-indexer <metallib>");
    MetalBackend backend(argv[1]);

    // Production geometry: 4 index heads of 128, blocks of 4 tokens. The
    // rotary width equals the index head dimension, which the upstream
    // configuration validates.
    const Params params{4096, 4, 128, 4, 128, 1e-6f};
    const uint32_t tokens = params.blocks * params.compressRatio;

    std::mt19937 engine(20260919);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto sample = [&](size_t count, float scale) {
      std::vector<uint16_t> out(count);
      for (uint16_t &value : out) value = toBf16(normal(engine) * scale);
      return out;
    };
    std::vector<uint16_t> queries =
        sample(size_t(params.heads) * params.headDimension, 0.4f);
    std::vector<uint16_t> keys =
        sample(size_t(tokens) * params.headDimension, 0.5f);
    std::vector<uint16_t> gain(params.headDimension);
    for (uint16_t &value : gain) value = toBf16(normal(engine) * 0.05f);
    std::vector<uint16_t> cosTable(size_t(tokens) * params.rotaryDimension);
    std::vector<uint16_t> sinTable(cosTable.size());
    for (size_t i = 0; i < cosTable.size(); ++i) {
      const float angle = float(i % 977) * 0.013f;
      cosTable[i] = toBf16(std::cos(angle));
      sinTable[i] = toBf16(std::sin(angle));
    }
    std::vector<uint32_t> starts(params.blocks);
    for (uint32_t block = 0; block < params.blocks; ++block)
      starts[block] = block * params.compressRatio;

    MetalBuffer queryBuffer = upload(backend, queries.data(),
                                     queries.size() * 2, "qsa-queries");
    MetalBuffer keyBuffer =
        upload(backend, keys.data(), keys.size() * 2, "qsa-keys");
    MetalBuffer startBuffer =
        upload(backend, starts.data(), starts.size() * 4, "qsa-starts");
    MetalBuffer gainBuffer =
        upload(backend, gain.data(), gain.size() * 2, "qsa-gain");
    MetalBuffer cosBuffer =
        upload(backend, cosTable.data(), cosTable.size() * 2, "qsa-cos");
    MetalBuffer sinBuffer =
        upload(backend, sinTable.data(), sinTable.size() * 2, "qsa-sin");
    MetalBuffer scoreBuffer = backend.allocateBuffer(
        size_t(params.blocks) * 4, BufferStorage::Shared, "qsa-scores");

    CommandGraph graph;
    graph.add("qsa_indexer_score",
              {queryBuffer, keyBuffer, startBuffer, gainBuffer, cosBuffer,
               sinBuffer, scoreBuffer},
              params, {params.blocks, 1, 1}, {128, 1, 1});
    (void)backend.submitCommand(graph.dispatches());

    const std::vector<float> want =
        cpuScores(params, queries, keys, starts, gain, cosTable, sinTable);
    const float *got = static_cast<const float *>(scoreBuffer.contents());

    double worst = 0.0;
    uint32_t nonZero = 0;
    for (uint32_t block = 0; block < params.blocks; ++block) {
      if (want[block] > 0.0f) ++nonZero;
      const double scale =
          std::max({std::abs(double(got[block])), std::abs(double(want[block])),
                    1e-3});
      worst = std::max(worst, std::abs(got[block] - want[block]) / scale);
    }
    std::cout << "qsa indexer, " << params.blocks << " blocks of "
              << params.compressRatio << ", " << params.heads << " heads of "
              << params.headDimension << '\n'
              << "  blocks with a positive score " << nonZero << " of "
              << params.blocks << '\n'
              << "  worst relative error " << worst << '\n';

    // relu zeroes many scores, so require that the test is actually
    // exercising the scoring path rather than comparing zeroes.
    require(nonZero > params.blocks / 4,
            "too few blocks scored above zero to be a meaningful comparison");
    require(worst < 0.02, "block scores do not match the reference");

    std::cout << "qsa indexer metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "qsa indexer metal test failed: " << error.what() << '\n';
    return 1;
  }
}

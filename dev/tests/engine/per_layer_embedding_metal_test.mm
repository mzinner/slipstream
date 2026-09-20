// Checks the gate and convolution half of the qwen4exp per-layer embedding.
//
// The CPU reference transcribes Qwen4ExpTextPLELayer from the upstream
// modelling code (transformers 5.16.1). That transcription was checked once
// against the real module run under torch, where it agreed to 3e-08, so what
// remains to test here is the kernels against it.

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
  uint32_t rows;
  uint32_t hidden;
  uint32_t count;
  uint32_t taps;
  uint32_t dilation;
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

MetalBuffer upload(MetalBackend &backend, const std::vector<uint16_t> &data,
                   const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(data.size() * 2,
                                              BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data.data(), data.size() * 2);
  return buffer;
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: per-layer-embedding <metallib>");
    MetalBackend backend(argv[1]);

    // Production geometry: four streams of 2560, four taps dilated by the
    // n-gram order of three, so nine rows of state.
    const Params params{12, 2560, 4, 4, 3, 1e-6f};
    const uint32_t width = params.count * params.hidden;
    const uint32_t state = (params.taps - 1) * params.dilation;

    std::mt19937 engine(20260919);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto sample = [&](size_t count, float scale) {
      std::vector<uint16_t> out(count);
      for (uint16_t &value : out) value = toBf16(normal(engine) * scale);
      return out;
    };
    std::vector<uint16_t> keys = sample(size_t(params.rows) * width, 0.5f);
    std::vector<uint16_t> values =
        sample(size_t(params.rows) * params.hidden, 0.4f);
    std::vector<uint16_t> hidden = sample(size_t(params.rows) * width, 0.5f);
    std::vector<uint16_t> keyGain = sample(width, 0.05f);
    std::vector<uint16_t> queryGain = sample(width, 0.05f);
    std::vector<uint16_t> convGain = sample(width, 0.05f);
    std::vector<uint16_t> convWeights = sample(size_t(width) * params.taps, 0.2f);

    MetalBuffer keyBuffer = upload(backend, keys, "ple-keys");
    MetalBuffer valueBuffer = upload(backend, values, "ple-values");
    MetalBuffer hiddenBuffer = upload(backend, hidden, "ple-hidden");
    MetalBuffer keyGainBuffer = upload(backend, keyGain, "ple-key-gain");
    MetalBuffer queryGainBuffer = upload(backend, queryGain, "ple-query-gain");
    MetalBuffer convGainBuffer = upload(backend, convGain, "ple-conv-gain");
    MetalBuffer convBuffer = upload(backend, convWeights, "ple-conv");
    MetalBuffer gated = backend.allocateBuffer(
        size_t(params.rows) * width * 2, BufferStorage::Shared, "ple-gated");
    // The leading state rows are the caller's; zero here means a fresh
    // sequence, which is what the reference pads with.
    MetalBuffer normalized = backend.allocateBuffer(
        size_t(params.rows + state) * width * 2, BufferStorage::Shared,
        "ple-normalized");
    std::memset(normalized.contents(), 0, normalized.sizeBytes());

    CommandGraph graph;
    graph.add("per_layer_embedding_gate",
              {keyBuffer, valueBuffer, hiddenBuffer, keyGainBuffer,
               queryGainBuffer, convGainBuffer, gated, normalized},
              params, {params.rows, 1, 1}, {256, 1, 1});
    graph.add("per_layer_embedding_convolve",
              {normalized, convBuffer, gated}, params, {64, 1, 1},
              {256, 1, 1});
    (void)backend.submitCommand(graph.dispatches());

    // ---- reference ----
    std::vector<float> reference(size_t(params.rows) * width);
    std::vector<float> normedReference(size_t(params.rows + state) * width, 0.0f);
    for (uint32_t row = 0; row < params.rows; ++row) {
      std::vector<float> gates(params.count);
      for (uint32_t stream = 0; stream < params.count; ++stream) {
        double keySquare = 0.0, querySquare = 0.0;
        for (uint32_t i = 0; i < params.hidden; ++i) {
          const size_t index = size_t(row) * width + stream * params.hidden + i;
          keySquare += double(fromBf16(keys[index])) * fromBf16(keys[index]);
          querySquare += double(fromBf16(hidden[index])) * fromBf16(hidden[index]);
        }
        const float keyScale =
            1.0f / std::sqrt(float(keySquare / params.hidden) + params.epsilon);
        const float queryScale =
            1.0f / std::sqrt(float(querySquare / params.hidden) + params.epsilon);
        float dot = 0.0f;
        for (uint32_t i = 0; i < params.hidden; ++i) {
          const uint32_t within = stream * params.hidden + i;
          const size_t index = size_t(row) * width + within;
          dot += (fromBf16(keys[index]) * keyScale *
                  (1.0f + fromBf16(keyGain[within]))) *
                 (fromBf16(hidden[index]) * queryScale *
                  (1.0f + fromBf16(queryGain[within])));
        }
        dot /= std::sqrt(float(params.hidden));
        const float magnitude = std::sqrt(std::max(std::abs(dot), 1e-6f));
        gates[stream] = dot < 0.0f ? -magnitude : magnitude;
      }
      for (uint32_t i = 0; i < width; ++i) {
        const float gate = 1.0f / (1.0f + std::exp(-gates[i / params.hidden]));
        reference[size_t(row) * width + i] = fromBf16(toBf16(
            gate * fromBf16(values[size_t(row) * params.hidden +
                                   i % params.hidden])));
      }
      for (uint32_t stream = 0; stream < params.count; ++stream) {
        double square = 0.0;
        for (uint32_t i = 0; i < params.hidden; ++i) {
          const float value =
              reference[size_t(row) * width + stream * params.hidden + i];
          square += double(value) * value;
        }
        const float scale =
            1.0f / std::sqrt(float(square / params.hidden) + params.epsilon);
        for (uint32_t i = 0; i < params.hidden; ++i) {
          const uint32_t within = stream * params.hidden + i;
          normedReference[size_t(row + state) * width + within] =
              fromBf16(toBf16(reference[size_t(row) * width + within] * scale *
                              (1.0f + fromBf16(convGain[within]))));
        }
      }
    }
    for (uint32_t row = 0; row < params.rows; ++row)
      for (uint32_t channel = 0; channel < width; ++channel) {
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < params.taps; ++tap) {
          const uint32_t source =
              row + state - (params.taps - 1 - tap) * params.dilation;
          sum += normedReference[size_t(source) * width + channel] *
                 fromBf16(convWeights[size_t(channel) * params.taps + tap]);
        }
        reference[size_t(row) * width + channel] +=
            sum / (1.0f + std::exp(-sum));
      }

    const uint16_t *got = static_cast<const uint16_t *>(gated.contents());
    double worst = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
      const double scale = std::max({std::abs(double(fromBf16(got[i]))),
                                     std::abs(double(reference[i])), 1e-3});
      worst = std::max(worst, std::abs(fromBf16(got[i]) - reference[i]) / scale);
    }

    std::cout << "per-layer embedding, " << params.count << " streams of "
              << params.hidden << ", " << params.taps << " taps dilated by "
              << params.dilation << '\n'
              << "  convolution state rows  " << state << '\n'
              << "  worst relative error    " << worst << '\n';
    require(worst < 0.02, "the gated output does not match the reference");

    std::cout << "per layer embedding metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "per layer embedding metal test failed: " << error.what()
              << '\n';
    return 1;
  }
}

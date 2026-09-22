// Checks the qwen4exp hyper-connection kernels against a CPU reference.
//
// The CPU reference transcribes Qwen4ExpTextGatedResidual from the upstream
// modelling code (transformers 5.16.1); the formulas are also recorded on
// Qwen4ExpHyperConnection. That transcription was checked once against the
// real module run under torch and agreed exactly, so what remains to test
// here is the kernels against the transcription. Inputs are generated
// deterministically, so the test needs no data files.
//
// Note the norm gain is stored as an offset from one and is zero-initialized
// upstream: the neutral weight is zero, not one. Using it directly silently
// doubles every gain.

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
  uint32_t lowRank;
  float epsilon;
  uint32_t withInject;
  uint32_t rowStep;
  uint32_t splits;
};

uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint32_t rounding = ((bits >> 16) & 1) + 0x7FFF;
  return uint16_t((bits + rounding) >> 16);
}
float fromBf16(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

MetalBuffer bufferFrom(MetalBackend &backend, const std::vector<uint16_t> &data,
                       const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(
      data.size() * sizeof(uint16_t), BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data.data(), data.size() * sizeof(uint16_t));
  return buffer;
}
uint16_t *hostOf(MetalBuffer &buffer) {
  return static_cast<uint16_t *>(buffer.contents());
}

// The package's 8-bit mix weights, in the matrix kernels' tiled layout: rows
// padded to a multiple of 256, codes as [256-row tile][group of 64][row][64],
// and per (tile, group, row) a bf16 bias (the group minimum) and scale
// (range / 255), as convert_qwen4exp.quantized_q8_tile writes them.
// `dequantized` (unpadded, row-major) gets the exact values the kernels use.
struct Quantized {
  std::vector<uint8_t> codes;
  std::vector<uint16_t> scales, biases;
  std::vector<float> dequantized;
};
Quantized quantize8(const std::vector<uint16_t> &values, size_t rows, size_t columns) {
  const size_t padded = (rows + 255) / 256 * 256, groups = columns / 64;
  Quantized q;
  q.codes.assign(padded * columns, 0);
  q.scales.assign(padded * groups, toBf16(1.0f));
  q.biases.assign(padded * groups, 0);
  q.dequantized.resize(rows * columns);
  for (size_t row = 0; row < rows; ++row)
    for (size_t group = 0; group < groups; ++group) {
      const size_t first = row * columns + group * 64;
      float low = INFINITY, high = -INFINITY;
      for (size_t i = 0; i < 64; ++i) {
        low = std::min(low, fromBf16(values[first + i]));
        high = std::max(high, fromBf16(values[first + i]));
      }
      const float scale = fromBf16(toBf16(high > low ? (high - low) / 255.0f : 1.0f));
      const float bias = fromBf16(toBf16(low));
      const size_t parameter = ((row / 256) * groups + group) * 256 + row % 256;
      q.scales[parameter] = toBf16(scale);
      q.biases[parameter] = toBf16(bias);
      for (size_t i = 0; i < 64; ++i) {
        const float code = std::clamp(
            std::nearbyint((fromBf16(values[first + i]) - bias) / scale), 0.0f, 255.0f);
        q.codes[parameter * 64 + i] = static_cast<uint8_t>(code);
        q.dequantized[first + i] = code * scale + bias;
      }
    }
  return q;
}

template <class T>
MetalBuffer bufferOf(MetalBackend &backend, const std::vector<T> &data, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(data.size() * sizeof(T), BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data.data(), data.size() * sizeof(T));
  return buffer;
}

// Transcribed from Qwen4ExpTextGatedResidual.forward.
struct Reference {
  std::vector<float> mixed, injection, updated;
};
Reference cpuReference(const Params &p, const std::vector<uint16_t> &x,
                       const std::vector<uint16_t> &gain,
                       const std::vector<float> &down,
                       const std::vector<float> &up,
                       const std::vector<uint16_t> &inject,
                       const std::vector<uint16_t> &block) {
  const uint32_t width = p.count * p.hidden;
  Reference out;
  out.mixed.assign(size_t(p.rows) * p.hidden, 0.0f);
  out.injection.assign(size_t(p.rows) * p.count, 0.0f);
  out.updated.assign(size_t(p.rows) * width, 0.0f);

  std::vector<float> normalized(width);
  for (uint32_t row = 0; row < p.rows; ++row) {
    const uint16_t *xr = x.data() + size_t(row) * width;
    for (uint32_t stream = 0; stream < p.count; ++stream) {
      double sum = 0.0;
      for (uint32_t i = 0; i < p.hidden; ++i) {
        const float value = fromBf16(xr[stream * p.hidden + i]);
        sum += double(value) * value;
      }
      const float scale =
          1.0f / std::sqrt(float(sum / double(p.hidden)) + p.epsilon);
      for (uint32_t i = 0; i < p.hidden; ++i) {
        const uint32_t index = stream * p.hidden + i;
        normalized[index] =
            fromBf16(xr[index]) * scale * (1.0f + fromBf16(gain[index]));
      }
    }
    // Metal rounds the normalized row to bf16 before reusing it.
    for (float &value : normalized) value = fromBf16(toBf16(value));

    std::vector<float> low(p.lowRank);
    for (uint32_t o = 0; o < p.lowRank; ++o) {
      float sum = 0.0f;
      for (uint32_t i = 0; i < width; ++i)
        sum += down[size_t(o) * width + i] * normalized[i];
      const float v = sum / float(p.count);
      low[o] = fromBf16(toBf16(v / (1.0f + std::exp(-v))));
    }
    for (uint32_t dimension = 0; dimension < p.hidden; ++dimension) {
      float accumulated = 0.0f;
      for (uint32_t stream = 0; stream < p.count; ++stream) {
        const uint32_t index = stream * p.hidden + dimension;
        float sum = 0.0f;
        for (uint32_t r = 0; r < p.lowRank; ++r)
          sum += up[size_t(index) * p.lowRank + r] * low[r];
        accumulated += (1.0f / (1.0f + std::exp(-sum))) * normalized[index];
      }
      out.mixed[size_t(row) * p.hidden + dimension] =
          accumulated / float(p.count);
    }
    for (uint32_t stream = 0; stream < p.count; ++stream) {
      float sum = 0.0f;
      for (uint32_t i = 0; i < width; ++i)
        sum += fromBf16(inject[size_t(stream) * width + i]) * normalized[i];
      const float v = sum / float(p.count);
      out.injection[size_t(row) * p.count + stream] =
          2.0f / (1.0f + std::exp(-v));
    }
    for (uint32_t stream = 0; stream < p.count; ++stream)
      for (uint32_t i = 0; i < p.hidden; ++i) {
        const uint32_t index = stream * p.hidden + i;
        out.updated[size_t(row) * width + index] =
            fromBf16(xr[index]) +
            fromBf16(toBf16(out.injection[size_t(row) * p.count + stream])) *
                fromBf16(block[size_t(row) * p.hidden + i]);
      }
  }
  return out;
}

double worstRelative(const uint16_t *got, const std::vector<float> &want,
                     size_t count, const char *label) {
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double a = fromBf16(got[i]), b = want[i];
    const double scale = std::max({std::abs(a), std::abs(b), 1e-3});
    worst = std::max(worst, std::abs(a - b) / scale);
  }
  // Relative error on near-zero values measures cancellation, not the
  // kernel, so the check is the worst absolute error against the typical
  // size of the output.
  double squares = 0.0, absolute = 0.0;
  for (size_t i = 0; i < count; ++i) {
    squares += double(want[i]) * want[i];
    absolute = std::max(absolute, std::abs(fromBf16(got[i]) - double(want[i])));
  }
  const double normalized = absolute / std::sqrt(squares / double(count));
  std::cout << "  " << label << " worst relative error " << worst
            << ", worst error / rms " << normalized << '\n';
  return normalized;
}

void runCase(MetalBackend &backend, uint32_t rows) {
    // Production geometry: four streams of 2560, low rank 320.
    // Decode-sized calls split the down projection 10 ways, as production does.
    const uint32_t splits = rows <= 32 ? 10 : 1;
    const Params params{rows, 2560, 4, 320, 1e-6f, 1, 1, splits};
    const uint32_t width = params.count * params.hidden;

    std::mt19937 engine(20260919);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto sample = [&](size_t count, float scale) {
      std::vector<uint16_t> out(count);
      for (uint16_t &value : out) value = toBf16(normal(engine) * scale);
      return out;
    };
    std::vector<uint16_t> x = sample(size_t(params.rows) * width, 0.7f);
    std::vector<uint16_t> gain(width);
    // Upstream stores an offset from one, zero-initialized.
    for (uint16_t &value : gain) value = toBf16(normal(engine) * 0.05f);
    std::vector<uint16_t> down = sample(size_t(params.lowRank) * width, 0.05f);
    std::vector<uint16_t> up = sample(size_t(width) * params.lowRank, 0.05f);
    std::vector<uint16_t> inject = sample(size_t(params.count) * width, 0.05f);
    std::vector<uint16_t> block = sample(size_t(params.rows) * params.hidden,
                                         0.3f);

    MetalBuffer xBuffer = bufferFrom(backend, x, "hc-residual");
    MetalBuffer gainBuffer = bufferFrom(backend, gain, "hc-gain");
    const Quantized down8 = quantize8(down, params.lowRank, width);
    const Quantized up8 = quantize8(up, width, params.lowRank);
    MetalBuffer downBuffer = bufferOf(backend, down8.codes, "hc-down");
    MetalBuffer downScales = bufferOf(backend, down8.scales, "hc-down-s");
    MetalBuffer downBiases = bufferOf(backend, down8.biases, "hc-down-b");
    MetalBuffer upBuffer = bufferOf(backend, up8.codes, "hc-up");
    MetalBuffer upScales = bufferOf(backend, up8.scales, "hc-up-s");
    MetalBuffer upBiases = bufferOf(backend, up8.biases, "hc-up-b");
    const uint32_t outputs = params.lowRank + params.count;
    MetalBuffer partials = backend.allocateBuffer(
        size_t(10) * std::max(rows, 1u) * outputs * 4, BufferStorage::Shared, "hc-partials");
    MetalBuffer injectBuffer = bufferFrom(backend, inject, "hc-inject");
    MetalBuffer blockBuffer = bufferFrom(backend, block, "hc-block");
    MetalBuffer normalized = backend.allocateBuffer(
        size_t(params.rows) * width * 2, BufferStorage::Shared, "hc-normed");
    MetalBuffer reduced = backend.allocateBuffer(
        size_t(params.rows) * params.lowRank * 2, BufferStorage::Shared,
        "hc-reduced");
    MetalBuffer mixed = backend.allocateBuffer(
        size_t(params.rows) * params.hidden * 2, BufferStorage::Shared,
        "hc-mixed");
    MetalBuffer injection = backend.allocateBuffer(
        size_t(params.rows) * params.count * 2, BufferStorage::Shared,
        "hc-injection");

    CommandGraph graph;
    // The production path: normalize, then down with the injection gates,
    // then up and the stream mix, then the residual update.
    graph.add("hyper_connection_rms", {xBuffer, gainBuffer, normalized}, params,
              {params.rows, params.count, 1}, {256, 1, 1});
    graph.add("hyper_connection_down",
              {normalized, downBuffer, downScales, downBiases, injectBuffer,
               reduced, injection, partials},
              params, {(outputs + 7) / 8, splits, 1}, {256, 1, 1});
    if (splits > 1)
      graph.add("hyper_connection_down_finish", {partials, reduced, injection},
                params, {(rows * outputs + 255) / 256, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_up_mix",
              {normalized, reduced, upBuffer, upScales, upBiases, mixed}, params,
              {params.hidden / 8, 1, 1}, {256, 1, 1});
    graph.add("hyper_connection_update",
              {xBuffer, blockBuffer, injection}, params, {64, 1, 1},
              {256, 1, 1});
    (void)backend.submitCommand(graph.dispatches());

    const Reference reference =
        cpuReference(params, x, gain, down8.dequantized, up8.dequantized, inject, block);

    std::cout << "hyper-connection, " << rows << " rows, " << params.count << " streams of "
              << params.hidden << ", low rank " << params.lowRank << '\n';
    const double mixedError =
        worstRelative(hostOf(mixed), reference.mixed,
                      size_t(params.rows) * params.hidden, "mixed input   ");
    const double injectionError =
        worstRelative(hostOf(injection), reference.injection,
                      size_t(params.rows) * params.count, "injection gate");
    const double updatedError =
        worstRelative(hostOf(xBuffer), reference.updated,
                      size_t(params.rows) * width, "residual update");

    // bf16 carries eight mantissa bits, and these reduce over 10240 terms.
    require(mixedError < 0.02, "mixed input does not match the reference");
    require(injectionError < 0.02, "injection gate does not match");
    require(updatedError < 0.02, "residual update does not match");

}


} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: hyper-connection <metallib>");
    MetalBackend backend(argv[1]);

    for (uint32_t rows : {1u, 8u, 37u})
      runCase(backend, rows);
    std::cout << "hyper connection metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hyper connection metal test failed: " << error.what() << '\n';
    return 1;
  }
}

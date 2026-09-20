// Checks the hashed n-gram lookup for the qwen4exp per-layer embedding.
//
// The CPU reference transcribes Qwen4ExpTextNGramEmbedding from the upstream
// modelling code (transformers 5.16.1). That transcription was checked once
// against the real module run under torch, where the gathered rows matched
// exactly, so what remains to test here is the kernel against it.
//
// The table is quantized in 32-element groups rather than the usual 64,
// because a head is 160 wide and 160 is not a whole number of 64s.

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
  uint32_t heads;
  uint32_t headDimension;
  uint32_t ngramSize;
  uint32_t headsPerOrder;
  uint32_t groupElements;
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

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: ngram-embedding <metallib>");
    MetalBackend backend(argv[1]);

    // Production shape, on a table small enough to hold: sixteen heads of
    // 160 for an n-gram order of three, eight heads per order.
    const Params params{24, 16, 160, 3, 8, 32};
    const uint32_t tableRows = 40'000;

    std::mt19937_64 engine(20260919);
    std::vector<uint32_t> shifted(size_t(params.ngramSize) * params.rows);
    for (uint32_t &value : shifted) value = uint32_t(engine() % 248320u);

    // Multipliers of the magnitude the real ones have, so the products land
    // near the top of the signed 64-bit range as they do in production.
    std::vector<int64_t> multipliers(params.ngramSize);
    for (int64_t &value : multipliers)
      value = int64_t(engine() % 30'000'000'000'000ULL) + 1'000'000'000'000LL;

    std::vector<int64_t> vocabulary(params.heads), offsets(params.heads);
    int64_t running = 0;
    for (uint32_t head = 0; head < params.heads; ++head) {
      vocabulary[head] = 2003 + 2 * int64_t(head);
      offsets[head] = running;
      running += vocabulary[head];
    }
    require(running <= tableRows, "the synthetic table must hold every head");

    const size_t elements = size_t(tableRows) * params.headDimension;
    std::vector<uint8_t> table(elements / 2);
    for (uint8_t &value : table) value = uint8_t(engine() & 0xFF);
    std::vector<uint16_t> scales(elements / params.groupElements);
    std::vector<uint16_t> biases(scales.size());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::mt19937 small(7);
    for (size_t i = 0; i < scales.size(); ++i) {
      scales[i] = toBf16(normal(small) * 0.01f);
      biases[i] = toBf16(normal(small) * 0.1f);
    }

    const size_t outputCount =
        size_t(params.rows) * params.heads * params.headDimension;
    MetalBuffer shiftedBuffer =
        upload(backend, shifted.data(), shifted.size() * 4, "ngram-shifted");
    MetalBuffer multiplierBuffer = upload(backend, multipliers.data(),
                                          multipliers.size() * 8, "ngram-mult");
    MetalBuffer vocabularyBuffer = upload(backend, vocabulary.data(),
                                          vocabulary.size() * 8, "ngram-vocab");
    MetalBuffer offsetBuffer =
        upload(backend, offsets.data(), offsets.size() * 8, "ngram-offsets");
    MetalBuffer tableBuffer =
        upload(backend, table.data(), table.size(), "ngram-table");
    MetalBuffer scaleBuffer =
        upload(backend, scales.data(), scales.size() * 2, "ngram-scales");
    MetalBuffer biasBuffer =
        upload(backend, biases.data(), biases.size() * 2, "ngram-biases");
    MetalBuffer outputBuffer = backend.allocateBuffer(
        outputCount * 2, BufferStorage::Shared, "ngram-output");

    CommandGraph graph;
    graph.add("ngram_embedding_gather",
              {shiftedBuffer, multiplierBuffer, vocabularyBuffer, offsetBuffer,
               tableBuffer, scaleBuffer, biasBuffer, outputBuffer},
              params, {params.rows, 1, 1}, {128, 1, 1});
    (void)backend.submitCommand(graph.dispatches());

    const uint16_t *got = static_cast<const uint16_t *>(outputBuffer.contents());
    double worst = 0.0;
    uint32_t distinctRows = 0;
    std::vector<int64_t> seen;
    for (uint32_t row = 0; row < params.rows; ++row) {
      for (uint32_t head = 0; head < params.heads; ++head) {
        const uint32_t order = head / params.headsPerOrder + 2;
        int64_t mixed = int64_t(shifted[size_t(0) * params.rows + row]) *
                        multipliers[0];
        for (uint32_t position = 1; position < order; ++position)
          mixed ^= int64_t(shifted[size_t(position) * params.rows + row]) *
                   multipliers[position];
        int64_t remainder = mixed % vocabulary[head];
        if (remainder < 0) remainder += vocabulary[head];
        const int64_t entry = remainder + offsets[head];
        require(entry >= 0 && entry < tableRows, "a gathered row is out of range");
        if (std::find(seen.begin(), seen.end(), entry) == seen.end()) {
          seen.push_back(entry);
          ++distinctRows;
        }
        for (uint32_t i = 0; i < params.headDimension; ++i) {
          const size_t weightIndex =
              size_t(entry) * params.headDimension / 2 + i / 2;
          const uint8_t packed = table[weightIndex];
          const float quantized = float((packed >> ((i & 1) * 4)) & 0xF);
          const size_t parameter =
              size_t(entry) * (params.headDimension / params.groupElements) +
              i / params.groupElements;
          // The kernel dequantizes in float and stores bf16, so round the
          // reference the same way rather than comparing across widths.
          const float want = fromBf16(toBf16(quantized *
                                                 fromBf16(scales[parameter]) +
                                             fromBf16(biases[parameter])));
          const size_t index =
              (size_t(row) * params.heads + head) * params.headDimension + i;
          const double scale = std::max({std::abs(double(want)),
                                         std::abs(double(fromBf16(got[index]))),
                                         1e-3});
          worst = std::max(worst, std::abs(fromBf16(got[index]) - want) / scale);
        }
      }
    }

    std::cout << "ngram embedding, " << params.heads << " heads of "
              << params.headDimension << ", order " << params.ngramSize << '\n'
              << "  distinct rows gathered  " << distinctRows << " of "
              << params.rows * params.heads << " lookups\n"
              << "  worst relative error    " << worst << '\n';
    // If the hash collapsed, the comparison would be near-vacuous.
    require(distinctRows > params.rows * params.heads / 2,
            "the hash is collapsing rows; the comparison would be weak");
    require(worst < 1e-6, "gathered rows do not match the reference");
    // A wrong row would differ by far more than a rounding step, so this is a
    // gather check, not a precision one.

    std::cout << "ngram embedding metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ngram embedding metal test failed: " << error.what() << '\n';
    return 1;
  }
}

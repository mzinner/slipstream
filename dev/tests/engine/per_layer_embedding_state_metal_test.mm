// Checks the history the qwen4exp per-layer embedding carries between steps.
//
// Two kernels: per_layer_embedding_shift builds each row's n-gram window from
// the stored token pair and this step's tokens, and per_layer_embedding_commit
// writes the history the next step starts from, keeping only committed rows.
//
// The window is checked against a transcription of the reference's own
// algorithm (Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos: segments
// restart after an end-of-sequence token), not against the one-line rule the
// kernel reduces it to, so the reduction itself is under test.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/PerLayerEmbedding.h"

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
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kEos = 248044;
constexpr uint32_t kHeaderBytes = 16;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// The reference: history = [older, newer] (both end-of-sequence when fresh)
// followed by the tokens; shifted by k within the current segment, else the
// end-of-sequence token. A segment starts after the latest end-of-sequence
// token strictly before the position.
std::vector<uint32_t> referenceShift(uint32_t older, uint32_t newer,
                                     const std::vector<uint32_t> &tokens) {
  std::vector<uint32_t> history{older, newer};
  history.insert(history.end(), tokens.begin(), tokens.end());
  const uint32_t rows = static_cast<uint32_t>(tokens.size());
  std::vector<uint32_t> out(3 * rows);
  for (uint32_t row = 0; row < rows; ++row) {
    const int position = static_cast<int>(row) + 2;
    int previousEos = -1;
    for (int p = 0; p < position; ++p)
      if (history[p] == kEos)
        previousEos = p;
    const int segmentStart = previousEos + 1;
    for (int shift = 0; shift < 3; ++shift) {
      const int source = position - shift;
      const bool valid = position - segmentStart >= shift && source >= 0;
      out[shift * rows + row] = valid ? history[source] : kEos;
    }
  }
  return out;
}

void checkShift(MetalBackend &backend, bool fresh, uint32_t older,
                uint32_t newer, const std::vector<uint32_t> &tokens,
                const std::string &label) {
  const uint32_t rows = static_cast<uint32_t>(tokens.size());
  MetalBuffer tokenBuffer = shared(backend, rows * 4, "tokens");
  std::memcpy(tokenBuffer.contents(), tokens.data(), rows * 4);
  MetalBuffer state = shared(backend, kHeaderBytes, "state");
  auto *header = static_cast<uint32_t *>(state.contents());
  if (!fresh) {
    header[0] = 1;
    header[1] = older;
    header[2] = newer;
  }
  MetalBuffer shifted = shared(backend, 3 * rows * 4, "shifted");
  const PerLayerEmbeddingStateParams params{rows, 0, 0, kEos, rows, 0, 0, 0};
  CommandGraph graph;
  graph.add("per_layer_embedding_shift", {tokenBuffer, state, shifted}, params,
            {(rows + 63) / 64, 1, 1}, {64, 1, 1});
  (void)backend.submitCommand(graph.dispatches());
  const auto expected = referenceShift(fresh ? kEos : older,
                                       fresh ? kEos : newer, tokens);
  const auto *got = static_cast<const uint32_t *>(shifted.contents());
  for (uint32_t i = 0; i < 3 * rows; ++i)
    require(got[i] == expected[i],
            label + ": shift " + std::to_string(i / rows) + " row " +
                std::to_string(i % rows) + " is " + std::to_string(got[i]) +
                ", reference " + std::to_string(expected[i]));
  std::cout << label << ": window matches the reference\n";
}

void checkCommit(MetalBackend &backend, bool fromBuffer, uint32_t kept,
                 const std::string &label) {
  constexpr uint32_t rows = 8, width = 96, history = 9, lane = 1;
  std::mt19937 random(kept * 7 + fromBuffer);
  std::vector<uint32_t> tokens(rows);
  for (auto &token : tokens)
    token = random() % 1000;
  std::vector<uint16_t> normalized((history + rows) * width);
  for (auto &value : normalized)
    value = static_cast<uint16_t>(random());
  MetalBuffer tokenBuffer = shared(backend, rows * 4, "tokens");
  std::memcpy(tokenBuffer.contents(), tokens.data(), rows * 4);
  MetalBuffer normalizedBuffer =
      shared(backend, normalized.size() * 2, "normalized");
  std::memcpy(normalizedBuffer.contents(), normalized.data(),
              normalized.size() * 2);
  const uint64_t stateBytes = kHeaderBytes + uint64_t{history} * width * 2;
  MetalBuffer in = shared(backend, stateBytes, "state-in");
  auto *header = static_cast<uint32_t *>(in.contents());
  header[0] = 1;
  header[1] = 111;
  header[2] = 222;
  MetalBuffer out = shared(backend, stateBytes, "state-out");
  MetalBuffer retained = shared(backend, 4 * 4, "retained");
  static_cast<uint32_t *>(retained.contents())[lane] = kept;
  const PerLayerEmbeddingStateParams params{
      rows, width, history, kEos, fromBuffer ? 0 : kept,
      fromBuffer ? 1u : 0u, lane, 0};
  CommandGraph graph;
  graph.add("per_layer_embedding_commit",
            {tokenBuffer, normalizedBuffer, in, out, retained}, params,
            {4, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(graph.dispatches());

  const auto *outHeader = static_cast<const uint32_t *>(out.contents());
  std::vector<uint32_t> sequence{111, 222};
  sequence.insert(sequence.end(), tokens.begin(), tokens.begin() + kept);
  require(outHeader[0] == 1, label + ": history not marked valid");
  require(outHeader[1] == sequence[sequence.size() - 2] &&
              outHeader[2] == sequence.back(),
          label + ": token pair is not the last two kept");
  const auto *outRows = reinterpret_cast<const uint16_t *>(
      static_cast<const uint8_t *>(out.contents()) + kHeaderBytes);
  require(std::memcmp(outRows, normalized.data() + uint64_t{kept} * width,
                      uint64_t{history} * width * 2) == 0,
          label + ": convolution history is not rows kept..kept+9");
  std::cout << label << ": keeps " << kept << " rows correctly\n";
}

void run(const std::string &metallib) {
  MetalBackend backend(metallib);
  checkShift(backend, true, 0, 0, {5, 6, 7, 8}, "fresh sequence");
  checkShift(backend, false, 41, 42, {5, 6, 7}, "continuing sequence");
  checkShift(backend, false, 41, kEos, {5, 6, 7}, "history ends in eos");
  checkShift(backend, false, kEos, 42, {5, 6}, "history older is eos");
  checkShift(backend, false, 41, 42, {5, kEos, 7, 8, kEos, kEos, 9},
             "eos inside the rows");
  checkShift(backend, false, 41, 42, {kEos}, "single eos row");
  for (uint32_t kept : {1u, 3u, 8u}) {
    checkCommit(backend, false, kept, "constant retention");
    checkCommit(backend, true, kept, "retention from the verifier");
  }
  std::cout << "per_layer_embedding_state_metal_test passed\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: per-layer-embedding-state <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "per_layer_embedding_state_metal_test failed: "
                << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}

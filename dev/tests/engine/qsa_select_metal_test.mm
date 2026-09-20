// Checks block selection for the qwen4exp sparse-attention indexer.
//
// The selection is exact: the kernel binary-searches the IEEE bit pattern for
// the k-th largest score, which is valid because a score is a sum of relu
// terms and so is never negative. The reference here sorts, which is only
// affordable because the test is small.
//
// Three properties matter and each is checked separately: the selected set is
// the true top k, ties resolve by lowest index, and the result does not change
// between runs.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
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

struct Params {
  uint32_t blocks;
  uint32_t budget;
};

std::vector<uint32_t> runSelection(MetalBackend &backend,
                                   const std::vector<float> &scores,
                                   uint32_t budget) {
  const Params params{uint32_t(scores.size()), budget};
  MetalBuffer scoreBuffer = backend.allocateBuffer(
      scores.size() * 4, BufferStorage::Shared, "qsa-select-scores");
  std::memcpy(scoreBuffer.contents(), scores.data(), scores.size() * 4);
  MetalBuffer selected = backend.allocateBuffer(
      size_t(budget) * 4, BufferStorage::Shared, "qsa-selected");
  std::memset(selected.contents(), 0xFF, selected.sizeBytes());
  MetalBuffer count =
      backend.allocateBuffer(4, BufferStorage::Shared, "qsa-selected-count");
  std::memset(count.contents(), 0, 4);

  CommandGraph graph;
  graph.add("qsa_select_blocks", {scoreBuffer, selected, count}, params,
            {1, 1, 1}, {1024, 1, 1});
  (void)backend.submitCommand(graph.dispatches());

  const uint32_t written = *static_cast<const uint32_t *>(count.contents());
  const uint32_t *indices = static_cast<const uint32_t *>(selected.contents());
  return std::vector<uint32_t>(indices, indices + written);
}

// Descending score, ascending index on a tie.
std::vector<uint32_t> reference(const std::vector<float> &scores,
                                uint32_t budget) {
  std::vector<uint32_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
  });
  order.resize(std::min<size_t>(budget, scores.size()));
  std::sort(order.begin(), order.end());
  return order;
}

void checkCase(MetalBackend &backend, const std::vector<float> &scores,
               uint32_t budget, const std::string &label) {
  std::vector<uint32_t> got = runSelection(backend, scores, budget);
  std::sort(got.begin(), got.end());
  const std::vector<uint32_t> want = reference(scores, budget);
  require(got.size() == want.size(),
          label + ": selected " + std::to_string(got.size()) + ", expected " +
              std::to_string(want.size()));
  require(got == want, label + ": the selected set is not the true top k");
  std::cout << "  " << label << ": " << got.size() << " of " << scores.size()
            << " selected, exact\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: qsa-select <metallib>");
    MetalBackend backend(argv[1]);

    std::mt19937 engine(20260919);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Production shape: 512 of 65,536.
    {
      std::vector<float> scores(65536);
      for (float &value : scores) value = std::abs(normal(engine)) * 3.0f;
      checkCase(backend, scores, 512, "production");
    }
    // Fewer blocks than the budget: everything is selected.
    {
      std::vector<float> scores(300);
      for (float &value : scores) value = std::abs(normal(engine));
      checkCase(backend, scores, 512, "short history");
    }
    // Heavy ties, which is where the index tie-break has to hold.
    {
      std::vector<float> scores(4096);
      for (size_t i = 0; i < scores.size(); ++i)
        scores[i] = float(engine() % 8);
      checkCase(backend, scores, 512, "heavy ties");
    }
    // Every score identical: the whole selection is decided by index order.
    {
      std::vector<float> scores(2048, 1.5f);
      checkCase(backend, scores, 512, "all equal");
    }
    // Zeros, which relu produces in quantity.
    {
      std::vector<float> scores(4096, 0.0f);
      for (size_t i = 0; i < 700; ++i) scores[i * 5] = std::abs(normal(engine));
      checkCase(backend, scores, 512, "mostly zero");
    }

    // The same input must give the same answer every time.
    std::vector<float> repeatable(8192);
    for (float &value : repeatable) value = std::abs(normal(engine)) * 2.0f;
    const std::vector<uint32_t> first = runSelection(backend, repeatable, 512);
    for (int attempt = 0; attempt < 4; ++attempt) {
      require(runSelection(backend, repeatable, 512) == first,
              "the selection is not reproducible between runs");
    }
    std::cout << "  reproducible across 5 runs, including slot order\n";

    std::cout << "qsa select metal test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "qsa select metal test failed: " << error.what() << '\n';
    return 1;
  }
}

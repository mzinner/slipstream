#include "engine/Types.hpp"
#include "model/Runtime.hpp"
#include "ops/Q8PageStorage.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

std::vector<uint32_t> pageRange(uint32_t first, uint32_t count) {
  std::vector<uint32_t> result(count);
  for (uint32_t index = 0; index < count; ++index)
    result[index] = first + index;
  return result;
}

uint32_t parseCount(std::string_view text, std::string_view label) {
  uint32_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !value) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

std::vector<uint32_t> parseTokens(std::string_view text) {
  std::vector<uint32_t> tokens;
  size_t start = 0;
  while (start < text.size()) {
    while (start < text.size() &&
           (text[start] == ',' || text[start] == ' ' || text[start] == '\n' ||
            text[start] == '\r'))
      ++start;
    if (start >= text.size())
      break;
    size_t end = start;
    while (end < text.size() && text[end] != ',' && text[end] != ' ' &&
           text[end] != '\n' && text[end] != '\r')
      ++end;
    uint32_t id = 0;
    auto res = std::from_chars(text.data() + start, text.data() + end, id);
    if (res.ec == std::errc{}) {
      tokens.push_back(id);
    }
    start = end;
  }
  return tokens;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: generate-sample METALLIB MODEL_ROOT [MAX_TOKENS] "
                     "[TOKEN_ID_CSV or stdin]\n";
        return 2;
      }
      const char *metallibPath = argv[1];
      const char *modelRoot = argv[2];
      uint32_t maxTokens = 32;
      if (argc >= 4) {
        maxTokens = parseCount(argv[3], "max_tokens");
      }
      std::vector<uint32_t> prompt;
      if (argc >= 5) {
        prompt = parseTokens(argv[4]);
      } else {
        std::string input;
        std::string line;
        while (std::getline(std::cin, line)) {
          input += line + " ";
        }
        prompt = parseTokens(input);
      }
      if (prompt.empty()) {
        std::cerr << "error: prompt tokens cannot be empty\n";
        return 1;
      }

      metal::MetalBackend backend(metallibPath);
      model::ModelPackage model = model::loadModelPackage(
          backend, std::filesystem::path(modelRoot));
      ops::ExecutionPlans operators(backend.capabilities());
      model::ModelMemoryPlan executorPlan =
          model::plannedRuntimeMemory(backend.capabilities(), model, operators);

      const uint32_t pagesPerLane =
          (prompt.size() + maxTokens +
           model::ExecutionLimits::targetVerifyRows) /
              kv::kPageTokens +
          4;
      const uint32_t pageCount =
          (pagesPerLane + model.targetKvLayout().sparseMappingBatchPages() - 1) /
          model.targetKvLayout().sparseMappingBatchPages() *
          model.targetKvLayout().sparseMappingBatchPages();

      const uint64_t governorLimit = std::max(
          backend.capabilities().recommendedMaxWorkingSetBytes,
          backend.memoryStats().allocatedBytes + 4ULL * 1024 * 1024 * 1024);
      MemoryGovernor governor(backend, governorLimit, 1);
      kv::Q8PageStorage pages(backend, governor.allocationAdmission(),
                              model.targetKvLayout(), pageCount);
      for (uint32_t page = 0; page < pageCount; ++page) {
        if (!pages.ensureResident(page))
          throw std::runtime_error("could not back the KV pages");
      }
      model::QwenStateStorage states(backend,
                                      governor.allocationAdmission(),
                                      model.stateLayout());
      model::RuntimeContext context{
          backend, governor.allocationAdmission(), model, pages, states, operators,
          16384, executorPlan.pipelineReserveBytes,
          executorPlan.runtimeOverheadReserveBytes};
      model::Runtime executor(context);

      const uint64_t laneId = 1;
      const uint32_t slot = 0;
      std::vector<uint32_t> lanePages = pageRange(0, pagesPerLane);

      EngineRequest request;
      request.id = laneId;
      request.prompt.assign(prompt.begin(), prompt.end());
      request.maxNewTokens = maxTokens;
      executor.beginColdRequest(request.modelView(), slot);

      // Prefill phase
      const auto prefillStart = std::chrono::steady_clock::now();
      uint32_t offset = 0;
      std::vector<uint32_t> generatedTokens;
      while (offset < prompt.size()) {
        const uint32_t count = std::min<uint32_t>(
            model::ExecutionLimits::prefillTokenBudget,
            static_cast<uint32_t>(prompt.size()) - offset);
        BatchPlan plan{WorkKind::Prefill, BatchCohort::Greedy,
                       {{laneId, count, offset}}, DecodeStage::Regular};
        ModelBatchItem item{laneId, slot, offset, offset, count, lanePages};
        item.inputTokens =
            std::span<const uint32_t>(prompt).subspan(offset, count);
        auto results =
            executor.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
        if (results.size() != 1 || results[0].consumedPromptTokens != count)
          throw std::runtime_error("prefill consumed wrong row count");
        offset += count;

        for (uint32_t token : results[0].outputTokens) {
          generatedTokens.push_back(token);
        }
        if (results[0].finished) {
          std::cerr << "Prefill emitted stop token\n";
          break;
        }
      }
      const auto prefillFinish = std::chrono::steady_clock::now();
      const double prefillMs = std::chrono::duration<double, std::milli>(
                                   prefillFinish - prefillStart)
                                   .count();
      std::cerr << "Prefill completed in " << prefillMs << " ms. Prompt tokens: " << prompt.size() << "\n";

      std::vector<double> decodeStepMs;

      uint64_t position = prompt.size();
      uint32_t totalGenerated = static_cast<uint32_t>(generatedTokens.size());
      while (totalGenerated < maxTokens) {
        BatchPlan plan{WorkKind::Decode, BatchCohort::Greedy,
                       {{laneId, 0, 0}}, DecodeStage::Regular};
        ModelBatchItem item{laneId, slot, position, 0, 0, lanePages};

        const auto stepStart = std::chrono::steady_clock::now();
        auto results =
            executor.decode(plan, std::span<const ModelBatchItem>(&item, 1));
        const auto stepFinish = std::chrono::steady_clock::now();
        const double stepMs = std::chrono::duration<double, std::milli>(
                                  stepFinish - stepStart)
                                  .count();
        decodeStepMs.push_back(stepMs);

        if (results.empty())
          break;
        const auto &res = results[0];
        if (res.outputTokens.empty())
          break;

        for (uint32_t token : res.outputTokens) {
          generatedTokens.push_back(token);
          ++totalGenerated;
          std::cerr << "Token: " << token << " (step: " << decodeStepMs.size() << ", total: " << totalGenerated << ")\n";
        }

        position += res.outputTokens.size() - res.outputTokensWithoutKv;

        if (res.finished) {
          std::cerr << "Finished by stop token\n";
          break;
        }
      }

      executor.end(laneId);

      // Output JSON
      std::cout << "{\n";
      std::cout << "  \"prefill_ms\": " << prefillMs << ",\n";
      std::cout << "  \"prompt_tokens\": " << prompt.size() << ",\n";
      std::cout << "  \"decode_steps\": " << decodeStepMs.size() << ",\n";
      std::cout << "  \"generated_tokens\": [";
      for (size_t i = 0; i < generatedTokens.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << generatedTokens[i];
      }
      std::cout << "],\n";
      std::cout << "  \"step_ms\": [";
      for (size_t i = 0; i < decodeStepMs.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << decodeStepMs[i];
      }
      std::cout << "]\n";
      std::cout << "}\n";

      return 0;
    } catch (const std::exception &error) {
      std::cerr << "generate-sample error: " << error.what() << '\n';
      return 1;
    }
  }
}

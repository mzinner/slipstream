#include "engine/RuntimeResources.hpp"

#import <Foundation/Foundation.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace splash;
using namespace splash::engine;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

class TemporaryModelRoot final {
public:
  TemporaryModelRoot() {
    path = std::filesystem::temp_directory_path() /
           ("splash-budget-" +
            std::string([NSUUID UUID].UUIDString.UTF8String));
    for (const char *component : {"target", "draft", "vision"}) {
      std::filesystem::create_directories(path / component);
      const auto file = path / component / "placeholder.bin";
      std::ofstream(file).put('\0');
      std::filesystem::resize_file(file, fileBytes);
    }
  }

  ~TemporaryModelRoot() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  static constexpr uint64_t fileBytes = 16 * 1024;
  static constexpr uint64_t packageBytes = 3 * fileBytes;
  std::filesystem::path path;
};

void testWeightBudgetBeforeLoading(const char *metallibPath) {
  TemporaryModelRoot root;
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root.path;
  // The placeholder draft the qwen4exp package declares.
  model::DFlashDraftLayout draft;
  draft.layers = 5;
  draft.hiddenSize = 2560;
  draft.dynamicSize = 768;
  draft.qkvSize = 3072;
  draft.attentionSize = 2048;
  draft.intermediateSize = 8704;
  draft.targetHiddenSize = model::Qwen4ExpLayout{}.capturedHiddenSize();
  ops::VisionLayout vision;
  vision.outputHiddenSize = draft.hiddenSize;
  config.model = model::makeModelDescriptor(
      "budget-test", model::Qwen4ExpLayout{}, draft, vision);
  config.buildId = "budget-test";

  // Each low ceiling fits two components, so every directory must be counted.
  // The other ceilings must reach the real loader, whose expected weight files
  // are deliberately absent. No actual model package is needed for this test.
  for (uint64_t ceiling : {root.packageBytes - 1, root.packageBytes,
                           uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config);
      throw std::runtime_error("placeholder model unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == root.packageBytes - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        require(std::string(error.what()).find("[memory_planning]") !=
                    std::string::npos &&
                    error.message().find("model weights require 49152 bytes") !=
                        std::string::npos &&
                    error.message().find("budget is 49151 bytes") !=
                        std::string::npos,
                "weight loading began before checking the memory ceiling");
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
        require(std::string(error.what()).find("[model_loading]") !=
                    std::string::npos &&
                    error.message().find("unable to open") !=
                        std::string::npos &&
                    error.message().find((root.path / "target").string()) !=
                        std::string::npos,
                "a sufficient weight budget did not reach the model loader");
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "expected metallib path");
      testWeightBudgetBeforeLoading(argv[1]);
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}

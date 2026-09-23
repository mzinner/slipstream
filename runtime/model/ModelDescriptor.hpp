#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen4Exp.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

using TargetLayout =
    std::variant<Qwen4ExpLayout>;

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Q8Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // The draft is zero-filled, present only because the format requires one
  // (qwen4exp has no DFlash 2 draft). Its proposals are never worth keeping,
  // so the runtime keeps only each step's anchor row.
  bool draftPlaceholder = false;
  // Exact bytes parsed during package inspection, including artifact digests.
  // Synthetic descriptors retain zero; this is separate from layout identity.
  std::array<uint8_t, 32> packageManifestSha256{};

  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target and draft layouts.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target,
                                                  DFlashDraftLayout draft,
                                                  ops::VisionLayout vision);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

} // namespace splash::model

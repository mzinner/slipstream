#pragma once

#include "metal/MetalBackend.hpp"
#include "model/Model.hpp"
#include "model/StateLayout.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace splash::model {

struct GdnParityBuffers final {
  metal::MetalBuffer stateBase;
  metal::MetalBuffer convolutionBase;
  metal::MetalBuffer recurrentBase;
  metal::MetalBuffer auxiliary; // Empty unless the layout reserves some.
  std::vector<metal::MetalBuffer> convolutionLayers;
  std::vector<metal::MetalBuffer> recurrentLayers;
};

class QwenGdnCell final {
public:
  ~QwenGdnCell();
  QwenGdnCell(const QwenGdnCell &) = delete;
  QwenGdnCell &operator=(const QwenGdnCell &) = delete;

  [[nodiscard]] const GdnParityBuffers &buffers() const noexcept {
    return buffers_;
  }
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return actualAllocatedBytes_;
  }

private:
  QwenGdnCell(metal::MetalBackend &backend,
              std::shared_ptr<StateAllocationTracker> tracker,
              GdnStateLayout layout,
              std::string_view label);

  std::shared_ptr<StateAllocationTracker> tracker_;
  GdnParityBuffers buffers_;
  uint64_t actualAllocatedBytes_ = 0;

  friend class QwenStateStorage;
};

struct QwenSlotBuffers final {
  // Views retain the slot's GDN allocations without copying data.
  std::array<GdnParityBuffers, 2> gdn;
};

struct QwenLogicalLengths final {
  uint64_t targetTokens = 0;

  bool operator==(const QwenLogicalLengths &) const = default;
};

struct QwenSlotMetadata final {
  bool assigned = false;
  uint64_t requestId = 0;
  uint32_t activeParity = 0;
  QwenLogicalLengths lengths;
};

class QwenStateStorage;

// One GDN cell: the buffers a cached state occupies.
struct QwenCacheSlot final {
  std::shared_ptr<QwenGdnCell> gdn;
};

// Free buffers available for reuse: a lane takes two cells, a cached state
// one. Both return them here; idle buffers are released only by explicit
// reclaim.
struct QwenBufferPool final {
  std::vector<std::shared_ptr<QwenGdnCell>> cells;
  // Cleared when the storage goes away; late returns then just free.
  bool open = true;
};

// An immutable composite snapshot owns a private copy of the state that
// cannot be recovered from sealed Q8 pages. No mutating buffers are exposed
// after construction; its buffers return to the pool when it is dropped.
class QwenCompositeState final : public CompositeState {
public:
  ~QwenCompositeState() override;
  QwenCompositeState(const QwenCompositeState &) = delete;
  QwenCompositeState &operator=(const QwenCompositeState &) = delete;

  [[nodiscard]] uint64_t bytes() const noexcept override {
    return layout_.cachedBytes();
  }

private:
  QwenCompositeState(std::shared_ptr<QwenBufferPool> pool, QwenCacheSlot slot,
                     CompositeStateLayout layout, QwenLogicalLengths lengths);

  std::shared_ptr<QwenBufferPool> pool_;
  QwenCacheSlot slot_;
  CompositeStateLayout layout_;
  QwenLogicalLengths lengths_;

  friend class QwenStateStorage;
};

// Live cells retain stable backing; only idle buffers may be reclaimed.
class QwenStateStorage final : public model::StateStorage {
public:
  QwenStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   CompositeStateLayout layout);

  ~QwenStateStorage() override;
  QwenStateStorage(const QwenStateStorage &) = delete;
  QwenStateStorage &operator=(const QwenStateStorage &) = delete;

  [[nodiscard]] const QwenSlotBuffers &buffers(uint32_t slot) const;
  [[nodiscard]] const QwenSlotMetadata &metadata(uint32_t slot) const;

  // Activation reuses pooled buffers and admits any missing allocations.
  // It clears parity-zero GDN state and resets logical lengths; later
  // transitions overwrite the remaining data. Refusals retain their cause.
  // Release returns the lane's buffers to the pool.
  [[nodiscard]] metal::AllocationResult tryActivateSlot(uint32_t slot, uint64_t requestId);
  void releaseSlot(uint32_t slot, uint64_t requestId);

  // Returns pooled buffers beyond the kept counts to macOS. Active lanes and
  // cached states are never moved or reclaimed.
  [[nodiscard]] uint64_t releaseIdle(uint32_t keepCells) noexcept override;
  [[nodiscard]] uint32_t idleCells() const noexcept;

  // Hot-path metadata operations; neither performs a buffer copy.
  void updateLengths(uint32_t slot, QwenLogicalLengths lengths);
  void swapParity(uint32_t slot);

  // Copies committed state into a pooled or newly admitted cache slot while
  // the lane retains its own cells. Returns nullptr on capacity pressure;
  // dropping a cached state makes its slot available for retry.
  [[nodiscard]] std::shared_ptr<const QwenCompositeState>
  snapshot(uint32_t slot);
  void restore(uint32_t slot, const CompositeState &state);

  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept override {
    return allocations_->bytes.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t actualSlotBytes(uint32_t slot) const;
  [[nodiscard]] CompositeStateLayout layout() const noexcept { return layout_; }

private:
  struct Slot final {
    QwenSlotBuffers buffers;
    QwenSlotMetadata metadata;
    std::array<std::shared_ptr<QwenGdnCell>, 2> gdn;
  };

  [[nodiscard]] Slot &slot(uint32_t index);
  [[nodiscard]] const Slot &slot(uint32_t index) const;
  void validateLengths(const QwenLogicalLengths &lengths,
                       bool cacheSnapshot) const;
  static void requireAssigned(const Slot &slot);
  [[nodiscard]] metal::AllocationResult allocateSlot(uint32_t index);
  [[nodiscard]] std::shared_ptr<QwenGdnCell> acquireCell(std::string_view label);
  [[nodiscard]] std::shared_ptr<QwenGdnCell>
  allocateGdnCell(std::string_view label,
                    metal::AllocationFailure *failure = nullptr);
  static void refreshViews(Slot &slot);

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  CompositeStateLayout layout_;
  std::shared_ptr<StateAllocationTracker> allocations_;
  std::shared_ptr<QwenBufferPool> pool_;
  std::array<Slot, ExecutionLimits::maximumBatchWidth> slots_;
};

} // namespace splash::model

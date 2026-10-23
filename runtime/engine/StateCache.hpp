#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvCache.hpp"
#include "model/Model.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>

namespace splash::engine {

class StateCache;

struct StateCheckpoint final {
  uint64_t kvBlock = 0;
  uint64_t publication = 0;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

class CompositeStateLease final {
public:
  CompositeStateLease(const CompositeStateLease &) = delete;
  CompositeStateLease &operator=(const CompositeStateLease &) = delete;
  CompositeStateLease(CompositeStateLease &&other) noexcept;
  CompositeStateLease &operator=(CompositeStateLease &&other) noexcept;
  ~CompositeStateLease() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr;
  }
  [[nodiscard]] uint64_t kvBlock() const noexcept { return kvBlock_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return boundary_; }
  [[nodiscard]] const std::shared_ptr<const CompositeState> &
  state() const noexcept {
    return state_;
  }
  void reset() noexcept;

private:
  friend class StateCache;
  CompositeStateLease(StateCache &owner, uint64_t kvBlock, uint32_t boundary,
                      std::shared_ptr<const CompositeState> state) noexcept;

  StateCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
  uint32_t boundary_ = 0;
  std::shared_ptr<const CompositeState> state_;
};

struct StateCacheSnapshot {
  uint32_t entries = 0;
  uint32_t pinned = 0;
  uint64_t bytes = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t publications = 0;
  uint64_t deduplicatedPublications = 0;
  uint64_t evictions = 0;
  uint32_t checkpointEntries = 0;
  uint64_t checkpointBytes = 0;
  uint64_t checkpointRetirements = 0;
  // Pressure and logical eviction; rolling retirements are counted separately.
  uint64_t checkpointEvictions = 0;
  // States at prompt message boundaries, kept ahead of all others.
  uint32_t boundaryEntries = 0;
  uint32_t boundaryCapacity = 0;
  uint64_t boundaryDemotions = 0;
};

// What a published state is for, which decides when it is evicted:
// checkpoints first, then ordinary and hint states by recency (a hint, a
// state the server guessed later prompts may resume from, starts as the
// oldest until one does), then boundary states.
enum class StateRole : uint8_t { Ordinary, Checkpoint, Hint, Boundary };

struct StateEviction final {
  bool evicted = false;
  uint64_t reclaimedBytes = 0;
};

// Attaches one immutable target-recurrent + draft-context state to a complete
// target-KV block. The pair is restored atomically.
class StateCache final {
public:
  // At most boundaryCapacity boundary states are kept as such; publishing one
  // more demotes the least recently used to an ordinary state.
  StateCache(KvCache &kv, CacheRecency &recency, uint32_t boundaryCapacity)
      : kv_(kv), recency_(recency), boundaryCapacity_(boundaryCapacity) {}
  StateCache(const StateCache &) = delete;
  StateCache &operator=(const StateCache &) = delete;

  [[nodiscard]] std::optional<CompositeStateLease>
  acquireDeepest(std::span<const uint64_t> kvChain);
  // Acquisition pins backing; accounting occurs only when admission succeeds.
  void recordLookup(bool hit) noexcept { hit ? ++hits_ : ++misses_; }

  // Reuses a published state without a restore pin or lookup accounting.
  // Any other role upgrades a checkpoint, and a boundary role upgrades any
  // state; nothing downgrades. Restoration alone never changes a role.
  [[nodiscard]] bool touchIfResident(uint64_t kvBlock,
                                     StateRole role = StateRole::Ordinary);

  void publish(uint64_t kvBlock, std::shared_ptr<const CompositeState> state,
               StateRole role = StateRole::Ordinary);
  // Publication identity protects replacement states from stale handles.
  [[nodiscard]] StateCheckpoint checkpoint(uint64_t kvBlock) const noexcept;
  // Ensures this publication is no longer a disposable checkpoint. Returns
  // false only when the matching checkpoint is pinned; absent, replaced and
  // upgraded publications already satisfy the postcondition.
  bool retireCheckpoint(StateCheckpoint checkpoint) noexcept;
  // Refreshes recency within the state's class. No-op when absent or pinned.
  void touch(uint64_t kvBlock) noexcept;

  [[nodiscard]] bool contains(uint64_t kvBlock) const noexcept;
  // Unpinned checkpoints precede ordinary and hint states, which precede
  // boundary states, regardless of recency. A boundary candidate is marked
  // protectedState.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate() const noexcept;
  [[nodiscard]] StateEviction evict(uint64_t kvBlock) noexcept;
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  friend class CompositeStateLease;

  struct Entry {
    std::shared_ptr<const CompositeState> state;
    uint32_t pins = 0;
    uint64_t lastUsed = 0;
    uint64_t previousEvictable = 0;
    uint64_t nextEvictable = 0;
    bool evictable = false;
    StateRole role = StateRole::Ordinary;
    uint64_t publication = 0;
  };

  struct EvictionQueue {
    uint64_t oldest = 0;
    uint64_t newest = 0;
  };

  [[nodiscard]] StateEviction erase(uint64_t kvBlock, bool retirement) noexcept;
  void release(uint64_t kvBlock) noexcept;
  [[nodiscard]] std::optional<CompositeStateLease>
  acquireBlock(uint64_t kvBlock);
  // Coldest inserts at the oldest end of the queue, as old as its oldest.
  void insertEvictable(uint64_t kvBlock, bool coldest = false) noexcept;
  void removeEvictable(uint64_t kvBlock) noexcept;
  [[nodiscard]] EvictionQueue &queueOf(const Entry &entry) noexcept;
  // Changes an entry's role, keeping the role counts and queues in step.
  void setRole(uint64_t kvBlock, StateRole role) noexcept;
  // Makes room for one more boundary state; false when none can be demoted.
  [[nodiscard]] bool reserveBoundary() noexcept;

  KvCache &kv_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Entry> entries_;
  uint64_t bytes_ = 0;
  EvictionQueue ordinaryEviction_;
  EvictionQueue checkpointEviction_;
  EvictionQueue boundaryEviction_;
  uint32_t pinnedEntries_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t publications_ = 0;
  uint64_t deduplicatedPublications_ = 0;
  uint64_t evictions_ = 0;
  uint64_t checkpointEntries_ = 0;
  uint64_t checkpointBytes_ = 0;
  uint64_t checkpointRetirements_ = 0;
  uint64_t checkpointEvictions_ = 0;
  uint32_t boundaryEntries_ = 0;
  uint32_t boundaryCapacity_ = 0;
  uint64_t boundaryDemotions_ = 0;
};

} // namespace splash::engine

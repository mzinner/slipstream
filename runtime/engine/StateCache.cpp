#include "engine/StateCache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

CompositeStateLease::CompositeStateLease(
    StateCache &owner, uint64_t kvBlock, uint32_t boundary,
    std::shared_ptr<const CompositeState> state) noexcept
    : owner_(&owner), kvBlock_(kvBlock), boundary_(boundary),
      state_(std::move(state)) {}

CompositeStateLease::CompositeStateLease(CompositeStateLease &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      kvBlock_(std::exchange(other.kvBlock_, 0)),
      boundary_(std::exchange(other.boundary_, 0)),
      state_(std::move(other.state_)) {}

CompositeStateLease &
CompositeStateLease::operator=(CompositeStateLease &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  owner_ = std::exchange(other.owner_, nullptr);
  kvBlock_ = std::exchange(other.kvBlock_, 0);
  boundary_ = std::exchange(other.boundary_, 0);
  state_ = std::move(other.state_);
  return *this;
}

CompositeStateLease::~CompositeStateLease() noexcept { reset(); }

void CompositeStateLease::reset() noexcept {
  if (owner_)
    owner_->release(kvBlock_);
  owner_ = nullptr;
  kvBlock_ = 0;
  boundary_ = 0;
  state_.reset();
}

std::optional<CompositeStateLease>
StateCache::acquireDeepest(std::span<const uint64_t> kvChain) {
  for (auto block = kvChain.rbegin(); block != kvChain.rend(); ++block) {
    if (auto lease = acquireBlock(*block))
      return lease;
  }
  return std::nullopt;
}

std::optional<CompositeStateLease> StateCache::acquireBlock(uint64_t kvBlock) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end())
    return std::nullopt;
  Entry &entry = found->second;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("composite state outlived its target KV block");
  }
  if (entry.pins == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("composite state pin count overflowed");
  }
  if (!entry.pins && pinnedEntries_ == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("composite state pinned entry count overflowed");
  }
  kv_.retainActive(kvBlock);
  if (!entry.pins) {
    removeEvictable(kvBlock);
    ++pinnedEntries_;
  }
  ++entry.pins;
  const uint32_t boundary = kv_.chainLength(kvBlock) * KvCache::pageTokens;
  return CompositeStateLease(*this, kvBlock, boundary, entry.state);
}

StateCache::EvictionQueue &StateCache::queueOf(const Entry &entry) noexcept {
  switch (entry.role) {
  case StateRole::Checkpoint:
    return checkpointEviction_;
  case StateRole::Boundary:
    return boundaryEviction_;
  case StateRole::Ordinary:
  case StateRole::Hint:
    break;
  }
  return ordinaryEviction_;
}

void StateCache::setRole(uint64_t kvBlock, StateRole role) noexcept {
  Entry &entry = entries_.at(kvBlock);
  if (entry.role == role)
    return;
  const bool evictable = entry.evictable;
  if (evictable)
    removeEvictable(kvBlock);
  const uint64_t bytes = entry.state->bytes();
  if (entry.role == StateRole::Checkpoint) {
    --checkpointEntries_;
    checkpointBytes_ -= bytes;
  } else if (entry.role == StateRole::Boundary) {
    --boundaryEntries_;
  }
  entry.role = role;
  if (role == StateRole::Checkpoint) {
    ++checkpointEntries_;
    checkpointBytes_ += bytes;
  } else if (role == StateRole::Boundary) {
    ++boundaryEntries_;
  }
  if (evictable)
    insertEvictable(kvBlock);
}

bool StateCache::reserveBoundary() noexcept {
  if (!boundaryCapacity_)
    return false;
  if (boundaryEntries_ < boundaryCapacity_)
    return true;
  const uint64_t oldest = boundaryEviction_.oldest;
  if (!oldest)
    return false;
  setRole(oldest, StateRole::Ordinary);
  ++boundaryDemotions_;
  return true;
}

bool StateCache::touchIfResident(uint64_t kvBlock, StateRole role) {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end())
    return false;
  if (!kv_.contains(kvBlock)) {
    throw std::logic_error("composite state outlived its target KV block");
  }
  const StateRole current = found->second.role;
  if (role == StateRole::Boundary && current != StateRole::Boundary &&
      reserveBoundary())
    setRole(kvBlock, StateRole::Boundary);
  else if (current == StateRole::Checkpoint && role != StateRole::Checkpoint)
    setRole(kvBlock, StateRole::Ordinary);
  else
    touch(kvBlock);
  ++deduplicatedPublications_;
  return true;
}

void StateCache::touch(uint64_t kvBlock) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.evictable)
    return;
  removeEvictable(kvBlock);
  insertEvictable(kvBlock);
}

void StateCache::publish(uint64_t kvBlock,
                         std::shared_ptr<const CompositeState> state,
                         StateRole role) {
  if (!state || !state->bytes()) {
    throw std::invalid_argument("composite state payload is empty");
  }
  if (!kv_.contains(kvBlock)) {
    throw std::invalid_argument("composite state KV block is not resident");
  }
  const uint64_t stateBytes = state->bytes();
  if (bytes_ > std::numeric_limits<uint64_t>::max() - stateBytes) {
    throw std::overflow_error("composite state byte count overflowed");
  }

  if (publications_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("composite state publication count overflowed");
  Entry entry;
  entry.publication = publications_ + 1;
  entry.state = std::move(state);
  if (entries_.contains(kvBlock))
    throw std::logic_error("duplicate composite state key");
  if (role == StateRole::Boundary && !reserveBoundary())
    role = StateRole::Ordinary;
  entry.role = role;
  entries_.emplace(kvBlock, std::move(entry));
  bytes_ += stateBytes;
  if (role == StateRole::Checkpoint) {
    ++checkpointEntries_;
    checkpointBytes_ += stateBytes;
  } else if (role == StateRole::Boundary) {
    ++boundaryEntries_;
  }
  insertEvictable(kvBlock, role == StateRole::Hint);
  ++publications_;
}

StateCheckpoint StateCache::checkpoint(uint64_t kvBlock) const noexcept {
  const auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.role != StateRole::Checkpoint)
    return {};
  return {kvBlock, found->second.publication};
}

bool StateCache::retireCheckpoint(StateCheckpoint checkpoint) noexcept {
  const auto found = entries_.find(checkpoint.kvBlock);
  if (found == entries_.end() || found->second.role != StateRole::Checkpoint ||
      found->second.publication != checkpoint.publication)
    return true;
  return erase(checkpoint.kvBlock, true).evicted;
}

bool StateCache::contains(uint64_t kvBlock) const noexcept {
  return entries_.contains(kvBlock);
}

std::optional<CacheEvictionCandidate>
StateCache::evictionCandidate() const noexcept {
  const uint64_t oldest = checkpointEviction_.oldest ? checkpointEviction_.oldest
                          : ordinaryEviction_.oldest ? ordinaryEviction_.oldest
                                                     : boundaryEviction_.oldest;
  if (!oldest)
    return std::nullopt;
  const Entry &entry = entries_.at(oldest);
  return CacheEvictionCandidate{oldest, entry.lastUsed,
                                entry.role == StateRole::Boundary};
}

StateEviction StateCache::evict(uint64_t kvBlock) noexcept {
  return erase(kvBlock, false);
}

StateEviction StateCache::erase(uint64_t kvBlock, bool retirement) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || found->second.pins)
    return {};
  const uint64_t reclaimed = found->second.state->bytes();
  removeEvictable(kvBlock);
  if (found->second.role == StateRole::Checkpoint) {
    --checkpointEntries_;
    checkpointBytes_ -= reclaimed;
    if (!retirement)
      ++checkpointEvictions_;
  } else if (found->second.role == StateRole::Boundary) {
    --boundaryEntries_;
  }
  bytes_ -= reclaimed;
  entries_.erase(found);
  if (retirement)
    ++checkpointRetirements_;
  else
    ++evictions_;
  return {true, reclaimed};
}

StateCacheSnapshot StateCache::snapshot() const noexcept {
  StateCacheSnapshot result;
  result.entries = static_cast<uint32_t>(std::min<uint64_t>(
      entries_.size(), std::numeric_limits<uint32_t>::max()));
  result.pinned = pinnedEntries_;
  result.bytes = bytes_;
  result.hits = hits_;
  result.misses = misses_;
  result.publications = publications_;
  result.deduplicatedPublications = deduplicatedPublications_;
  result.evictions = evictions_;
  result.checkpointEntries = static_cast<uint32_t>(std::min<uint64_t>(
      checkpointEntries_, std::numeric_limits<uint32_t>::max()));
  result.checkpointBytes = checkpointBytes_;
  result.checkpointRetirements = checkpointRetirements_;
  result.checkpointEvictions = checkpointEvictions_;
  result.boundaryEntries = boundaryEntries_;
  result.boundaryCapacity = boundaryCapacity_;
  result.boundaryDemotions = boundaryDemotions_;
  return result;
}

void StateCache::release(uint64_t kvBlock) noexcept {
  auto found = entries_.find(kvBlock);
  if (found == entries_.end() || !found->second.pins)
    std::terminate();
  --found->second.pins;
  if (!found->second.pins) {
    if (!pinnedEntries_)
      std::terminate();
    --pinnedEntries_;
    insertEvictable(kvBlock);
  }
  kv_.releaseActive(kvBlock);
}

void StateCache::insertEvictable(uint64_t kvBlock, bool coldest) noexcept {
  Entry &entry = entries_.at(kvBlock);
  if (entry.evictable || entry.pins)
    std::terminate();
  EvictionQueue &queue = queueOf(entry);
  entry.evictable = true;
  if (coldest && queue.oldest) {
    Entry &oldest = entries_.at(queue.oldest);
    entry.lastUsed = oldest.lastUsed;
    entry.previousEvictable = 0;
    entry.nextEvictable = queue.oldest;
    oldest.previousEvictable = kvBlock;
    queue.oldest = kvBlock;
    return;
  }
  entry.previousEvictable = queue.newest;
  entry.nextEvictable = 0;
  entry.lastUsed = recency_.next();
  if (queue.newest)
    entries_.at(queue.newest).nextEvictable = kvBlock;
  else
    queue.oldest = kvBlock;
  queue.newest = kvBlock;
}

void StateCache::removeEvictable(uint64_t kvBlock) noexcept {
  Entry &entry = entries_.at(kvBlock);
  if (!entry.evictable)
    std::terminate();
  EvictionQueue &queue = queueOf(entry);
  if (entry.previousEvictable)
    entries_.at(entry.previousEvictable).nextEvictable = entry.nextEvictable;
  else
    queue.oldest = entry.nextEvictable;
  if (entry.nextEvictable)
    entries_.at(entry.nextEvictable).previousEvictable =
        entry.previousEvictable;
  else
    queue.newest = entry.previousEvictable;
  entry.previousEvictable = 0;
  entry.nextEvictable = 0;
  entry.evictable = false;
}

} // namespace splash::engine

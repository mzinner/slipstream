#include "metal/abi/KernelABI.h"

// Block selection for the qwen4exp sparse-attention indexer.
//
// The scoring kernel leaves one score per visible block; this takes the
// highest `budget / compress_ratio` of them. At the production budget that
// is 512 blocks out of as many as 65,536, which is well past the point where
// sorting is sensible.
//
// Scores are a sum of relu terms and so are never negative, which is what
// makes this simple: for non-negative floats the IEEE bit pattern orders the
// same way the value does, so the k-th largest score can be found by a binary
// search over the bit pattern. Thirty-two counting passes settle it exactly,
// with no partial sort and no ordering assumptions between ties.
//
// Ties at the threshold are broken by block index, ascending, so a query's
// selection does not depend on which thread happened to get there first.

struct QsaSelectParams {
  uint blocks;
  uint budget;
};

constant constexpr uint kThreads = 1024;
constant constexpr uint kSimdgroups = kThreads / 32;

kernel void qsa_select_blocks(
    device const float *scores [[buffer(0)]],
    device uint *selected [[buffer(1)]],
    device atomic_uint *selected_count [[buffer(2)]],
    constant QsaSelectParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint partial[kSimdgroups];
  threadgroup uint threshold_shared;
  threadgroup uint above_shared;
  // Lowest index a tie may still be taken from, so repeated rounds advance.
  threadgroup uint tie_cursor;

  const uint wanted = min(params.budget, params.blocks);
  if (thread_index == 0) {
    atomic_store_explicit(selected_count, 0u, memory_order_relaxed);
    threshold_shared = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Largest bit pattern whose population is still at least `wanted`.
  uint threshold = 0u;
  for (int bit = 31; bit >= 0; --bit) {
    const uint candidate = threshold | (1u << uint(bit));
    uint count = 0u;
    for (uint i = thread_index; i < params.blocks; i += kThreads) {
      if (as_type<uint>(scores[i]) >= candidate) ++count;
    }
    count = simd_sum(count);
    if (simd_lane == 0) partial[simd_group] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      uint total = 0u;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      // Keep the bit only while the candidate still admits enough blocks.
      if (total >= wanted) {
        threshold_shared = candidate;
        above_shared = total;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threshold = threshold_shared;
  }

  // Everything strictly above the threshold is in. Blocks exactly at it fill
  // the remainder in index order, so ties resolve the same way every run.
  uint strictly_above = 0u;
  for (uint i = thread_index; i < params.blocks; i += kThreads) {
    if (as_type<uint>(scores[i]) > threshold) ++strictly_above;
  }
  strictly_above = simd_sum(strictly_above);
  if (simd_lane == 0) partial[simd_group] = strictly_above;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    uint total = 0u;
    for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
    above_shared = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint base = min(above_shared, wanted);
  const uint ties_wanted = wanted - base;
  if (thread_index == 0) {
    tie_cursor = 0u;
    // The parallel claim below can overshoot when more blocks sit above the
    // threshold than were asked for; the count is clamped either way.
    atomic_store_explicit(selected_count, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Everything strictly above the threshold goes out in parallel, but each
  // thread is given its slots up front by a prefix sum rather than racing for
  // them with an atomic. Claiming atomically would leave the buffer in a
  // different order on every run for the same input, and this engine does not
  // do that anywhere else. Thread t always owns blocks t, t + kThreads, ...,
  // so the layout is fixed by the input alone.
  uint mine = 0u;
  for (uint i = thread_index; i < params.blocks; i += kThreads) {
    if (as_type<uint>(scores[i]) > threshold) ++mine;
  }
  const uint within_simd = simd_prefix_exclusive_sum(mine);
  if (simd_lane == 31) partial[simd_group] = within_simd + mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup uint simd_bases[kSimdgroups];
  if (thread_index == 0) {
    uint running = 0u;
    for (uint s = 0; s < kSimdgroups; ++s) {
      simd_bases[s] = running;
      running += partial[s];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint slot = simd_bases[simd_group] + within_simd;
  for (uint i = thread_index; i < params.blocks; i += kThreads) {
    if (as_type<uint>(scores[i]) > threshold) {
      if (slot < wanted) selected[slot] = i;
      ++slot;
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // Blocks sitting exactly at the threshold fill what remains, lowest index
  // first, so a tie resolves the same way on every run. There is normally one
  // of these - the k-th block itself - so this is a reduction, not a scan.
  for (uint taken = 0; taken < ties_wanted; ++taken) {
    uint best = ~0u;
    for (uint i = thread_index; i < params.blocks; i += kThreads) {
      if (as_type<uint>(scores[i]) == threshold && i >= tie_cursor)
        best = min(best, i);
    }
    best = simd_min(best);
    if (simd_lane == 0) partial[simd_group] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      uint winner = ~0u;
      for (uint s = 0; s < kSimdgroups; ++s) winner = min(winner, partial[s]);
      if (winner != ~0u) {
        selected[base + taken] = winner;
        tie_cursor = winner + 1u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (thread_index == 0) {
    atomic_store_explicit(selected_count, min(base + ties_wanted, wanted),
                          memory_order_relaxed);
  }
}

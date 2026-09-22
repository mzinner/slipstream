#include "metal/abi/KernelABI.h"

#include <metal_stdlib>
using namespace metal;

// The MTP head's pick, first half: each threadgroup takes one slice of the
// vocabulary and leaves its 16 largest logits (highest first, ties to the
// lower id) plus the slice's max and sum of exp(logit - max). The host merges
// the slices' candidates, which is far cheaper than scanning 248K logits.
constant constexpr uint kPickSlices = 64;
constant constexpr uint kPickCandidates = 16;
constant constexpr uint kPickThreads = 256;
constant constexpr uint kPickSliceMax = 4096;

kernel void mtp_pick_slices(device const bfloat *logits [[buffer(0)]],
                            device uint *ids [[buffer(1)]],
                            device float *values [[buffer(2)]],
                            device float *mass [[buffer(3)]],
                            constant uint &vocabulary [[buffer(4)]],
                            uint slice [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint per = (vocabulary + kPickSlices - 1) / kPickSlices;
  const uint first = slice * per, last = min(first + per, vocabulary);
  const uint count = last > first ? last - first : 0;
  threadgroup float staged[kPickSliceMax];
  threadgroup float best_value[kPickThreads / 32];
  threadgroup uint best_index[kPickThreads / 32];
  float local_max = -INFINITY;
  for (uint i = thread_index; i < count; i += kPickThreads) {
    const float v = float(logits[first + i]);
    staged[i] = v;
    local_max = max(local_max, v);
  }
  local_max = simd_max(local_max);
  if (simd_lane == 0) best_value[simd_group] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint s = 0; s < kPickThreads / 32; ++s) m = max(m, best_value[s]);
  float sum = 0.0f;
  for (uint i = thread_index; i < count; i += kPickThreads)
    sum += exp(staged[i] - m);
  sum = simd_sum(sum);
  threadgroup float sums[kPickThreads / 32];
  if (simd_lane == 0) sums[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint s = 0; s < kPickThreads / 32; ++s) total += sums[s];
    mass[2 * slice] = m;
    mass[2 * slice + 1] = total;
  }
  // Sixteen rounds of (value, lowest index) argmax, removing each winner.
  for (uint round = 0; round < kPickCandidates; ++round) {
    float v = -INFINITY;
    uint index = 0xFFFFFFFFu;
    for (uint i = thread_index; i < count; i += kPickThreads)
      if (staged[i] > v) { v = staged[i]; index = i; }
    for (uint offset = 16; offset > 0; offset /= 2) {
      const float other_v = simd_shuffle_down(v, offset);
      const uint other_i = simd_shuffle_down(index, offset);
      if (other_v > v || (other_v == v && other_i < index)) { v = other_v; index = other_i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) { best_value[simd_group] = v; best_index[simd_group] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float bv = -INFINITY;
      uint bi = 0xFFFFFFFFu;
      for (uint s = 0; s < kPickThreads / 32; ++s)
        if (best_value[s] > bv || (best_value[s] == bv && best_index[s] < bi)) {
          bv = best_value[s];
          bi = best_index[s];
        }
      ids[slice * kPickCandidates + round] = bi == 0xFFFFFFFFu ? 0xFFFFFFFFu : first + bi;
      values[slice * kPickCandidates + round] = bv;
      if (bi != 0xFFFFFFFFu) staged[bi] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

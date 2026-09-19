#include "metal/abi/KernelABI.h"

// Block scoring for the qwen4exp sparse-attention indexer.
//
// The indexer compresses the visible keys into blocks of CompressRatio
// tokens, scores every block against the query, and keeps the best
// BlockTopK of them; attention then runs over the tokens of those blocks
// plus the ragged tail that does not fill a block. This kernel is the
// scoring half - the part that dominates the cost, since it touches every
// visible block - and leaves selection and the attention walk to callers.
//
// Per block b, over head dimension D and heads H:
//
//   pooled  = mean of the block's raw keys, accumulated in fp32
//   normed  = rms(pooled) * (1 + k_gain)        one group, like the model
//   rotated = rope(normed, cos[start(b)], sin[start(b)])
//   score   = sum over heads of relu(dot(q[h], rotated)) / sqrt(D)
//
// The query arrives already normalized and rotated, matching the reference,
// which prepares it once per position rather than once per block.

struct QsaIndexerParams {
  uint blocks;
  uint heads;
  uint head_dimension;
  uint compress_ratio;
  uint rotary_dimension;
  float epsilon;
};

constant constexpr uint kThreads = 128;
constant constexpr uint kMaxHeadDimension = 256;

// One threadgroup per block.
kernel void qsa_indexer_score(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *raw_keys [[buffer(1)]],
    device const uint *block_starts [[buffer(2)]],
    device const bfloat *key_gain [[buffer(3)]],
    device const bfloat *cos_table [[buffer(4)]],
    device const bfloat *sin_table [[buffer(5)]],
    device float *scores [[buffer(6)]],
    constant QsaIndexerParams &params [[buffer(7)]],
    uint block [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint dimension = params.head_dimension;
  const uint start = block_starts[block];

  threadgroup float pooled[kMaxHeadDimension];
  threadgroup float rotated[kMaxHeadDimension];
  threadgroup float partial[kThreads / 32];

  // Mean over the block's tokens. The reference pools in fp32 before it
  // narrows back, so the accumulation width matters here.
  for (uint i = thread_index; i < dimension; i += kThreads) {
    float sum = 0.0f;
    for (uint token = 0; token < params.compress_ratio; ++token)
      sum += float(raw_keys[ulong(start + token) * dimension + i]);
    pooled[i] = sum / float(params.compress_ratio);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float square = 0.0f;
  for (uint i = thread_index; i < dimension; i += kThreads)
    square += pooled[i] * pooled[i];
  square = simd_sum(square);
  if (simd_lane == 0) partial[simd_group] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint s = 0; s < kThreads / 32; ++s) total += partial[s];
    partial[0] = rsqrt(total / float(dimension) + params.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = partial[0];

  // Normalize, then rotate the leading rotary_dimension lanes in halves.
  // `half` is a Metal type name, hence the longer name here.
  const uint rotary_half = params.rotary_dimension / 2;
  for (uint i = thread_index; i < dimension; i += kThreads) {
    // The stored gain is an offset from one, as everywhere in this model.
    const float value = pooled[i] * scale * (1.0f + float(key_gain[i]));
    rotated[i] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = thread_index; i < params.rotary_dimension; i += kThreads) {
    const float c = float(cos_table[ulong(start) * params.rotary_dimension + i]);
    const float s = float(sin_table[ulong(start) * params.rotary_dimension + i]);
    // rotate_half: the second half is negated and swapped with the first.
    const float partner = i < rotary_half ? -rotated[i + rotary_half]
                                    : rotated[i - rotary_half];
    pooled[i] = rotated[i] * c + partner * s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = thread_index; i < params.rotary_dimension; i += kThreads)
    rotated[i] = pooled[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One simdgroup per head; relu before the sum, as in the reference.
  float total = 0.0f;
  for (uint head = simd_group; head < params.heads;
       head += kThreads / 32) {
    device const bfloat *q = queries + ulong(head) * dimension;
    float dot = 0.0f;
    for (uint i = simd_lane; i < dimension; i += 32)
      dot += float(q[i]) * rotated[i];
    dot = simd_sum(dot);
    if (simd_lane == 0) total += max(dot, 0.0f);
  }
  if (simd_lane == 0) partial[simd_group] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float sum = 0.0f;
    for (uint s = 0; s < kThreads / 32; ++s) sum += partial[s];
    scores[block] = sum / sqrt(float(dimension));
  }
}

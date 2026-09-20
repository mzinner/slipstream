#include "metal/abi/KernelABI.h"

// Hyper-connections for qwen4exp (Qwen3.8-Flash-Next). The residual is Count
// parallel streams carried as one row of Count*Hidden. Over a row x:
//
//   xn    = rms_per_stream(x) * (1 + gain)   gain spans the whole row
//   w     = silu(down(xn) / Count)       width -> LowRank
//   w     = sigmoid(up(w))               LowRank -> width
//   input = mean over streams of reshape(w) * reshape(xn)
//   inj   = 2 * sigmoid(block_inject(xn) / Count)
//
// and after the block produces y of width Hidden the residual update is an
// outer product against the *unnormalized* x:
//
//   x = x + outer(inj, y)
//
// Three dispatches: normalize-and-reduce, mix-and-inject, then the update.
// The gate weights are bf16 rather than Q4 because LowRank is 320, which is
// not a multiple of either packed tile width.


constant constexpr uint kThreads = 256;
constant constexpr uint kSimdWidth = 32;
constant constexpr uint kSimdgroups = kThreads / kSimdWidth;

// Per-stream RMS, then the low-rank reduction. One threadgroup per row.
kernel void hyper_connection_normalize(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *gain [[buffer(1)]],
    device const bfloat *down [[buffer(2)]],
    device bfloat *normalized [[buffer(3)]],
    device bfloat *reduced [[buffer(4)]],
    constant HyperConnectionParams &params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  device const bfloat *x = input + ulong(row) * width;
  device bfloat *xn = normalized + ulong(row) * width;

  threadgroup float scales[8];
  threadgroup float partial[kSimdgroups];

  // One stream at a time: each is an independent RMS group.
  for (uint stream = 0; stream < params.count; ++stream) {
    float sum = 0.0f;
    for (uint i = thread_index; i < hidden; i += kThreads) {
      const float value = float(x[stream * hidden + i]);
      sum += value * value;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      scales[stream] = rsqrt(total / float(hidden) + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = thread_index; i < width; i += kThreads)
    // The stored gain is an offset from one, and is zero-initialized
    // upstream, so the neutral norm is a zero weight rather than a one.
    xn[i] = bfloat(float(x[i]) * scales[i / hidden] *
                   (1.0f + float(gain[i])));
  threadgroup_barrier(mem_flags::mem_device);

  // down is [low_rank, width]; one simdgroup per output, lanes stride the row.
  device bfloat *out = reduced + ulong(row) * params.low_rank;
  for (uint o = simd_group; o < params.low_rank; o += kSimdgroups) {
    device const bfloat *weights = down + ulong(o) * width;
    float sum = 0.0f;
    for (uint i = simd_lane; i < width; i += kSimdWidth)
      sum += float(weights[i]) * float(xn[i]);
    sum = simd_sum(sum);
    if (simd_lane == 0) {
      // silu(v) = v * sigmoid(v)
      const float v = sum / float(params.count);
      out[o] = bfloat(v / (1.0f + exp(-v)));
    }
  }
}

// The up projection, the stream mix, and the injection gate.
kernel void hyper_connection_mix(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *reduced [[buffer(1)]],
    device const bfloat *up [[buffer(2)]],
    device const bfloat *inject [[buffer(3)]],
    device bfloat *mixed [[buffer(4)]],
    device bfloat *injection [[buffer(5)]],
    constant HyperConnectionParams &params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  const uint low_rank = params.low_rank;
  device const bfloat *xn = normalized + ulong(row) * width;
  device const bfloat *low = reduced + ulong(row) * low_rank;

  threadgroup float staged[64];
  for (uint r = thread_index; r < low_rank && r < 64; r += kThreads)
    staged[r] = float(low[r]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each thread owns whole output dimensions, so the mean over streams needs
  // no cross-thread reduction: it reads all `count` entries for its own index.
  device bfloat *out = mixed + ulong(row) * hidden;
  for (uint dimension = thread_index; dimension < hidden;
       dimension += kThreads) {
    float accumulated = 0.0f;
    for (uint stream = 0; stream < params.count; ++stream) {
      const uint index = stream * hidden + dimension;
      device const bfloat *weights = up + ulong(index) * low_rank;
      float sum = 0.0f;
      for (uint r = 0; r < low_rank; ++r) {
        const float value = r < 64 ? staged[r] : float(low[r]);
        sum += float(weights[r]) * value;
      }
      const float gate = 1.0f / (1.0f + exp(-sum));
      accumulated += gate * float(xn[index]);
    }
    out[dimension] = bfloat(accumulated / float(params.count));
  }

  // inject is [count, width]; one simdgroup per stream.
  device bfloat *gates = injection + ulong(row) * params.count;
  for (uint stream = simd_group; stream < params.count;
       stream += kSimdgroups) {
    device const bfloat *weights = inject + ulong(stream) * width;
    float sum = 0.0f;
    for (uint i = simd_lane; i < width; i += kSimdWidth)
      sum += float(weights[i]) * float(xn[i]);
    sum = simd_sum(sum);
    if (simd_lane == 0) {
      const float v = sum / float(params.count);
      gates[stream] = bfloat(2.0f / (1.0f + exp(-v)));
    }
  }
}

// residual += outer(injection, block output)
kernel void hyper_connection_update(
    device bfloat *residual [[buffer(0)]],
    device const bfloat *block [[buffer(1)]],
    device const bfloat *injection [[buffer(2)]],
    constant HyperConnectionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  const uint total = params.rows * width;
  for (uint element = index; element < total; element += grid_size) {
    const uint row = element / width;
    const uint offset = element % width;
    const float gate = float(injection[row * params.count + offset / hidden]);
    const float value = float(block[row * hidden + offset % hidden]);
    residual[element] = bfloat(float(residual[element]) + gate * value);
  }
}

kernel void hyper_connection_update_out(
    device const bfloat *residual_in [[buffer(0)]],
    device bfloat *residual_out [[buffer(1)]],
    device const bfloat *block [[buffer(2)]],
    device const bfloat *injection [[buffer(3)]],
    constant HyperConnectionParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  const uint total = params.rows * width;
  for (uint element = index; element < total; element += grid_size) {
    const uint row = element / width;
    const uint offset = element % width;
    const float gate = float(injection[row * params.count + offset / hidden]);
    const float value = float(block[row * hidden + offset % hidden]);
    residual_out[element] = bfloat(float(residual_in[element]) + gate * value);
  }
}

// Final mixer: collapses the streams before lm_head without block injection.
kernel void hyper_connection_mix_no_inject(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *reduced [[buffer(1)]],
    device const bfloat *up [[buffer(2)]],
    device bfloat *mixed [[buffer(3)]],
    constant HyperConnectionParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  (void)simd_lane;
  (void)simd_group;
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  const uint low_rank = params.low_rank;
  device const bfloat *xn = normalized + ulong(row) * width;
  device const bfloat *low = reduced + ulong(row) * low_rank;

  threadgroup float staged[64];
  for (uint r = thread_index; r < low_rank && r < 64; r += kThreads)
    staged[r] = float(low[r]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat *out = mixed + ulong(row) * hidden;
  for (uint dimension = thread_index; dimension < hidden;
       dimension += kThreads) {
    float accumulated = 0.0f;
    for (uint stream = 0; stream < params.count; ++stream) {
      const uint index = stream * hidden + dimension;
      device const bfloat *weights = up + ulong(index) * low_rank;
      float sum = 0.0f;
      for (uint r = 0; r < low_rank; ++r) {
        const float value = r < 64 ? staged[r] : float(low[r]);
        sum += float(weights[r]) * value;
      }
      const float gate = 1.0f / (1.0f + exp(-sum));
      accumulated += gate * float(xn[index]);
    }
    out[dimension] = bfloat(accumulated / float(params.count));
  }
}

// Broadcasts a [rows, hidden] embedding to [rows, count * hidden] streams.
kernel void hyper_connection_broadcast(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant HyperConnectionParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint width = params.count * params.hidden;
  const uint total = params.rows * width;
  for (uint element = index; element < total; element += grid_size) {
    const uint row = element / width;
    const uint dim = element % params.hidden;
    output[element] = input[row * params.hidden + dim];
  }
}

// In-place vector addition: residual += delta
kernel void hyper_connection_accumulate(
    device bfloat *residual [[buffer(0)]],
    device const bfloat *delta [[buffer(1)]],
    constant HyperConnectionParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint total = params.rows * params.count * params.hidden;
  for (uint element = index; element < total; element += grid_size) {
    residual[element] = bfloat(float(residual[element]) + float(delta[element]));
  }
}


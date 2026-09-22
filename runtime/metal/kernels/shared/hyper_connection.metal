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

inline ulong hc_row(constant HyperConnectionParams &params, uint row) {
  return ulong(row) * (params.row_step ? params.row_step : 1);
}

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
  const uint hidden4 = hidden / 4;
  const uint width4 = width / 4;
  device const bfloat *x = input + ulong(row) * width;
  device bfloat *xn = normalized + ulong(row) * width;

  threadgroup float scales[8];
  threadgroup float partial[kSimdgroups];

  // One stream at a time: each is an independent RMS group.
  for (uint stream = 0; stream < params.count; ++stream) {
    device const bfloat4 *xs4 =
        reinterpret_cast<device const bfloat4 *>(x + stream * hidden);
    float sum = 0.0f;
    for (uint i = thread_index; i < hidden4; i += kThreads) {
      const bfloat4 val = xs4[i];
      sum += float(val.x) * float(val.x) + float(val.y) * float(val.y) +
             float(val.z) * float(val.z) + float(val.w) * float(val.w);
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

  device const bfloat4 *x4_in = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *gain4 = reinterpret_cast<device const bfloat4 *>(gain);
  device bfloat4 *xn4 = reinterpret_cast<device bfloat4 *>(xn);
  for (uint i = thread_index; i < width4; i += kThreads) {
    const bfloat4 val = x4_in[i];
    const bfloat4 g = gain4[i];
    const float scale = scales[i / hidden4];
    xn4[i] = bfloat4(
        bfloat(float(val.x) * scale * (1.0f + float(g.x))),
        bfloat(float(val.y) * scale * (1.0f + float(g.y))),
        bfloat(float(val.z) * scale * (1.0f + float(g.z))),
        bfloat(float(val.w) * scale * (1.0f + float(g.w))));
  }
  threadgroup_barrier(mem_flags::mem_device);

  // down is [low_rank, width]; one simdgroup per output, lanes stride the row.
  device bfloat *out = reduced + ulong(row) * params.low_rank;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(xn);
  for (uint o = simd_group; o < params.low_rank; o += kSimdgroups) {
    device const bfloat4 *w4 =
        reinterpret_cast<device const bfloat4 *>(down + ulong(o) * width);
    float sum = 0.0f;
    for (uint i = simd_lane; i < width4; i += kSimdWidth) {
      const bfloat4 w = w4[i];
      const bfloat4 vx = x4[i];
      sum += float(w.x) * float(vx.x) + float(w.y) * float(vx.y) +
             float(w.z) * float(vx.z) + float(w.w) * float(vx.w);
    }
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
  const uint low_rank4 = low_rank / 4;
  const uint width4 = width / 4;
  device const bfloat *xn = normalized + ulong(row) * width;
  device const bfloat *low = reduced + ulong(row) * low_rank;

  threadgroup float staged[320];
  for (uint r = thread_index; r < low_rank && r < 320; r += kThreads)
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
      device const bfloat4 *up4 =
          reinterpret_cast<device const bfloat4 *>(up + ulong(index) * low_rank);
      float sum = 0.0f;
      for (uint r = 0; r < low_rank4; ++r) {
        const bfloat4 w = up4[r];
        sum += float(w.x) * staged[r * 4 + 0] +
               float(w.y) * staged[r * 4 + 1] +
               float(w.z) * staged[r * 4 + 2] +
               float(w.w) * staged[r * 4 + 3];
      }
      const float gate = 1.0f / (1.0f + exp(-sum));
      accumulated += gate * float(xn[index]);
    }
    out[dimension] = bfloat(accumulated / float(params.count));
  }

  // inject is [count, width]; one simdgroup per stream.
  device bfloat *gates = injection + ulong(row) * params.count;
  device const bfloat4 *xn4 = reinterpret_cast<device const bfloat4 *>(xn);
  for (uint stream = simd_group; stream < params.count;
       stream += kSimdgroups) {
    device const bfloat4 *weights4 =
        reinterpret_cast<device const bfloat4 *>(inject + ulong(stream) * width);
    float sum = 0.0f;
    for (uint i = simd_lane; i < width4; i += kSimdWidth) {
      const bfloat4 w = weights4[i];
      const bfloat4 vx = xn4[i];
      sum += float(w.x) * float(vx.x) + float(w.y) * float(vx.y) +
             float(w.z) * float(vx.z) + float(w.w) * float(vx.w);
    }
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
    const ulong row = hc_row(params, element / width);
    const uint offset = element % width;
    const float gate = float(injection[row * params.count + offset / hidden]);
    const float value = float(block[row * hidden + offset % hidden]);
    const ulong at = row * width + offset;
    residual[at] = bfloat(float(residual[at]) + gate * value);
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
    const ulong row = hc_row(params, element / width);
    const uint offset = element % width;
    const float gate = float(injection[row * params.count + offset / hidden]);
    const float value = float(block[row * hidden + offset % hidden]);
    const ulong at = row * width + offset;
    residual_out[at] = bfloat(float(residual_in[at]) + gate * value);
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

// ---------------------------------------------------------------------------
// Weight-stationary hyper-connection: the same math in three dispatches that
// read each weight once for every row, instead of once per row.
//
// The per-row kernels above ran one threadgroup per row. With the 8 rows of a
// decode step that kept 8 threadgroups busy on a GPU with far more cores, and
// each re-read the whole 6.5 MB down and up matrices: about 2% of the
// machine's bandwidth, and 75% of all decode GPU time. Here parallelism comes
// from the outputs (324 down rows, 2560 mixed positions), and rows are
// processed in blocks against weights already in registers.
// ---------------------------------------------------------------------------

constant constexpr uint kRowBlock = 8;


// 1. xn = rms_per_stream(x) * (1 + gain). One threadgroup per (row, stream).
kernel void hyper_connection_rms(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *gain [[buffer(1)]],
    device bfloat *normalized [[buffer(2)]],
    constant HyperConnectionParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, stream = group.y;
  if (row >= params.rows || stream >= params.count)
    return;
  const uint hidden = params.hidden, width = params.count * hidden;
  const ulong base = hc_row(params, row) * width + stream * hidden;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(input + base);
  float sum = 0.0f;
  for (uint i = thread_index; i < hidden / 4; i += kThreads) {
    const float4 v = float4(x4[i]);
    sum += dot(v, v);
  }
  threadgroup float partial[kSimdgroups];
  sum = simd_sum(sum);
  if (simd_lane == 0)
    partial[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < kSimdgroups; ++s)
    total += partial[s];
  const float scale = rsqrt(total / float(hidden) + params.epsilon);
  device const bfloat4 *g4 =
      reinterpret_cast<device const bfloat4 *>(gain + stream * hidden);
  device bfloat4 *out4 = reinterpret_cast<device bfloat4 *>(normalized + base);
  for (uint i = thread_index; i < hidden / 4; i += kThreads)
    out4[i] = bfloat4(float4(x4[i]) * scale * (1.0f + float4(g4[i])));
}

// 2. reduced = silu(down . xn / count), injection = 2 sigmoid(inject . xn / count).
// A threadgroup owns 8 outputs (one per simdgroup) for up to 8 rows. The
// rows' normalized input is staged in threadgroup memory a slice at a time,
// so it is read from device memory once per threadgroup rather than once
// per output, and each weight element is read exactly once.
constant constexpr uint kDownSlice = 512;
inline float hc_down_activation(float total, bool is_inject, uint count) {
  const float v = total / float(count);
  return is_inject ? 2.0f / (1.0f + exp(-v)) : v / (1.0f + exp(-v));
}

kernel void hyper_connection_down(
    device const bfloat *normalized [[buffer(0)]],
    device const uchar *down [[buffer(1)]],
    device const bfloat *down_scales [[buffer(2)]],
    device const bfloat *down_biases [[buffer(3)]],
    device const bfloat *inject [[buffer(4)]],
    device bfloat *reduced [[buffer(5)]],
    device bfloat *injection [[buffer(6)]],
    device float *partials [[buffer(7)]],
    constant HyperConnectionParams &params [[buffer(8)]],
    uint2 group2 [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint width = params.count * params.hidden;
  const uint outputs = params.low_rank + (params.with_inject ? params.count : 0);
  const uint group = group2.x;
  const uint splits = max(params.splits, 1u);
  const uint per_split = width / splits;
  const uint output = group * kSimdgroups + simd_group;
  const bool active = output < outputs;
  const bool is_inject = output >= params.low_rank;
  // Low-rank rows are 8-bit in the matrix kernels' tiled layout - codes as
  // [256-row tile][group of 64 inputs][row][64], a scale and bias per (tile,
  // group, row) - so prompts can run them as a matrix product; the few inject
  // rows stay bf16. The branch is uniform across a simdgroup (one output each).
  device const bfloat *w = inject + ulong(active && is_inject ? output - params.low_rank : 0) * width;
  const uint down_row = active && !is_inject ? output : 0;
  const uint down_groups = width / 64;
  const ulong tile_base = ulong(down_row >> 8) * down_groups;  // (tile, group 0)
  const uint tile_row = down_row & 255u;
  threadgroup float4 staged[kRowBlock][kDownSlice / 4];
  for (uint first = 0; first < params.rows; first += kRowBlock) {
    const uint count = min(kRowBlock, params.rows - first);
    float acc[kRowBlock] = {0};
    for (uint slice = group2.y * per_split; slice < (group2.y + 1) * per_split;
         slice += kDownSlice) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = thread_index; e < kRowBlock * kDownSlice / 4; e += kThreads) {
        const uint r = e / (kDownSlice / 4), c = e % (kDownSlice / 4);
        staged[r][c] = r < count
            ? float4(reinterpret_cast<device const bfloat4 *>(
                  normalized + hc_row(params, first + r) * width + slice)[c])
            : float4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active && is_inject) {
        device const bfloat4 *w4 = reinterpret_cast<device const bfloat4 *>(w + slice);
        for (uint i = simd_lane; i < kDownSlice / 4; i += kSimdWidth) {
          const float4 wv = float4(w4[i]);
          for (uint r = 0; r < kRowBlock; ++r)
            acc[r] += dot(wv, staged[r][i]);
        }
      } else if (active) {
        for (uint i = simd_lane; i < kDownSlice / 4; i += kSimdWidth) {
          const uint k = slice + 4 * i;
          const ulong parameter = (tile_base + k / 64) * 256 + tile_row;
          const float4 wv = float4(*reinterpret_cast<device const uchar4 *>(
                                down + parameter * 64 + k % 64)) *
                                float(down_scales[parameter]) +
                            float(down_biases[parameter]);
          for (uint r = 0; r < kRowBlock; ++r)
            acc[r] += dot(wv, staged[r][i]);
        }
      }
    }
    for (uint r = 0; r < kRowBlock; ++r) {
      const float total = simd_sum(acc[r]);
      if (active && simd_lane == 0 && r < count) {
        if (splits > 1) {
          partials[(ulong(group2.y) * params.rows + first + r) * outputs + output] = total;
        } else if (is_inject) {
          injection[hc_row(params, first + r) * params.count + (output - params.low_rank)] =
              bfloat(hc_down_activation(total, true, params.count));
        } else {
          reduced[hc_row(params, first + r) * params.low_rank + output] =
              bfloat(hc_down_activation(total, false, params.count));
        }
      }
    }
  }
}

// 2b. Sums hyper_connection_down's split partials and applies its activation.
// One thread per (row, output).
kernel void hyper_connection_down_finish(
    device const float *partials [[buffer(0)]],
    device bfloat *reduced [[buffer(1)]],
    device bfloat *injection [[buffer(2)]],
    constant HyperConnectionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint outputs = params.low_rank + (params.with_inject ? params.count : 0);
  if (index >= params.rows * outputs)
    return;
  const uint row = index / outputs, output = index % outputs;
  float total = 0.0f;
  for (uint split = 0; split < params.splits; ++split)
    total += partials[(ulong(split) * params.rows + row) * outputs + output];
  const bool is_inject = output >= params.low_rank;
  const bfloat value = bfloat(hc_down_activation(total, is_inject, params.count));
  if (is_inject)
    injection[hc_row(params, row) * params.count + (output - params.low_rank)] = value;
  else
    reduced[hc_row(params, row) * params.low_rank + output] = value;
}

// 3. mixed = mean over streams of sigmoid(up . reduced) * xn.
// One simdgroup per hidden position: it owns that position's up row in each
// of the 4 streams, splits the 320-long dot products across its 32 lanes
// (10 each), and averages the streams itself. The rows' reduced vectors are
// staged once per threadgroup.
kernel void hyper_connection_up_mix(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *reduced [[buffer(1)]],
    device const uchar *up [[buffer(2)]],
    device const bfloat *up_scales [[buffer(3)]],
    device const bfloat *up_biases [[buffer(4)]],
    device bfloat *mixed [[buffer(5)]],
    constant HyperConnectionParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kLowRank = 320;
  constexpr uint kPerLane = kLowRank / kSimdWidth;   // 10
  constexpr uint kStreams = 4;
  const uint hidden = params.hidden, width = params.count * hidden;
  const uint position = group * kSimdgroups + simd_group;
  threadgroup float staged[kRowBlock][kLowRank];
  // This lane's slice of the position's four up rows.
  float w[kStreams][kPerLane];
  // Tiled like the down projection: 256-row tiles, 5 groups of 64 inputs.
  for (uint k = 0; k < kStreams; ++k) {
    const uint row = k * hidden + position;
    const ulong tile_base = ulong(row >> 8) * (kLowRank / 64);
    for (uint j = 0; j < kPerLane / 2; ++j) {
      // The pair never straddles a group: 64 and the lane's 10 are even.
      const uint input = simd_lane * kPerLane + 2 * j;
      const ulong parameter = (tile_base + input / 64) * 256 + (row & 255u);
      const float2 v = float2(*reinterpret_cast<device const uchar2 *>(
          up + parameter * 64 + input % 64));
      const float scale = float(up_scales[parameter]), bias = float(up_biases[parameter]);
      w[k][2 * j] = v.x * scale + bias;
      w[k][2 * j + 1] = v.y * scale + bias;
    }
  }
  for (uint first = 0; first < params.rows; first += kRowBlock) {
    const uint count = min(kRowBlock, params.rows - first);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = thread_index; e < kRowBlock * kLowRank; e += kThreads) {
      const uint r = e / kLowRank, c = e % kLowRank;
      staged[r][c] = r < count ? float(reduced[hc_row(params, first + r) * kLowRank + c]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < count; ++r) {
      threadgroup const float *v = staged[r] + simd_lane * kPerLane;
      float mean = 0.0f;
      for (uint k = 0; k < kStreams; ++k) {
        float sum = 0.0f;
        for (uint j = 0; j < kPerLane; ++j)
          sum += w[k][j] * v[j];
        sum = simd_sum(sum);
        const float gate = 1.0f / (1.0f + exp(-sum));
        mean += gate * float(normalized[hc_row(params, first + r) * width + k * hidden + position]);
      }
      if (simd_lane == 0)
        mixed[hc_row(params, first + r) * hidden + position] = bfloat(mean / float(params.count));
    }
  }
}

// ---------------------------------------------------------------------------
// Prompt path. With thousands of rows the down and up projections run as
// matrix products (prefill_linear_q8 over the tiled weights); these kernels
// finish around them. Rows here are physical (row_step 1).
// ---------------------------------------------------------------------------

// reduced = silu(raw / count) from the down product, whose rows are padded to
// kDownPadded outputs (kHyperDownPadded on the host). One thread per (row,
// low-rank output).
constant constexpr uint kDownPadded = 512;
kernel void hyper_connection_prompt_reduce(
    device const bfloat *raw [[buffer(0)]],
    device bfloat *reduced [[buffer(1)]],
    constant HyperConnectionParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint padded = kDownPadded;
  if (index >= params.rows * params.low_rank)
    return;
  const uint row = index / params.low_rank, output = index % params.low_rank;
  reduced[ulong(row) * params.low_rank + output] = bfloat(hc_down_activation(
      float(raw[ulong(row) * padded + output]), false, params.count));
}

// injection = 2 sigmoid(inject . xn / count). One threadgroup per (row, stream).
kernel void hyper_connection_prompt_inject(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *inject [[buffer(1)]],
    device bfloat *injection [[buffer(2)]],
    constant HyperConnectionParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, stream = group.y;
  if (row >= params.rows || stream >= params.count)
    return;
  const uint width = params.count * params.hidden;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(normalized + ulong(row) * width);
  device const bfloat4 *w4 = reinterpret_cast<device const bfloat4 *>(inject + ulong(stream) * width);
  float sum = 0.0f;
  for (uint i = thread_index; i < width / 4; i += kThreads)
    sum += dot(float4(x4[i]), float4(w4[i]));
  threadgroup float partial[kSimdgroups];
  sum = simd_sum(sum);
  if (simd_lane == 0)
    partial[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint s = 0; s < kSimdgroups; ++s)
      total += partial[s];
    injection[ulong(row) * params.count + stream] =
        bfloat(hc_down_activation(total, true, params.count));
  }
}

// mixed = mean over streams of sigmoid(up logit) * xn. One thread per
// (row, hidden position).
kernel void hyper_connection_prompt_mix(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *logits [[buffer(1)]],
    device bfloat *mixed [[buffer(2)]],
    constant HyperConnectionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.rows * params.hidden)
    return;
  const uint row = index / params.hidden, position = index % params.hidden;
  const uint width = params.count * params.hidden;
  float mean = 0.0f;
  for (uint k = 0; k < params.count; ++k) {
    const ulong at = ulong(row) * width + k * params.hidden + position;
    mean += float(normalized[at]) / (1.0f + exp(-float(logits[at])));
  }
  mixed[ulong(row) * params.hidden + position] = bfloat(mean / float(params.count));
}

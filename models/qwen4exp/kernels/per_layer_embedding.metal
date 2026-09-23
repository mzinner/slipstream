#include "metal/abi/KernelABI.h"
#include "models/qwen4exp/abi/PerLayerEmbedding.h"

// The gate and convolution half of the qwen4exp per-layer embedding, which
// runs after the hashed n-gram rows have been gathered and projected.
//
// Over a row, with Count streams of Hidden and width W = Count * Hidden:
//
//   key   = rms_per_stream(key_projection)  * (1 + key_gain)
//   query = rms_per_stream(hidden)          * (1 + query_gain)
//   gate  = (key . query) per stream / sqrt(Hidden)
//   gate  = sign(gate) * sqrt(|gate|)         clamped away from zero
//   value = sigmoid(gate) * value_projection  broadcast across the stream
//   out   = value + silu(conv(rms(value) * (1 + conv_gain)))
//
// Two details are worth stating because neither is visible in the shapes.
// The gate takes a signed square root, which compresses a dot product over
// 2560 terms into a range sigmoid can use. And the convolution is depthwise
// and dilated by the n-gram order, so its state spans (taps - 1) * order
// positions rather than taps - 1.

// PerLayerEmbeddingParams is in metal/abi/PerLayerEmbedding.h.

constant constexpr uint kThreads = 256;
constant constexpr uint kSimdgroups = kThreads / 32;

// One threadgroup per row. `history` holds the (taps - 1) * dilation
// positions preceding this row's block, already normalized by the caller.
kernel void per_layer_embedding_gate(
    device const bfloat *keys [[buffer(0)]],
    device const bfloat *values [[buffer(1)]],
    device const bfloat *hidden_states [[buffer(2)]],
    device const bfloat *key_gain [[buffer(3)]],
    device const bfloat *query_gain [[buffer(4)]],
    device const bfloat *convolution_gain [[buffer(5)]],
    device bfloat *gated [[buffer(6)]],
    device bfloat *normalized [[buffer(7)]],
    constant PerLayerEmbeddingParams &params [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint hidden = params.hidden;
  const uint width = params.count * hidden;
  device const bfloat *key_row = keys + ulong(row) * width;
  device const bfloat *query_row = hidden_states + ulong(row) * width;
  device const bfloat *value_row = values + ulong(row) * hidden;
  device bfloat *out = gated + ulong(row) * width;
  // The convolution reads history before the current rows, so the
  // normalized copy is written past that history. The caller owns the
  // leading (taps - 1) * dilation rows.
  device bfloat *out_normed =
      normalized + (ulong(row) + (params.taps - 1) * params.dilation) * width;

  threadgroup float key_scale[8];
  threadgroup float query_scale[8];
  threadgroup float gates[8];
  threadgroup float partial[kSimdgroups];

  // Both norms are per stream, as everywhere else in this model.
  for (uint stream = 0; stream < params.count; ++stream) {
    float key_square = 0.0f, query_square = 0.0f;
    for (uint i = thread_index; i < hidden; i += kThreads) {
      const float k = float(key_row[stream * hidden + i]);
      const float q = float(query_row[stream * hidden + i]);
      key_square += k * k;
      query_square += q * q;
    }
    key_square = simd_sum(key_square);
    query_square = simd_sum(query_square);
    if (simd_lane == 0) partial[simd_group] = key_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      key_scale[stream] = rsqrt(total / float(hidden) + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) partial[simd_group] = query_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      query_scale[stream] = rsqrt(total / float(hidden) + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // One gate per stream: the normalized key against the normalized query.
  for (uint stream = 0; stream < params.count; ++stream) {
    float dot = 0.0f;
    for (uint i = thread_index; i < hidden; i += kThreads) {
      const uint index = stream * hidden + i;
      const float k = float(key_row[index]) * key_scale[stream] *
                      (1.0f + float(key_gain[index]));
      const float q = float(query_row[index]) * query_scale[stream] *
                      (1.0f + float(query_gain[index]));
      dot += k * q;
    }
    dot = simd_sum(dot);
    if (simd_lane == 0) partial[simd_group] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      total /= sqrt(float(hidden));
      // Signed square root, held away from zero so the sign survives.
      const float magnitude = sqrt(max(abs(total), 1e-6f));
      gates[stream] = total < 0.0f ? -magnitude : magnitude;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The value is shared across streams; only the gate differs.
  for (uint i = thread_index; i < width; i += kThreads) {
    const uint stream = i / hidden;
    const float gate = 1.0f / (1.0f + exp(-gates[stream]));
    out[i] = bfloat(gate * float(value_row[i % hidden]));
  }
  threadgroup_barrier(mem_flags::mem_device);

  // The convolution reads a normalized copy, while the residual keeps the
  // unnormalized one.
  for (uint stream = 0; stream < params.count; ++stream) {
    float square = 0.0f;
    for (uint i = thread_index; i < hidden; i += kThreads) {
      const float value = float(out[stream * hidden + i]);
      square += value * value;
    }
    square = simd_sum(square);
    if (simd_lane == 0) partial[simd_group] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint s = 0; s < kSimdgroups; ++s) total += partial[s];
      key_scale[stream] = rsqrt(total / float(hidden) + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = thread_index; i < width; i += kThreads)
    out_normed[i] = bfloat(float(out[i]) * key_scale[i / hidden] *
                           (1.0f + float(convolution_gain[i])));
}

// Depthwise, dilated by the n-gram order, then silu, added to the residual.
// `normalized` carries the preceding state rows before the current rows, so
// row r reads positions r + state - tap * dilation.
kernel void per_layer_embedding_convolve(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device bfloat *gated [[buffer(2)]],
    constant PerLayerEmbeddingParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint width = params.count * params.hidden;
  const uint state = (params.taps - 1) * params.dilation;
  const uint total = params.rows * width;
  for (uint element = index; element < total; element += grid_size) {
    const uint row = element / width;
    const uint channel = element % width;
    float sum = 0.0f;
    for (uint tap = 0; tap < params.taps; ++tap) {
      // The last tap is the current position, earlier taps reach back by a
      // whole dilation each.
      const uint source = row + state - (params.taps - 1 - tap) * params.dilation;
      sum += float(normalized[ulong(source) * width + channel]) *
             float(weights[ulong(channel) * params.taps + tap]);
    }
    const float activated = sum / (1.0f + exp(-sum));
    gated[element] = bfloat(float(gated[element]) + activated);
  }
}

// ---------------------------------------------------------------------------
// History carried between steps.
//
// The state region holds a 16-byte header - valid, older token, newer token,
// padding - then `history` normalized rows of `width`. A zeroed region is a
// fresh sequence: two end-of-sequence tokens and a zero convolution history,
// exactly what the reference starts from.
// ---------------------------------------------------------------------------

constant constexpr uint kPleHeaderWords = 4;

inline uint ple_history_token(device const uint *header, uint index, uint eos) {
  return header[0] ? header[1 + index] : eos;
}

// The shifted token rows the n-gram gather reads, [3][rows]: each token, its
// predecessor, and the one before that.
//
// The reference restarts the window after an end-of-sequence token. With a
// window of three that reduces to one rule: two back is replaced by the
// end-of-sequence token whenever one back is one. (One back being itself the
// end-of-sequence token needs no rule, and neither does two back.)
kernel void per_layer_embedding_shift(
    device const uint *tokens [[buffer(0)]],
    device const uchar *state [[buffer(1)]],
    device uint *shifted [[buffer(2)]],
    constant PerLayerEmbeddingStateParams &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows)
    return;
  device const uint *header = reinterpret_cast<device const uint *>(state);
  const uint older = ple_history_token(header, 0, params.eos);
  const uint newer = ple_history_token(header, 1, params.eos);
  const uint one_back = row >= 1 ? tokens[row - 1] : newer;
  const uint two_back_raw =
      row >= 2 ? tokens[row - 2] : (row == 1 ? newer : older);
  shifted[row] = tokens[row];
  shifted[params.rows + row] = one_back;
  shifted[2 * params.rows + row] =
      one_back == params.eos ? params.eos : two_back_raw;
}

// Load the stored convolution history into the rows ahead of this step's.
kernel void per_layer_embedding_history(
    device const uchar *state [[buffer(0)]],
    device bfloat *normalized [[buffer(1)]],
    constant PerLayerEmbeddingStateParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  device const bfloat *rows = reinterpret_cast<device const bfloat *>(
      state + kPleHeaderWords * sizeof(uint));
  const uint total = params.history * params.width;
  for (uint element = index; element < total; element += grid_size)
    normalized[element] = rows[element];
}

// Add the embedding's output into the residual the layer reads.
kernel void per_layer_embedding_add(
    device const bfloat *gated [[buffer(0)]],
    device bfloat *residual [[buffer(1)]],
    constant PerLayerEmbeddingStateParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint total = params.rows * params.width;
  for (uint element = index; element < total; element += grid_size)
    residual[element] = bfloat(float(residual[element]) + float(gated[element]));
}

// Write the history the next step starts from, keeping only the rows that
// were committed. With r rows kept, the convolution history is normalized
// rows [r, r + history) - the last `history` of the stored history followed
// by the kept rows - and the two tokens are the last two of the stored pair
// followed by the kept tokens.
kernel void per_layer_embedding_commit(
    device const uint *tokens [[buffer(0)]],
    device const bfloat *normalized [[buffer(1)]],
    device const uchar *state_in [[buffer(2)]],
    device uchar *state_out [[buffer(3)]],
    device const uint *retained_counts [[buffer(4)]],
    constant PerLayerEmbeddingStateParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint kept = params.retained_from_buffer
                        ? min(retained_counts[params.lane], params.rows)
                        : params.retained;
  device bfloat *rows_out = reinterpret_cast<device bfloat *>(
      state_out + kPleHeaderWords * sizeof(uint));
  const uint total = params.history * params.width;
  for (uint element = index; element < total; element += grid_size)
    rows_out[element] = normalized[ulong(kept) * params.width + element];
  if (index == 0) {
    device const uint *header = reinterpret_cast<device const uint *>(state_in);
    // The sequence [older, newer, tokens...]; the new pair is its last two.
    uint sequence[2];
    for (uint i = 0; i < 2; ++i) {
      const uint position = kept + i; // index into [older, newer, tokens...]
      sequence[i] = position < 2 ? ple_history_token(header, position, params.eos)
                                 : tokens[position - 2];
    }
    device uint *out = reinterpret_cast<device uint *>(state_out);
    out[0] = 1;
    out[1] = sequence[0];
    out[2] = sequence[1];
    out[3] = 0;
  }
}

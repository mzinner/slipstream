#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

template <uint QHeads, uint KHeads>
inline void full_qkv_decode_phase(
    device const bfloat *qkv, device const bfloat *q_norm,
    device const bfloat *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values,
    constant FullDecodeBatchParams &params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint2 group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint HeadDim = 256, RotaryPairs = 32, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  ulong kv_lane_stride = ulong(KHeads) * params.row_stride * HeadDim;
  FullPrefillParams lane_params{params.tokens, params.cache_stride,
                                params.row_stride};
  full_qkv_storage_phase<QHeads, KHeads>(
      qkv + ulong(batch) * params.tokens * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * params.tokens * RotaryPairs,
      rope_sin + ulong(batch) * params.tokens * RotaryPairs,
      queries + ulong(batch) * QHeads * params.row_stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group);
}

kernel void verify_attention_qkv(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *q_norm [[buffer(1)]],
    device const bfloat *k_norm [[buffer(2)]],
    device const float *rope_cos [[buffer(3)]],
    device const float *rope_sin [[buffer(4)]],
    device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant FullDecodeBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[256];
  full_qkv_decode_phase<24, 4>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values, params,
      reductions, normalized, group, thread_index, lane, simd_group);
}

kernel void verify_attention_qkv_kv2_g8(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *q_norm [[buffer(1)]],
    device const bfloat *k_norm [[buffer(2)]],
    device const float *rope_cos [[buffer(3)]],
    device const float *rope_sin [[buffer(4)]],
    device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant FullDecodeBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[256];
  full_qkv_decode_phase<16, 2>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values, params,
      reductions, normalized, group, thread_index, lane, simd_group);
}

kernel void verify_attention_qkv_kv2_g12(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *q_norm [[buffer(1)]],
    device const bfloat *k_norm [[buffer(2)]],
    device const float *rope_cos [[buffer(3)]],
    device const float *rope_sin [[buffer(4)]],
    device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant FullDecodeBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[256];
  full_qkv_decode_phase<24, 2>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values, params,
      reductions, normalized, group, thread_index, lane, simd_group);
}

template <uint QHeads, uint KHeads>
inline void full_attention_gate_decode_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullDecodeBatchParams &params, uint index,
    uint grid_size) {
  constexpr uint HeadDim = 256, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  uint per_lane = params.tokens * QHeads * HeadDim;
  uint count = params.lanes * per_lane;
  for (uint element = index; element < count; element += grid_size) {
    uint batch = element / per_lane;
    uint lane_element = element % per_lane;
    uint row = lane_element / (QHeads * HeadDim);
    uint remainder = lane_element % (QHeads * HeadDim);
    uint query_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    float gate = float(
        packed_qkv[(ulong(batch) * params.tokens + row) * PackedStride +
                   query_head * QStride + HeadDim + dim]);
    float sigmoid = 1.0f / (1.0f + fast::exp2(-1.44269504089f * gate));
    uint kv_head = query_head / HeadsPerKV;
    uint local_head = query_head % HeadsPerKV;
    ulong attention_index =
        (((ulong(batch) * KHeads + kv_head) * params.row_stride + row) *
             HeadsPerKV +
         local_head) *
            HeadDim +
        dim;
    hidden[(ulong(batch) * params.tokens + row) * QHeads * HeadDim +
           query_head * HeadDim + dim] =
        bfloat(float(attention[attention_index]) * sigmoid);
  }
}

kernel void verify_attention_gate(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<24, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void verify_attention_gate_kv2_g8(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<16, 2>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void verify_attention_gate_kv2_g12(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<24, 2>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

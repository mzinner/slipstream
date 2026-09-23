#include "metal/abi/KernelABI.h"

// Sorted top-K lists: descending value, ascending token id on ties.
template <uint K>
inline void top_insert(thread float *values, thread uint *ids, float value,
                       uint token) {
  if (!(value > values[K - 1] ||
        (value == values[K - 1] && token < ids[K - 1])))
    return;
  uint slot = K - 1;
  while (slot > 0 && (value > values[slot - 1] ||
                      (value == values[slot - 1] && token < ids[slot - 1]))) {
    values[slot] = values[slot - 1];
    ids[slot] = ids[slot - 1];
    --slot;
  }
  values[slot] = value;
  ids[slot] = token;
}

// Each simdgroup drains its sorted top-K lists into threadgroup memory;
// thread 0 merges the eight lists into the shard's partial.
template <uint K>
__attribute__((always_inline)) inline void top_shard_store(
    thread float (&local_values)[K], thread uint (&local_ids)[K],
    threadgroup float (&group_values)[8 * K],
    threadgroup uint (&group_ids)[8 * K], device uint *partial_ids,
    device float *partial_values, uint group, uint thread_index, uint lane,
    uint simd_group) {
  uint cursor = 0;
  for (uint rank = 0; rank < K; ++rank) {
    float value = cursor < K ? local_values[cursor] : -INFINITY;
    uint token = cursor < K ? local_ids[cursor] : 0xffffffffu;
    float simd_best = simd_max(value);
    uint simd_id = simd_min(value == simd_best ? token : 0xffffffffu);
    uint winner =
        simd_min(value == simd_best && token == simd_id ? lane : 0xffffffffu);
    if (lane == 0) {
      group_values[simd_group * K + rank] = simd_best;
      group_ids[simd_group * K + rank] = simd_id;
    }
    if (lane == winner)
      ++cursor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_index == 0) {
    float final_values[K];
    uint final_ids[K];
    for (uint i = 0; i < K; ++i) {
      final_values[i] = -INFINITY;
      final_ids[i] = 0xffffffffu;
    }
    for (uint item = 0; item < 8 * K; ++item)
      top_insert<K>(final_values, final_ids, group_values[item],
                    group_ids[item]);
    for (uint rank = 0; rank < K; ++rank) {
      partial_ids[group * K + rank] = final_ids[rank];
      partial_values[group * K + rank] = final_values[rank];
    }
  }
}

// Merges the Shards partials of one row into its sorted top-K.
template <uint K, uint Shards>
__attribute__((always_inline)) inline void top_partials_reduce(
    device const uint *partial_ids, device const float *partial_values,
    uint row, thread float (&values)[K], thread uint (&ids)[K]) {
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  uint origin = row * Shards * K;
  for (uint item = 0; item < Shards * K; ++item)
    top_insert<K>(values, ids, partial_values[origin + item],
                  partial_ids[origin + item]);
}

// One row's sparse target distribution: the merged top-32 is softmaxed at
// temperature over its first top_k entries, truncated by top_p, renormalized
// and written in ascending token-id order; slots past the valid count carry
// ~0u. The constant references keep the divisions inside the loops.
__attribute__((always_inline)) inline void top32_probs_row(
    device const uint *partial_ids, device const float *partial_values,
    device uint *top_ids, device float *top_probs, uint row,
    constant uint &top_k, constant float &temperature, constant float &top_p) {
  float values[32];
  uint ids[32];
  top_partials_reduce<32, SPLASH_TARGET_SAMPLING_SHARDS>(partial_ids, partial_values, row, values, ids);

  uint valid_count = 0;
  while (valid_count < 32 && ids[valid_count] != 0xffffffffu)
    ++valid_count;
  uint selected_count = min(top_k, valid_count);

  float probabilities[32];
  float sum = 0.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    float probability = rank < selected_count
                            ? exp((values[rank] - values[0]) / temperature)
                            : 0.0f;
    probabilities[rank] = probability;
    sum += probability;
  }
  float prefix = 0.0f;
  float kept_sum = 0.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    float probability = probabilities[rank] / sum;
    bool keep = rank < selected_count && prefix <= top_p;
    probabilities[rank] = keep ? probability : 0.0f;
    prefix += probability;
    kept_sum += probabilities[rank];
  }
  uint used = 0;
  for (uint output = 0; output < 32; ++output) {
    ulong destination = ulong(row) * 32 + output;
    if (output >= valid_count) {
      top_ids[destination] = 0xffffffffu;
      top_probs[destination] = 0.0f;
      continue;
    }
    uint best = 32;
    uint best_id = 0xffffffffu;
    for (uint rank = 0; rank < valid_count; ++rank) {
      if ((used & (1u << rank)) == 0 && ids[rank] < best_id) {
        best = rank;
        best_id = ids[rank];
      }
    }
    used |= 1u << best;
    top_ids[destination] = best_id;
    top_probs[destination] = probabilities[best] / kept_sum;
  }
}

kernel void
decode_sample_top32_sharded(device const bfloat *logits [[buffer(0)]],
                     device uint *partial_ids [[buffer(1)]],
                     device float *partial_values [[buffer(2)]],
                     device const uint *token_mask [[buffer(3)]],
                     constant TargetSamplingParams &params [[buffer(4)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint thread_index [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  uint row = group / Shards;
  uint shard = group % Shards;
  threadgroup float group_values[8 * 32];
  threadgroup uint group_ids[8 * 32];
  device const bfloat *source =
      logits + ulong(params.row_offset + row) * params.vocabulary;
  float local_values[32];
  uint local_ids[32];
  for (uint i = 0; i < 32; ++i) {
    local_values[i] = -INFINITY;
    local_ids[i] = 0xffffffffu;
  }
  for (uint token = shard * 256 + thread_index; token < params.vocabulary;
       token += Shards * 256) {
    uint mask_row = params.mask_row_offset + row;
    if (params.constrained &&
        (token_mask[mask_row * params.mask_words + token / 32] &
         (1u << (token % 32))) == 0)
      continue;
    top_insert<32>(local_values, local_ids, float(source[token]), token);
  }
  top_shard_store<32>(local_values, local_ids, group_values, group_ids,
                      partial_ids, partial_values, group, thread_index, lane,
                      simd_group);
}

kernel void decode_sample_sparse_top1(device const uint *top_ids [[buffer(0)]],
                        device const float *top_probs [[buffer(1)]],
                        device uint *tokens [[buffer(2)]],
                        uint row [[thread_position_in_grid]]) {
  uint best_id = 0xffffffffu;
  float best_probability = -1.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    uint index = row * 32 + rank;
    float probability = top_probs[index];
    uint token = top_ids[index];
    if (probability > best_probability ||
        (probability == best_probability && token < best_id)) {
      best_probability = probability;
      best_id = token;
    }
  }
  tokens[row] = best_id;
}

kernel void decode_sample_top32_probs(device const uint *partial_ids [[buffer(0)]],
                               device const float *partial_values [[buffer(1)]],
                               device uint *top_ids [[buffer(2)]],
                               device float *top_probs [[buffer(3)]],
                               constant TargetSamplingParams &params
                               [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint thread_index
                               [[thread_index_in_threadgroup]]) {
  if (thread_index != 0)
    return;
  top32_probs_row(partial_ids, partial_values, top_ids, top_probs, row,
                  params.top_k, params.temperature, params.top_p);
}

kernel void decode_sample_top32_sharded_batch(
    device const bfloat *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device const uint *token_mask [[buffer(3)]],
    constant TargetSamplingBatchParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  uint global_row = group / Shards;
  uint shard = group % Shards;
  uint batch = global_row / params.rows_per_lane;
  uint row = global_row % params.rows_per_lane;
  if (batch >= params.lanes)
    return;
  threadgroup float group_values[8 * 32];
  threadgroup uint group_ids[8 * 32];
  device const bfloat *source = logits + ulong(global_row) * params.vocabulary;
  float local_values[32];
  uint local_ids[32];
  for (uint i = 0; i < 32; ++i) {
    local_values[i] = -INFINITY;
    local_ids[i] = 0xffffffffu;
  }
  bool constrained = (params.constrained_mask & (1u << batch)) != 0;
  ulong mask_origin =
      ulong(batch) * (SPLASH_TARGET_VERIFY_ROWS + 1) * params.mask_words +
      ulong(row + 1) * params.mask_words;
  for (uint token = shard * 256 + thread_index; token < params.vocabulary;
       token += Shards * 256) {
    if (constrained &&
        (token_mask[mask_origin + token / 32] & (1u << (token % 32))) == 0)
      continue;
    top_insert<32>(local_values, local_ids, float(source[token]), token);
  }
  top_shard_store<32>(local_values, local_ids, group_values, group_ids,
                      partial_ids, partial_values, group, thread_index, lane,
                      simd_group);
}

kernel void decode_sample_top32_probs_batch(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device uint *top_ids [[buffer(2)]], device float *top_probs [[buffer(3)]],
    constant TargetSamplingBatchParams &params [[buffer(4)]],
    uint global_row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (thread_index != 0)
    return;
  uint batch = global_row / params.rows_per_lane;
  if (batch >= params.lanes)
    return;
  top32_probs_row(partial_ids, partial_values, top_ids, top_probs, global_row,
                  params.top_k[batch], params.temperature[batch],
                  params.top_p[batch]);
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition.
inline float sparse_lookup(device const uint *ids,
                           device const float *probabilities, uint count,
                           uint token) {
  for (uint i = 0; i < count; ++i) {
    if (ids[i] == token)
      return probabilities[i];
  }
  return 0.0f;
}

inline uint sparse_sample(device const uint *ids,
                          device const float *probabilities, float uniform) {
  float total = 0.0f;
  for (uint i = 0; i < 32; ++i)
    total += probabilities[i];
  float threshold = uniform * total;
  float cumulative = 0.0f;
  uint fallback = ids[0];
  for (uint i = 0; i < 32; ++i) {
    if (!(probabilities[i] > 0.0f))
      continue;
    fallback = ids[i];
    cumulative += probabilities[i];
    if (cumulative > threshold)
      return ids[i];
  }
  return fallback;
}

inline uint sparse_residual_sample(device const uint *target_ids,
                                   device const float *target_probs,
                                   device const uint *draft_ids,
                                   device const float *draft_probs,
                                   float uniform) {
  float total = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    float q = sparse_lookup(draft_ids, draft_probs, 16, target_ids[i]);
    total += max(target_probs[i] - q, 0.0f);
  }
  if (!(total > 0.0f)) {
    return sparse_sample(target_ids, target_probs, uniform);
  }
  float threshold = uniform * total;
  float cumulative = 0.0f;
  uint fallback = target_ids[0];
  for (uint i = 0; i < 32; ++i) {
    float q = sparse_lookup(draft_ids, draft_probs, 16, target_ids[i]);
    float residual = max(target_probs[i] - q, 0.0f);
    if (!(residual > 0.0f))
      continue;
    fallback = target_ids[i];
    cumulative += residual;
    if (cumulative > threshold)
      return target_ids[i];
  }
  return fallback;
}

kernel void decode_sample_sparse_draw(device const uint *top_ids [[buffer(0)]],
                                 device const float *top_probs [[buffer(1)]],
                                 device const float *uniforms [[buffer(2)]],
                                 device uint *tokens [[buffer(3)]]) {
  tokens[0] = sparse_sample(top_ids, top_probs, uniforms[0]);
}

// Keeps at most params.remaining of the accepted tokens plus the correction,
// cut after the first stop token, and records the count and the next anchor.
inline void finish_acceptance(device const uint *tokens, uint accepted,
                              AcceptParams params, device uint &retained,
                              device uint &next_anchor,
                              device uint &accepted_count) {
  accepted_count = accepted;
  retained = min(accepted + 1, params.remaining);
  for (uint i = 0; i < retained; ++i) {
    if (tokens[i] == params.stop_token_0 || tokens[i] == params.stop_token_1) {
      retained = i + 1;
      break;
    }
  }
  next_anchor = tokens[retained - 1];
}

inline void accept_sampled_lane(device const uint *draft_tokens,
                                device const uint *draft_ids,
                                device const float *draft_probs,
                                device const uint *target_ids,
                                device const float *target_probs,
                                device const float *uniforms,
                                device uint *output_tokens,
                                device uint &retained,
                                device uint &next_anchor,
                                device uint &accepted_count,
                                AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS) {
    uint token = draft_tokens[accepted];
    float q = sparse_lookup(draft_ids + accepted * 16,
                            draft_probs + accepted * 16, 16, token);
    float p = sparse_lookup(target_ids + accepted * 32,
                            target_probs + accepted * 32, 32, token);
    if (!(uniforms[accepted + SPLASH_TARGET_VERIFY_ROWS] * q < p))
      break;
    output_tokens[accepted] = token;
    ++accepted;
  }
  if (accepted == SPLASH_DRAFT_PROPOSAL_TOKENS) {
    output_tokens[accepted] =
        sparse_sample(target_ids + SPLASH_DRAFT_PROPOSAL_TOKENS * 32,
                      target_probs + SPLASH_DRAFT_PROPOSAL_TOKENS * 32,
                      uniforms[2 * SPLASH_TARGET_VERIFY_ROWS - 1]);
  } else {
    output_tokens[accepted] = sparse_residual_sample(
        target_ids + accepted * 32, target_probs + accepted * 32,
        draft_ids + accepted * 16, draft_probs + accepted * 16,
        uniforms[2 * SPLASH_TARGET_VERIFY_ROWS - 1]);
  }
  finish_acceptance(output_tokens, accepted, params, retained, next_anchor,
                    accepted_count);
}

kernel void decode_sample_argmax_sharded(device const bfloat *logits [[buffer(0)]],
                              device float *partial_values [[buffer(1)]],
                              device uint *partial_indices [[buffer(2)]],
                              constant uint &vocabulary [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint thread_index
                              [[thread_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  uint row = group / Shards;
  uint shard = group % Shards;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  device const bfloat *source = logits + ulong(row) * vocabulary;
  for (uint token = shard * 256 + thread_index; token < vocabulary;
       token += Shards * 256) {
    float value = float(source[token]);
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_values[group] = group_best;
      partial_indices[group] = index;
    }
  }
}

kernel void decode_sample_argmax_reduce(device const float *partial_values [[buffer(0)]],
                             device const uint *partial_indices [[buffer(1)]],
                             device uint *tokens [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  index = simd_min(index);
  if (lane == 0)
    tokens[row] = index;
}

kernel void verify_input_tokens(
    device const uint *draft_input [[buffer(0)]],
    device const uint *draft_tokens [[buffer(1)]],
    device uint *verify_input [[buffer(2)]],
    constant VerifyInputBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS;
  if (index < count) {
    uint batch = index / SPLASH_TARGET_VERIFY_ROWS;
    uint row = index % SPLASH_TARGET_VERIFY_ROWS;
    uint token = row == 0
                     ? draft_input[batch * SPLASH_TARGET_VERIFY_ROWS]
                     : draft_tokens[batch * SPLASH_DRAFT_PROPOSAL_TOKENS +
                                    row - 1];
    verify_input[index] = min(token, params.vocabulary - 1u);
  }
}

inline void accept_greedy_lane(device const uint *draft_tokens,
                               device uint *target_tokens,
                               device uint &retained,
                               device uint &next_anchor,
                               device uint &accepted_count,
                               AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS &&
         draft_tokens[accepted] == target_tokens[accepted]) {
    ++accepted;
  }
  finish_acceptance(target_tokens, accepted, params, retained, next_anchor,
                    accepted_count);
}

kernel void decode_accept_dflash(
    device const uint *draft_tokens [[buffer(0)]],
    device const uint *draft_ids [[buffer(1)]],
    device const float *draft_probs [[buffer(2)]],
    device const uint *target_ids [[buffer(3)]],
    device const float *target_probs [[buffer(4)]],
    device const float *uniforms [[buffer(5)]],
    device uint *target_tokens [[buffer(6)]],
    device uint *retained [[buffer(7)]],
    device uint *next_anchor [[buffer(8)]],
    device uint *accepted_count [[buffer(9)]],
    constant AcceptBatchParams &params [[buffer(10)]],
    uint batch [[threadgroup_position_in_grid]]) {
  if (batch >= params.lanes)
    return;
  uint remaining = params.remaining[batch];
  AcceptParams lane_params{remaining, params.stop_token_0,
                           params.stop_token_1};
  device const uint *lane_draft =
      draft_tokens + batch * SPLASH_DRAFT_PROPOSAL_TOKENS;
  device uint *lane_target =
      target_tokens + batch * SPLASH_TARGET_VERIFY_ROWS;
  if (params.sampling_mask & (1u << batch)) {
    accept_sampled_lane(
        lane_draft,
        draft_ids + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * 16,
        draft_probs + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * 16,
        target_ids + batch * SPLASH_TARGET_VERIFY_ROWS * 32,
        target_probs + batch * SPLASH_TARGET_VERIFY_ROWS * 32,
        uniforms + batch * 2 * SPLASH_TARGET_VERIFY_ROWS, lane_target,
        retained[batch], next_anchor[batch], accepted_count[batch],
        lane_params);
  } else {
    accept_greedy_lane(lane_draft, lane_target, retained[batch],
                       next_anchor[batch], accepted_count[batch], lane_params);
  }
}

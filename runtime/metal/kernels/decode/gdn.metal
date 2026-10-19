#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"

// Decode threadgroups are 256 threads: one simdgroup per verify row in the
// prologue, and in the scan the head's 128 state rows strided over the eight
// simdgroups (each advances two of its sixteen rows at a time).
//
// Which of a step's rows a lane retains is known only after the step, so the
// state a step writes is the one before its rows: it leaves the rows on the
// lane's tape, and the lane's next step folds the retained ones into the
// state before its own rows. A step reads and writes the state once.
constant uint kDecodeSimdgroups = 8;

inline void grid_completion(device atomic_uint &arrived,
                            device atomic_uint &generation, uint group_count,
                            uint thread_index) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    uint ticket = atomic_fetch_add_explicit(&arrived, 1, memory_order_relaxed);
    if (ticket + 1 == group_count) {
      atomic_store_explicit(&arrived, 0, memory_order_relaxed);
      atomic_fetch_add_explicit(&generation, 1, memory_order_relaxed);
    }
  }
}

// The threadgroup operands of one value head: the rows' prepared q/k, the
// head's v, the gates, and the k, v and gates of the pending rows, which the
// head folds into its state first.
template <uint HeadDim> struct GdnDecodeShared {
  bfloat queries[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat keys[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat values[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  float decay[SPLASH_TARGET_VERIFY_ROWS];
  bfloat beta[SPLASH_TARGET_VERIFY_ROWS];
  bfloat pending_keys[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat pending_values[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  float pending_decay[SPLASH_TARGET_VERIFY_ROWS];
  bfloat pending_beta[SPLASH_TARGET_VERIFY_ROWS];
};

// One layer's tape of one lane's step (metal/abi/GDN.h): the rows'
// convolution inputs, their k rows (each key head's) then v rows, decay and
// beta.
struct GdnTape {
  device bfloat *inputs;
  device bfloat *kv;
  device float *decay;
  device bfloat *beta;
};

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline GdnTape gdn_tape(device uchar *tape, ulong offset) {
  constexpr uint KvWidth = (KeyHeads + ValueHeads) * HeadDim;
  device uchar *layer = tape + offset;
  return {reinterpret_cast<device bfloat *>(layer),
          reinterpret_cast<device bfloat *>(layer + gdn_tape_kv_offset(ConvDim)),
          reinterpret_cast<device float *>(
              layer + gdn_tape_decay_offset(ConvDim, KvWidth)),
          reinterpret_cast<device bfloat *>(
              layer + gdn_tape_beta_offset(ConvDim, KvWidth, ValueHeads))};
}

// Eight verify rows' conv+SiLU, q/k RMS norms and gates for one value head.
// One simdgroup per row holds channels 32g + lane (g = 0..3). RMS reduction
// sums each 32-channel group, then adds the four partials in channel order.
// The convolution reads the inputs before the rows from `history`: the
// carried rows of the state the lane's pending rows leave, which also go to
// conv_state_out. q/k/v and the gates go to threadgroup memory for the scan,
// beside the pending rows' k, v and gates from their tape; the rows'
// convolution inputs, k, v and gates go to this step's tape, from which the
// lane's next step folds in the rows the lane retains.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    GdnConvHistory history, device bfloat *conv_state_out,
    GdnTape pending_tape, uint pending, GdnTape step_tape,
    device const float *a_scale, device const bfloat *dt_bias,
    uint packed_width, threadgroup GdnDecodeShared<HeadDim> &shared,
    uint value_head, uint lane, uint simd_group, uint tree_parents = 0) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint KvWidth = KeyWidth + ValueWidth;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens == kDecodeSimdgroups && HeadDim == 128,
                "one simdgroup per verify row, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const bool carrier = simd_group < SPLASH_GDN_CONVOLUTION_TAPS - 1;
  const uint token = simd_group;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  const uint k_column = key_head * HeadDim + lane;
  const uint v_column = KeyWidth + value_head * HeadDim + lane;

  if (token < pending) {
    for (uint g = 0; g < Groups; ++g) {
      shared.pending_keys[token * HeadDim + 32 * g + lane] =
          pending_tape.kv[token * KvWidth + k_column + 32 * g];
      shared.pending_values[token * HeadDim + 32 * g + lane] =
          pending_tape.kv[token * KvWidth + v_column + 32 * g];
    }
    if (lane == 0) {
      shared.pending_decay[token] =
          pending_tape.decay[token * ValueHeads + value_head];
      shared.pending_beta[token] =
          pending_tape.beta[token * ValueHeads + value_head];
    }
  }

  float q[Groups], k[Groups];
  bfloat v[Groups], carry_v[Groups], carry_q[Groups], carry_k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, history, conv_weights, packed_width,
                               token, q_channel + 32 * g, tree_parents));
    k[g] = float(gdn_conv_silu(packed, history, conv_weights, packed_width,
                               token, k_channel + 32 * g, tree_parents));
    v[g] = gdn_conv_silu(packed, history, conv_weights, packed_width, token,
                         v_channel + 32 * g, tree_parents);
    if (carrier) {
      carry_v[g] = history.at(simd_group, v_channel + 32 * g);
      carry_q[g] = shared_writer ? history.at(simd_group, q_channel + 32 * g)
                                 : bfloat(0.0f);
      carry_k[g] = shared_writer ? history.at(simd_group, k_channel + 32 * g)
                                 : bfloat(0.0f);
    }
  }
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
  const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
  device const bfloat *inputs = packed + token * packed_width;
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    shared.queries[token * HeadDim + dim] = query;
    shared.keys[token * HeadDim + dim] = key;
    shared.values[token * HeadDim + dim] = v[g];
    step_tape.inputs[token * ConvDim + v_channel + 32 * g] =
        inputs[v_channel + 32 * g];
    step_tape.kv[token * KvWidth + v_column + 32 * g] = v[g];
    if (shared_writer) {
      step_tape.inputs[token * ConvDim + q_channel + 32 * g] =
          inputs[q_channel + 32 * g];
      step_tape.inputs[token * ConvDim + k_channel + 32 * g] =
          inputs[k_channel + 32 * g];
      step_tape.kv[token * KvWidth + k_column + 32 * g] = key;
    }
  }
  if (lane == 0) {
    const uint gate_index = token * ValueHeads + value_head;
    gdn_write_gates(inputs, dt_bias, a_scale, BOffset, AOffset, value_head,
                    step_tape.beta[gate_index], step_tape.decay[gate_index]);
    shared.beta[token] = step_tape.beta[gate_index];
    shared.decay[token] = step_tape.decay[gate_index];
  }
  if (carrier) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] = carry_v[g];
      if (shared_writer) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] = carry_q[g];
        conv_state_out[row * ConvDim + k_channel + 32 * g] = carry_k[g];
      }
    }
  }
}

// Delta-rule recurrence over 128 state rows, four fp32 columns per lane.
// RowsInFlight rows advance together to overlap their reductions and
// arithmetic. The pending rows advance the incoming state to the committed
// one, which goes to state_out; the step's own rows then advance it only for
// their outputs, which go to recurrent. Each row preserves the decay, memory,
// delta, update, output operation order.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight>
inline void gdn_decode_scan(device const float *state_in,
                            device float *state_out, device bfloat *output,
                            threadgroup GdnDecodeShared<HeadDim> &shared,
                            uint pending, uint value_head, uint lane,
                            uint simd_group, uint tree_parents = 0) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const auto base = [=](uint batch, uint r) {
    const uint value_dim = (batch + r) * kDecodeSimdgroups + simd_group;
    return (ulong(value_head) * HeadDim + value_dim) * HeadDim + lane * 4;
  };
  if (tree_parents == 0) {
    float committed[RowsInFlight][4];
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        committed[r][i] = state_in[base(0, r) + i];
    for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
      uint value_dim[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r)
        value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      for (uint token = 0; token < pending; ++token) {
        const float d = shared.pending_decay[token];
        const float b = float(shared.pending_beta[token]);
        threadgroup const bfloat *key =
            shared.pending_keys + token * HeadDim + lane * 4;
        float memory[RowsInFlight];
        for (uint r = 0; r < RowsInFlight; ++r) {
          memory[r] = 0.0f;
          for (uint i = 0; i < 4; ++i) {
            committed[r][i] *= d;
            memory[r] += committed[r][i] * float(key[i]);
          }
          memory[r] = simd_sum(memory[r]);
        }
        for (uint r = 0; r < RowsInFlight; ++r) {
          const float delta =
              (float(shared.pending_values[token * HeadDim + value_dim[r]]) -
               memory[r]) *
              b;
          for (uint i = 0; i < 4; ++i)
            committed[r][i] += float(key[i]) * delta;
        }
      }
      float upcoming[RowsInFlight][4] = {};
      if (batch + RowsInFlight < Batches) {
        for (uint r = 0; r < RowsInFlight; ++r)
          for (uint i = 0; i < 4; ++i)
            upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
      }
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          state_out[base(batch, r) + i] = committed[r][i];
      for (uint token = 0; token < Tokens; ++token) {
        const float d = shared.decay[token];
        const float b = float(shared.beta[token]);
        threadgroup const bfloat *key =
            shared.keys + token * HeadDim + lane * 4;
        threadgroup const bfloat *query =
            shared.queries + token * HeadDim + lane * 4;
        float memory[RowsInFlight];
        for (uint r = 0; r < RowsInFlight; ++r) {
          memory[r] = 0.0f;
          for (uint i = 0; i < 4; ++i) {
            committed[r][i] *= d;
            memory[r] += committed[r][i] * float(key[i]);
          }
          memory[r] = simd_sum(memory[r]);
        }
        float result[RowsInFlight];
        for (uint r = 0; r < RowsInFlight; ++r) {
          const float delta =
              (float(shared.values[token * HeadDim + value_dim[r]]) -
               memory[r]) *
              b;
          result[r] = 0.0f;
          for (uint i = 0; i < 4; ++i) {
            committed[r][i] += float(key[i]) * delta;
            result[r] += committed[r][i] * float(query[i]);
          }
          result[r] = simd_sum(result[r]);
        }
        if (lane == 0) {
          for (uint r = 0; r < RowsInFlight; ++r) {
            output[(ulong(token) * ValueHeads + value_head) * HeadDim +
                   value_dim[r]] = bfloat(result[r]);
          }
        }
      }
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          committed[r][i] = upcoming[r][i];
    }
    return;
  }

  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    float committed[RowsInFlight][4];
    float state[Tokens][RowsInFlight][4];
    uint value_dim[RowsInFlight];
    ulong state_base[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      state_base[r] = base(batch, r);
      for (uint i = 0; i < 4; ++i)
        committed[r][i] = state_in[state_base[r] + i];
    }
    for (uint token = 0; token < pending; ++token) {
      const float d = shared.pending_decay[token];
      const float b = float(shared.pending_beta[token]);
      threadgroup const bfloat *key =
          shared.pending_keys + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          float st =
              (token == 0) ? committed[r][i]
                           : state[uint(tree_parent(tree_parents, int(token)))][r][i];
          st *= d;
          state[token][r][i] = st;
          memory[r] += st * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(shared.pending_values[token * HeadDim + value_dim[r]]) -
             memory[r]) *
            b;
        for (uint i = 0; i < 4; ++i)
          state[token][r][i] += float(key[i]) * delta;
      }
    }
    if (pending > 0) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          committed[r][i] = state[pending - 1][r][i];
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[state_base[r] + i] = committed[r][i];
    for (uint token = 0; token < Tokens; ++token) {
      const float d = shared.decay[token];
      const float b = float(shared.beta[token]);
      threadgroup const bfloat *key = shared.keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query =
          shared.queries + token * HeadDim + lane * 4;
      int p = tree_parent(tree_parents, int(token));
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          float st = (p < 0) ? committed[r][i] : state[uint(p)][r][i];
          st *= d;
          state[token][r][i] = st;
          memory[r] += st * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(shared.values[token * HeadDim + value_dim[r]]) - memory[r]) *
            b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[token][r][i] += float(key[i]) * delta;
          result[r] += state[token][r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r) {
          output[(ulong(token) * ValueHeads + value_head) * HeadDim +
                 value_dim[r]] = bfloat(result[r]);
        }
      }
    }
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device uchar *tape, device const float *a_scale,
    device const bfloat *dt_bias,
    device bfloat *recurrent, device const bfloat *gdn_norm_weight,
    device bfloat *gdn_hidden, device atomic_uint *arrived,
    device atomic_uint *generation, constant GDNDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane, uint simd_group,
    threadgroup float *scratch, threadgroup GdnDecodeShared<HeadDim> &shared) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  if (batch >= params.lanes || group.x >= ValueHeads)
    return;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  packed += ulong(batch) * Rows * params.packed_width;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);

  const uint pending = params.pending_rows[batch];
  const GdnTape pending_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(
          tape, params.pending_tape[batch]);
  const GdnTape step_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(
          tape, params.step_tape[batch]);
  const GdnConvHistory history{conv_state_in, pending_tape.inputs, ConvDim,
                               ConvDim, pending, params.tree_parents};

  device bfloat *lane_recurrent =
      recurrent + ulong(batch) * Rows * ValueWidth;
  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, conv_weights, history, conv_state_out, pending_tape, pending,
      step_tape, a_scale, dt_bias, params.packed_width, shared, group.x, lane,
      simd_group, params.tree_parents);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_decode_scan<KeyHeads, ValueHeads, HeadDim, ConvDim, RowsInFlight>(
      state_in, state_out, lane_recurrent, shared, pending, group.x, lane,
      simd_group, params.tree_parents);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  gdn_gate_phase<ValueHeads, HeadDim, ConvDim, kDecodeSimdgroups>(
      lane_recurrent, packed, gdn_norm_weight, lane_hidden, Rows * ValueHeads,
      ValueHeads, params.packed_width, params.sigmoid_gate != 0, scratch,
      group.x, thread_index, lane, simd_group);
  grid_completion(arrived[batch], generation[batch], ValueHeads,
                  thread_index);
}

#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim,         \
                         RowsInFlight)                                         \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const uchar *current0 [[buffer(2)]],                             \
      device const uchar *current1 [[buffer(3)]],                             \
      device const uchar *current2 [[buffer(4)]],                             \
      device const uchar *current3 [[buffer(5)]],                             \
      device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],   \
      device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],   \
      device uchar *tape [[buffer(10)]],                                      \
      device const float *a_scale [[buffer(11)]],                             \
      device const bfloat *dt_bias [[buffer(12)]],                            \
      device bfloat *recurrent [[buffer(13)]],                                \
      device const bfloat *gdn_norm_weight [[buffer(14)]],                    \
      device bfloat *gdn_hidden [[buffer(15)]],                               \
      device atomic_uint *arrived [[buffer(16)]],                             \
      device atomic_uint *generation [[buffer(17)]],                          \
      constant GDNDecodeBatchParams &params [[buffer(18)]],                   \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float scratch[kDecodeSimdgroups];                             \
    threadgroup GdnDecodeShared<HeadDim> shared;                              \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim,            \
                           RowsInFlight>(                                     \
        packed, conv_weights, current0, current1, current2, current3, next0,  \
        next1, next2, next3, tape, a_scale, dt_bias, recurrent,               \
        gdn_norm_weight, gdn_hidden, arrived, generation, params, group,       \
        thread_index, lane, simd_group, scratch, shared);                     \
  }

// Two rows overlap reductions and arithmetic without the register cost of four.
GDN_DECODE_ENTRY(verify_gdn_fused, 16, 48, 128, 10240, 2)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32, 16, 32, 128, 8192, 2)
#undef GDN_DECODE_ENTRY

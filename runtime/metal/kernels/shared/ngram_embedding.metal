#include "metal/abi/KernelABI.h"

// Hashed n-gram lookup for the qwen4exp per-layer embedding.
//
// Each head hashes a window of the token history into its own prime-sized
// table and gathers one row. The heads partition the embedding width, so the
// sixteen gathered rows of 160 concatenate to the 2560 the layer consumes.
//
// For a head belonging to n-gram order n, over the token history shifted
// right by each position within the window:
//
//   mixed = shift[0] * multiplier[0]
//   mixed ^= shift[p] * multiplier[p]        for p in 1 .. n-1
//   row   = mixed mod vocabulary[head] + offset[head]
//
// The arithmetic is 64-bit and stays that way: the multipliers are around
// 1e13 and a token id reaches the low hundreds of thousands, so a product
// lands near 6e18 - inside a signed 64-bit range, but not inside a 32-bit
// one. The modulus follows the divisor's sign, as the reference's remainder
// does, so a wrapped product cannot produce a negative row.
//
// Shifted history arrives already resolved by the caller: the reference
// restarts the window at an end-of-sequence token, which is a sequence
// property rather than a per-row one.

// NgramEmbeddingParams is in metal/abi/PerLayerEmbedding.h.

// One threadgroup per row; one simdgroup per head while heads remain.
kernel void ngram_embedding_gather(
    device const uint *shifted_tokens [[buffer(0)]],
    device const long *multipliers [[buffer(1)]],
    device const long *head_vocabulary [[buffer(2)]],
    device const long *head_offsets [[buffer(3)]],
    device const uchar *table [[buffer(4)]],
    device const bfloat *scales [[buffer(5)]],
    device const bfloat *biases [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant NgramEmbeddingParams &params [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const uint dimension = params.head_dimension;
  const uint groups = params.group_elements;

  for (uint head = 0; head < params.heads; ++head) {
    // Heads are laid out in blocks, one block per n-gram order, and an order
    // mixes one more shifted copy of the history than the order below it.
    const uint order = head / params.heads_per_order + 2;
    long mixed = long(shifted_tokens[0 * params.rows + row]) * multipliers[0];
    for (uint position = 1; position < order; ++position) {
      mixed ^= long(shifted_tokens[position * params.rows + row]) *
               multipliers[position];
    }
    const long vocabulary = head_vocabulary[head];
    long remainder = mixed % vocabulary;
    // Metal's % truncates toward zero; the reference's remainder follows the
    // divisor, so a negative mix has to be brought back up.
    if (remainder < 0) remainder += vocabulary;
    const ulong entry = ulong(remainder + head_offsets[head]);

    // One row of the table, half a byte per weight, with a scale and bias per
    // group. The rows are a whole number of groups wide, so a row's first
    // element starts a group and the parameters index directly.
    const ulong weight_base = entry * ulong(dimension) / 2;
    const ulong parameter_base = entry * ulong(dimension / groups);
    device bfloat *destination = output + ulong(row) * params.heads * dimension +
                                 ulong(head) * dimension;
    for (uint i = thread_index; i < dimension; i += threads) {
      const uchar packed = table[weight_base + i / 2];
      const float quantized = float((packed >> ((i & 1) * 4)) & 0xF);
      const ulong parameter = parameter_base + i / groups;
      destination[i] = bfloat(quantized * float(scales[parameter]) +
                              float(biases[parameter]));
    }
  }
}

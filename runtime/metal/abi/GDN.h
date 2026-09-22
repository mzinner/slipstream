#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

// sigmoid_gate selects the output gate: 0 is silu (Qwen3.8), 1 is sigmoid
// (qwen4exp, whose config sets output_gate_type). The two models share every
// GDN dimension, so the activation cannot be told apart by the compiled shape.
struct GDNPreparePrefillParams {
  uint32_t tokens;
  uint32_t packed_width;
  uint32_t sigmoid_gate;
};

static_assert(sizeof(GDNPreparePrefillParams) == 12,
              "GDN prefill prepare parameters are 12 bytes on both sides");

struct GDNDecodeBatchParams {
  uint32_t sigmoid_gate; // As in GDNPreparePrefillParams; also pads the strides.
  uint32_t packed_width;
  uint32_t lanes;
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNDecodeBatchParams) == 40,
              "GDN decode parameters are 40 bytes on both sides");

// Explicit padding aligns the uint64_t state strides and keeps transmitted
// bytes initialized.
struct GDNBatchCommitParams {
  uint32_t groups;
  uint32_t packed_width;
  uint32_t lanes;
  uint32_t packed_stride;
  uint32_t mixed_stride;
  uint32_t decay_stride;
  uint32_t beta_stride;
  uint32_t reserved0;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNBatchCommitParams) == 56,
              "GDN commit parameters are 56 bytes on both sides");

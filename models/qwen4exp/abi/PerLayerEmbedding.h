#pragma once

// Parameter layouts for the qwen4exp per-layer embedding, shared by host
// dispatch code and the Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct NgramEmbeddingParams {
  uint32_t rows;
  uint32_t heads;
  uint32_t head_dimension;
  uint32_t ngram_size;
  uint32_t heads_per_order;
  uint32_t group_elements;
};

static_assert(sizeof(NgramEmbeddingParams) == 24,
              "NgramEmbeddingParams must be 24 bytes on both sides");

struct PerLayerEmbeddingParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t count;
  uint32_t taps;
  uint32_t dilation;
  float epsilon;
};

static_assert(sizeof(PerLayerEmbeddingParams) == 24,
              "PerLayerEmbeddingParams must be 24 bytes on both sides");

// The step-to-step history, and how many of this step's rows it keeps.
//
// rows       rows this step processed for the sequence
// width      hyper-connection width of one normalized row
// history    normalized rows the convolution reaches back over
// eos        the reference's end-of-sequence token, which both fills a fresh
//            history and restarts the n-gram window
// retained   rows to commit; when retained_from_buffer is set it is read
//            per lane from the verifier's retained-count buffer instead
struct PerLayerEmbeddingStateParams {
  uint32_t rows;
  uint32_t width;
  uint32_t history;
  uint32_t eos;
  uint32_t retained;
  uint32_t retained_from_buffer;
  uint32_t lane;
  uint32_t reserved;
};

static_assert(sizeof(PerLayerEmbeddingStateParams) == 32,
              "PerLayerEmbeddingStateParams must be 32 bytes on both sides");

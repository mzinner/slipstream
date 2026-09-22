#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct HyperConnectionParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t count;
  uint32_t low_rank;
  float epsilon;
  // hyper_connection_down also computes the injection gates. Zero for the
  // final mixer, which has none.
  uint32_t with_inject;
  // Logical row r is physical row r * row_step (0 means 1). Decode with one
  // live row per 8-row lane processes only rows 0, 8, 16, ...
  uint32_t row_step;
  // hyper_connection_down splits each output's dot product across this many
  // threadgroups (0 or 1: no split) and leaves float partial sums for
  // hyper_connection_down_finish. Decode's few rows need the parallelism.
  uint32_t splits;
};

static_assert(sizeof(HyperConnectionParams) == 32,
              "HyperConnectionParams must be 32 bytes on both sides");

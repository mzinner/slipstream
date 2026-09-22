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
  uint32_t reserved;
};

static_assert(sizeof(HyperConnectionParams) == 32,
              "HyperConnectionParams must be 32 bytes on both sides");

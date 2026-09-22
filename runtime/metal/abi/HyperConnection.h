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
};

static_assert(sizeof(HyperConnectionParams) == 24,
              "HyperConnectionParams must be 24 bytes on both sides");

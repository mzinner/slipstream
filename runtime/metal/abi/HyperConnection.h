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
};

static_assert(sizeof(HyperConnectionParams) == 20,
              "HyperConnectionParams must be 20 bytes on both sides");

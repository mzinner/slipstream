#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct LastHiddenRowsParams {
  uint32_t rows;
  uint32_t width;
};

static_assert(sizeof(LastHiddenRowsParams) == 8,
              "Last hidden row parameters are 8 bytes on both sides");

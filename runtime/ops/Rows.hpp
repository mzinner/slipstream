#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

class Rows final {
public:
  // Copies the last min(rows, 8) rows of `width` bf16 values from source to
  // the start of destination.
  static void gatherLast(metal::CommandGraph &graph, metal::MetalBuffer source,
                         metal::MetalBuffer destination, uint32_t rows,
                         uint32_t width);
};

} // namespace splash::ops

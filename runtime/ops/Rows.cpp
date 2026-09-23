#include "ops/Rows.hpp"

#include "metal/abi/Rows.h"

#include <stdexcept>
#include <utility>

namespace splash::ops {

void Rows::gatherLast(metal::CommandGraph &graph, metal::MetalBuffer source,
                      metal::MetalBuffer destination, uint32_t rows,
                      uint32_t width) {
  if (!rows || !width)
    throw std::invalid_argument("invalid last-row gather");
  graph.add("prefill_gather_last_hidden_rows8",
            {std::move(source), std::move(destination)},
            LastHiddenRowsParams{rows, width}, {(width + 127) / 128, 1, 1});
}

} // namespace splash::ops

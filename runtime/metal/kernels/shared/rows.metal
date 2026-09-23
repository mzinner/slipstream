#include "metal/abi/KernelABI.h"

// Copies the last min(rows, 8) rows of a prompt chunk into the decode arena:
// the draft head starts from the prompt's final hidden rows.
kernel void prefill_gather_last_hidden_rows8(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant LastHiddenRowsParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  uint kept = min(params.rows, 8u);
  uint start = params.rows - kept;
  for (uint element = index; element < kept * params.width;
       element += grid_size) {
    uint row = element / params.width;
    uint dim = element % params.width;
    output[element] = input[(start + row) * params.width + dim];
  }
}

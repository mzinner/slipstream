#include <metal_stdlib>
using namespace metal;

// Measurement only (SPLASH_CAPTURE_LAYERS): copies a prompt chunk's residual
// after a chosen layer, so guesser training can read the model's inner state.
kernel void capture_rows(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint index [[thread_position_in_grid]],
                         uint grid_size [[threads_per_grid]]) {
  for (uint element = index; element < count; element += grid_size)
    output[element] = input[element];
}

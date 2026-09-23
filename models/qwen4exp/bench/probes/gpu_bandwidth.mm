// Peak GPU read bandwidth: every thread streams float4s and folds them into
// one value, so the only cost is reading memory. This is the ceiling any
// decode kernel can approach, not a target it will hit.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

static const char *kSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void read_all(device const float4 *in [[buffer(0)]], device float *out [[buffer(1)]],
                     constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]],
                     uint grid [[threads_per_grid]]) {
  float4 acc = 0;
  for (uint i = id; i < count; i += grid) acc += in[i];
  if (acc.x == 1234.5f) out[0] = acc.y;   // keeps the loads alive
})";

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kSource) options:nil error:&error];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"read_all"] error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const uint64_t bytes = 4ull << 30;
    id<MTLBuffer> input = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> output = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    uint32_t count = uint32_t(bytes / 16);
    double best = 0;
    for (int run = 0; run < 8; ++run) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
      [encoder setBytes:&count length:4 atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(1 << 20, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      double seconds = command.GPUEndTime - command.GPUStartTime;
      double gbps = bytes / seconds / 1e9;
      if (run > 0 && gbps > best) best = gbps;
    }
    printf("GPU read bandwidth: %.0f GB/s (best of 7, 4 GiB buffer)\n", best);
  }
}

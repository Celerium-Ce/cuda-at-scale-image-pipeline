#ifndef CUDA_AT_SCALE_INCLUDE_SOBEL_CUH_
#define CUDA_AT_SCALE_INCLUDE_SOBEL_CUH_

#include <cuda_runtime.h>

// Launches the Sobel gradient-magnitude kernel on `stream`. `d_in` and
// `d_out` are tightly packed (row stride == width) device buffers of
// `width * height` 8-bit grayscale pixels; `d_out` may equal `d_in`'s
// companion buffer but must not alias `d_in` itself. Border pixels use
// clamp-to-edge (replicate) addressing, matching the NPP blur stage that
// typically precedes this kernel.
void LaunchSobelKernel(const unsigned char* d_in, unsigned char* d_out,
                        int width, int height, cudaStream_t stream);

#endif  // CUDA_AT_SCALE_INCLUDE_SOBEL_CUH_

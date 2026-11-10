#include "sobel.cuh"

#include <cmath>

#include "cuda_utils.h"

namespace {

// Thread block is kBlockDim x kBlockDim; the shared-memory tile adds a
// one-pixel halo on every side to supply the 3x3 Sobel stencil's neighbors
// without re-reading global memory per pixel.
constexpr int kBlockDim = 16;

__device__ __forceinline__ int ClampInt(int value, int lo, int hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

__global__ void SobelKernel(const unsigned char* __restrict__ in,
                             unsigned char* __restrict__ out, int width,
                             int height) {
  __shared__ unsigned char tile[kBlockDim + 2][kBlockDim + 2];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x = blockIdx.x * kBlockDim + tx;
  const int y = blockIdx.y * kBlockDim + ty;

  // Cooperatively load the tile plus its halo; out-of-bounds taps clamp to
  // the nearest edge pixel (replicate-border), matching the NPP blur stage
  // that normally feeds this kernel.
  for (int dy = ty; dy < kBlockDim + 2; dy += kBlockDim) {
    const int src_y = ClampInt(blockIdx.y * kBlockDim + dy - 1, 0, height - 1);
    for (int dx = tx; dx < kBlockDim + 2; dx += kBlockDim) {
      const int src_x = ClampInt(blockIdx.x * kBlockDim + dx - 1, 0, width - 1);
      tile[dy][dx] = in[src_y * width + src_x];
    }
  }
  __syncthreads();

  if (x >= width || y >= height) {
    return;
  }

  const int cx = tx + 1;
  const int cy = ty + 1;

  const int gx = -tile[cy - 1][cx - 1] + tile[cy - 1][cx + 1] -
                 2 * tile[cy][cx - 1] + 2 * tile[cy][cx + 1] -
                 tile[cy + 1][cx - 1] + tile[cy + 1][cx + 1];

  const int gy = -tile[cy - 1][cx - 1] - 2 * tile[cy - 1][cx] -
                 tile[cy - 1][cx + 1] + tile[cy + 1][cx - 1] +
                 2 * tile[cy + 1][cx] + tile[cy + 1][cx + 1];

  const int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
  out[y * width + x] = static_cast<unsigned char>(ClampInt(magnitude, 0, 255));
}

}  // namespace

void LaunchSobelKernel(const unsigned char* d_in, unsigned char* d_out,
                        int width, int height, cudaStream_t stream) {
  const dim3 block(kBlockDim, kBlockDim);
  const dim3 grid((width + kBlockDim - 1) / kBlockDim,
                   (height + kBlockDim - 1) / kBlockDim);
  SobelKernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height);
  CUDA_CHECK(cudaGetLastError());
}

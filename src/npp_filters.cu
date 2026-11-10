// Copyright 2026 Celerium-Ce
#include "include/npp_filters.cuh"

#include "include/cuda_utils.h"

void ApplyGaussianBlurNpp(const Npp8u* d_in, int in_step, Npp8u* d_out,
                           int out_step, int width, int height,
                           NppStreamContext ctx) {
  const NppiSize size{width, height};
  const NppiPoint offset{0, 0};

  NPP_CHECK(nppiFilterGaussBorder_8u_C1R_Ctx(
      d_in, in_step, size, offset, d_out, out_step, size,
      NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, ctx));
}

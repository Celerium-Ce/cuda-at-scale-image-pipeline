// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_NPP_FILTERS_CUH_
#define INCLUDE_NPP_FILTERS_CUH_

#include <npp.h>

// Applies a 5x5 Gaussian blur to an 8-bit single-channel image using NPP's
// nppiFilterGaussBorder, with replicate-edge border handling so the output
// has the same dimensions as the input. `in_step`/`out_step` are the row
// strides, in bytes, of `d_in`/`d_out` (equal to `width` for tightly packed
// buffers). The call is associated with `ctx.hStream`, so it may run
// concurrently with other streams' work.
void ApplyGaussianBlurNpp(const Npp8u* d_in, int in_step, Npp8u* d_out,
                           int out_step, int width, int height,
                           NppStreamContext ctx);

#endif  // INCLUDE_NPP_FILTERS_CUH_

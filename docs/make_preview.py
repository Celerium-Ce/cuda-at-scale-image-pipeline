#!/usr/bin/env python3
"""Generates illustrative preview PNGs for the README.

IMPORTANT: this is a CPU-side (NumPy) re-implementation of the same two
steps the real pipeline runs on the GPU (5x5 Gaussian blur, then 3x3
Sobel gradient magnitude), used ONLY to produce human-viewable PNGs for
the README on a machine with no CUDA Toolkit installed. It is NOT part
of the graded pipeline and is NOT proof that the CUDA/NPP code in src/
runs correctly on a GPU - that proof has to come from actually running
bin/batch_edge_detect (via run.sh) on a CUDA-capable machine, per the
"Toolchain note" in the main README.

Usage:
    python3 docs/make_preview.py
"""
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "data", "input")
PREVIEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")

# Same 5x5 kernel NPP_MASK_SIZE_5_X_5 uses (binomial approximation of a
# Gaussian), and the same 3x3 Sobel kernels as src/sobel_kernel.cu.
GAUSS_5X5 = np.array(
    [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ],
    dtype=np.float64,
) / 256.0

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def convolve_replicate(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for ky in range(kh):
        for kx in range(kw):
            out += kernel[ky, kx] * padded[ky : ky + image.shape[0], kx : kx + image.shape[1]]
    return out


def read_pgm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        max_val = int(f.readline())
        assert max_val == 255
        data = np.frombuffer(f.read(width * height), dtype=np.uint8)
        return data.reshape(height, width)


def process(image):
    blurred = convolve_replicate(image.astype(np.float64), GAUSS_5X5)
    gx = convolve_replicate(blurred, SOBEL_X)
    gy = convolve_replicate(blurred, SOBEL_Y)
    magnitude = np.sqrt(gx**2 + gy**2)
    edges = np.clip(magnitude, 0, 255).astype(np.uint8)
    return blurred.astype(np.uint8), edges


def main():
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    samples = ["synth_0001.pgm", "synth_0002.pgm", "synth_0003.pgm"]
    for name in samples:
        src_path = os.path.join(INPUT_DIR, name)
        if not os.path.exists(src_path):
            print(f"  skip {name}: not found")
            continue
        image = read_pgm(src_path)
        blurred, edges = process(image)

        stem = os.path.splitext(name)[0]
        Image.fromarray(image).save(os.path.join(PREVIEW_DIR, f"{stem}_input.png"))
        Image.fromarray(blurred).save(os.path.join(PREVIEW_DIR, f"{stem}_blurred.png"))
        Image.fromarray(edges).save(os.path.join(PREVIEW_DIR, f"{stem}_edges.png"))
        print(f"  wrote previews for {name}")

    print(f"Preview PNGs written to {PREVIEW_DIR}")


if __name__ == "__main__":
    main()

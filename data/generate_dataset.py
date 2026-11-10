#!/usr/bin/env python3
"""Synthesizes a batch of grayscale PGM images for the CUDA Gaussian-blur
+ Sobel edge-detection pipeline.

No external dataset or network access is required: this script generates
a deterministic set of 256x256 images (gradients, concentric rings,
checkerboards, and sinusoidal interference patterns, each with a little
added noise) so the project is reproducible anywhere a Python 3
interpreter is available, independent of the CUDA build itself.

Usage:
    python3 generate_dataset.py [num_images]
"""
import math
import os
import random
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
WIDTH = 256
HEIGHT = 256
NUM_IMAGES = 200


def write_pgm(path, pixels, width, height):
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(pixels))


def pattern_gradient(x, y, w, h, seed):
    angle = seed * 0.37
    nx, ny = x / w, y / h
    return (nx * math.cos(angle) + ny * math.sin(angle)) * 255


def pattern_rings(x, y, w, h, seed):
    cx = w / 2 + math.sin(seed) * w * 0.2
    cy = h / 2 + math.cos(seed * 1.3) * h * 0.2
    dist = math.hypot(x - cx, y - cy)
    freq = 0.15 + (seed % 7) * 0.03
    return 127 + 127 * math.sin(dist * freq)


def pattern_checker(x, y, w, h, seed):
    cell = 8 + (seed % 24)
    return 220 if ((x // cell) + (y // cell)) % 2 == 0 else 30


def pattern_interference(x, y, w, h, seed):
    f1 = 0.05 + (seed % 5) * 0.01
    f2 = 0.07 + (seed % 3) * 0.015
    v = math.sin(x * f1 + seed) + math.sin(y * f2 - seed * 0.5)
    return 127 + 100 * (v / 2)


PATTERNS = [pattern_gradient, pattern_rings, pattern_checker, pattern_interference]


def clamp8(value):
    return 0 if value < 0 else (255 if value > 255 else int(value))


def generate_image(width, height, seed):
    rng = random.Random(seed)
    pattern = PATTERNS[seed % len(PATTERNS)]
    noise_amplitude = rng.uniform(2.0, 12.0)
    pixels = bytearray(width * height)
    for y in range(height):
        row_base = y * width
        for x in range(width):
            base = pattern(x, y, width, height, seed)
            noise = rng.uniform(-noise_amplitude, noise_amplitude)
            pixels[row_base + x] = clamp8(base + noise)
    return pixels


def main():
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_IMAGES
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(num_images):
        pixels = generate_image(WIDTH, HEIGHT, seed=i)
        path = os.path.join(OUT_DIR, f"synth_{i:04d}.pgm")
        write_pgm(path, pixels, WIDTH, HEIGHT)
        if (i + 1) % 25 == 0 or i + 1 == num_images:
            print(f"  generated {i + 1}/{num_images} images")
    print(f"Wrote {num_images} PGM images ({WIDTH}x{HEIGHT}) to {OUT_DIR}")


if __name__ == "__main__":
    main()

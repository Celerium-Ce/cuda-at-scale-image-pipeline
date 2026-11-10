#ifndef CUDA_AT_SCALE_INCLUDE_PGM_IO_H_
#define CUDA_AT_SCALE_INCLUDE_PGM_IO_H_

#include <string>
#include <vector>

// In-memory representation of a binary (P5) grayscale PGM image.
struct Image {
  int width = 0;
  int height = 0;
  int max_val = 255;
  std::vector<unsigned char> data;
};

// Reads a binary PGM (P5) file from `path` into `*image`. Returns false and
// logs a message to stderr on any I/O or format error.
bool ReadPgm(const std::string& path, Image* image);

// Writes `image` to `path` as a binary PGM (P5) file. Returns false and
// logs a message to stderr if the file cannot be written.
bool WritePgm(const std::string& path, const Image& image);

#endif  // CUDA_AT_SCALE_INCLUDE_PGM_IO_H_

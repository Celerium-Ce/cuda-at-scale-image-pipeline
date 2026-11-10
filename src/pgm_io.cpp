#include "pgm_io.h"

#include <cctype>

#include <fstream>
#include <iostream>
#include <string>

namespace {

// Skips PGM whitespace and '#' comment lines, leaving the stream positioned
// at the next non-whitespace, non-comment character.
void SkipWhitespaceAndComments(std::istream* in) {
  int c;
  while ((c = in->peek()) != EOF) {
    if (std::isspace(c)) {
      in->get();
    } else if (c == '#') {
      std::string line;
      std::getline(*in, line);
    } else {
      break;
    }
  }
}

int ReadHeaderToken(std::istream* in) {
  SkipWhitespaceAndComments(in);
  int value = 0;
  *in >> value;
  return value;
}

}  // namespace

bool ReadPgm(const std::string& path, Image* image) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file: " << path << "\n";
    return false;
  }

  std::string magic;
  in >> magic;
  if (magic != "P5") {
    std::cerr << "Unsupported PGM format (expected P5): " << path << "\n";
    return false;
  }

  image->width = ReadHeaderToken(&in);
  image->height = ReadHeaderToken(&in);
  image->max_val = ReadHeaderToken(&in);
  in.get();  // Consume the single whitespace byte that follows max_val.

  if (image->width <= 0 || image->height <= 0 || image->max_val <= 0 ||
      image->max_val > 255) {
    std::cerr << "Invalid PGM header in: " << path << "\n";
    return false;
  }

  const size_t num_pixels =
      static_cast<size_t>(image->width) * static_cast<size_t>(image->height);
  image->data.resize(num_pixels);
  in.read(reinterpret_cast<char*>(image->data.data()), num_pixels);
  if (!in) {
    std::cerr << "Truncated PGM pixel data: " << path << "\n";
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, const Image& image) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "Cannot open file for writing: " << path << "\n";
    return false;
  }
  out << "P5\n" << image.width << " " << image.height << "\n"
      << image.max_val << "\n";
  out.write(reinterpret_cast<const char*>(image.data.data()),
            image.data.size());
  return out.good();
}

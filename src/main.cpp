// Copyright 2026 Celerium-Ce
//
// CUDA at Scale Independent Project.
//
// Batch image-processing pipeline: for every grayscale PGM image in an
// input directory, runs a 5x5 Gaussian blur (NVIDIA NPP) followed by a
// hand-written tiled CUDA Sobel gradient-magnitude kernel, writing the
// edge-detected result to an output directory. Images are distributed
// round-robin across several CUDA streams with pinned host buffers so
// transfers and compute for different images can overlap.
//
// Directory handling is done with POSIX dirent/sys-stat calls rather than
// std::filesystem, so the build has no dependency on a GCC version new
// enough to ship <filesystem> without a separate -lstdc++fs link (some
// CUDA base images still pair an older host compiler with a recent nvcc).

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/cuda_utils.h"
#include "include/npp_filters.cuh"
#include "include/pgm_io.h"
#include "include/sobel.cuh"

namespace {

struct Args {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  std::string log_file = "logs/run.log";
  int num_streams = 4;
  int64_t limit = -1;  // -1 means "process every image found".
};

void PrintUsage(const char* prog_name) {
  std::cout
      << "Usage: " << prog_name << " [options]\n"
      << "  --input DIR    Directory of input .pgm images (default: data/input)\n"
      << "  --output DIR   Directory to write processed .pgm images (default: data/output)\n"
      << "  --log FILE     Path to write the run log (default: logs/run.log)\n"
      << "  --streams N    Number of concurrent CUDA streams (default: 4)\n"
      << "  --limit N      Process at most N images (default: all)\n"
      << "  -h, --help     Show this message\n";
}

// Parses argv into `*args`. Returns false (after printing a message) on any
// unrecognized or malformed argument.
bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto NextValue = [&](const char* flag_name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag_name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      args->input_dir = NextValue("--input");
    } else if (arg == "--output") {
      args->output_dir = NextValue("--output");
    } else if (arg == "--log") {
      args->log_file = NextValue("--log");
    } else if (arg == "--streams") {
      args->num_streams = std::stoi(NextValue("--streams"));
    } else if (arg == "--limit") {
      args->limit = static_cast<int64_t>(std::stoll(NextValue("--limit")));
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (args->num_streams < 1) {
    std::cerr << "--streams must be >= 1\n";
    return false;
  }
  return true;
}

bool DirExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Creates `path` and any missing parent directories (like `mkdir -p`).
// Silently succeeds if a component already exists.
void MakeDirsRecursive(const std::string& path) {
  std::string partial;
  size_t pos = 0;
  while (pos < path.size()) {
    const size_t next = path.find('/', pos);
    const size_t end = (next == std::string::npos) ? path.size() : next;
    partial = path.substr(0, end);
    if (!partial.empty() && !DirExists(partial)) {
      mkdir(partial.c_str(), 0755);
    }
    if (next == std::string::npos) break;
    pos = next + 1;
  }
}

std::string ParentDir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? std::string() : path.substr(0, slash);
}

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

bool HasSuffix(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> CollectInputFiles(const std::string& input_dir,
                                            int64_t limit) {
  std::vector<std::string> files;
  DIR* dir = opendir(input_dir.c_str());
  if (dir == nullptr) {
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    const std::string name = entry->d_name;
    if (HasSuffix(name, ".pgm")) {
      files.push_back(input_dir + "/" + name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  if (limit >= 0 && static_cast<int64_t>(files.size()) > limit) {
    files.resize(limit);
  }
  return files;
}

// Processes a single image end-to-end on `stream`/`npp_ctx`: pinned H2D
// copy, NPP Gaussian blur, custom Sobel kernel, pinned D2H copy, then
// writes the result PGM. Returns the elapsed GPU time in milliseconds, or
// a negative value if the image could not be read.
double ProcessImage(const std::string& input_path,
                     const std::string& output_path, cudaStream_t stream,
                     NppStreamContext npp_ctx) {
  Image image;
  if (!ReadPgm(input_path, &image)) {
    return -1.0;
  }

  const size_t num_bytes =
      static_cast<size_t>(image.width) * static_cast<size_t>(image.height);

  unsigned char* h_pinned_in = nullptr;
  unsigned char* h_pinned_out = nullptr;
  unsigned char* d_in = nullptr;
  unsigned char* d_blurred = nullptr;
  unsigned char* d_edges = nullptr;

  CUDA_CHECK(cudaMallocHost(&h_pinned_in, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_pinned_out, num_bytes));
  std::memcpy(h_pinned_in, image.data.data(), num_bytes);

  CUDA_CHECK(cudaMalloc(&d_in, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_blurred, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_edges, num_bytes));

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaEventRecord(start_event, stream));

  CUDA_CHECK(cudaMemcpyAsync(d_in, h_pinned_in, num_bytes,
                              cudaMemcpyHostToDevice, stream));
  ApplyGaussianBlurNpp(d_in, image.width, d_blurred, image.width, image.width,
                        image.height, npp_ctx);
  LaunchSobelKernel(d_blurred, d_edges, image.width, image.height, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_edges, num_bytes,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

  Image out_image;
  out_image.width = image.width;
  out_image.height = image.height;
  out_image.max_val = 255;
  out_image.data.assign(h_pinned_out, h_pinned_out + num_bytes);
  WritePgm(output_path, out_image);

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFreeHost(h_pinned_in);
  cudaFreeHost(h_pinned_out);
  cudaFree(d_in);
  cudaFree(d_blurred);
  cudaFree(d_edges);

  return static_cast<double>(elapsed_ms);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    return EXIT_FAILURE;
  }

  if (!DirExists(args.input_dir)) {
    std::cerr << "Input directory does not exist: " << args.input_dir << "\n";
    return EXIT_FAILURE;
  }
  MakeDirsRecursive(args.output_dir);
  const std::string log_parent = ParentDir(args.log_file);
  if (!log_parent.empty()) {
    MakeDirsRecursive(log_parent);
  }

  const std::vector<std::string> files =
      CollectInputFiles(args.input_dir, args.limit);
  if (files.empty()) {
    std::cerr << "No .pgm files found in " << args.input_dir << "\n";
    return EXIT_FAILURE;
  }

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA-capable device found.\n";
    return EXIT_FAILURE;
  }
  cudaDeviceProp device_prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

  std::ofstream log(args.log_file);
  auto LogLine = [&](const std::string& line) {
    std::cout << line << "\n";
    log << line << "\n";
  };

  LogLine("CUDA at Scale - Batch Gaussian Blur + Sobel Edge Detection");
  LogLine(std::string("Device: ") + device_prop.name);
  LogLine("Images to process: " + std::to_string(files.size()));
  LogLine("Streams: " + std::to_string(args.num_streams));
  LogLine("---");

  std::vector<cudaStream_t> streams(args.num_streams);
  std::vector<NppStreamContext> npp_ctx(args.num_streams);
  for (int i = 0; i < args.num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    NPP_CHECK(nppGetStreamContext(&npp_ctx[i]));
    npp_ctx[i].hStream = streams[i];
  }

  const auto batch_start = std::chrono::high_resolution_clock::now();
  int64_t ok_count = 0;
  double total_gpu_ms = 0.0;

  for (size_t idx = 0; idx < files.size(); ++idx) {
    const int stream_idx = static_cast<int>(idx % args.num_streams);
    const std::string& input_path = files[idx];
    const std::string output_path = args.output_dir + "/" + BaseName(input_path);

    const double elapsed_ms = ProcessImage(input_path, output_path,
                                            streams[stream_idx],
                                            npp_ctx[stream_idx]);
    if (elapsed_ms < 0.0) {
      LogLine("  [SKIP] Failed to read " + input_path);
      continue;
    }

    total_gpu_ms += elapsed_ms;
    ++ok_count;
    LogLine("  [OK] " + BaseName(input_path) + " -> " +
            std::to_string(elapsed_ms) + " ms [stream " +
            std::to_string(stream_idx) + "]");
  }

  for (int i = 0; i < args.num_streams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    cudaStreamDestroy(streams[i]);
  }

  const auto batch_end = std::chrono::high_resolution_clock::now();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(batch_end - batch_start)
          .count();

  LogLine("---");
  LogLine("Processed " + std::to_string(ok_count) + "/" +
           std::to_string(files.size()) + " images successfully.");
  LogLine("Total wall-clock time: " + std::to_string(wall_ms) + " ms");
  LogLine("Sum of per-image GPU times: " + std::to_string(total_gpu_ms) + " ms");
  if (wall_ms > 0.0) {
    LogLine("Throughput: " +
             std::to_string(ok_count / (wall_ms / 1000.0)) + " images/sec");
  }

  return (ok_count == static_cast<int64_t>(files.size())) ? EXIT_SUCCESS
                                                          : EXIT_FAILURE;
}

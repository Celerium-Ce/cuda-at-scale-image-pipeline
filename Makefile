# CUDA at Scale Independent Project
# Builds bin/batch_edge_detect: a batch Gaussian-blur (NPP) + Sobel
# edge-detection (custom CUDA kernel) pipeline.
#
# Requirements: CUDA Toolkit 10.1+ (nvcc on PATH) with the split NPP
# libraries (nppc, nppif, nppisu) and a CUDA-capable GPU.
#
# Usage:
#   make                 # build bin/batch_edge_detect
#   make dataset         # (re)generate the synthetic data/input images
#   make run             # build, then process data/input -> data/output
#   make clean           # remove build artifacts
#
# Override the target GPU architecture(s) if the default list below does
# not cover your device, e.g.:
#   make ARCHES="-gencode arch=compute_86,code=sm_86"

NVCC      ?= nvcc

TARGET    := bin/batch_edge_detect
SRC_DIR   := src
INC_DIR   := include
OBJ_DIR   := bin/obj

CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS)) \
             $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

NVCCFLAGS := -std=c++17 -O2 -I$(INC_DIR)
LDFLAGS   := -lnppc -lnppif -lnppisu -lcudart

# Covers Maxwell (sm_52) through Ampere (sm_86); adjust for newer/older GPUs.
ARCHES ?= -gencode arch=compute_52,code=sm_52 \
          -gencode arch=compute_60,code=sm_60 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86

.PHONY: all dirs dataset run clean

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) bin data/output logs

$(TARGET): $(OBJS)
	$(NVCC) $(ARCHES) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) $(ARCHES) -c $< -o $@

dataset:
	python3 data/generate_dataset.py

run: all
	./$(TARGET) --input data/input --output data/output --log logs/run.log

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

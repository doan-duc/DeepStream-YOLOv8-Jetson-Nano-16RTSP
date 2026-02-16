################################################################################
# Makefile — Custom YOLO Parser Plugin for DeepStream 6.0
################################################################################
# Target:  Jetson Nano (aarch64, CUDA 10.2)
# Output:  libnvdsinfer_custom_impl_Yolo.so
#
# This plugin provides custom bounding box parsing for YOLOv8 output format,
# enabling DeepStream to correctly interpret YOLOv8 detection results.
#
# Usage:
#   make          # Build the shared library
#   make clean    # Remove compiled files
#
# Requirements:
#   - DeepStream SDK 6.0 installed at /opt/nvidia/deepstream/deepstream-6.0
#   - CUDA 10.2 toolkit
#   - GCC with C++11 support
################################################################################

# CUDA version (10.2 on Jetson Nano with JetPack 4.6)
CUDA_VER := 10.2

# Optional features (set to 1 to enable)
OPENCV ?= 0
GRAPH ?= 0

# Compilers
CC := g++
NVCC := /usr/local/cuda-$(CUDA_VER)/bin/nvcc

# DeepStream installation path
DS_PATH := /opt/nvidia/deepstream/deepstream-6.0

# C++ compiler flags
CFLAGS := -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I$(DS_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

# Optional: OpenCV support (for INT8 calibration)
ifeq ($(OPENCV), 1)
	COMMON += -DOPENCV
	CFLAGS += $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
	LIBS += $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

ifeq ($(GRAPH), 1)
	COMMON += -DGRAPH
endif

# CUDA compiler flags
CUFLAGS := -I$(DS_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

# Check for optional parser library
ifeq ($(shell ldconfig -p | grep -q libnvparsers && echo 1 || echo 0), 1)
	LIBS += -lnvparsers
endif

# Linker libraries
LIBS += -lnvinfer_plugin -lnvinfer -lnvonnxparser \
        -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS := -shared -Wl,--start-group $(LIBS) -Wl,--end-group

# Source files
INCS := $(wildcard layers/*.h) $(wildcard *.h)

SRCFILES := $(filter-out calibrator.cpp, $(wildcard *.cpp))
ifeq ($(OPENCV), 1)
	SRCFILES += calibrator.cpp
endif
SRCFILES += $(wildcard layers/*.cpp)
SRCFILES += $(wildcard *.cu)

# Output
TARGET_LIB := libnvdsinfer_custom_impl_Yolo.so

TARGET_OBJS := $(SRCFILES:.cpp=.o)
TARGET_OBJS := $(TARGET_OBJS:.cu=.o)

# Build rules
all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS) $<

$(TARGET_LIB): $(TARGET_OBJS)
	$(CC) -o $@ $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)

.PHONY: all clean

# ==============================================================================
# Dockerfile for DeepStream YOLOv8 on Jetson Nano
# ==============================================================================
# Base: NVIDIA DeepStream 6.0 for Jetson (L4T)
# Includes: TensorRT 8.0, CUDA 10.2, GStreamer, Python 3.6
# ==============================================================================

FROM nvcr.io/nvidia/deepstream-l4t:6.0-samples

# Set working directory
WORKDIR /opt/nvidia/deepstream/deepstream-6.0/sources/yolo_work

# Install additional dependencies (if needed)
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Build custom YOLO parser plugin
WORKDIR /opt/nvidia/deepstream/deepstream-6.0/sources/yolo_work/nvdsinfer_custom_impl_Yolo
RUN make clean && make

# Set back to main working directory
WORKDIR /opt/nvidia/deepstream/deepstream-6.0/sources/yolo_work

# Default command
CMD ["/bin/bash"]

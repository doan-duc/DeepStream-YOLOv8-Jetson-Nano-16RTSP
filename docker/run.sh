#!/bin/bash
set -e

# --- Configuration ---
DOCKER_IMAGE="ds-yolo"
CONTAINER_WORKDIR="/opt/nvidia/deepstream/deepstream-6.0/sources/yolo_work"

# Project directory (auto-detect relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo " DeepStream YOLOv8 — Jetson Nano Launcher"
echo "=============================================="
echo ""
echo "  Project Dir:  $PROJECT_DIR"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Mounted at:   $CONTAINER_WORKDIR"
echo ""

# --- Step 1: Grant X11 display access ---
echo "[1/2] Granting X11 display access for Docker..."
sudo xhost +local:root 2>/dev/null || true

# --- Step 2: Launch Docker container ---
echo "[2/2] Starting Docker container..."
echo ""

sudo docker run -it --rm \
    --net=host \
    --runtime nvidia \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PROJECT_DIR":"$CONTAINER_WORKDIR" \
    -w "$CONTAINER_WORKDIR" \
    "$DOCKER_IMAGE" \
    /bin/bash -c "chmod -R 777 $CONTAINER_WORKDIR && echo '' && echo '  Container ready. Run:' && echo '    deepstream-app -c configs/deepstream_app_config.txt' && echo '' && exec /bin/bash"

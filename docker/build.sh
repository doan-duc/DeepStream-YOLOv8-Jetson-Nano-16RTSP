#!/bin/bash
# ==============================================================================
# Build Docker Image for DeepStream YOLOv8 on Jetson Nano
# ==============================================================================
# This script builds the Docker image from the Dockerfile
#
# Usage:
#   bash docker/build.sh
#
# Prerequisites:
#   - Docker installed on Jetson Nano (included with JetPack)
#   - Internet connection to pull base image
# ==============================================================================

set -e

# --- Configuration ---
DOCKER_IMAGE="ds-yolo"

# Project directory (auto-detect relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo " Building DeepStream YOLOv8 Docker Image"
echo "=============================================="
echo ""
echo "  Project Dir:  $PROJECT_DIR"
echo "  Image Name:   $DOCKER_IMAGE"
echo ""

# --- Build Docker image ---
echo "[1/2] Building Docker image (this may take 10-20 minutes)..."
cd "$PROJECT_DIR"
sudo docker build -t "$DOCKER_IMAGE" -f docker/Dockerfile .

echo ""
echo "[2/2] Verifying image..."
sudo docker images | grep "$DOCKER_IMAGE"

echo ""
echo "=============================================="
echo " ✓ Build complete!"
echo "=============================================="
echo ""
echo "To run the container:"
echo "  bash docker/run.sh"
echo ""
echo "To save the image for offline use:"
echo "  sudo docker save $DOCKER_IMAGE -o docker/ds-yolo-package.tar"
echo ""

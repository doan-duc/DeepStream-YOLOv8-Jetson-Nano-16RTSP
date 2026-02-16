# DeepStream YOLOv8 — 16-Camera RTSP Object Detection on Jetson Nano

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-DeepStream%206.0-76B900?style=for-the-badge&logo=nvidia" alt="DeepStream"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Jetson-Nano-76B900?style=for-the-badge&logo=nvidia" alt="Jetson Nano"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

---

## Overview

A production-ready pipeline for **real-time object detection** of **C2 bottles** using a **custom-trained YOLOv8n** model, running on **NVIDIA Jetson Nano** via **DeepStream SDK 6.0**. The system processes **16 RTSP camera streams simultaneously** with hardware-accelerated inference using TensorRT FP16.

The entire runtime environment is containerized in Docker, ensuring consistent deployment across devices.

The DeepStream plugin and configuration files are based on [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) by Marcos Luciano, modified to support **YOLOv8 output format** and optimized for real-time bounding box rendering on 16 camera streams.

### Key Features

| Feature | Description |
|---|---|
| **16-stream support** | Process 16 RTSP camera feeds concurrently with batch inference |
| **TensorRT FP16** | Optimized inference engine for maximum performance on Jetson Nano |
| **IOU Tracker** | Built-in object tracking across frames |
| **OSD (On-Screen Display)** | Real-time bounding box overlay with confidence scores |
| **Tiled Display** | 4×4 grid view of all camera feeds in a single window |
| **Dockerized** | Fully containerized for reproducible deployment |
| **Custom YOLO Parser** | Modified DeepStream YOLO plugin compiled for Jetson Nano (aarch64) |

---

## Dataset

The model is trained to detect **C2 bottles** (C2 Freeze Icy Mint Lemon tea bottles).

<p align="center">
  <img src="data(C2).png" width="300" alt="C2 Bottle Sample"/>
</p>

| Property | Value |
|---|---|
| **Object class** | `C2` (C2 Freeze bottle) |
| **Number of classes** | 1 |
| **Label file** | `models/labels.txt` |

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────┐
│ RTSP Cameras │───>│  DeepStream  │───>│  TensorRT     │───>│   OSD   │
│    (16)      │    │  Streammux   │    │  YOLOv8n FP16 │    │ Display │
└──────────────┘    └──────────────┘    └───────────────┘    └─────────┘
                           │                    │
                    ┌──────┴──────┐      ┌──────┴──────┐
                    │ GStreamer   │      │ IOU Tracker │
                    │ Pipeline    │      │ (batch=16)  │
                    └─────────────┘      └─────────────┘
```

---

### Model Architecture and Training

This project uses a **custom YOLOv8n model** (Student) distilled from a larger **YOLOv8l** model (Teacher) using **Knowledge Distillation (KD)**.

**Why Knowledge Distillation?**
- **Teacher (YOLOv8l):** High accuracy but slow on Jetson Nano.
- **Student (YOLOv8n-p2):** Fast inference but lower accuracy.
- **Result:** The student model learns from the teacher's soft labels, achieving higher accuracy than standard YOLOv8n while maintaining the speed of the Nano architecture.

**Training Results:**
The `training_result/` folder contains metrics comparing the student and teacher models.

| Metric | Student (YOLOv8n-p2) | Teacher (YOLOv8l) |
|---|---|---|
| **mAP50** | High (see results.png) | Very High |
| **Speed** | ~15ms (Jetson Nano) | ~100ms+ (Too slow) |

### Custom Modifications

#### 1. **Reduced P5 Head Channels**
The P5 (large object) detection head has been reduced from 256 to **128 channels** for faster inference:
```yaml
# Standard YOLOv8n: C2f[256]
# Custom model:    C2f[128]  ← Reduced by 50%
```

#### 2. **Dropout Layer (10%)**
Added dropout layer after P5 head to reduce overfitting:
```yaml
- [-1, 1, nn.Dropout, [0.1]]  # 10% dropout rate
```

### Detection Heads

| Head | Resolution | Channels | Purpose |
|---|---|---|---|
| **P3** | 8×8 | 64 | Small object detection |
| **P4** | 16×16 | 128 | Medium object detection |
| **P5** | 32×32 | 128 | Large object detection (reduced) |

### Model Statistics (from Ultralytics summary)

| Metric | Value |
|---|---|
| **Layers** | 130 |
| **Parameters** | **1,099,523** (~1.1M) |
| **GFLOPs** | **3.6** |
| **ONNX Size** | **4.6 MB** |
| **PyTorch Size** | **2.4 MB** |

### Model Configuration File

The model architecture is defined in `models/yolov8n-p2.yaml`:

```yaml
# Parameters
nc: 1                    # Number of classes
depth_multiple: 0.3      # YOLOv8n depth
width_multiple: 0.2      # YOLOv8n width

# Backbone: Standard YOLOv8n
# Head: Custom with reduced P5 channels + dropout
```

---

## Project Structure

```
.
├── README.md                          # This file (English)
├── README_VI.md                       # Vietnamese version
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data(C2).png                       # Dataset sample image (C2 bottle)
│
├── training_result/                   # Training metrics & KD comparison
│   ├── results.png                    # Loss & mAP charts (Student)
│   ├── confusion_matrix.png           # Accuracy matrix (Student)
│   ├── val_batch0_pred.jpg            # Sample prediction visualization
│   └── teacher_result/                # Teacher (YOLOv8l) metrics
│
├── configs/                           # DeepStream configuration files
│   ├── deepstream_app_config.txt      # Main app config (16 cameras, batch=16)
│   └── config_infer_primary.txt       # Inference engine config (batch=16, FP16)
│
├── models/                            # Model files
│   ├── best.pt                        # ✅ YOLOv8n Student weights (KD trained)
│   ├── best.onnx                      # ✅ YOLOv8n ONNX model (batch=16, pre-exported)
│   ├── yolov8n-p2.yaml                # Custom model architecture definition
│   └── labels.txt                     # Class labels: "C2"
│
├── nvdsinfer_custom_impl_Yolo/        # Custom YOLO parser plugin (from DeepStream-Yolo)
│   ├── Makefile                       # Build script for recompilation
│   ├── libnvdsinfer_custom_impl_Yolo.so  # Pre-compiled shared library (aarch64)
│   ├── nvdsinfer_yolo_engine.cpp      # TensorRT engine builder
│   ├── yolo.cpp / yolo.h             # YOLO model parser
│   ├── yoloPlugins.cpp / yoloPlugins.h  # TensorRT custom plugins
│   └── layers/                        # Neural network layer implementations
│
├── docker/                            # Docker deployment files
    ├── Dockerfile                     # Docker image definition
    ├── build.sh                       # Build Docker image script
    └── run.sh                         # Run Docker container script
```

---

## Prerequisites

### Hardware
- **NVIDIA Jetson Nano** (4GB recommended)
- **16 IP Cameras** with RTSP support (H.264 or H.265)
- Ethernet connection to camera network
- Power supply: 5V 4A barrel jack (for stable performance under load)

### Software (on Jetson Nano)
- **JetPack 4.6** (L4T R32.6.1)
- **DeepStream SDK 6.0**
- **CUDA 10.2**
- **TensorRT 8.0+**
- **Docker** (pre-installed with JetPack)

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project_YOLOV8
```

### 2. Build the Docker Image

```bash
# Build image (takes 10-20 minutes first time)
bash docker/build.sh

# Verify image
sudo docker images | grep ds-yolo
```

The build script will:
- Pull NVIDIA DeepStream 6.0 base image (~4-5 GB download)
- Copy project files into the container
- Compile the custom YOLO parser plugin
- Create image named `ds-yolo`

### 3. Generate TensorRT Engine File

The ONNX model (`models/best.onnx`) is included in the repository. You must convert it to a TensorRT engine **on the target Jetson Nano**:

```bash
# Enter the Docker container
bash docker/run.sh

# Inside container: Generate TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/best.onnx \
    --saveEngine=models/best.engine \
    --fp16 \
    --workspace=2048
```

**Expected output:**
- File: `models/best.engine`
- Build time: ~5-10 minutes on Jetson Nano

> ⚠️ **Critical:** TensorRT engine files are **NOT portable** across different devices. They are compiled specifically for the GPU architecture, CUDA version, and TensorRT version. You **must** generate the `.engine` file on the target Jetson Nano device.

### 4. Configure Camera Sources

Edit the DeepStream app config to match your camera setup:

```bash
nano configs/deepstream_app_config.txt
```

Update the RTSP URIs for each `[source0]` through `[source15]` section:

```ini
[source0]
enable=1
type=4
uri=rtsp://admin:PASSWORD@192.168.1.101:554/ch1/main
num-sources=1
gpu-id=0

# ... repeat for source1 through source15
```

### 5. Run the Pipeline

```bash
# Enter Docker container (if not already inside)
bash docker/run.sh

# Inside container: Run DeepStream with 16 cameras
deepstream-app -c configs/deepstream_app_config.txt
```

You will see a 4×4 tiled display of all 16 camera feeds with real-time C2 bottle detection.

---

## Model Files

### Included Files

| File | Format | Size | Description |
|---|---|---|---|
| `best.pt` | PyTorch | 2.4 MB | ✅ **Pre-included** — Original trained weights (for custom batch export) |
| `best.onnx` | ONNX | 4.6 MB | ✅ **Pre-included** — Exported for batch=16 |
| `labels.txt` | Text | 3 B | ✅ **Pre-included** — Class labels: `C2` |
| `best.engine` | TensorRT | — | ⚠️ **Must generate on Jetson** — See Quick Start step 3 |

### Workflow Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  best.pt    │───>│  best.onnx   │───>│ best.engine   │───>│  DeepStream  │
│ (included)  │    │  (included)  │    │ (generate on  │    │  Inference   │
│  2.4 MB     │    │   4.6 MB     │    │   Jetson)     │    │              │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
     ↓                                          ↓
  Training          Export (done)          trtexec
  machine           batch=16               (5-10 min)
```

### Why Each File Exists

1. **`best.pt`** — Original PyTorch weights from training. Use this to re-export ONNX with a different batch size:
   ```bash
   python utils/export_yolov8.py -w models/best.pt --batch 8
   ```

2. **`best.onnx`** — Already exported with batch=16 for convenience. Portable across machines.

3. **`best.engine`** — **NOT portable.** Must be generated on the target Jetson Nano because TensorRT engines are compiled for specific GPU architecture (Maxwell), CUDA version (10.2), and TensorRT version (8.0).

---

## Configuration Details

### DeepStream App Config (`configs/deepstream_app_config.txt`)

| Section | Key Settings |
|---|---|
| `[tiled-display]` | 4×4 grid, 1280×720 output |
| `[streammux]` | batch-size=16, 640×640, live-source=1 |
| `[tracker]` | IOU tracker, 640×384 |
| `[primary-gie]` | interval=6 (process every 7th frame) |
| `[osd]` | border-width=5, text-size=20 |

### Inference Config (`configs/config_infer_primary.txt`)

| Property | Value |
|---|---|
| `model-engine-file` | `../models/best.engine` |
| `batch-size` | 16 |
| `network-mode` | 2 (FP16) |
| `num-detected-classes` | 1 |
| `parse-bbox-func-name` | `NvDsInferParseYolo` |
| `nms-iou-threshold` | 0.2 |
| `pre-cluster-threshold` | 0.2 |

---

## Docker Setup

### Building the Docker Image

```bash
bash docker/build.sh
```

The `Dockerfile` environment:
- **Base image:** `nvcr.io/nvidia/deepstream-l4t:6.0-samples`
- **Platform:** ARM64 (Jetson Nano)
- **Installs:** Python3, pip, project dependencies
- **Compiles:** Custom YOLO parser plugin for aarch64

### Running the Container

```bash
bash docker/run.sh
```

The run script mounts:
- Project directory → `/opt/nvidia/deepstream/deepstream-6.0/sources/yolo_work`
- X11 display for GUI output
- NVIDIA runtime for GPU access

### Saving / Loading Docker Image

```bash
# Save image to tar file
sudo docker save ds-yolo -o docker/ds-yolo-package.tar

# Load image from tar file
sudo docker load -i docker/ds-yolo-package.tar
```

---

## Recompiling the Custom YOLO Plugin

The `nvdsinfer_custom_impl_Yolo/` directory contains the custom YOLO parser plugin, originally from [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) and modified to support YOLOv8 output format on Jetson Nano.

The library is pre-compiled for Jetson Nano (aarch64, CUDA 10.2, DeepStream 6.0). If you need to recompile:

```bash
cd nvdsinfer_custom_impl_Yolo
make clean
make
```

This produces `libnvdsinfer_custom_impl_Yolo.so`, which handles:
- Custom bounding box parsing for YOLOv8 output format
- CUDA-accelerated post-processing
- TensorRT engine creation from ONNX

---

## Performance

| Configuration | Cameras | Batch Size | Precision | Approx. FPS |
|---|---|---|---|---|
| `deepstream_app_config.txt` | 16 | 16 | FP16 | ~10-15 per stream |

> Performance varies based on:
> - Network conditions and RTSP stream quality
> - Camera resolution (1080p vs 720p)
> - Jetson Nano thermal state (ensure adequate cooling)
> - Number of detected objects per frame

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `Permission denied` when running Docker | Run `sudo xhost +local:root` before starting |
| Camera feed shows black screen | Verify RTSP URL: `ffprobe rtsp://...` |
| Engine file not found | Regenerate `.engine` on the target Jetson device |
| Low FPS | Increase `interval` in `[primary-gie]` to skip frames |
| `libnvdsinfer_custom_impl_Yolo.so` error | Recompile: `cd nvdsinfer_custom_impl_Yolo && make` |
| Docker GPU not detected | Ensure `--runtime nvidia` flag is set |
| Out of memory errors | Reduce batch size or camera resolution |
| Thermal throttling | Add active cooling (fan) to Jetson Nano |

---

## Acknowledgments

This project is built upon the following open-source projects:

- **[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)** by Marcos Luciano — The `nvdsinfer_custom_impl_Yolo/` plugin code and DeepStream configuration templates were cloned from this repository and modified to support YOLOv8 output format with real-time bounding box rendering on 16 camera streams.
- **[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)** — Real-time video analytics framework.
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — YOLOv8 model architecture and training framework.

---

## Author

**Đoàn Sinh Đức**
Student at Hanoi University of Science and Technology (HUST)

---
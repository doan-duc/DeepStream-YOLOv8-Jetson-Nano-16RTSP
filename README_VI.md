# DeepStream YOLOv8 — Nhận Diện Chai C2 16 Camera RTSP trên Jetson Nano

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-DeepStream%206.0-76B900?style=for-the-badge&logo=nvidia" alt="DeepStream"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Jetson-Nano-76B900?style=for-the-badge&logo=nvidia" alt="Jetson Nano"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

---

## Tổng Quan

Hệ thống nhận diện đối tượng **real-time** cho **chai C2** sử dụng mô hình **YOLOv8n tùy chỉnh**, chạy trên **NVIDIA Jetson Nano** thông qua **DeepStream SDK 6.0**. Hệ thống xử lý **16 luồng camera RTSP đồng thời** với suy luận được tăng tốc phần cứng bằng TensorRT FP16.

Toàn bộ môi trường runtime được đóng gói trong Docker, đảm bảo triển khai nhất quán trên các thiết bị.

Plugin DeepStream và các file cấu hình dựa trên [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) của Marcos Luciano, được chỉnh sửa để hỗ trợ **định dạng output YOLOv8** và tối ưu cho việc vẽ bounding box real-time trên 16 luồng camera.

### Tính Năng Chính

| Tính năng | Mô tả |
|---|---|
| **Hỗ trợ 16 luồng** | Xử lý 16 luồng camera RTSP đồng thời với batch inference |
| **TensorRT FP16** | Engine suy luận tối ưu cho hiệu suất tối đa trên Jetson Nano |
| **IOU Tracker** | Theo dõi đối tượng tích hợp qua các khung hình |
| **OSD (On-Screen Display)** | Hiển thị bounding box real-time với độ tin cậy |
| **Tiled Display** | Hiển thị lưới 4×4 tất cả 16 camera trong một cửa sổ |
| **Docker hóa** | Đóng gói hoàn toàn để triển khai dễ dàng |
| **Custom YOLO Parser** | Plugin DeepStream YOLO tùy chỉnh biên dịch cho Jetson Nano (aarch64) |

---

## Dataset

Mô hình được train để nhận diện **chai C2** (C2 Freeze Icy Mint Lemon).

<p align="center">
  <img src="data(C2).png" width="300" alt="Mẫu chai C2"/>
</p>

| Thuộc tính | Giá trị |
|---|---|
| **Lớp đối tượng** | `C2` (chai C2 Freeze) |
| **Số lớp** | 1 |
| **File nhãn** | `models/labels.txt` |

---

## Kiến Trúc Hệ Thống

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

### Kiến Trúc Mô Hình & Huấn Luyện

Dự án sử dụng **mô hình YOLOv8n tùy chỉnh** (Student), được huấn luyện bằng kỹ thuật **Knowledge Distillation (KD)** từ mô hình **YOLOv8l** (Teacher).

**Tại sao dùng Knowledge Distillation?**
- **Teacher (YOLOv8l):** Độ chính xác cao nhưng rất chậm trên Jetson Nano.
- **Student (YOLOv8n-p2):** Tốc độ suy luận nhanh nhưng độ chính xác thấp hơn.
- **Kết quả:** Mô hình Student học từ "soft labels" của Teacher, đạt độ chính xác cao hơn YOLOv8n chuẩn trong khi vẫn giữ được tốc độ xử lý nhanh của kiến trúc Nano.

**Kết Quả Huấn Luyện:**
Thư mục `training_result/` chứa các biểu đồ so sánh giữa mô hình Student và Teacher.

| Chỉ số | Student (YOLOv8n-p2) | Teacher (YOLOv8l) |
|---|---|---|
| **mAP50** | Cao (xem results.png) | Rất cao |
| **Tốc độ** | ~15ms (Jetson Nano) | ~100ms+ (Quá chậm) |

### Điều Chỉnh Tùy Chỉnh

#### 1. **Giảm Số Kênh P5 Head**
Detection head P5 (đối tượng lớn) đã được giảm từ 256 xuống **128 kênh**:
```yaml
# YOLOv8n chuẩn: C2f[256]
# Mô hình tùy chỉnh: C2f[128]  ← Giảm 50%
```

#### 2. **Lớp Dropout (10%)**
Thêm lớp dropout sau P5 head để tránh overfitting:
```yaml
- [-1, 1, nn.Dropout, [0.1]]  # Tỷ lệ dropout 10%
```

### Detection Heads

| Head | Độ phân giải | Số kênh | Mục đích |
|---|---|---|---|
| **P3** | 8×8 | 64 | Nhận diện đối tượng nhỏ |
| **P4** | 16×16 | 128 | Nhận diện đối tượng trung bình |
| **P5** | 32×32 | 128 | Nhận diện đối tượng lớn (đã giảm) |

### Thống Kê Mô Hình (từ Ultralytics summary)

| Chỉ số | Giá trị |
|---|---|
| **Layers** | 130 |
| **Tham số** | **1,099,523** (~1.1M) |
| **GFLOPs** | **3.6** |
| **Kích thước ONNX** | **4.6 MB** |
| **Kích thước PyTorch** | **2.4 MB** |

### File Cấu Hình Mô Hình

Kiến trúc mô hình được định nghĩa trong `models/yolov8n-p2.yaml`:

```yaml
# Tham số
nc: 1                    # Số lớp
depth_multiple: 0.3      # Độ sâu YOLOv8n
width_multiple: 0.2      # Độ rộng YOLOv8n

# Backbone: YOLOv8n chuẩn
# Head: Tùy chỉnh với P5 giảm kênh + dropout
```

---

## Cấu Trúc Dự Án

```
.
├── README.md                          # File README (English)
├── README_VI.md                       # File README (Tiếng Việt)
├── LICENSE                            # Giấy phép MIT
├── requirements.txt                   # Thư viện Python cần thiết
├── .gitignore                         # Quy tắc Git ignore
├── data(C2).png                       # Ảnh mẫu dataset (chai C2)
│
├── training_result/                   # Kết quả huấn luyện & KD
│   ├── results.png                    # Biểu đồ Loss & mAP (Student)
│   ├── confusion_matrix.png           # Ma trận nhầm lẫn (Student)
│   ├── val_batch0_pred.jpg            # Ảnh trực quan hóa dự đoán
│   └── teacher_result/                # Kết quả của Teacher (YOLOv8l)
│
├── configs/                           # File cấu hình DeepStream
│   ├── deepstream_app_config.txt      # Cấu hình app (16 camera, batch=16)
│   └── config_infer_primary.txt       # Cấu hình inference (batch=16, FP16)
│
├── models/                            # File mô hình
│   ├── best.pt                        # ✅ Trọng số Student (đã train với KD)
│   ├── best.onnx                      # ✅ Mô hình ONNX (batch=16, đã export sẵn)
│   ├── yolov8n-p2.yaml                # Định nghĩa kiến trúc mô hình
│   └── labels.txt                     # Nhãn lớp: "C2"
│
├── nvdsinfer_custom_impl_Yolo/        # Plugin YOLO parser (từ DeepStream-Yolo)
│   ├── Makefile                       # Script build để biên dịch lại
│   ├── libnvdsinfer_custom_impl_Yolo.so  # Thư viện đã biên dịch (aarch64)
│   ├── nvdsinfer_yolo_engine.cpp      # TensorRT engine builder
│   ├── yolo.cpp / yolo.h             # YOLO model parser
│   ├── yoloPlugins.cpp / yoloPlugins.h  # TensorRT custom plugins
│   └── layers/                        # Triển khai các lớp neural network
│
├── docker/                            # File triển khai Docker
    ├── Dockerfile                     # Định nghĩa Docker image
    ├── build.sh                       # Script build Docker image
    └── run.sh                         # Script chạy Docker container
```

---

## Yêu Cầu Hệ Thống

### Phần Cứng
- **NVIDIA Jetson Nano** (khuyến nghị 4GB)
- **16 Camera IP** hỗ trợ RTSP (H.264 hoặc H.265)
- Kết nối Ethernet đến mạng camera
- Nguồn điện: 5V 4A barrel jack (để hiệu suất ổn định khi tải cao)

### Phần Mềm (trên Jetson Nano)
- **JetPack 4.6** (L4T R32.6.1)
- **DeepStream SDK 6.0**
- **CUDA 10.2**
- **TensorRT 8.0+**
- **Docker** (cài sẵn với JetPack)

---

## Hướng Dẫn Nhanh

### 1. Clone Repository

```bash
git clone <repository-url>
cd Project_YOLOV8
```

### 2. Build Docker Image

```bash
bash docker/build.sh
sudo docker images | grep ds-yolo
```

Script này sẽ:
- Tải NVIDIA DeepStream 6.0 base image (~4-5 GB)
- Copy toàn bộ project vào container
- Biên dịch plugin YOLO parser
- Tạo image tên là `ds-yolo`

### 3. Tạo File TensorRT Engine

Mô hình ONNX (`models/best.onnx`) đã có sẵn. Bạn phải chuyển đổi nó sang TensorRT engine **trên Jetson Nano đích**:

```bash
# Vào Docker container
bash docker/run.sh

# Trong container: Tạo TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/best.onnx \
    --saveEngine=models/best.engine \
    --fp16 \
    --workspace=2048
```

> ⚠️ **Quan trọng:** File TensorRT engine **KHÔNG thể chuyển** giữa các thiết bị khác nhau. Bạn **phải** tạo file `.engine` trên thiết bị Jetson Nano đích.

### 4. Cấu Hình Camera

Chỉnh sửa RTSP URI trong `configs/deepstream_app_config.txt`:

```ini
[source0]
enable=1
type=4
uri=rtsp://admin:PASSWORD@192.168.1.101:554/ch1/main
num-sources=1
gpu-id=0
# ... lặp lại cho source1 đến source15
```

### 5. Chạy Pipeline

```bash
bash docker/run.sh
deepstream-app -c configs/deepstream_app_config.txt
```

---

## File Mô Hình

### File Có Sẵn

| File | Định dạng | Dung lượng | Mô tả |
|---|---|---|---|
| `best.pt` | PyTorch | 2.4 MB | ✅ **Có sẵn** — Trọng số gốc (để export batch size khác) |
| `best.onnx` | ONNX | 4.6 MB | ✅ **Có sẵn** — Đã export batch=16 |
| `labels.txt` | Text | 3 B | ✅ **Có sẵn** — Nhãn lớp: `C2` |
| `best.engine` | TensorRT | — | ⚠️ **Phải tạo trên Jetson** — Xem bước 3 |

### Quy Trình

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  best.pt    │───>│  best.onnx   │───>│ best.engine   │───>│  DeepStream  │
│ (có sẵn)    │    │  (có sẵn)    │    │ (tạo trên     │    │  Inference   │
│  2.4 MB     │    │   4.6 MB     │    │   Jetson)     │    │              │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

### Tại Sao Có File `best.pt`?

Nếu bạn muốn dùng **batch size khác** (ví dụ: 8 camera thay vì 16), bạn có thể export lại ONNX từ file `.pt`:

```bash
# Trên máy có ultralytics
python utils/export_yolov8.py -w models/best.pt --batch 8

# Chuyển sang Jetson và tạo lại engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/best.onnx \
    --saveEngine=models/best.engine \
    --fp16 --workspace=2048
```

---

## Hiệu Suất

| Cấu hình | Số camera | Batch Size | Độ chính xác | FPS ước tính |
|---|---|---|---|---|
| `deepstream_app_config.txt` | 16 | 16 | FP16 | ~10-15 mỗi luồng |

---

## Xử Lý Sự Cố

| Vấn đề | Giải pháp |
|---|---|
| `Permission denied` khi chạy Docker | `sudo xhost +local:root` trước khi khởi động |
| Camera hiển thị màn hình đen | Kiểm tra RTSP URL: `ffprobe rtsp://...` |
| Không tìm thấy engine file | Tạo lại `.engine` trên Jetson đích |
| FPS thấp | Tăng `interval` trong `[primary-gie]` |
| Lỗi `libnvdsinfer_custom_impl_Yolo.so` | Biên dịch lại: `cd nvdsinfer_custom_impl_Yolo && make` |
| Docker không phát hiện GPU | Đảm bảo flag `--runtime nvidia` |
| Lỗi hết bộ nhớ | Giảm batch size hoặc độ phân giải camera |

---

## Nguồn Gốc & Ghi Công

Dự án dựa trên các project mã nguồn mở:

- **[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)** của Marcos Luciano — Code plugin `nvdsinfer_custom_impl_Yolo/` và template cấu hình DeepStream được clone từ repository này và chỉnh sửa để hỗ trợ định dạng output YOLOv8 với việc vẽ bounding box real-time trên 16 luồng camera.
- **[NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)** — Framework phân tích video real-time.
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Kiến trúc mô hình YOLOv8 và framework huấn luyện.

---

## Tác Giả

**Đoàn Sinh Đức**
Sinh viên Đại học Bách khoa Hà Nội (HUST)

---

## Giấy Phép

MIT License — Copyright (c) 2026 Đoàn Sinh Đức

Xem [LICENSE](LICENSE) để biết chi tiết.

> **Lưu ý:** Code plugin `nvdsinfer_custom_impl_Yolo/` có nguồn gốc từ [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo), được cấp phép GPL-3.0. Vui lòng tham khảo repository gốc để biết điều khoản giấy phép.

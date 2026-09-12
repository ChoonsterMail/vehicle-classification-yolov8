# 🚗 Hệ thống Thị giác Máy tính Phân loại và Nhận diện Phương tiện Giao thông

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io)

## 📋 Giới thiệu

Đồ án 2 — Khoa Công nghệ Thông tin, Trường Đại học Mỏ - Địa Chất.

Xây dựng hệ thống ứng dụng **Deep Learning** và **Computer Vision** để **phát hiện, phân loại và đếm lưu lượng** phương tiện giao thông từ video/ảnh giám sát. Hệ thống phân loại **5 nhóm phương tiện** phổ biến trên đường phố Việt Nam:

| # | Lớp | Class ID | Tên tiếng Anh |
|---|------|----------|---------------|
| 1 | Ô tô con | `0` | car |
| 2 | Xe máy | `1` | motorcycle |
| 3 | Xe buýt | `2` | bus |
| 4 | Xe tải | `3` | truck |
| 5 | Xe đạp | `4` | bicycle |

## 🎯 Mục tiêu

### Học thuật và nghiên cứu
- Làm chủ toàn diện pipeline thị giác máy tính: Thu thập, gán nhãn dữ liệu chuẩn YOLO format, tăng cường ảnh (Data Augmentation), huấn luyện mô hình và tối ưu hóa suy luận.
- Nghiên cứu chuyên sâu kiến trúc CNN và mô hình phát hiện đối tượng đơn tầng (YOLOv8/YOLOv9).
- Thiết lập phương pháp đánh giá thực nghiệm: Precision, Recall, F1-Score, Confusion Matrix, IoU, mAP@0.5, mAP@0.5:0.95, FPS.

### Ứng dụng và thực tiễn
- Phân loại chính xác **5 nhóm phương tiện** từ camera giám sát giao thông.
- Đạt tốc độ suy luận thời gian thực **≥ 30 FPS**.
- Tích hợp thuật toán **ByteTrack** + vạch kẻ ảo đếm lưu lượng phương tiện qua từng làn đường.
- Triển khai ứng dụng **Web Demo tương tác** bằng Streamlit.

## 📁 Cấu trúc dự án

```
Phân loại phương tiện giao thông/
├── README.md                   # Tài liệu dự án (file này)
├── requirements.txt            # Thư viện Python cần cài đặt
├── .gitignore                  # Danh sách file/thư mục bỏ qua
│
├── configs/                    # File cấu hình
│   └── data.yaml               # Cấu hình dataset cho YOLO
│
├── data/                       # Dữ liệu (không push lên Git)
│   ├── raw/                     # Ảnh/video gốc chưa xử lý
│   ├── train/
│   │   ├── images/              # Ảnh huấn luyện
│   │   └── labels/              # Nhãn YOLO (.txt)
│   ├── val/
│   │   ├── images/              # Ảnh validation
│   │   └── labels/              # Nhãn YOLO (.txt)
│   └── test/
│       ├── images/              # Ảnh kiểm thử
│       └── labels/              # Nhãn YOLO (.txt)
│
├── docs/                       # Tài liệu
│   ├── phan_cong_nhiem_vu.md    # Phân công nhiệm vụ nhóm
│   └── quy_chuan_gan_nhan.md   # Quy chuẩn gán nhãn YOLO
│
├── models/                     # Mô hình đã huấn luyện
│   ├── classification/          # Mô hình phân loại (ResNet, MobileNet)
│   └── detection/               # Mô hình phát hiện (YOLOv8)
│
├── notebooks/                  # Jupyter Notebooks
│   └── eda.ipynb                # Phân tích khám phá dữ liệu (EDA)
│
└── src/                        # Mã nguồn chính
    ├── data_preprocessing.py    # Tiền xử lý và tăng cường dữ liệu
    ├── train_classifier.py      # Huấn luyện mô hình phân loại
    ├── train_yolo.py            # Huấn luyện YOLOv8
    ├── evaluate.py              # Đánh giá mô hình
    ├── tracker.py               # ByteTrack theo dõi & đếm xe
    └── app.py                   # Ứng dụng Web Streamlit
```

## 🔧 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- CUDA 11.8+ (khuyến nghị cho GPU)
- RAM ≥ 8GB

### Cài đặt thư viện

```bash
# Clone repository
git clone https://github.com/<username>/vehicle-classification.git
cd vehicle-classification

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Huấn luyện mô hình phân loại (Baseline)
```bash
python src/train_classifier.py --model resnet50 --epochs 50 --batch-size 32
```

### 2. Huấn luyện YOLOv8
```bash
python src/train_yolo.py --model yolov8n --data configs/data.yaml --epochs 100
```

### 3. Đánh giá mô hình
```bash
python src/evaluate.py --model models/detection/best.pt --data configs/data.yaml
```

### 4. Chạy Web Demo
```bash
streamlit run src/app.py
```

## 📊 Dữ liệu

| Tập dữ liệu | Nguồn | Số lượng |
|-------------|-------|----------|
| BIT-Vehicle Dataset | Học thuật | 9.850 ảnh |
| Kaggle Vehicle Detection | Kaggle | Đang khảo sát |
| Video giao thông Hà Nội | Tự thu thập | Đang thu thập |

**Phân chia dữ liệu:** Train 70% / Val 20% / Test 10%

## 📈 Kết quả dự kiến

| Chỉ số | Mục tiêu |
|--------|----------|
| mAP@0.5 | ≥ 85% |
| mAP@0.5:0.95 | ≥ 60% |
| FPS (GPU) | ≥ 30 |
| FPS (CPU) | ≥ 10 |

## 👥 Thành viên nhóm

| Thành viên | Vai trò chính |
|-----------|--------------|
| **Nguyễn Thành Đạt** | Trưởng nhóm — Quản lý dự án, thiết kế pipeline, đánh giá thực nghiệm |
| **Đỗ Xuân Bách** | Dữ liệu & Baseline — Khảo sát dataset, xây dựng mô hình phân loại cơ sở |
| **Nguyễn Vũ Tùng** | Hạ tầng & Mô hình — Thiết lập môi trường, huấn luyện YOLOv8, tối ưu ONNX |
| **Phạm Văn Tưởng** | Nghiên cứu & Báo cáo — Lý thuyết CNN/YOLO, EDA, viết báo cáo |

**Giáo viên hướng dẫn:** Trần Thị Hòa

## 📚 Tài liệu tham khảo

1. Nguyễn Thanh Thủy, *Giáo trình Trí tuệ nhân tạo và Thị giác máy tính*, NXB Bách khoa Hà Nội, 2020.
2. Joseph Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, IEEE CVPR, 2016.
3. Kaiming He et al., *Deep Residual Learning for Image Recognition*, IEEE CVPR, 2016.
4. Glenn Jocher et al., *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
5. Yifu Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV, 2022.

## 📄 License

Dự án phục vụ mục đích học tập — Đồ án 2, Trường Đại học Mỏ - Địa Chất.

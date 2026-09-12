# Phân công Nhiệm vụ — Đồ án 2

## Thông tin chung
- **Đề tài:** Xây dựng hệ thống thị giác máy tính phân loại và nhận diện phương tiện giao thông
- **GVHD:** Trần Thị Hòa
- **Thời gian:** 10 tuần

---

## Tuần 1 — Khởi tạo & Nghiên cứu lý thuyết

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Khởi tạo GitHub Repo, phân công nhiệm vụ, viết phần Mục tiêu & Phạm vi nghiên cứu | Kho GitHub & file phân công | Cuối tuần 1 |
| Đỗ Xuân Bách | Khảo sát và tải các tập dữ liệu phương tiện giao thông (BIT-Vehicle, Kaggle Vehicle Dataset) | Báo cáo khảo sát dataset | Cuối tuần 1 |
| Nguyễn Vũ Tùng | Thiết lập môi trường huấn luyện (PyTorch, Ultralytics, CUDA, cuDNN, GPU Colab/Local) | Môi trường huấn luyện sẵn sàng | Cuối tuần 1 |
| Phạm Văn Tưởng | Nghiên cứu lý thuyết mạng CNN, Transfer Learning và kiến trúc Single-stage Detector YOLO | Báo cáo tổng quan lý thuyết | Cuối tuần 1 |

## Tuần 2 — Dữ liệu & Gán nhãn

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Thống nhất quy chuẩn gán nhãn 5 lớp phương tiện và định dạng YOLO format (.txt) | Tài liệu quy chuẩn gán nhãn | Cuối tuần 2 |
| Đỗ Xuân Bách | Kiểm tra và loại bỏ các mẫu ảnh hỏng; chuẩn hóa tọa độ Bounding Box về khoảng [0,1] | Bộ dữ liệu đã lọc | Cuối tuần 2 |
| Nguyễn Vũ Tùng | Thu thập video giao thông thực tế tại các nút giao ở Hà Nội và cắt frame bổ sung | Video + frames giao thông | Cuối tuần 2 |
| Phạm Văn Tưởng | Phân tích khám phá dữ liệu (EDA): đếm số lượng mẫu mỗi lớp, phân tích tỷ lệ khung hình | Jupyter Notebook EDA | Cuối tuần 2 |

## Tuần 3 — Tiền xử lý & Tăng cường dữ liệu

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Thiết kế pipeline phân chia Train/Val/Test (tỷ lệ 70:20:10) đảm bảo cân bằng phân bố lớp | Script chia dữ liệu | Cuối tuần 3 |
| Đỗ Xuân Bách | Lập trình module tăng cường dữ liệu ảnh (Albumentations: Random Brightness, CLAHE, Flip, Blur) | Module augmentation | Cuối tuần 3 |
| Nguyễn Vũ Tùng | Viết script tiền xử lý ảnh tự động (Letterboxing resize về 640×640, Normalization) | Script tiền xử lý | Cuối tuần 3 |
| Phạm Văn Tưởng | Thử nghiệm kỹ thuật Mosaic Augmentation và MixUp để xử lý hiện tượng xe bị che khuất | Báo cáo kết quả thử nghiệm | Cuối tuần 3 |

## Tuần 4 — Xây dựng mô hình Baseline

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Thiết lập kịch bản đánh giá Baseline và lập trình module tính toán Confusion Matrix, F1-Score | Module đánh giá | Cuối tuần 4 |
| Đỗ Xuân Bách | Xây dựng mô hình phân loại ảnh cơ sở (Baseline Classifier) sử dụng ResNet50 Transfer Learning | Mô hình ResNet50 | Cuối tuần 4 |
| Nguyễn Vũ Tùng | Xây dựng mô hình phân loại nhẹ (MobileNetV3 / EfficientNet) để so sánh tốc độ | Mô hình MobileNetV3 | Cuối tuần 4 |
| Phạm Văn Tưởng | Huấn luyện mô hình phân loại, vẽ đồ thị hàm Loss và Accuracy qua các Epoch | Biểu đồ Loss/Accuracy | Cuối tuần 4 |

## Tuần 5–6 — Huấn luyện YOLOv8 & Tinh chỉnh

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Cấu hình data.yaml + kiến trúc YOLOv8; Điều phối Hyperparameter Tuning (LR, Optimizer, Weight decay) | File cấu hình + báo cáo tuning | Cuối tuần 6 |
| Đỗ Xuân Bách | Huấn luyện YOLOv8n (nano); Error Analysis phân tích nhầm lẫn xe buýt/xe tải | Mô hình YOLOv8n + báo cáo lỗi | Cuối tuần 6 |
| Nguyễn Vũ Tùng | Huấn luyện YOLOv8s/YOLOv8m; Tinh chỉnh Confidence Threshold và NMS | Mô hình YOLOv8s/m tối ưu | Cuối tuần 6 |
| Phạm Văn Tưởng | Giám sát Loss hội tụ; Thử nghiệm Focal Loss xử lý mất cân bằng lớp | Biểu đồ loss + checkpoint | Cuối tuần 6 |

## Tuần 7 — Đánh giá thực nghiệm

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Tổng hợp kết quả thực nghiệm trên tập Test, vẽ Precision-Recall Curves | Báo cáo thực nghiệm | Cuối tuần 7 |
| Đỗ Xuân Bách | Đo đạc mAP@0.5 và mAP@0.5:0.95 cho từng lớp phương tiện | Bảng kết quả chi tiết | Cuối tuần 7 |
| Nguyễn Vũ Tùng | Benchmark FPS và Latency trên CPU và GPU | Bảng benchmark | Cuối tuần 7 |
| Phạm Văn Tưởng | Đánh giá độ bền vững dưới các điều kiện ánh sáng (ngày, nắng, đêm) | Báo cáo robustness | Cuối tuần 7 |

## Tuần 8 — Tích hợp ByteTrack & Export ONNX

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Thiết kế thuật toán vạch kẻ ảo (Virtual Line Counting) đếm xe qua từng làn | Module đếm xe | Cuối tuần 8 |
| Đỗ Xuân Bách | Tích hợp ByteTrack với YOLO để gán Tracking ID ổn định | Module tracking | Cuối tuần 8 |
| Nguyễn Vũ Tùng | Xuất mô hình sang ONNX và kiểm thử tăng tốc với ONNX Runtime | File model.onnx | Cuối tuần 8 |
| Phạm Văn Tưởng | Xây dựng module xử lý luồng video camera liên tục qua OpenCV | Module video pipeline | Cuối tuần 8 |

## Tuần 9 — Web Demo & Báo cáo

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Thiết kế Layout ứng dụng Web Streamlit (upload video/ảnh, dashboard thống kê) | Web app layout | Cuối tuần 9 |
| Đỗ Xuân Bách | Lập trình nhận diện ảnh tĩnh, hiển thị Bounding Box và xác suất nhãn | Tính năng nhận diện ảnh | Cuối tuần 9 |
| Nguyễn Vũ Tùng | Lập trình xử lý video tải lên, hiển thị bounding box + bộ đếm xe | Tính năng xử lý video | Cuối tuần 9 |
| Phạm Văn Tưởng | Viết bản thảo báo cáo toàn văn Đồ án 2 (Draft 1: Chương 1–4) | Bản thảo báo cáo | Cuối tuần 9 |

## Tuần 10 — Hoàn thiện & Thuyết trình

| Thành viên | Công việc | Sản phẩm đầu ra | Deadline |
|-----------|-----------|-----------------|----------|
| Nguyễn Thành Đạt | Rà soát, chuẩn hóa định dạng toàn bộ cuốn Báo cáo (.docx, .pdf) | Báo cáo hoàn chỉnh | Cuối tuần 10 |
| Đỗ Xuân Bách | Chuẩn bị video demo nhận diện trên video thực tế, hoàn thiện GitHub | Video demo + GitHub | Cuối tuần 10 |
| Nguyễn Vũ Tùng | Soạn kịch bản thuyết trình phần Dữ liệu, Mô hình YOLOv8, kết quả thực nghiệm | Slides thuyết trình | Cuối tuần 10 |
| Phạm Văn Tưởng | Soạn kịch bản thuyết trình phần ứng dụng, đếm phương tiện, câu hỏi phản biện | Slides + Q&A | Cuối tuần 10 |

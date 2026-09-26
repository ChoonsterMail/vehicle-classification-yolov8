<!--
=============================================================================
PULL REQUEST TEMPLATE
Đồ án 2: Phân loại và Nhận diện Phương tiện Giao thông
Khoa Công nghệ Thông tin - Trường Đại học Mỏ - Địa chất
=============================================================================
-->

## 📌 1. Tiêu đề Pull Request (Theo chuẩn Conventional Commits)
> *Ví dụ: `feat(preprocess): tích hợp letterbox và chuẩn hóa ảnh 640x640`*
> *Ví dụ: `fix(yolo): sửa lỗi nhầm nhãn class bus và truck trong data.yaml`*

---

## 👤 2. Thông tin Thành viên Thực hiện & Tuần làm việc
- **Thành viên:** 
  - [ ] Nguyễn Thành Đạt (Trưởng nhóm)
  - [ ] Đỗ Xuân Bách
  - [ ] Nguyễn Vũ Tùng
  - [ ] Phạm Văn Tưởng
- **Tuần thực hiện:** Tuần _____ (theo Đề cương)
- **Issue liên quan (nếu có):** Closes #_____

---

## 📝 3. Tóm tắt Nội dung Thay đổi
*(Mô tả chi tiết những gì bạn đã làm, lý do thay đổi và cách giải quyết)*

- 

---

## 🏷️ 4. Phân loại Thay đổi
*(Chọn ít nhất một mục phù hợp)*

- [ ] ✨ `feat`: Tính năng / Module mã nguồn mới
- [ ] 🐛 `fix`: Sửa lỗi mã nguồn / Lỗi script
- [ ] 🧹 `refactor`: Tối ưu / Tái cấu trúc mã nguồn (không đổi tính năng)
- [ ] 📊 `data`: Cập nhật cấu hình dữ liệu / Scripts tiền xử lý / Nhãn
- [ ] 🤖 `model`: Huấn luyện / Xuất mô hình / Benchmark
- [ ] 📖 `docs`: Cập nhật tài liệu kỹ thuật / Báo cáo tuần / Hướng dẫn
- [ ] ⚙️ `chore`: Cập nhật cấu hình môi trường, thư viện (`requirements.txt`, CI/CD)

---

## 📂 5. Các Tệp/Thư mục Bị Ảnh Hưởng
- [ ] `src/` (Mã nguồn chính)
- [ ] `src/utils/` (Thư viện dùng chung)
- [ ] `configs/` (Tệp cấu hình dataset & hyperparameter)
- [ ] `notebooks/` (Jupyter Notebooks EDA, Visualization)
- [ ] `docs/` (Quy chuẩn, phân công, hướng dẫn)
- [ ] `tests/` hoặc `.github/` (CI/CD workflows)

---

## 🧪 6. Kết quả Kiểm thử & Minh chứng (Test & Proof)
*(Cung cấp lệnh đã chạy thử, kết quả in ra màn hình hoặc hình ảnh đính kèm)*

```bash
# Lệnh đã chạy thử nghiệm thành công trên máy local hoặc Colab:

```

- **Kết quả / Metrics (nếu có):**
  - Accuracy: _____%
  - F1-Score: _____%
  - FPS / Latency: _____ ms

---

## ✅ 7. Checklist Kiểm tra trước khi yêu cầu Merge
- [ ] Mã nguồn chạy thử nghiệm thành công, không phát sinh lỗi ngoại lệ (`Exception / Traceback`).
- [ ] Tuân thủ quy chuẩn đặt tên biến, hàm theo chuẩn PEP8 của Python.
- [ ] Không vô tình commit tệp dữ liệu nặng (`*.jpg`, `*.png` trong `data/`, `*.zip`, `*.mp4`).
- [ ] Không commit tệp trọng số mô hình lớn (`*.pt`, `*.pth`, `*.onnx`) vào kho Git (phải dùng Google Drive hoặc lưu trữ riêng).
- [ ] Đã thêm chú thích / docstring rõ ràng cho các hàm và module mới.
- [ ] Đã kéo mã nguồn mới nhất từ nhánh `develop` về nhánh cá nhân và giải quyết xung đột (nếu có).

---

## 👥 8. Người Review chéo (Peer Reviewer)
- Được chỉ định review: @_____
- [ ] Đồng ý Merge vào nhánh đích.

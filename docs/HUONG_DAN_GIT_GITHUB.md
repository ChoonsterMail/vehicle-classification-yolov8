# 📖 Cẩm Nang Thực Chiến Git & GitHub Dành Cho Thành Viên Nhóm

> **Dành riêng cho 4 thành viên:** Nguyễn Thành Đạt, Đỗ Xuân Bách, Nguyễn Vũ Tùng, Phạm Văn Tưởng.  
> **Mục đích:** Hướng dẫn từng bước từ cài đặt, thiết lập ban đầu cho đến các thao tác Git hàng ngày và xử lý xung đột mã nguồn.

---

## 🚀 PHẦN 1: THIẾT LẬP LẦN ĐẦU (Chỉ làm 1 lần)

### 1.1. Cấu hình danh tính Git trên máy tính cá nhân
Mở Terminal (macOS/Linux) hoặc Git Bash (Windows) và chạy 2 lệnh sau:

```bash
# Thay thế bằng tên và email tài khoản GitHub của bạn
git config --global user.name "Nguyen Thanh Dat"
git config --global user.email "your-email@gmail.com"

# Kiểm tra lại cấu hình
git config --global --list
```

### 1.2. Clone kho lưu trữ về máy tính
```bash
# Thay URL bằng đường dẫn kho GitHub chính thức của nhóm
git clone https://github.com/ChoonsterMail/vehicle-classification-yolov8.git

# Di chuyển vào thư mục dự án
cd vehicle-classification-yolov8
```

### 1.3. Khởi tạo môi trường ảo Python và cài đặt thư viện
```bash
# Trên macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Trên Windows (Command Prompt hoặc PowerShell):
python -m venv venv
venv\Scripts\activate

# Nâng cấp pip và cài đặt thư viện đồ án
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 PHẦN 2: QUY TRÌNH LÀM VIỆC HÀNG NGÀY (Daily Workflow)

Mỗi khi bắt đầu làm một công việc mới (ví dụ: viết script, huấn luyện model, vẽ đồ thị):

### Bước 1: Luôn cập nhật mã nguồn mới nhất từ nhánh `develop`
```bash
# Chuyển về nhánh develop
git checkout develop

# Kéo toàn bộ code mới nhất mà các bạn khác vừa merge
git pull origin develop
```

### Bước 2: Tạo nhánh mới cho nhiệm vụ của mình
> **Quy tắc đặt tên:** `feat/<tuần>-<tên-bạn>-<tên-công-việc>`

```bash
# Ví dụ bạn Bách làm module ResNet50 Tuần 4:
git checkout -b feat/w4-bach-resnet50

# Ví dụ bạn Tùng làm module MobileNetV3 Tuần 4:
git checkout -b feat/w4-tung-mobilenetv3

# Ví dụ bạn Tưởng làm Notebook Visualization Tuần 4:
git checkout -b feat/w4-tuong-visualization
```

### Bước 3: Lập trình và kiểm tra trạng thái
Trong khi code, thường xuyên kiểm tra xem mình đã sửa những file nào:
```bash
git status
```
Chỉ đưa các file mã nguồn liên quan vào vùng chuẩn bị commit (staging):
```bash
# Thêm file cụ thể
git add src/train_classifier.py configs/data.yaml

# TUYỆT ĐỐI KHÔNG dùng 'git add .' bừa bãi nếu trong thư mục có file ảnh nặng hoặc file .pth
```

### Bước 4: Tạo Commit với thông điệp rõ ràng
```bash
git commit -m "feat(resnet50): hoàn thành chiến lược transfer learning 2 giai đoạn freeze và unfreeze"
```

### Bước 5: Đẩy nhánh lên GitHub
```bash
git push -u origin feat/w4-bach-resnet50
```

### Bước 6: Mở Pull Request (PR) trên web GitHub
1. Vào link GitHub của dự án, bạn sẽ thấy thông báo màu vàng: `feat/w4-bach-resnet50 had recent pushes`. Bấm **Compare & pull request**.
2. **Lưu ý quan trọng:** Đảm bảo nhánh đích (base) là **`develop`**, không chọn `main`.
3. Điền vào form mô tả Pull Request (đã có mẫu sẵn).
4. Tag người bạn muốn nhờ review (ví dụ Trưởng nhóm Đạt).
5. Bấm **Create pull request**.

---

## ⚡ PHẦN 3: CÁCH XỬ LÝ KHI BỊ XUNG ĐỘT (MERGE CONFLICT)

Khi hai thành viên cùng sửa một file hoặc nhánh của bạn bị tụt hậu so với `develop`:

```bash
# 1. Đảm bảo bạn đã commit toàn bộ code dang dở trên nhánh của mình
git add .
git commit -m "chore: lưu tạm trước khi rebase"

# 2. Cập nhật nhánh develop trên máy
git checkout develop
git pull origin develop

# 3. Quay lại nhánh của bạn và hòa trộn code từ develop vào
git checkout feat/w4-bach-resnet50
git merge develop
```

- Nếu có thông báo `CONFLICT (content): Merge conflict in ...`:
  1. Mở file bị báo đỏ trong VSCode.
  2. Bạn sẽ thấy các khối đánh dấu:
     ```text
     <<<<<<< HEAD (Code hiện tại của bạn)
     learning_rate = 0.001
     =======
     learning_rate = 0.0001
     >>>>>>> develop (Code của bạn khác trên develop)
     ```
  3. Chọn đoạn code đúng nhất (bấm nút *Accept Current Change*, *Accept Incoming Change* hoặc tự sửa tay cho thống nhất).
  4. Lưu file lại, sau đó chạy:
     ```bash
     git add <tên_file_vừa_sửa>
     git commit -m "fix(conflict): giải quyết xung đột mã nguồn với develop"
     git push origin feat/w4-bach-resnet50
     ```

---

## 📋 BẢNG TRA CỨU NHANH CÁC LỆNH GIT (Cheat Sheet)

| Lệnh Git | Tác dụng |
| :--- | :--- |
| `git status` | Xem trạng thái các file bị thay đổi, file mới tạo |
| `git diff` | Xem chi tiết từng dòng code đã sửa đổi |
| `git branch -a` | Xem danh sách tất cả các nhánh (cục bộ và trên server) |
| `git checkout -b <tên-nhánh>` | Tạo và chuyển sang nhánh mới |
| `git checkout <tên-nhánh>` | Chuyển sang nhánh đã có |
| `git log --oneline -n 5` | Xem lịch sử 5 commit gần nhất gọn gàng |
| `git stash` | Tạm cất các thay đổi chưa commit để chuyển nhánh gấp |
| `git stash pop` | Lấy lại các thay đổi vừa tạm cất |
| `git checkout -- <file>` | Hủy bỏ các sửa đổi chưa commit trên một file |

---

## 📞 HỖ TRỢ VÀ LIÊN HỆ NỘI BỘ
- Nếu gặp lỗi Git không giải quyết được: Chụp ảnh màn hình lỗi gửi lên nhóm Zalo của đề tài để Trưởng nhóm (Nguyễn Thành Đạt) hỗ trợ ngay, **tránh gõ các lệnh `git push --force` hay xóa thư mục `.git`**.

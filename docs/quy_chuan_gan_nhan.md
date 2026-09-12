# Quy chuẩn Gán nhãn Phương tiện Giao thông — YOLO Format

## 1. Định nghĩa 5 lớp phương tiện

| Class ID | Tên lớp | Mô tả | Ví dụ cụ thể |
|----------|---------|-------|--------------|
| `0` | **car** (Ô tô con) | Xe 4 bánh chở người, kích thước nhỏ–trung bình | Sedan, SUV, hatchback, crossover, taxi, xe công nghệ (Grab) |
| `1` | **motorcycle** (Xe máy) | Xe 2 bánh có động cơ | Xe số, xe tay ga, xe côn tay. **Không** bao gồm xe đạp điện |
| `2` | **bus** (Xe buýt) | Xe chở khách cỡ lớn, thường > 16 chỗ | Xe buýt công cộng, xe khách liên tỉnh, xe du lịch |
| `3` | **truck** (Xe tải) | Xe chở hàng hóa, có thùng hàng | Xe tải nhẹ, xe tải nặng, xe container, xe ben, xe bồn |
| `4` | **bicycle** (Xe đạp) | Xe 2 bánh không động cơ hoặc xe đạp điện | Xe đạp thường, xe đạp điện, xe đạp thể thao |

### Lưu ý phân biệt

| Trường hợp dễ nhầm | Quy tắc xử lý |
|--------------------|---------------|
| Xe buýt vs Xe tải | Xe buýt có **cửa sổ hành khách** dọc thân xe; Xe tải có **thùng hàng kín/hở** |
| Xe bán tải (pickup) | Phân loại là **car** (`0`) nếu kích thước nhỏ; **truck** (`3`) nếu có thùng hàng lớn |
| Xe đạp điện | Phân loại là **bicycle** (`4`) |
| Xe 3 bánh (xích lô, xe ba gác) | Phân loại là **motorcycle** (`1`) nếu có động cơ; **bicycle** (`4`) nếu không |
| Xe cứu thương, xe cảnh sát | Phân loại theo loại xe gốc: thường là **car** (`0`) |

---

## 2. Định dạng YOLO (.txt)

### 2.1 Cấu trúc file nhãn

Mỗi ảnh `image_001.jpg` có một file nhãn tương ứng `image_001.txt` cùng tên, khác đuôi.

**Format mỗi dòng:**
```
<class_id> <x_center> <y_center> <width> <height>
```

| Trường | Kiểu | Phạm vi | Mô tả |
|--------|------|---------|-------|
| `class_id` | int | 0–4 | ID lớp phương tiện (xem bảng trên) |
| `x_center` | float | [0, 1] | Tọa độ X tâm bounding box / chiều rộng ảnh |
| `y_center` | float | [0, 1] | Tọa độ Y tâm bounding box / chiều cao ảnh |
| `width` | float | [0, 1] | Chiều rộng bounding box / chiều rộng ảnh |
| `height` | float | [0, 1] | Chiều cao bounding box / chiều cao ảnh |

### 2.2 Công thức chuẩn hóa

Nếu tọa độ bounding box gốc là `(x_min, y_min, x_max, y_max)` pixel và ảnh có kích thước `(img_w, img_h)`:

```
x_center = ((x_min + x_max) / 2) / img_w
y_center = ((y_min + y_max) / 2) / img_h
width    = (x_max - x_min) / img_w
height   = (y_max - y_min) / img_h
```

### 2.3 Ví dụ

**Ảnh:** `hanoi_intersection_001.jpg` (1920 × 1080 px)

Có 3 phương tiện: 1 ô tô con, 1 xe máy, 1 xe buýt.

**File nhãn:** `hanoi_intersection_001.txt`
```
0 0.4521 0.3750 0.0938 0.1389
1 0.7031 0.6204 0.0365 0.0833
2 0.2344 0.4444 0.1563 0.2778
```

**Giải thích dòng 1:**
- Class `0` = ô tô con
- Tâm tại (0.4521, 0.3750) tương đương pixel (868, 405)
- Kích thước (0.0938, 0.1389) tương đương 180×150 pixel

---

## 3. Quy tắc gán nhãn Bounding Box

### 3.1 Nguyên tắc chung

1. **Bao kín toàn bộ phương tiện**: Bounding box phải bao trọn phương tiện, bao gồm cả bánh xe và gương chiếu hậu.
2. **Sát viền nhất có thể**: Giảm thiểu khoảng trống (padding) giữa bounding box và phương tiện. Tối đa 5% padding.
3. **Mỗi phương tiện 1 box**: Không vẽ trùng box cho cùng 1 phương tiện.
4. **Bỏ qua phương tiện quá nhỏ**: Nếu phương tiện chiếm < 20×20 pixel (trên ảnh gốc), **không gán nhãn**.
5. **Phương tiện bị che khuất**: Nếu ≥ 50% phương tiện nhìn thấy được → **gán nhãn**. Nếu < 50% → **bỏ qua**.

### 3.2 Các trường hợp đặc biệt

| Trường hợp | Xử lý |
|-----------|-------|
| Phương tiện cắt ngang viền ảnh | Gán nhãn nếu ≥ 50% thân xe hiện trong ảnh |
| Phương tiện đang rẽ/xiên góc | Vẽ box bao kín phần nhìn thấy |
| Nhiều xe chồng chéo | Gán box riêng cho **từng xe** nhìn thấy được ≥ 50% |
| Phương tiện đỗ yên | Vẫn gán nhãn bình thường |
| Ảnh ban đêm (tối, mờ) | Gán nhãn nếu nhận diện được loại xe bằng mắt thường |
| Xe kéo rơ moóc | Gán 1 box cho toàn bộ (đầu kéo + rơ moóc) → class `truck` |

---

## 4. Công cụ gán nhãn khuyến nghị

| Công cụ | Ưu điểm | Link |
|---------|---------|------|
| **LabelImg** | Nhẹ, hỗ trợ YOLO format trực tiếp | https://github.com/HumanSignal/labelImg |
| **CVAT** | Web-based, hỗ trợ team collaboration | https://www.cvat.ai/ |
| **Roboflow** | Auto-augmentation, export nhiều format | https://roboflow.com/ |
| **Label Studio** | Đa năng, hỗ trợ nhiều loại annotation | https://labelstud.io/ |

**Khuyến nghị:** Sử dụng **Roboflow** cho quản lý dataset nhóm + auto export YOLO format.

---

## 5. Cấu trúc thư mục dữ liệu

```
data/
├── train/
│   ├── images/
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_00001.txt
│       ├── img_00002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Quy ước đặt tên file:**
- Ảnh: `<nguồn>_<số thứ tự>.jpg` (ví dụ: `bit_00001.jpg`, `hanoi_00001.jpg`)
- Nhãn: Trùng tên ảnh, đổi đuôi `.txt`
- Chỉ dùng chữ thường, số và dấu gạch dưới `_`

---

## 6. Checklist kiểm tra chất lượng nhãn

- [ ] Tất cả file nhãn `.txt` đều có file ảnh `.jpg` tương ứng
- [ ] Không có file ảnh thiếu file nhãn (trừ ảnh không có phương tiện)
- [ ] Class ID chỉ nằm trong {0, 1, 2, 3, 4}
- [ ] Tọa độ x_center, y_center, width, height đều trong khoảng [0, 1]
- [ ] Không có dòng trống hoặc ký tự lạ trong file nhãn
- [ ] Bounding box không vượt ra ngoài biên ảnh
- [ ] Phương tiện quá nhỏ (< 20×20 px) đã được bỏ qua

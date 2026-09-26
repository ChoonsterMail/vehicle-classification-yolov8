"""
src/mosaic_mixup.py
====================
Phụ trách: Phạm Văn Tưởng — Tuần 3

Thử nghiệm và so sánh 2 kỹ thuật augmentation mạnh:
  • Mosaic Augmentation : ghép 4 ảnh thành 1 ảnh 640×640
  • MixUp Augmentation  : blend 2 ảnh với trọng số Beta(α, α)

Đây là 2 kỹ thuật chủ chốt của YOLOv8 giúp xử lý hiện tượng xe bị che khuất
và tăng cường ngữ cảnh đa dạng.

Cách dùng:
    python src/mosaic_mixup.py --demo          # Demo với ảnh synthetic
    python src/mosaic_mixup.py --compare --n 100 --input data/train
    python src/mosaic_mixup.py --help
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_SIZE = 640   # Kích thước ảnh Mosaic/MixUp output


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Đọc file nhãn YOLO → (bboxes [xc,yc,w,h] normalized, class_ids)."""
    bboxes, class_ids = [], []
    if not label_path.exists():
        return bboxes, class_ids
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_ids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_ids


def write_yolo_label(label_path: Path, bboxes: List[List[float]], class_ids: List[int]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, (xc, yc, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Mosaic Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def create_mosaic(
    images: List[np.ndarray],
    labels_list: List[Tuple[List[List[float]], List[int]]],
    output_size: int = OUTPUT_SIZE,
    center_range: Tuple[float, float] = (0.3, 0.7),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Ghép 4 ảnh thành 1 ảnh Mosaic kích thước output_size × output_size.

    Sơ đồ ghép:
        ┌─────────┬─────────┐
        │  img[0] │  img[1] │
        │  (TL)   │  (TR)   │
        ├─────────┼─────────┤
        │  img[2] │  img[3] │
        │  (BL)   │  (BR)   │
        └─────────┴─────────┘

    Parameters
    ----------
    images : list 4 ảnh np.ndarray HxWxC (uint8)
    labels_list : list 4 (bboxes, class_ids) tương ứng
    output_size : kích thước output (vuông)
    center_range : khoảng ngẫu nhiên của điểm ghép trung tâm (tỷ lệ)
    pad_color : màu nền nếu vùng ảnh không đủ lớn

    Returns
    -------
    mosaic_image : np.ndarray output_size × output_size
    merged_bboxes : list [xc,yc,w,h] normalized theo mosaic
    merged_class_ids : list int
    """
    assert len(images) == 4 and len(labels_list) == 4, \
        "Mosaic yêu cầu đúng 4 ảnh và 4 nhãn"

    rng = random.Random(seed)

    S = output_size
    # Điểm trung tâm ngẫu nhiên
    cx = int(rng.uniform(center_range[0], center_range[1]) * S)
    cy = int(rng.uniform(center_range[0], center_range[1]) * S)

    # Canvas nền
    mosaic = np.full((S, S, 3), pad_color, dtype=np.uint8)
    merged_bboxes: List[List[float]] = []
    merged_class_ids: List[int] = []

    # Vị trí 4 ảnh: (x1_canvas, y1_canvas, x2_canvas, y2_canvas)
    placement_info = [
        (0,  0,  cx, cy),   # TL — img[0]
        (cx, 0,  S,  cy),   # TR — img[1]
        (0,  cy, cx, S),    # BL — img[2]
        (cx, cy, S,  S),    # BR — img[3]
    ]

    for idx, (img, (bboxes, class_ids), (x1c, y1c, x2c, y2c)) in enumerate(
        zip(images, labels_list, placement_info)
    ):
        h, w = img.shape[:2]
        region_w = x2c - x1c  # chiều rộng vùng canvas
        region_h = y2c - y1c  # chiều cao vùng canvas

        if region_w <= 0 or region_h <= 0:
            continue

        # Scale ảnh vừa vào vùng canvas
        scale = min(region_w / w, region_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Đặt ảnh vào góc của vùng canvas
        # TL: góc dưới-phải; TR: góc dưới-trái; BL: góc trên-phải; BR: góc trên-trái
        if idx == 0:  # TL → đặt sát góc dưới-phải (cx, cy)
            x_offset = x2c - new_w
            y_offset = y2c - new_h
        elif idx == 1:  # TR → đặt sát góc dưới-trái (cx, cy)
            x_offset = x1c
            y_offset = y2c - new_h
        elif idx == 2:  # BL → đặt sát góc trên-phải (cx, cy)
            x_offset = x2c - new_w
            y_offset = y1c
        else:  # BR → đặt sát góc trên-trái (cx, cy)
            x_offset = x1c
            y_offset = y1c

        # Clip về trong canvas
        x_offset = max(x1c, min(x_offset, x2c - new_w))
        y_offset = max(y1c, min(y_offset, y2c - new_h))
        x_end = min(x_offset + new_w, S)
        y_end = min(y_offset + new_h, S)

        # Copy vào mosaic
        mosaic[y_offset:y_end, x_offset:x_end] = resized[:y_end - y_offset, :x_end - x_offset]

        # Cập nhật bboxes
        for (xc_n, yc_n, w_n, h_n), cls_id in zip(bboxes, class_ids):
            # Pixel trong resized
            xc_px = xc_n * new_w
            yc_px = yc_n * new_h
            bw_px = w_n  * new_w
            bh_px = h_n  * new_h

            # Pixel trong mosaic
            xc_mos = xc_px + x_offset
            yc_mos = yc_px + y_offset

            # Clip bounding box vào canvas
            x_min_mos = max(x1c, xc_mos - bw_px / 2)
            y_min_mos = max(y1c, yc_mos - bh_px / 2)
            x_max_mos = min(x2c, xc_mos + bw_px / 2)
            y_max_mos = min(y2c, yc_mos + bh_px / 2)

            bw_clip = x_max_mos - x_min_mos
            bh_clip = y_max_mos - y_min_mos

            if bw_clip < 5 or bh_clip < 5:
                continue  # Bỏ qua bbox quá nhỏ

            # Normalize theo mosaic
            xc_new = ((x_min_mos + x_max_mos) / 2) / S
            yc_new = ((y_min_mos + y_max_mos) / 2) / S
            w_new  = bw_clip / S
            h_new  = bh_clip / S

            # Clip [0,1]
            xc_new = max(0.0, min(1.0, xc_new))
            yc_new = max(0.0, min(1.0, yc_new))
            w_new  = max(0.0, min(1.0, w_new))
            h_new  = max(0.0, min(1.0, h_new))

            merged_bboxes.append([xc_new, yc_new, w_new, h_new])
            merged_class_ids.append(cls_id)

    return mosaic, merged_bboxes, merged_class_ids


# ──────────────────────────────────────────────────────────────────────────────
# MixUp Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def create_mixup(
    img1: np.ndarray,
    labels1: Tuple[List[List[float]], List[int]],
    img2: np.ndarray,
    labels2: Tuple[List[List[float]], List[int]],
    alpha: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Blend 2 ảnh với trọng số lấy mẫu từ phân phối Beta(alpha, alpha).

    λ ~ Beta(alpha, alpha)
    mixed_image = λ * img1 + (1-λ) * img2

    Nhãn: kết hợp tất cả bounding box của cả 2 ảnh (không blend nhãn).
    Ảnh đầu tiên chiếm tỷ lệ λ ≥ 0.5 (được làm ảnh "chính").

    Parameters
    ----------
    img1, img2 : np.ndarray HxWxC (uint8) — cùng kích thước
    labels1, labels2 : (bboxes, class_ids) cho mỗi ảnh
    alpha : tham số Beta distribution (0.8 = blend nhẹ, 0.3 = blend mạnh)

    Returns
    -------
    mixed_image, merged_bboxes, merged_class_ids
    """
    if seed is not None:
        np.random.seed(seed)

    # Đảm bảo cùng kích thước
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

    # Lấy lambda từ Beta distribution
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Đảm bảo lam >= 0.5 (img1 là chính)

    # Blend ảnh
    mixed = (img1.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam))
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)

    # Kết hợp nhãn (không blend, lấy tất cả)
    bboxes1, class_ids1 = labels1
    bboxes2, class_ids2 = labels2
    merged_bboxes    = list(bboxes1)    + list(bboxes2)
    merged_class_ids = list(class_ids1) + list(class_ids2)

    return mixed, merged_bboxes, merged_class_ids


# ──────────────────────────────────────────────────────────────────────────────
# Demo & Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _make_synthetic_image(color: Tuple[int, int, int], size: int = 640) -> np.ndarray:
    """Tạo ảnh synthetic với màu nền và hình chữ nhật giả xe."""
    img = np.full((size, size, 3), (114, 114, 114), dtype=np.uint8)
    # Vẽ "xe"
    cv2.rectangle(img, (50, 80), (200, 200), color, -1)
    cv2.rectangle(img, (300, 150), (550, 350), tuple(max(0, c-50) for c in color), -1)
    return img


def run_demo(save_figures: bool = True) -> None:
    """Demo Mosaic và MixUp trên ảnh synthetic."""
    print("\n=== DEMO: Mosaic & MixUp Augmentation ===\n")

    # Tạo 4 ảnh synthetic
    colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)]
    images = [_make_synthetic_image(c) for c in colors]

    # Nhãn giả: 2 xe mỗi ảnh
    labels = [
        ([
            [0.195, 0.219, 0.234, 0.250],
            [0.664, 0.391, 0.391, 0.313],
        ], [0, 1]),
    ] * 4

    # ── Mosaic ──
    print("[1] Tạo Mosaic từ 4 ảnh...")
    t0 = time.time()
    mosaic_img, mosaic_boxes, mosaic_cls = create_mosaic(images, labels, seed=42)
    t_mosaic = time.time() - t0
    print(f"    Thời gian: {t_mosaic*1000:.1f} ms")
    print(f"    Bboxes: {len(mosaic_boxes)} (gốc: {sum(len(l[0]) for l in labels)})")
    print(f"    Shape: {mosaic_img.shape}")

    # ── MixUp ──
    print("\n[2] Tạo MixUp từ 2 ảnh...")
    t0 = time.time()
    for alpha in [0.8, 0.5, 0.3]:
        mixed_img, mixed_boxes, mixed_cls = create_mixup(
            images[0], labels[0], images[1], labels[1], alpha=alpha, seed=42
        )
        t_mixup = time.time() - t0
        print(f"    alpha={alpha}: {t_mixup*1000:.1f} ms | bboxes={len(mixed_boxes)}")
        t0 = time.time()

    # ── Visualize ──
    if save_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: 4 ảnh gốc và Mosaic
            for i in range(4):
                ax = axes[0][i % 2 + (0 if i < 2 else 0)]  # sẽ dùng row 1
                pass

            # Layout đơn giản hơn
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for i, (img, ax) in enumerate(zip(images, axes[:4])):
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Ảnh gốc {i+1}")
                ax.axis("off")
            plt.suptitle("4 ảnh gốc dùng cho Mosaic", fontsize=13)
            plt.tight_layout()
            import os
            os.makedirs("figures", exist_ok=True)
            plt.savefig("figures/mosaic_source_images.png", dpi=100, bbox_inches="tight")
            plt.close()

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].imshow(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Mosaic (4→1) | {len(mosaic_boxes)} bboxes", fontsize=12)
            axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(mixed_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"MixUp (α=0.3) | {len(mixed_boxes)} bboxes", fontsize=12)
            axes[1].axis("off")
            plt.suptitle("So sánh Mosaic vs MixUp Augmentation", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig("figures/mosaic_vs_mixup.png", dpi=120, bbox_inches="tight")
            plt.close()
            print("\n[Saved] figures/mosaic_vs_mixup.png")

        except ImportError:
            print("[WARN] matplotlib chưa cài — bỏ qua visualize")

    print("\n✅ Demo hoàn thành!")


def run_benchmark(input_dir: str, n: int = 100, seed: int = 42) -> None:
    """
    Benchmark Mosaic và MixUp trên n ảnh thực từ input_dir.
    In thống kê: thời gian trung bình, số bbox trung bình.
    """
    img_dir = Path(input_dir) / "images"
    lbl_dir = Path(input_dir) / "labels"

    image_files = sorted(
        p for p in img_dir.glob("*.jpg")
        if p.is_file()
    )[:n * 4]  # Cần đủ ảnh cho mosaic

    if len(image_files) < 4:
        print(f"[ERROR] Cần ≥ 4 ảnh trong {img_dir}")
        return

    rng = random.Random(seed)
    print(f"\n=== Benchmark trên {min(n, len(image_files)//4)} mẫu ===")

    # ── Mosaic benchmark ──
    t0 = time.time()
    total_boxes = 0
    count = 0

    for i in range(0, min(n * 4, len(image_files)), 4):
        batch_imgs = []
        batch_labels = []

        for j in range(4):
            img_path = image_files[i + j]
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.full((640, 640, 3), 114, dtype=np.uint8)
            bboxes, class_ids = read_yolo_label(lbl_path)
            batch_imgs.append(img)
            batch_labels.append((bboxes, class_ids))

        _, boxes, _ = create_mosaic(batch_imgs, batch_labels, seed=seed + i)
        total_boxes += len(boxes)
        count += 1

    t_mosaic = time.time() - t0
    print(f"\n[Mosaic]")
    print(f"  Số mẫu     : {count}")
    print(f"  Thời gian   : {t_mosaic:.3f}s tổng | {t_mosaic/count*1000:.1f} ms/mẫu")
    print(f"  Bbox TB     : {total_boxes/count:.1f} bboxes/ảnh")

    # ── MixUp benchmark ──
    t0 = time.time()
    total_boxes = 0
    count = 0

    for i in range(0, min(n * 2, len(image_files)), 2):
        img1 = cv2.imread(str(image_files[i]))
        img2 = cv2.imread(str(image_files[i + 1]))
        lbl1 = read_yolo_label(lbl_dir / (image_files[i].stem + ".txt"))
        lbl2 = read_yolo_label(lbl_dir / (image_files[i + 1].stem + ".txt"))

        if img1 is None or img2 is None:
            continue

        _, boxes, _ = create_mixup(img1, lbl1, img2, lbl2, alpha=0.8)
        total_boxes += len(boxes)
        count += 1

    t_mixup = time.time() - t0
    print(f"\n[MixUp]")
    print(f"  Số mẫu     : {count}")
    print(f"  Thời gian   : {t_mixup:.3f}s tổng | {t_mixup/count*1000:.1f} ms/mẫu")
    print(f"  Bbox TB     : {total_boxes/count:.1f} bboxes/ảnh")

    print(f"\n📊 So sánh tốc độ:")
    print(f"  Mosaic: {t_mosaic/count*1000:.1f} ms/mẫu")
    print(f"  MixUp : {t_mixup/count*1000:.1f} ms/mẫu")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Thử nghiệm Mosaic và MixUp Augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo",    action="store_true", help="Chạy demo với ảnh synthetic")
    parser.add_argument("--compare", action="store_true", help="Chạy benchmark so sánh")
    parser.add_argument("--input",   default="data/train", help="Thư mục dữ liệu cho benchmark")
    parser.add_argument("--n",       type=int, default=100,  help="Số mẫu benchmark")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--no-save", action="store_true",    help="Không lưu figure")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo or not (args.demo or args.compare):
        run_demo(save_figures=not args.no_save)

    if args.compare:
        run_benchmark(args.input, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()

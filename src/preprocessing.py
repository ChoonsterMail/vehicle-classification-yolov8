"""
src/preprocessing.py
=====================
Phụ trách: Nguyễn Vũ Tùng — Tuần 3

Script tiền xử lý ảnh tự động theo chuẩn YOLO:
  • Letterboxing: resize về 640×640 không méo, thêm padding xám (114,114,114)
  • Cập nhật tọa độ bounding box sau khi resize
  • Normalization: [0,255] → [0.0, 1.0] (chuẩn PyTorch, lưu dưới dạng float32 .npy)
  • Hỗ trợ xử lý hàng loạt với multiprocessing

Cách dùng:
    python src/preprocessing.py --input data/train/images --output data/train_processed
    python src/preprocessing.py --verify  # Kiểm tra ảnh đã xử lý
    python src/preprocessing.py --demo    # Demo letterbox trên ảnh ngẫu nhiên
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SIZE = 640          # Kích thước output mặc định (640×640)
PAD_COLOR   = (114, 114, 114)   # Màu padding chuẩn YOLOv8
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# Letterboxing
# ──────────────────────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    new_shape: int = TARGET_SIZE,
    color: Tuple[int, int, int] = PAD_COLOR,
    auto: bool = False,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize ảnh về new_shape × new_shape giữ nguyên tỷ lệ, thêm padding.

    Parameters
    ----------
    image     : np.ndarray HxWxC (uint8)
    new_shape : kích thước output (vuông)
    color     : màu padding (B,G,R)
    auto      : nếu True, padding tối thiểu chia hết cho stride
    stride    : dùng khi auto=True

    Returns
    -------
    padded_image : np.ndarray new_shape × new_shape (uint8)
    ratio        : tỷ lệ scale (< 1.0 = thu nhỏ)
    pad          : (pad_w, pad_h) — padding thêm vào mỗi phía
    """
    h, w = image.shape[:2]

    # Tỷ lệ scale (lấy min để không vượt target)
    ratio = min(new_shape / h, new_shape / w)

    # Kích thước mới sau scale
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tính padding (chia đều 2 phía)
    pad_w = (new_shape - new_w) / 2
    pad_h = (new_shape - new_h) / 2

    if auto:
        # Điều chỉnh để chia hết cho stride
        pad_w = pad_w - (pad_w % stride)
        pad_h = pad_h - (pad_h % stride)

    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, ratio, (int(left), int(top))


def update_bboxes_after_letterbox(
    bboxes: list,
    original_shape: Tuple[int, int],
    new_shape: int = TARGET_SIZE,
) -> list:
    """
    Cập nhật tọa độ bounding box YOLO sau letterboxing.

    Parameters
    ----------
    bboxes : list of [x_center, y_center, width, height] (normalized)
    original_shape : (height, width) của ảnh gốc
    new_shape : kích thước target

    Returns
    -------
    list of [x_center, y_center, width, height] (normalized theo ảnh mới)

    Ghi chú:
        Sau letterbox, tọa độ pixel thực thay đổi vì có padding.
        Cần chuyển về pixel, áp dụng scale + offset, rồi normalize lại.
    """
    h_orig, w_orig = original_shape
    ratio = min(new_shape / h_orig, new_shape / w_orig)
    new_w = int(round(w_orig * ratio))
    new_h = int(round(h_orig * ratio))
    pad_w = (new_shape - new_w) / 2
    pad_h = (new_shape - new_h) / 2

    updated = []
    for bbox in bboxes:
        xc_n, yc_n, w_n, h_n = bbox

        # Chuyển về pixel trong ảnh gốc
        xc_px = xc_n * w_orig
        yc_px = yc_n * h_orig
        w_px  = w_n  * w_orig
        h_px  = h_n  * h_orig

        # Áp dụng scale và offset padding
        xc_new = (xc_px * ratio + pad_w) / new_shape
        yc_new = (yc_px * ratio + pad_h) / new_shape
        w_new  = (w_px  * ratio)         / new_shape
        h_new  = (h_px  * ratio)         / new_shape

        # Clip về [0,1]
        xc_new = max(0.0, min(1.0, xc_new))
        yc_new = max(0.0, min(1.0, yc_new))
        w_new  = max(0.0, min(1.0, w_new))
        h_new  = max(0.0, min(1.0, h_new))

        if w_new > 0 and h_new > 0:
            updated.append([xc_new, yc_new, w_new, h_new])
        else:
            updated.append(None)  # bbox quá nhỏ sau scale

    return updated


# ──────────────────────────────────────────────────────────────────────────────
# Normalize
# ──────────────────────────────────────────────────────────────────────────────

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values từ [0,255] uint8 về [0.0, 1.0] float32.
    Đây là chuẩn PyTorch (trước khi áp dụng ImageNet mean/std).

    Parameters
    ----------
    image : np.ndarray HxWxC uint8

    Returns
    -------
    np.ndarray HxWxC float32 trong [0.0, 1.0]
    """
    return image.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Xử lý 1 file
# ──────────────────────────────────────────────────────────────────────────────

def process_single_image(
    img_path: Path,
    lbl_path: Optional[Path],
    out_img_dir: Path,
    out_lbl_dir: Optional[Path],
    target_size: int = TARGET_SIZE,
    save_normalized: bool = False,
) -> dict:
    """
    Xử lý 1 ảnh: letterbox + optionally normalize + lưu.

    Returns
    -------
    dict với status, shape, ratio
    """
    # Đọc ảnh
    image = cv2.imread(str(img_path))
    if image is None:
        return {"status": "error", "reason": f"Không đọc được ảnh: {img_path}"}

    h_orig, w_orig = image.shape[:2]

    # Letterbox
    padded, ratio, pad = letterbox(image, new_shape=target_size)
    assert padded.shape[:2] == (target_size, target_size), \
        f"Letterbox output sai shape: {padded.shape}"

    # Lưu ảnh đã letterbox
    out_img_path = out_img_dir / img_path.name
    cv2.imwrite(str(out_img_path), padded, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Optionally lưu float32 normalized
    if save_normalized:
        normed = normalize_image(padded)
        npy_path = out_img_dir / (img_path.stem + ".npy")
        np.save(str(npy_path), normed)

    # Xử lý nhãn
    if lbl_path and lbl_path.exists() and out_lbl_dir:
        bboxes, class_ids = _read_yolo_label(lbl_path)
        if bboxes:
            updated = update_bboxes_after_letterbox(bboxes, (h_orig, w_orig), target_size)
            out_lbl_path = out_lbl_dir / lbl_path.name
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                for cls_id, bbox in zip(class_ids, updated):
                    if bbox is not None:
                        xc, yc, w, h = bbox
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return {
        "status": "ok",
        "file": img_path.name,
        "orig_shape": (h_orig, w_orig),
        "ratio": ratio,
        "pad": pad,
    }


def _read_yolo_label(label_path: Path):
    """Helper đọc nhãn YOLO → (bboxes, class_ids)."""
    bboxes, class_ids = [], []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            bboxes.append([xc, yc, w, h])
            class_ids.append(cls_id)
    return bboxes, class_ids


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_batch(
    image_dir: str,
    label_dir: Optional[str] = None,
    output_dir: str = "data/processed",
    target_size: int = TARGET_SIZE,
    num_workers: int = 4,
    save_normalized: bool = False,
) -> None:
    """
    Xử lý hàng loạt: letterbox tất cả ảnh trong image_dir.

    Parameters
    ----------
    image_dir : thư mục ảnh gốc
    label_dir : thư mục nhãn YOLO (None = bỏ qua nhãn)
    output_dir : thư mục output
    target_size : kích thước target (mặc định 640)
    num_workers : số process song song
    save_normalized : lưu thêm file .npy float32
    """
    img_dir = Path(image_dir)
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels" if label_dir else None

    out_img_dir.mkdir(parents=True, exist_ok=True)
    if out_lbl_dir:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    lbl_dir = Path(label_dir) if label_dir else None

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"[WARN] Không tìm thấy ảnh nào trong: {img_dir}")
        return

    print(f"\n{'=' * 55}")
    print(f"  Letterbox Preprocessing ({target_size}×{target_size})")
    print(f"  Input : {img_dir.resolve()}")
    print(f"  Output: {Path(output_dir).resolve()}")
    print(f"  Tổng  : {len(image_files)} ảnh | Workers: {num_workers}")
    print(f"{'=' * 55}\n")

    t0 = time.time()
    ok_count = 0
    err_count = 0

    # Xây dựng task list
    tasks = []
    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt") if lbl_dir else None
        tasks.append((img_path, lbl_path, out_img_dir, out_lbl_dir, target_size, save_normalized))

    if num_workers > 1 and len(tasks) > 10:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_image, *task): task[0].name
                for task in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["status"] == "ok":
                    ok_count += 1
                else:
                    err_count += 1
                    print(f"  [ERR] {result.get('reason', '')}")

                if i % 100 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    fps = i / elapsed if elapsed > 0 else 0
                    print(f"  [{i:5d}/{len(tasks)}] {fps:.1f} ảnh/giây", end="\r")
    else:
        for i, task in enumerate(tasks, 1):
            result = process_single_image(*task)
            if result["status"] == "ok":
                ok_count += 1
            else:
                err_count += 1
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                fps = i / elapsed if elapsed > 0 else 0
                print(f"  [{i:5d}/{len(tasks)}] {fps:.1f} ảnh/giây", end="\r")

    elapsed = time.time() - t0
    print(f"\n\n✅ Hoàn thành!")
    print(f"   Thành công : {ok_count}")
    print(f"   Lỗi        : {err_count}")
    print(f"   Thời gian  : {elapsed:.1f}s ({ok_count / elapsed:.1f} ảnh/giây)")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(save_demo: bool = True) -> None:
    """Demo letterbox trên ảnh synthetic."""
    print("\n=== DEMO: Letterboxing ===\n")

    # Tạo ảnh giả lập 1920×1080 (16:9)
    image = np.full((1080, 1920, 3), (114, 114, 114), dtype=np.uint8)
    cv2.rectangle(image, (300, 200), (800, 600), (0, 100, 200), -1)   # xe xanh
    cv2.rectangle(image, (1000, 400), (1500, 700), (200, 50, 0), -1)  # xe đỏ
    cv2.putText(image, "1920x1080 original", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    print(f"Ảnh gốc: {image.shape}")

    # Letterbox
    padded, ratio, pad = letterbox(image, new_shape=640)
    print(f"Sau letterbox: {padded.shape}, ratio={ratio:.4f}, pad={pad}")

    # Kiểm tra bboxes
    bboxes = [[0.2865, 0.3704, 0.2604, 0.3704], [0.6510, 0.5093, 0.2604, 0.2778]]
    updated = update_bboxes_after_letterbox(bboxes, (1080, 1920), 640)
    print(f"\nBboxes gốc    : {bboxes}")
    print(f"Bboxes updated: {updated}")

    # Normalize
    normed = normalize_image(padded)
    print(f"\nNormalize: dtype={normed.dtype}, min={normed.min():.3f}, max={normed.max():.3f}")

    if save_demo:
        os.makedirs("figures", exist_ok=True)
        # Vẽ demo so sánh
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.imshow(cv2.cvtColor(image[:540:, ::2], cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Ảnh gốc: 1920×1080", fontsize=12)
        ax1.axis("off")
        ax2.imshow(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Sau Letterbox: 640×640 (ratio={ratio:.3f})", fontsize=12)
        ax2.axis("off")
        plt.suptitle("Minh họa Letterboxing", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("figures/letterbox_demo.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n[Saved] figures/letterbox_demo.png")

    print("\n✅ Demo hoàn thành!")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Letterboxing preprocessing cho YOLO dataset (640×640)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", default="data/train/images", help="Thư mục ảnh gốc")
    parser.add_argument("--labels", "-l", default=None,                help="Thư mục nhãn YOLO (optional)")
    parser.add_argument("--output", "-o", default="data/train_processed", help="Thư mục output")
    parser.add_argument("--size",   type=int, default=640,              help="Kích thước target")
    parser.add_argument("--workers", type=int, default=4,               help="Số process song song")
    parser.add_argument("--save-npy", action="store_true",              help="Lưu thêm file .npy float32")
    parser.add_argument("--demo",   action="store_true",                help="Chạy demo trên ảnh synthetic")
    parser.add_argument("--verify", action="store_true",                help="Kiểm tra ảnh output có shape đúng không")
    return parser.parse_args()


def verify_output(output_dir: str, expected_size: int = 640) -> None:
    """Kiểm tra tất cả ảnh output có shape TARGET_SIZE × TARGET_SIZE."""
    img_dir = Path(output_dir) / "images"
    errors = 0
    checked = 0
    for p in img_dir.glob("*.jpg"):
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (expected_size, expected_size):
            print(f"[ERR] {p.name}: shape={img.shape if img is not None else 'None'}")
            errors += 1
        checked += 1
    print(f"\nKiểm tra {checked} ảnh: {errors} lỗi")
    if errors == 0:
        print("✅ Tất cả ảnh có shape đúng!")


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return

    if args.verify:
        verify_output(args.output, args.size)
        return

    preprocess_batch(
        image_dir=args.input,
        label_dir=args.labels,
        output_dir=args.output,
        target_size=args.size,
        num_workers=args.workers,
        save_normalized=args.save_npy,
    )


if __name__ == "__main__":
    main()

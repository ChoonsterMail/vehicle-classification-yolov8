"""
src/augmentation.py
====================
Phụ trách: Đỗ Xuân Bách — Tuần 3

Module tăng cường dữ liệu ảnh dùng thư viện Albumentations.
Tương thích với định dạng bounding box YOLO (normalized xywh).

Kỹ thuật augmentation:
  • RandomBrightnessContrast   — bù sáng/tối camera giám sát
  • CLAHE                      — tăng tương phản vùng tối
  • HorizontalFlip             — đối xứng chiều đi
  • Blur / MotionBlur          — mô phỏng xe chuyển động
  • HueSaturationValue         — đa dạng màu sắc xe
  • RandomRain / RandomFog     — điều kiện thời tiết xấu (p thấp)

Cách dùng:
    python src/augmentation.py --input data/train --output data/train_aug --n 3
    python src/augmentation.py --demo  # Chạy demo không cần dữ liệu thực
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Albumentations import (graceful fallback nếu chưa cài)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import albumentations as A
    from albumentations.core.composition import Compose
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations chưa được cài đặt. Chạy: pip install albumentations")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# VehicleAugmentor
# ──────────────────────────────────────────────────────────────────────────────

class VehicleAugmentor:
    """
    Bộ tăng cường dữ liệu ảnh phương tiện giao thông.

    Parameters
    ----------
    mode : 'standard' | 'heavy' | 'light'
        Mức độ augmentation:
        - 'light'    : Chỉ flip + brightness (an toàn)
        - 'standard' : Đầy đủ, khuyến nghị mặc định
        - 'heavy'    : Thêm rain, fog, mạnh hơn

    seed : int, optional
        Seed ngẫu nhiên (None = không cố định)
    """

    def __init__(self, mode: str = "standard", seed: Optional[int] = None):
        if not HAS_ALBUMENTATIONS:
            raise RuntimeError("Cài đặt albumentations trước: pip install albumentations")

        self.mode = mode
        self.seed = seed
        self._transform = self._build_transform(mode)

    def _build_transform(self, mode: str) -> "Compose":
        """Xây dựng pipeline Albumentations theo mode."""
        common = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4
            ),
        ]

        if mode == "light":
            transforms_list = common[:2]

        elif mode == "standard":
            transforms_list = common + [
                A.OneOf([
                    A.Blur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT, value=114, p=0.3
                ),
            ]

        elif mode == "heavy":
            transforms_list = common + [
                A.OneOf([
                    A.Blur(blur_limit=7, p=1.0),
                    A.MotionBlur(blur_limit=11, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                ], p=0.4),
                A.GaussNoise(var_limit=(10, 80), p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT, value=114, p=0.4
                ),
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=15, drop_width=1,
                    drop_color=(200, 200, 200), blur_value=3,
                    brightness_coefficient=0.9, rain_type=None, p=0.1
                ),
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.1),
                A.RandomShadow(p=0.15),
            ]

        else:
            raise ValueError(f"mode phải là 'light' | 'standard' | 'heavy', nhận: '{mode}'")

        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format="yolo",           # [x_center, y_center, width, height] normalized
                label_fields=["class_ids"],
                min_area=100,            # Loại bỏ bbox quá nhỏ sau augment
                min_visibility=0.3,      # Loại bỏ bbox bị che > 70%
            ),
        )

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_ids: List[int],
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Áp dụng augmentation lên 1 ảnh và các bounding box.

        Parameters
        ----------
        image : np.ndarray HxWxC (uint8, BGR hoặc RGB)
        bboxes : list of [x_center, y_center, width, height] (normalized [0,1])
        class_ids : list of int, cùng chiều dài với bboxes

        Returns
        -------
        (aug_image, aug_bboxes, aug_class_ids)
        """
        result = self._transform(
            image=image,
            bboxes=bboxes if bboxes else [],
            class_ids=class_ids if class_ids else [],
        )
        return result["image"], list(result["bboxes"]), list(result["class_ids"])

    # ── Batch augment ─────────────────────────────────────────────────────────

    def augment_image_file(
        self,
        img_path: Path,
        lbl_path: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        n_augment: int = 3,
        prefix: str = "aug",
    ) -> int:
        """
        Đọc 1 ảnh + nhãn, tạo n_augment phiên bản augmented và lưu.

        Returns
        -------
        int : số ảnh đã lưu thành công
        """
        # Đọc ảnh
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Không đọc được: {img_path}")
            return 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Đọc nhãn YOLO
        bboxes, class_ids = parse_yolo_label(lbl_path)

        saved = 0
        for i in range(n_augment):
            aug_img, aug_boxes, aug_cls = self.apply(image, bboxes, class_ids)

            # Tên file output
            stem = f"{prefix}_{i:02d}_{img_path.stem}"
            out_img_path = out_img_dir / f"{stem}.jpg"
            out_lbl_path = out_lbl_dir / f"{stem}.txt"

            # Lưu ảnh
            aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Lưu nhãn
            write_yolo_label(out_lbl_path, aug_boxes, aug_cls)
            saved += 1

        return saved


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_yolo_label(
    label_path: Path,
) -> Tuple[List[List[float]], List[int]]:
    """Đọc file nhãn YOLO → (bboxes, class_ids)."""
    bboxes, class_ids = [], []
    if not label_path.exists():
        return bboxes, class_ids

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Clip về [0,1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            if w > 0 and h > 0:
                bboxes.append([xc, yc, w, h])
                class_ids.append(cls_id)

    return bboxes, class_ids


def write_yolo_label(
    label_path: Path,
    bboxes: List[List[float]],
    class_ids: List[int],
) -> None:
    """Ghi file nhãn YOLO từ (bboxes, class_ids)."""
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, (xc, yc, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích: augment_dataset
# ──────────────────────────────────────────────────────────────────────────────

def augment_dataset(
    input_dir: str,
    output_dir: str,
    n_augment: int = 3,
    mode: str = "standard",
    seed: int = 42,
    copy_originals: bool = True,
) -> None:
    """
    Tăng cường toàn bộ dataset: đọc ảnh từ input_dir, sinh thêm n_augment ảnh,
    lưu vào output_dir. Có thể copy ảnh gốc vào output_dir cùng.

    Parameters
    ----------
    input_dir : thư mục có images/ và labels/
    output_dir : thư mục output
    n_augment : số ảnh augment mỗi ảnh gốc
    mode : 'light' | 'standard' | 'heavy'
    seed : random seed
    copy_originals : nếu True, copy ảnh gốc vào output_dir
    """
    import shutil

    inp = Path(input_dir)
    out = Path(output_dir)

    inp_img = inp / "images"
    inp_lbl = inp / "labels"
    out_img = out / "images"
    out_lbl = out / "labels"

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    augmentor = VehicleAugmentor(mode=mode, seed=seed)

    image_files = sorted(
        p for p in inp_img.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {inp_img}")

    total_saved = 0

    for img_path in image_files:
        lbl_path = inp_lbl / (img_path.stem + ".txt")

        # Copy ảnh gốc
        if copy_originals:
            shutil.copy2(img_path, out_img / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)

        # Augment
        saved = augmentor.augment_image_file(
            img_path=img_path,
            lbl_path=lbl_path,
            out_img_dir=out_img,
            out_lbl_dir=out_lbl,
            n_augment=n_augment,
        )
        total_saved += saved

    orig_count = len(image_files) if copy_originals else 0
    print(f"\n✅ Augmentation hoàn thành!")
    print(f"   Ảnh gốc copy: {orig_count}")
    print(f"   Ảnh augmented: {total_saved}")
    print(f"   Tổng output: {orig_count + total_saved}")
    print(f"   Thư mục: {out.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo (không cần dữ liệu thực)
# ──────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Chạy demo trên ảnh synthetic để minh họa augmentation."""
    print("\n=== DEMO: Augmentation trên ảnh ngẫu nhiên ===\n")

    # Tạo ảnh giả lập (gradient màu)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(50, 200, 640, dtype=np.uint8)
    img[:, :, 1] = np.linspace(100, 150, 480, dtype=np.uint8).reshape(-1, 1)
    img[:, :, 2] = 120

    # Vẽ hình chữ nhật giả xe
    cv2.rectangle(img, (100, 150), (300, 280), (0, 120, 255), -1)
    cv2.rectangle(img, (400, 200), (580, 320), (255, 80, 0), -1)

    bboxes = [
        [0.3125, 0.4479, 0.3125, 0.2708],  # xe 1
        [0.7656, 0.5417, 0.2813, 0.2500],  # xe 2
    ]
    class_ids = [0, 1]  # car, motorcycle

    print(f"Ảnh gốc: {img.shape}, bboxes: {bboxes}")

    for mode in ("light", "standard", "heavy"):
        aug = VehicleAugmentor(mode=mode, seed=42)
        aug_img, aug_boxes, aug_cls = aug.apply(img.copy(), bboxes, class_ids)
        print(f"[{mode:8s}] boxes sau: {len(aug_boxes)}, classes: {aug_cls}")

    print("\n✅ Demo thành công! Augmentor hoạt động đúng.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tăng cường dữ liệu ảnh phương tiện giao thông (Albumentations)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", default="data/train",     help="Thư mục input (có images/ + labels/)")
    parser.add_argument("--output", "-o", default="data/train_aug", help="Thư mục output")
    parser.add_argument("--n",  type=int, default=3,  help="Số ảnh augment mỗi ảnh gốc")
    parser.add_argument("--mode",   default="standard", choices=["light", "standard", "heavy"])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--no-copy-originals", action="store_true",
                        help="Không copy ảnh gốc vào output")
    parser.add_argument("--demo",   action="store_true", help="Chạy demo không cần dữ liệu")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        if not HAS_ALBUMENTATIONS:
            print("[ERROR] Cần cài albumentations: pip install albumentations")
            return
        run_demo()
        return

    if not HAS_ALBUMENTATIONS:
        print("[ERROR] Cần cài albumentations: pip install albumentations")
        return

    print(f"\n{'=' * 50}")
    print(f"  Vehicle Data Augmentation")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Mode  : {args.mode}")
    print(f"  n_aug : {args.n} ảnh/ảnh gốc")
    print(f"{'=' * 50}\n")

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        n_augment=args.n,
        mode=args.mode,
        seed=args.seed,
        copy_originals=not args.no_copy_originals,
    )


if __name__ == "__main__":
    main()

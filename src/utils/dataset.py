"""
src/utils/dataset.py
====================
VehicleDataset — PyTorch Dataset dùng chung cho train_classifier.py và train_mobilenet.py.

Cấu trúc thư mục kỳ vọng:
    data/
    ├── train/
    │   ├── images/   *.jpg
    │   └── labels/   *.txt  (YOLO format)
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Nhãn YOLO (.txt): mỗi dòng là <class_id> <x_center> <y_center> <width> <height>
Với bài toán CLASSIFICATION, ta lấy class_id của đối tượng đầu tiên trong file nhãn.
"""

import os
import glob
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Tên 5 lớp phương tiện
CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
NUM_CLASSES = 5

# ──────────────────────────────────────────────────────────────────────────────
# Hàm đọc nhãn YOLO
# ──────────────────────────────────────────────────────────────────────────────

def read_yolo_label(label_path: str) -> int:
    """
    Đọc file nhãn YOLO và trả về class_id của đối tượng đầu tiên.
    Trả về -1 nếu file rỗng (ảnh không có phương tiện).
    """
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return -1
    # Lấy class_id đầu tiên
    class_id = int(lines[0].split()[0])
    return class_id


# ──────────────────────────────────────────────────────────────────────────────
# VehicleDataset
# ──────────────────────────────────────────────────────────────────────────────

class VehicleDataset(Dataset):
    """
    PyTorch Dataset cho bài toán phân loại ảnh phương tiện giao thông.

    Parameters
    ----------
    root_dir : str
        Thư mục gốc của split (ví dụ: 'data/train').
        Phải có 2 thư mục con: images/ và labels/.
    transform : callable, optional
        Torchvision transforms áp dụng lên ảnh PIL.
    skip_empty : bool
        Nếu True, bỏ qua ảnh có file nhãn rỗng (không có phương tiện).
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        skip_empty: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.skip_empty = skip_empty

        image_dir = self.root_dir / "images"
        label_dir = self.root_dir / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục nhãn: {label_dir}")

        # Liệt kê tất cả ảnh và ghép với file nhãn tương ứng
        self.samples: List[Tuple[Path, int]] = []

        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_paths = []
        for ext in image_exts:
            image_paths.extend(image_dir.glob(ext))
        image_paths = sorted(image_paths)

        missing_labels = 0
        empty_labels = 0

        for img_path in image_paths:
            label_path = label_dir / (img_path.stem + ".txt")

            if not label_path.exists():
                missing_labels += 1
                continue

            class_id = read_yolo_label(str(label_path))

            if class_id == -1:
                empty_labels += 1
                if skip_empty:
                    continue
                else:
                    # Dùng class_id=0 làm mặc định — không lý tưởng
                    class_id = 0

            if not (0 <= class_id < NUM_CLASSES):
                print(f"[WARN] class_id={class_id} nằm ngoài [{0},{NUM_CLASSES}): {img_path.name}")
                continue

            self.samples.append((img_path, class_id))

        print(f"[Dataset] {root_dir}: {len(self.samples)} ảnh hợp lệ "
              f"| bỏ qua: {missing_labels} thiếu nhãn, {empty_labels} nhãn rỗng")

    # ── Thống kê ──────────────────────────────────────────────────────────────

    def class_counts(self) -> dict:
        """Trả về dict {class_name: count} cho từng lớp."""
        from collections import Counter
        counter = Counter(label for _, label in self.samples)
        return {CLASS_NAMES[i]: counter.get(i, 0) for i in range(NUM_CLASSES)}

    # ── PyTorch interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_id = self.samples[idx]

        # Đọc ảnh
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_id


# ──────────────────────────────────────────────────────────────────────────────
# Default Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_default_transforms(split: str = "train", img_size: int = 224):
    """
    Trả về transform mặc định cho từng split.

    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    img_size : kích thước ảnh đầu vào mô hình (224 cho ResNet/MobileNet)
    """
    # Chuẩn hóa ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train / val / test.

    Parameters
    ----------
    data_dir : str
        Thư mục gốc chứa train/, val/, test/
    batch_size : int
    img_size : int
        Kích thước ảnh đầu vào (224 cho ResNet/MobileNet)
    num_workers : int
    pin_memory : bool

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_dataset = VehicleDataset(
        root_dir=data_dir / "train",
        transform=get_default_transforms("train", img_size),
    )
    val_dataset = VehicleDataset(
        root_dir=data_dir / "val",
        transform=get_default_transforms("val", img_size),
    )
    test_dataset = VehicleDataset(
        root_dir=data_dir / "test",
        transform=get_default_transforms("test", img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Quick test (chạy standalone)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"\n=== Kiểm tra VehicleDataset từ: {data_dir} ===\n")

    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"[SKIP] {split_dir} không tồn tại")
            continue
        ds = VehicleDataset(split_dir, transform=get_default_transforms(split))
        print(f"  Phân bố lớp: {ds.class_counts()}\n")

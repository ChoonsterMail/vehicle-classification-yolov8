"""
src/train_classifier.py
========================
Phụ trách: Đỗ Xuân Bách — Tuần 4

Huấn luyện mô hình phân loại ảnh ResNet50 dùng Transfer Learning từ ImageNet.

Chiến lược 2 giai đoạn:
  Giai đoạn 1 (Warm-up): Freeze toàn bộ backbone, chỉ train lớp FC cuối (10 epoch)
  Giai đoạn 2 (Fine-tune): Unfreeze toàn bộ, train với LR thấp hơn 10× (40 epoch)

Cách dùng:
    python src/train_classifier.py --model resnet50 --epochs 50 --batch-size 32
    python src/train_classifier.py --model resnet50 --data-dir data/ --epochs 50
    python src/train_classifier.py --demo  # Chạy demo 2 epoch với dữ liệu giả
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torchvision import models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch chưa cài đặt: pip install torch torchvision")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
NUM_CLASSES = 5


# ──────────────────────────────────────────────────────────────────────────────
# Model Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> "nn.Module":
    """
    Xây dựng ResNet50 với đầu phân loại tùy chỉnh.
    Thay lớp FC cuối (1000 → num_classes).
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Thay lớp FC cuối
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def freeze_backbone(model: "nn.Module") -> None:
    """Freeze toàn bộ backbone, chỉ để fc layer trainable."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_all(model: "nn.Module") -> None:
    """Unfreeze toàn bộ model cho fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Unfreeze] Trainable params: {trainable:,}")


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: "nn.Module",
    loader,
    criterion: "nn.Module",
    optimizer: "optim.Optimizer",
    device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Huấn luyện 1 epoch.

    Returns
    -------
    (avg_loss, accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            batch_acc = (preds == labels).float().mean().item() * 100
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={batch_acc:.1f}%", end="\r")

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_one_epoch(
    model: "nn.Module",
    loader,
    criterion: "nn.Module",
    device,
) -> Tuple[float, float]:
    """Đánh giá 1 epoch (không backward)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Ghi log huấn luyện ra file CSV và lưu history."""

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List] = {
            "epoch": [], "phase": [],
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "lr": [],
        }
        self._file = open(self.save_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["epoch", "phase", "train_loss", "val_loss",
                        "train_acc", "val_acc", "lr"]
        )
        self._writer.writeheader()

    def log(self, epoch, phase, train_loss, val_loss, train_acc, val_acc, lr):
        row = dict(epoch=epoch, phase=phase,
                   train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
                   train_acc=round(train_acc, 4),   val_acc=round(val_acc, 4),
                   lr=round(lr, 8))
        self._writer.writerow(row)
        self._file.flush()
        for k, v in row.items():
            self.history.setdefault(k, []).append(v)

    def close(self):
        self._file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────────────────────

def train_resnet50(
    data_dir: str,
    save_dir: str = "models/classification",
    epochs: int = 50,
    batch_size: int = 32,
    warmup_epochs: int = 10,
    lr_warmup: float = 1e-3,
    lr_finetune: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    img_size: int = 224,
    num_workers: int = 4,
    device_str: str = "auto",
    pretrained: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Huấn luyện ResNet50 với Transfer Learning — 2 giai đoạn.

    Returns
    -------
    dict history {'train_loss': [...], 'val_loss': [...], 'train_acc': [...], 'val_acc': [...]}
    """
    if not HAS_TORCH:
        raise RuntimeError("Cài đặt PyTorch trước: pip install torch torchvision")

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"\n{'=' * 60}")
    print(f"  Huấn luyện ResNet50 — Transfer Learning")
    print(f"  Device   : {device}")
    print(f"  Data dir : {data_dir}")
    print(f"  Epochs   : {epochs} ({warmup_epochs} warmup + {epochs-warmup_epochs} finetune)")
    print(f"  Batch    : {batch_size}")
    print(f"{'=' * 60}\n")

    # DataLoaders
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.dataset import get_dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir, batch_size=batch_size, img_size=img_size, num_workers=num_workers
    )

    # Model
    model = build_resnet50(NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    # Loss với Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    log_path = save_dir / "resnet50_training_log.csv"
    logger = TrainingLogger(str(log_path))

    best_val_acc = 0.0
    best_epoch = 0

    # ── Giai đoạn 1: Warmup (Freeze backbone) ──────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  GIAI ĐOẠN 1: Warm-up ({warmup_epochs} epochs)")
    print(f"  LR = {lr_warmup} | Chỉ train lớp FC")
    print(f"{'─'*50}\n")

    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_warmup, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=lr_warmup * 0.1)

    for epoch in range(1, warmup_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc     = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.log(epoch, "warmup", train_loss, val_loss, train_acc, val_acc, lr_now)

        print(f"  Epoch {epoch:3d}/{warmup_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
              f"lr={lr_now:.2e} | {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, save_dir / "resnet50_best.pth")
            print(f"  ✅ Best model saved (val_acc={val_acc:.2f}%)")

    # ── Giai đoạn 2: Fine-tuning (Unfreeze all) ────────────────────────────
    finetune_epochs = epochs - warmup_epochs
    print(f"\n{'─'*50}")
    print(f"  GIAI ĐOẠN 2: Fine-tuning ({finetune_epochs} epochs)")
    print(f"  LR = {lr_finetune} | Train toàn bộ model")
    print(f"{'─'*50}\n")

    unfreeze_all(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_finetune, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=lr_finetune * 0.01)

    for epoch in range(warmup_epochs + 1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc     = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.log(epoch, "finetune", train_loss, val_loss, train_acc, val_acc, lr_now)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
              f"lr={lr_now:.2e} | {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, save_dir / "resnet50_best.pth")
            print(f"  ✅ Best model saved (val_acc={val_acc:.2f}%)")

    # Lưu model cuối
    torch.save(model.state_dict(), save_dir / "resnet50_last.pth")

    logger.close()

    # Lưu summary
    summary = {
        "model": "resnet50",
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "params": {
            "lr_warmup": lr_warmup,
            "lr_finetune": lr_finetune,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
        }
    }
    with open(save_dir / "resnet50_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Huấn luyện hoàn thành!")
    print(f"  Best val accuracy : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Model lưu tại     : {save_dir.resolve()}")
    print(f"{'=' * 60}")

    return logger.history


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Demo 2 epoch với dữ liệu giả lập để kiểm tra pipeline."""
    if not HAS_TORCH:
        print("[ERROR] PyTorch chưa cài đặt")
        return

    print("\n=== DEMO: Train ResNet50 (2 epochs, dữ liệu giả) ===\n")

    device = torch.device("cpu")
    model = build_resnet50(NUM_CLASSES, pretrained=False).to(device)

    # Dữ liệu giả
    batch_size = 4
    x_fake = torch.randn(batch_size, 3, 224, 224)
    y_fake = torch.randint(0, NUM_CLASSES, (batch_size,))
    fake_dataset = torch.utils.data.TensorDataset(x_fake, y_fake)
    loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)

    freeze_backbone(model)
    for epoch in range(1, 3):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate_one_epoch(model, loader, criterion, device)
        print(f"  Epoch {epoch}: train_loss={loss:.4f} train_acc={acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")

    print("\n✅ Demo thành công! Pipeline hoạt động đúng.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện ResNet50 Transfer Learning cho phân loại phương tiện",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",     default="resnet50",              help="Tên model (mặc định resnet50)")
    parser.add_argument("--data-dir",  default="data",                  help="Thư mục dữ liệu")
    parser.add_argument("--save-dir",  default="models/classification", help="Thư mục lưu model")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch-size",type=int, default=32)
    parser.add_argument("--warmup",    type=int, default=10,            help="Số epoch warm-up (freeze backbone)")
    parser.add_argument("--lr",        type=float, default=1e-3,        help="LR giai đoạn warm-up")
    parser.add_argument("--lr-ft",     type=float, default=1e-4,        help="LR giai đoạn fine-tune")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--img-size",  type=int, default=224)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--device",    default="auto")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-pretrain", action="store_true",           help="Không dùng pretrained weights")
    parser.add_argument("--demo",      action="store_true",             help="Chạy demo 2 epoch với dữ liệu giả")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return

    history = train_resnet50(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup,
        lr_warmup=args.lr,
        lr_finetune=args.lr_ft,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        img_size=args.img_size,
        num_workers=args.workers,
        device_str=args.device,
        pretrained=not args.no_pretrain,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

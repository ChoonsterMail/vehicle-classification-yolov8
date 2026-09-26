"""
src/train_mobilenet.py
=======================
Phụ trách: Nguyễn Vũ Tùng — Tuần 4

Huấn luyện mô hình nhẹ MobileNetV3-Large để so sánh tốc độ với ResNet50.

Mục tiêu so sánh:
  ┌─────────────────┬──────────┬───────────────┐
  │ Metric          │ ResNet50 │ MobileNetV3-L │
  ├─────────────────┼──────────┼───────────────┤
  │ Params          │ ~25M     │ ~5.5M         │
  │ Top-1 Acc (kỳ vọng) │ ≥90%  │ ≥87%        │
  │ FPS trên CPU    │ baseline │ 3–5× nhanh hơn│
  └─────────────────┴──────────┴───────────────┘

Cách dùng:
    python src/train_mobilenet.py --epochs 50 --batch-size 32
    python src/train_mobilenet.py --demo  # Demo 2 epoch với dữ liệu giả
    python src/train_mobilenet.py --benchmark  # Đo FPS so với ResNet50
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
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
# Model Builders
# ──────────────────────────────────────────────────────────────────────────────

def build_mobilenetv3_large(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> "nn.Module":
    """
    Xây dựng MobileNetV3-Large với đầu phân loại tùy chỉnh.

    MobileNetV3-Large có kiến trúc classifier:
        Linear(960, 1280) → Hardswish → Dropout → Linear(1280, 1000)
    Ta thay Linear cuối: 1000 → num_classes
    """
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Thay lớp output cuối
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> "nn.Module":
    """
    Xây dựng EfficientNet-B0 (thay thế nếu cần model nhẹ hơn).
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def build_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> "nn.Module":
    """Factory function chọn model theo tên."""
    builders = {
        "mobilenetv3": build_mobilenetv3_large,
        "mobilenetv3_large": build_mobilenetv3_large,
        "efficientnet_b0": build_efficientnet_b0,
    }
    if model_name not in builders:
        raise ValueError(f"Model không hỗ trợ: {model_name}. Chọn: {list(builders.keys())}")
    return builders[model_name](num_classes=num_classes, pretrained=pretrained)


def count_params(model: "nn.Module") -> Dict[str, int]:
    """Đếm số parameters của model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop (tái sử dụng từ train_classifier.py)
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping để ổn định training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            batch_acc = (preds == labels).float().mean().item() * 100
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={batch_acc:.1f}%", end="\r")

    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total * 100


# ──────────────────────────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────────────────────────

def train_mobilenet(
    data_dir: str,
    save_dir: str = "models/classification",
    model_name: str = "mobilenetv3",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    label_smoothing: float = 0.1,
    img_size: int = 224,
    num_workers: int = 4,
    device_str: str = "auto",
    pretrained: bool = True,
    use_onecycle: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Huấn luyện MobileNetV3-Large.

    Khác với ResNet50, MobileNetV3 nhỏ hơn nhiều nên:
    - Không cần giai đoạn freeze riêng (train toàn bộ ngay từ đầu)
    - Dùng OneCycleLR (thay vì CosineAnnealingLR) để hội tụ nhanh hơn
    - Gradient clipping để ổn định

    Returns history dict.
    """
    if not HAS_TORCH:
        raise RuntimeError("Cài đặt PyTorch trước")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(device_str)

    # Model
    model = build_model(model_name, NUM_CLASSES, pretrained)
    model = model.to(device)
    params_info = count_params(model)

    print(f"\n{'=' * 60}")
    print(f"  Huấn luyện {model_name.upper()} (lightweight)")
    print(f"  Device     : {device}")
    print(f"  Params     : {params_info['total']:,} tổng | {params_info['trainable']:,} trainable")
    print(f"  Data dir   : {data_dir}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {lr} (OneCycleLR)" if use_onecycle else f"  LR : {lr}")
    print(f"{'=' * 60}\n")

    # DataLoaders
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.dataset import get_dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir, batch_size=batch_size, img_size=img_size, num_workers=num_workers
    )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if use_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.2,  # 20% epoch đầu warm-up LR
            anneal_strategy="cos",
        )
        step_scheduler_per_batch = True
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        step_scheduler_per_batch = False

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log CSV
    log_path = save_dir / f"{model_name}_training_log.csv"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
        )
        writer.writeheader()

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train (OneCycleLR step per batch)
            if step_scheduler_per_batch:
                # Custom loop để step scheduler mỗi batch
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += images.size(0)
                train_loss = running_loss / total
                train_acc  = correct / total * 100
            else:
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch
                )
                scheduler.step()

            val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)
            lr_now = optimizer.param_groups[0]["lr"]

            # Log
            row = dict(epoch=epoch, train_loss=round(train_loss, 6),
                       val_loss=round(val_loss, 6),
                       train_acc=round(train_acc, 4), val_acc=round(val_acc, 4),
                       lr=round(lr_now, 8))
            writer.writerow(row)
            f.flush()
            for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]:
                history[k].append(row[k])

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
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                    "model_name": model_name,
                }, save_dir / f"{model_name}_best.pth")
                print(f"  ✅ Best model saved (val_acc={val_acc:.2f}%)")

    # Lưu model cuối
    torch.save(model.state_dict(), save_dir / f"{model_name}_last.pth")

    # Summary
    summary = {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "num_params": params_info["total"],
    }
    with open(save_dir / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Huấn luyện {model_name} hoàn thành!")
    print(f"  Best val accuracy : {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Model lưu tại     : {save_dir.resolve()}")
    print(f"{'=' * 60}")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# FPS Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_fps(
    model_paths: Dict[str, Tuple[str, str]],
    img_size: int = 224,
    batch_size: int = 1,
    n_runs: int = 200,
    device_str: str = "cpu",
) -> None:
    """
    Đo FPS của nhiều mô hình.

    Parameters
    ----------
    model_paths : {'ResNet50': ('path.pth', 'resnet50'), 'MobileNetV3': ('path.pth', 'mobilenetv3')}
    n_runs : số lần chạy inference để tính trung bình
    """
    if not HAS_TORCH:
        return

    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  FPS Benchmark — {device} | batch={batch_size} | img={img_size}×{img_size}")
    print(f"{'='*60}")

    dummy = torch.randn(batch_size, 3, img_size, img_size).to(device)

    results = {}
    for name, (path, mtype) in model_paths.items():
        if path and Path(path).exists():
            if mtype == "resnet50":
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            else:
                model = build_model(mtype, NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            # Tạo model mới nếu không có file
            if mtype == "resnet50":
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            else:
                model = build_model(mtype, NUM_CLASSES, pretrained=False)

        model = model.to(device).eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy)

        # Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy)
        elapsed = time.perf_counter() - t0

        fps = n_runs * batch_size / elapsed
        latency_ms = elapsed / n_runs * 1000

        params = count_params(model)
        results[name] = {"fps": fps, "latency_ms": latency_ms, "params": params["total"]}

        print(f"\n  [{name}]")
        print(f"    Params   : {params['total']:,}")
        print(f"    FPS      : {fps:.1f}")
        print(f"    Latency  : {latency_ms:.2f} ms/image")

    if len(results) >= 2:
        names = list(results.keys())
        fps_0 = results[names[0]]["fps"]
        fps_1 = results[names[1]]["fps"]
        ratio = fps_1 / fps_0 if fps_0 > 0 else 0
        print(f"\n  📊 {names[1]} nhanh hơn {names[0]}: {ratio:.1f}×")

    print(f"\n{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    if not HAS_TORCH:
        print("[ERROR] PyTorch chưa cài đặt")
        return

    print("\n=== DEMO: Train MobileNetV3-Large (2 epochs, dữ liệu giả) ===\n")

    device = torch.device("cpu")
    model = build_mobilenetv3_large(NUM_CLASSES, pretrained=False).to(device)
    params = count_params(model)
    print(f"MobileNetV3-L params: {params['total']:,}")

    # Dữ liệu giả
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, NUM_CLASSES, (8,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, 3):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate_one_epoch(model, loader, criterion, device)
        print(f"  Epoch {epoch}: train_loss={loss:.4f} acc={acc:.1f}% | val_acc={val_acc:.1f}%")

    # Benchmark so sánh
    print("\n--- Quick FPS benchmark (CPU, no real models) ---")
    benchmark_fps(
        model_paths={
            "ResNet50":     (None, "resnet50"),
            "MobileNetV3-L": (None, "mobilenetv3"),
        },
        n_runs=50,
        device_str="cpu",
    )
    print("\n✅ Demo hoàn thành!")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện MobileNetV3-Large cho phân loại phương tiện",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",     default="mobilenetv3",
                        choices=["mobilenetv3", "mobilenetv3_large", "efficientnet_b0"])
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--save-dir",  default="models/classification")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch-size",type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--img-size",  type=int,   default=224)
    parser.add_argument("--workers",   type=int,   default=4)
    parser.add_argument("--device",    default="auto")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--no-onecycle", action="store_true", help="Dùng CosineAnnealingLR thay OneCycleLR")
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--demo",      action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Chạy FPS benchmark so sánh MobileNetV3 vs ResNet50")
    parser.add_argument("--resnet-path",  default=None, help="Đường dẫn ResNet50 .pth cho benchmark")
    parser.add_argument("--mobile-path",  default=None, help="Đường dẫn MobileNetV3 .pth cho benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo()
        return

    if args.benchmark:
        benchmark_fps(
            model_paths={
                "ResNet50":      (args.resnet_path,  "resnet50"),
                "MobileNetV3-L": (args.mobile_path, "mobilenetv3"),
            },
            device_str="cpu",
        )
        return

    train_mobilenet(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        img_size=args.img_size,
        num_workers=args.workers,
        device_str=args.device,
        pretrained=not args.no_pretrain,
        use_onecycle=not args.no_onecycle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

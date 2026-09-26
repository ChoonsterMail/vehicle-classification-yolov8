"""
src/evaluate.py
================
Phụ trách: Nguyễn Thành Đạt — Tuần 4

Module đánh giá mô hình toàn diện — dùng chung cho classifier và YOLO.
Tính toán: Confusion Matrix, F1-Score, Precision, Recall, Classification Report.

Cách dùng:
    # Đánh giá mô hình phân loại (ResNet50 / MobileNetV3):
    python src/evaluate.py --model models/classification/resnet50_best.pth \
                           --data-dir data/ \
                           --model-type resnet50

    # Chạy demo không cần model thực:
    python src/evaluate.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Deep Learning imports (graceful)
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch chưa cài đặt.")

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn chưa cài đặt: pip install scikit-learn")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
NUM_CLASSES = 5


# ──────────────────────────────────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────────────────────────────────

def load_classifier(model_path: str, model_type: str = "resnet50", device: str = "cpu"):
    """
    Load mô hình phân loại đã huấn luyện từ file .pth.

    Parameters
    ----------
    model_path : đường dẫn file model (.pth)
    model_type : 'resnet50' | 'mobilenetv3' | 'efficientnet'
    device : 'cpu' | 'cuda' | 'mps'

    Returns
    -------
    model : torch.nn.Module ở chế độ eval
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch chưa cài đặt")

    device = torch.device(device)

    if model_type == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_type in ("mobilenetv3", "mobilenetv3_large"):
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, NUM_CLASSES
        )
    else:
        raise ValueError(f"model_type không hỗ trợ: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)
    # Hỗ trợ cả 2 kiểu save: state_dict trực tiếp hoặc dict có key 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"[Load] Model {model_type} từ: {model_path}")
    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    dataloader,
    device,
    verbose: bool = True,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Chạy inference trên toàn bộ dataloader.

    Returns
    -------
    (all_preds, all_labels, all_probs_max)
    """
    all_preds  = []
    all_labels = []
    all_probs  = []

    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.max(dim=1).values.cpu().numpy().tolist())

            if verbose and (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                n_done = (batch_idx + 1) * dataloader.batch_size
                print(f"  [{n_done}/{len(dataloader.dataset)}] {elapsed:.1f}s", end="\r")

    if verbose:
        print(f"\n  Tổng: {len(all_preds)} ảnh | {time.time()-t0:.2f}s")

    return all_preds, all_labels, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Tính toán toàn bộ metrics đánh giá.

    Returns
    -------
    dict với các key:
        accuracy, f1_macro, f1_micro, f1_weighted, f1_per_class,
        precision_macro, recall_macro, confusion_matrix, report
    """
    if not HAS_SKLEARN:
        raise RuntimeError("pip install scikit-learn")

    if class_names is None:
        class_names = CLASS_NAMES

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy  = float((y_true == y_pred).mean()) * 100

    # F1 Scores
    f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_micro    = float(f1_score(y_true, y_pred, average="micro",    zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Precision & Recall
    precision_macro = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    recall_macro    = float(recall_score(   y_true, y_pred, average="macro",    zero_division=0))
    precision_per   = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    recall_per      = recall_score(   y_true, y_pred, average=None, zero_division=0).tolist()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Classification Report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy":          round(accuracy, 4),
        "f1_macro":          round(f1_macro * 100, 4),
        "f1_micro":          round(f1_micro * 100, 4),
        "f1_weighted":       round(f1_weighted * 100, 4),
        "f1_per_class":      {class_names[i]: round(v * 100, 4) for i, v in enumerate(f1_per_class)},
        "precision_macro":   round(precision_macro * 100, 4),
        "recall_macro":      round(recall_macro * 100, 4),
        "precision_per_class": {class_names[i]: round(v * 100, 4) for i, v in enumerate(precision_per)},
        "recall_per_class":    {class_names[i]: round(v * 100, 4) for i, v in enumerate(recall_per)},
        "confusion_matrix":  cm.tolist(),
        "classification_report": report,
    }

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Print & Save Results
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict, model_name: str = "Model") -> None:
    """In kết quả đánh giá ra console theo định dạng đẹp."""
    print(f"\n{'=' * 60}")
    print(f"  📊 Kết quả đánh giá: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy      : {metrics['accuracy']:.2f}%")
    print(f"  F1 Macro      : {metrics['f1_macro']:.2f}%")
    print(f"  F1 Weighted   : {metrics['f1_weighted']:.2f}%")
    print(f"  Precision Macro: {metrics['precision_macro']:.2f}%")
    print(f"  Recall Macro  : {metrics['recall_macro']:.2f}%")
    print(f"\n  F1 theo lớp:")
    for cls_name, f1 in metrics["f1_per_class"].items():
        bar = "█" * int(f1 / 5) + "░" * (20 - int(f1 / 5))
        print(f"    {cls_name:12s}: {bar} {f1:.1f}%")
    print(f"{'=' * 60}")


def save_results(
    metrics: Dict,
    y_true: List[int],
    y_pred: List[int],
    save_dir: str = "results",
    model_name: str = "model",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Lưu kết quả đánh giá:
      - JSON report
      - Confusion matrix heatmap (PNG)
      - F1 per class bar chart (PNG)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Lưu JSON
    json_path = save_dir / f"{model_name}_eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")

    # Confusion matrix
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.visualization import plot_confusion_matrix, plot_class_distribution
        import numpy as np

        cm = np.array(metrics["confusion_matrix"])
        cm_path = str(fig_dir / f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            save_path=cm_path,
            normalize=True,
            title=f"Confusion Matrix — {model_name}",
        )

        # F1 per class bar
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f1_vals = list(metrics["f1_per_class"].values())
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
        bars = ax.bar(class_names, f1_vals, color=colors, alpha=0.85)
        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("F1-Score (%)")
        ax.set_title(f"F1-Score theo lớp — {model_name}")
        ax.set_ylim(0, 115)
        ax.axhline(85, color="red", linestyle="--", alpha=0.5, label="Mục tiêu 85%")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        f1_path = str(fig_dir / f"{model_name}_f1_per_class.png")
        plt.tight_layout()
        plt.savefig(f1_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {f1_path}")

    except Exception as e:
        print(f"[WARN] Không thể tạo figure: {e}")

    print(f"\n✅ Kết quả đánh giá lưu tại: {save_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Demo tính metrics với dữ liệu giả lập."""
    print("\n=== DEMO: Evaluation với dữ liệu giả lập ===\n")
    rng = np.random.default_rng(42)

    n = 500
    # Giả lập: mô hình khá tốt, accuracy ~88%
    y_true = rng.integers(0, NUM_CLASSES, n).tolist()
    noise = rng.random(n) < 0.12  # 12% sai
    y_pred = [
        rng.integers(0, NUM_CLASSES) if noise[i] else y_true[i]
        for i in range(n)
    ]

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Demo Model")

    save_results(metrics, y_true, y_pred, save_dir="results/demo", model_name="demo")
    print("\n✅ Demo hoàn thành!")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình phân loại phương tiện giao thông",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default=None,                   help="Đường dẫn file model .pth")
    parser.add_argument("--model-type", default="resnet50",
                        choices=["resnet50", "mobilenetv3", "efficientnet_b0"],
                        help="Kiến trúc mô hình")
    parser.add_argument("--data-dir",   default="data",                 help="Thư mục dữ liệu (có train/val/test)")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"],               help="Split cần đánh giá")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="auto",                 help="cpu | cuda | mps | auto")
    parser.add_argument("--save-dir",   default="results",              help="Thư mục lưu kết quả")
    parser.add_argument("--demo",       action="store_true",            help="Chạy demo với dữ liệu giả lập")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo or args.model is None:
        run_demo()
        return

    if not HAS_TORCH or not HAS_SKLEARN:
        print("[ERROR] Cần cài PyTorch và scikit-learn")
        return

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load model
    model, device_obj = load_classifier(args.model, args.model_type, device)

    # DataLoader
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.dataset import get_dataloaders
    _, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    loader = test_loader if args.split == "test" else val_loader

    # Inference
    print(f"\nĐang chạy inference trên tập {args.split}...")
    y_pred, y_true, _ = run_inference(model, loader, device_obj)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    model_name = Path(args.model).stem
    print_metrics(metrics, model_name)

    # Save
    save_results(metrics, y_true, y_pred, save_dir=args.save_dir, model_name=model_name)


if __name__ == "__main__":
    main()

"""
src/utils/visualization.py
===========================
Các hàm vẽ đồ thị dùng chung: phân bố lớp, loss/accuracy curves,
confusion matrix, bounding box overlay.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Chạy không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# Tên lớp mặc định
CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

# ──────────────────────────────────────────────────────────────────────────────
# 1. Phân bố lớp
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    counts_dict: Dict[str, Dict[str, int]],
    save_path: str = "figures/class_distribution.png",
    title: str = "Phân bố số mẫu theo lớp phương tiện",
) -> None:
    """
    Vẽ biểu đồ cột phân bố lớp cho nhiều split.

    Parameters
    ----------
    counts_dict : {'train': {'car': 500, ...}, 'val': {...}, 'test': {...}}
    save_path : đường dẫn lưu ảnh
    """
    splits = list(counts_dict.keys())
    classes = CLASS_NAMES

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, split in enumerate(splits):
        counts = [counts_dict[split].get(c, 0) for c in classes]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize(),
                      color=CLASS_COLORS[i % len(CLASS_COLORS)], alpha=0.85)
        # Ghi số lên đầu cột
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Lớp phương tiện", fontsize=12)
    ax.set_ylabel("Số lượng ảnh", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels([f"{c}\n(class {i})" for i, c in enumerate(classes)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loss & Accuracy curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_accuracy(
    history: Dict[str, List[float]],
    save_dir: str = "figures",
    model_name: str = "model",
) -> None:
    """
    Vẽ 2 biểu đồ: Loss và Accuracy qua từng epoch.

    Parameters
    ----------
    history : {
        'train_loss': [...], 'val_loss': [...],
        'train_acc':  [...], 'val_acc':  [...]
    }
    save_dir : thư mục lưu ảnh
    model_name : tên mô hình (dùng trong tên file)
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Loss ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    loss_path = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {loss_path}")

    # ── Accuracy ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax.plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} — Training & Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    acc_path = os.path.join(save_dir, f"{model_name}_accuracy.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {acc_path}")


def plot_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "val_acc",
    save_path: str = "figures/model_comparison.png",
    title: str = "So sánh Val Accuracy các mô hình",
) -> None:
    """
    Vẽ đồ thị so sánh nhiều mô hình trên cùng 1 figure.

    Parameters
    ----------
    histories : {'ResNet50': {...}, 'MobileNetV3': {...}}
    metric : key trong history cần so sánh
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    for i, (name, hist) in enumerate(histories.items()):
        values = hist.get(metric, [])
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, f"-o", color=colors[i % len(colors)],
                markersize=4, label=name)

    ax.set_xlabel("Epoch")
    ylabel = "Accuracy (%)" if "acc" in metric else "Loss"
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "figures/confusion_matrix.png",
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """
    Vẽ confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray shape (N, N)
    class_names : danh sách tên lớp
    normalize : nếu True, hiển thị phần trăm thay vì số nguyên
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Visualize Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def visualize_augmentation(
    original: np.ndarray,
    augmented_list: List[Tuple[str, np.ndarray]],
    save_path: str = "figures/augmentation_demo.png",
) -> None:
    """
    Hiển thị ảnh gốc và các phiên bản augmented.

    Parameters
    ----------
    original : np.ndarray HxWxC (uint8, RGB)
    augmented_list : [('HorizontalFlip', aug_img), ('Brightness', aug_img), ...]
    """
    n = len(augmented_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    for i, (name, aug_img) in enumerate(augmented_list):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(name, fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Minh họa Data Augmentation", fontsize=13, fontweight="bold", y=1.01)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualize Predictions với Bounding Box
# ──────────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    image: np.ndarray,
    boxes: List[List[float]],
    class_ids: List[int],
    scores: Optional[List[float]] = None,
    save_path: str = "figures/predictions.png",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Vẽ bounding box và nhãn lên ảnh.

    Parameters
    ----------
    image : np.ndarray HxWxC (uint8, RGB)
    boxes : list of [x_min, y_min, x_max, y_max] (pixel coordinates)
    class_ids : list of int
    scores : list of float (confidence), optional
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        cls = class_ids[i]
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label = class_names[cls] if cls < len(class_names) else str(cls)
        if scores is not None:
            label += f" {scores[i]:.2f}"

        ax.text(
            x_min, y_min - 4, label,
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
        )

    ax.axis("off")
    ax.set_title("Kết quả phát hiện phương tiện", fontsize=13)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")

"""
src/split_dataset.py
=====================
Phụ trách: Nguyễn Thành Đạt — Tuần 3

Pipeline phân chia dữ liệu từ data/raw/ sang train/val/test với tỷ lệ 70:20:10.
Sử dụng Stratified Split để đảm bảo phân bố lớp đồng đều trong cả 3 tập.

Cách dùng:
    python src/split_dataset.py --raw-dir data/raw --seed 42
    python src/split_dataset.py --raw-dir data/raw --dry-run   # Xem trước, không copy
    python src/split_dataset.py --help
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
NUM_CLASSES = 5

DEFAULT_RATIOS = (0.70, 0.20, 0.10)  # train / val / test
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# Đọc nhãn YOLO
# ──────────────────────────────────────────────────────────────────────────────

def get_primary_class(label_path: Path) -> int:
    """
    Đọc file nhãn YOLO và trả về class_id của đối tượng đầu tiên.
    Trả về -1 nếu file rỗng.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return -1
    return int(lines[0].split()[0])


# ──────────────────────────────────────────────────────────────────────────────
# Thu thập samples từ raw/
# ──────────────────────────────────────────────────────────────────────────────

def collect_samples(raw_dir: Path) -> Dict[int, List[Tuple[Path, Path]]]:
    """
    Duyệt thư mục raw_dir và thu thập cặp (image_path, label_path) theo lớp.

    Parameters
    ----------
    raw_dir : thư mục chứa ảnh và nhãn (có thể ảnh/nhãn cùng folder hoặc images/ + labels/)

    Returns
    -------
    dict {class_id: [(img_path, lbl_path), ...]}
    """
    # Hỗ trợ cả 2 kiểu tổ chức:
    # 1. raw/images/*.jpg + raw/labels/*.txt
    # 2. raw/*.jpg + raw/*.txt (flat)

    img_dir = raw_dir / "images" if (raw_dir / "images").exists() else raw_dir
    lbl_dir = raw_dir / "labels" if (raw_dir / "labels").exists() else raw_dir

    samples_by_class: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)
    missing_label = 0
    empty_label = 0
    invalid_class = 0

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong: {img_dir}\n"
            f"Đảm bảo thư mục chứa file .jpg/.png hoặc có thư mục con images/"
        )

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            missing_label += 1
            continue

        cls_id = get_primary_class(lbl_path)
        if cls_id == -1:
            empty_label += 1
            continue
        if not (0 <= cls_id < NUM_CLASSES):
            invalid_class += 1
            continue

        samples_by_class[cls_id].append((img_path, lbl_path))

    print(f"\n[Collect] Tổng ảnh quét: {len(image_files)}")
    print(f"  ✓ Hợp lệ: {sum(len(v) for v in samples_by_class.values())}")
    print(f"  ✗ Thiếu nhãn: {missing_label}")
    print(f"  ✗ Nhãn rỗng: {empty_label}")
    print(f"  ✗ Class ID lạ: {invalid_class}\n")

    for cls_id, items in sorted(samples_by_class.items()):
        print(f"  [{CLASS_NAMES[cls_id]:12s}] (class {cls_id}): {len(items):5d} ảnh")

    return samples_by_class


# ──────────────────────────────────────────────────────────────────────────────
# Stratified Split
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(
    samples_by_class: Dict[int, List[Tuple[Path, Path]]],
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Chia dữ liệu theo stratified sampling — mỗi lớp được chia theo đúng tỷ lệ.

    Returns
    -------
    (train_samples, val_samples, test_samples) — mỗi phần là list of (img_path, lbl_path)
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Tổng tỷ lệ phải bằng 1.0"

    rng = random.Random(seed)
    train_all, val_all, test_all = [], [], []

    for cls_id, items in sorted(samples_by_class.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        # test lấy phần còn lại
        n_test  = n - n_train - n_val

        train_all.extend(shuffled[:n_train])
        val_all.extend(shuffled[n_train:n_train + n_val])
        test_all.extend(shuffled[n_train + n_val:])

        print(f"  [{CLASS_NAMES[cls_id]:12s}] train={n_train:4d} | val={n_val:4d} | test={n_test:4d}")

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


# ──────────────────────────────────────────────────────────────────────────────
# Copy files
# ──────────────────────────────────────────────────────────────────────────────

def copy_split(
    samples: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str,
    dry_run: bool = False,
) -> None:
    """
    Copy (image, label) pairs vào output_dir/images/ và output_dir/labels/.
    """
    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in samples:
        dst_img = img_out / img_path.name
        dst_lbl = lbl_out / lbl_path.name

        if not dry_run:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}[{split_name:5s}] → {output_dir} : {len(samples)} cặp ảnh-nhãn")


# ──────────────────────────────────────────────────────────────────────────────
# Báo cáo thống kê
# ──────────────────────────────────────────────────────────────────────────────

def print_report(train, val, test) -> None:
    """In bảng báo cáo phân bố lớp sau khi chia."""
    from collections import Counter

    def class_counter(samples):
        # Đọc class_id từ file nhãn
        counts = Counter()
        for img_path, lbl_path in samples:
            if lbl_path.exists():
                cls = get_primary_class(lbl_path)
                if cls >= 0:
                    counts[cls] += 1
        return counts

    splits = [("train", train), ("val", val), ("test", test)]

    print("\n" + "=" * 70)
    print(f"{'Kết quả phân chia dữ liệu':^70}")
    print("=" * 70)
    header = f"{'Lớp':>14}" + "".join(f"  {s:>8}" for s, _ in splits) + f"  {'Tổng':>8}"
    print(header)
    print("-" * 70)

    totals = {s: len(samples) for s, samples in splits}
    class_totals = defaultdict(int)

    counters = [(s, class_counter(samples)) for s, samples in splits]

    for cls_id in range(NUM_CLASSES):
        row = f"  {CLASS_NAMES[cls_id]:12s}"
        for s, cnt in counters:
            v = cnt.get(cls_id, 0)
            class_totals[cls_id] += v
            row += f"  {v:>8d}"
        row += f"  {class_totals[cls_id]:>8d}"
        print(row)

    print("-" * 70)
    total_row = f"  {'Tổng':12s}"
    grand_total = 0
    for s, _ in splits:
        total_row += f"  {totals[s]:>8d}"
        grand_total += totals[s]
    total_row += f"  {grand_total:>8d}"
    print(total_row)
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phân chia dữ liệu Stratified Split 70/20/10 cho YOLO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir",  default="data/raw",  help="Thư mục chứa dữ liệu gốc")
    parser.add_argument("--data-dir", default="data",      help="Thư mục gốc output (chứa train/val/test)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.20)
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed để tái tạo kết quả")
    parser.add_argument("--dry-run",     action="store_true",       help="Chạy thử, không copy file")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir  = Path(args.raw_dir)
    data_dir = Path(args.data_dir)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio > 1.0")
    ratios = (args.train_ratio, args.val_ratio, test_ratio)

    print(f"\n{'=' * 50}")
    print(f"  Stratified Dataset Split")
    print(f"  Tỷ lệ: train={ratios[0]:.0%} / val={ratios[1]:.0%} / test={test_ratio:.0%}")
    print(f"  Seed: {args.seed}")
    print(f"  Raw dir: {raw_dir.resolve()}")
    print(f"  Output: {data_dir.resolve()}")
    if args.dry_run:
        print("  ⚠️  DRY RUN — không copy file thực sự")
    print(f"{'=' * 50}\n")

    # 1. Thu thập
    samples_by_class = collect_samples(raw_dir)

    if not samples_by_class:
        print("[ERROR] Không tìm thấy sample hợp lệ. Kiểm tra lại thư mục raw/")
        return

    # 2. Chia
    print("\nPhân chia theo lớp:")
    train_samples, val_samples, test_samples = stratified_split(
        samples_by_class, ratios=ratios, seed=args.seed
    )

    # 3. Copy
    print("\nCopy files:")
    copy_split(train_samples, data_dir / "train", "train", dry_run=args.dry_run)
    copy_split(val_samples,   data_dir / "val",   "val",   dry_run=args.dry_run)
    copy_split(test_samples,  data_dir / "test",  "test",  dry_run=args.dry_run)

    # 4. Báo cáo
    if not args.dry_run:
        print_report(
            [(img, lbl) for img, lbl in train_samples],
            [(img, lbl) for img, lbl in val_samples],
            [(img, lbl) for img, lbl in test_samples],
        )
        print("\n✅ Hoàn thành phân chia dữ liệu!")
    else:
        print(f"\n[DRY RUN] Sẽ tạo: train={len(train_samples)} | val={len(val_samples)} | test={len(test_samples)}")


if __name__ == "__main__":
    main()

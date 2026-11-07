"""
scripts/split_dataset.py — Task 1.4 (apply)
=============================================
Produces deterministic train/val/test split CSV files.

Strategy:
  - Stratified split (equal class proportions in each fold)
  - Seeded for reproducibility (SEED=42)
  - Test set is fixed and NEVER touched during training or tuning
  - If subsampling is used (--max-per-class), it is applied BEFORE splitting
    so test set remains proportional; seed and size are logged in the CSV header

Outputs:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv

Each CSV has columns: path, label (0=Real, 1=AI-generated)

Usage:
  python scripts/split_dataset.py --data-dir data/raw [--max-per-class 2000]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CLASS_MAP = {
    # Will be detected automatically from folder names
    # Positive class (AI-generated): label=1
    # Negative class (Real): label=0
}


def detect_classes(data_dir: Path) -> dict[str, int]:
    """Auto-detect class folders and assign 0/1 labels."""
    dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(dirs) != 2:
        print(f"WARNING: Expected 2 class folders, found {len(dirs)}: {[d.name for d in dirs]}")

    # Assign label=1 to whichever folder name contains 'ai' or 'fake' or 'generated'
    # label=0 to 'real' or 'authentic'
    label_map = {}
    for d in dirs:
        name_lower = d.name.lower()
        if any(k in name_lower for k in ["ai", "fake", "generated", "artificial", "synthetic"]):
            label_map[d.name] = 1
        else:
            label_map[d.name] = 0

    if len(set(label_map.values())) < 2:
        # fallback: alphabetical order
        for i, d in enumerate(dirs):
            label_map[d.name] = i

    return label_map


def collect_files(data_dir: Path, label_map: dict[str, int],
                  max_per_class: int | None) -> list[tuple[str, int]]:
    rng = np.random.default_rng(SEED)
    all_samples = []
    for cls_name, label in label_map.items():
        cls_dir = data_dir / cls_name
        files = sorted([
            str(f) for f in cls_dir.rglob("*")
            if f.suffix.lower() in EXTENSIONS
        ])
        if max_per_class and len(files) > max_per_class:
            indices = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(indices)]
            print(f"  Subsampled '{cls_name}': {max_per_class} of {len(cls_dir.rglob('*'))}")
        all_samples.extend([(f, label) for f in files])
        print(f"  Class '{cls_name}' (label={label}): {len(files)} images")
    return all_samples


def write_split(samples: list[tuple[str, int]], path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(samples)
    print(f"  → {path}  ({len(samples)} rows)")


def run_split(data_dir: Path, max_per_class: int | None,
              val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
    label_map = detect_classes(data_dir)
    print(f"\nClass → label map: {label_map}")

    samples = collect_files(data_dir, label_map, max_per_class)
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=SEED,
    )

    # Then split val from trainval
    val_ratio_adj = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adj,
        stratify=y_trainval,
        random_state=SEED,
    )

    note = (
        f"SEED={SEED}  max_per_class={max_per_class}  "
        f"label_map={label_map}  "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    out = Path("data/splits")
    print("\nWriting split files:")
    write_split(list(zip(X_train, y_train)), out / "train.csv", note)
    write_split(list(zip(X_val, y_val)),   out / "val.csv",   note)
    write_split(list(zip(X_test, y_test)), out / "test.csv",  note)

    print(f"\nSplit summary (SEED={SEED}):")
    print(f"  Train : {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Label map: {label_map}")
    if max_per_class:
        print(f"  Subsampled to {max_per_class} per class before splitting")
    print("  Test set is fixed. Do not use it until final evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified dataset split for VisionGuard")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap per class (for CPU-budget subsampling)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    run_split(data_dir, args.max_per_class, args.val_ratio, args.test_ratio)


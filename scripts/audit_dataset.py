"""
scripts/audit_dataset.py — Task 1.2
====================================
Walks the dataset directory and emits a CSV row per image file with:
  class, bytes, width, height, colour_mode, format, sha256

Also detects:
  - Unreadable / corrupt files   → reported in summary
  - Exact duplicates by SHA-256  → reported in summary

Outputs:
  data/audit_report.csv       — per-image CSV
  data/audit_summary.csv      — per-class aggregate stats
  report/figures/audit_summary_table.txt — human-readable table for report

Usage:
  python scripts/audit_dataset.py --data-dir data/raw

Evidence: Task 1.2 — printed summary + CSVs are referenced in report §1.2
"""
import argparse
import csv
import hashlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_image(path: Path) -> dict:
    """Return a dict of image metadata, or flag as corrupt."""
    size_bytes = path.stat().st_size
    sha = sha256_of_file(path)
    try:
        with Image.open(path) as img:
            img.load()           # fully decode — catches truncated files
            width, height = img.size
            mode = img.mode
            fmt = img.format or "unknown"
        return dict(
            path=str(path),
            cls=path.parent.name,
            size_bytes=size_bytes,
            width=width,
            height=height,
            colour_mode=mode,
            format=fmt,
            sha256=sha,
            corrupt=False,
        )
    except (UnidentifiedImageError, Exception):
        return dict(
            path=str(path),
            cls=path.parent.name,
            size_bytes=size_bytes,
            width=None,
            height=None,
            colour_mode=None,
            format=None,
            sha256=sha,
            corrupt=True,
        )


def run_audit(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = [f for f in cls_dir.rglob("*") if f.suffix.lower() in EXTENSIONS]
        print(f"  Scanning class '{cls_dir.name}': {len(files)} files …")
        for f in sorted(files):
            rows.append(audit_image(f))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "audit_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull report saved → {csv_path}  ({len(df)} rows)")

    # ── Duplicates ─────────────────────────────────────────────────────
    dup_hashes = df[df.duplicated("sha256", keep=False)]["sha256"].unique()
    dup_count = len(df[df["sha256"].isin(dup_hashes)])

    # ── Corrupt ────────────────────────────────────────────────────────
    corrupt_count = df["corrupt"].sum()

    # ── Per-class summary ──────────────────────────────────────────────
    clean = df[~df["corrupt"]].copy()
    summary = (
        clean.groupby("cls")
        .agg(
            count=("path", "len"),
            min_width=("width", "min"),
            median_width=("width", "median"),
            max_width=("width", "max"),
            min_height=("height", "min"),
            median_height=("height", "median"),
            max_height=("height", "max"),
            min_size_bytes=("size_bytes", "min"),
            median_size_bytes=("size_bytes", "median"),
            max_size_bytes=("size_bytes", "max"),
        )
        .reset_index()
    )
    summary_path = out_dir / "audit_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ── Print ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATASET AUDIT SUMMARY")
    print("="*60)
    print(f"Total images   : {len(df)}")
    print(f"Corrupt files  : {int(corrupt_count)}")
    print(f"Duplicate files: {dup_count} images share a hash with another")
    print(f"Classes found  : {sorted(df['cls'].unique().tolist())}")
    print()
    print(summary.to_string(index=False))
    print("="*60)

    # ── Text table for report ──────────────────────────────────────────
    txt_path = Path("report/figures/audit_summary_table.txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(f"Dataset Audit Summary\n")
        f.write(f"Total images   : {len(df)}\n")
        f.write(f"Corrupt files  : {int(corrupt_count)}\n")
        f.write(f"Duplicate files: {dup_count}\n\n")
        f.write(summary.to_string(index=False))
    print(f"\nText table saved → {txt_path}")

    if int(corrupt_count) > 0:
        print("\nCorrupt files:")
        for _, r in df[df["corrupt"]].iterrows():
            print(f"  {r['path']}")

    if dup_count > 0:
        print(f"\nDuplicate SHA-256 groups ({len(dup_hashes)}):")
        for h in dup_hashes[:5]:
            paths = df[df["sha256"] == h]["path"].tolist()
            print(f"  {h[:12]}… → {paths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit VisionGuard dataset")
    parser.add_argument("--data-dir", default="data/raw", help="Root directory containing class folders")
    parser.add_argument("--out-dir", default="data", help="Output directory for CSVs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    run_audit(data_dir, Path(args.out_dir))


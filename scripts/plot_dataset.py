"""
scripts/plot_dataset.py — Task 1.3
=====================================
Generates:
  1. Sample image grid per class (4 images each)
  2. Width/height/filesize histograms per class

Reads data/audit_report.csv (produced by audit_dataset.py).
Saves all figures to report/figures/.

Usage:
  python scripts/plot_dataset.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image


FIGURES_DIR = Path("report/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CSV   = Path("data/audit_report.csv")
DATA_DIR    = Path("data/raw")

N_SAMPLES_PER_CLASS = 4


def sample_image_grid() -> None:
    """Figure 1: 4 sample images per class in a grid."""
    if not AUDIT_CSV.exists():
        print("audit_report.csv not found — run audit_dataset.py first.")
        return

    df = pd.read_csv(AUDIT_CSV)
    df = df[~df["corrupt"]]
    classes = sorted(df["cls"].unique())
    n_cls = len(classes)

    fig, axes = plt.subplots(n_cls, N_SAMPLES_PER_CLASS,
                             figsize=(3.5 * N_SAMPLES_PER_CLASS, 3.5 * n_cls))
    if n_cls == 1:
        axes = [axes]

    fig.suptitle("Sample Images per Class", fontsize=15, fontweight="bold")

    for row_idx, cls in enumerate(classes):
        cls_df = df[df["cls"] == cls].sample(
            min(N_SAMPLES_PER_CLASS, len(df[df["cls"] == cls])),
            random_state=42,
        )
        for col_idx, (_, row) in enumerate(cls_df.iterrows()):
            ax = axes[row_idx][col_idx]
            try:
                img = Image.open(row["path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#222")
                ax.text(0.5, 0.5, "?", ha="center", va="center",
                        transform=ax.transAxes, fontsize=20, color="white")
            ax.set_title(cls, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / "sample_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample grid → {out}")


def histogram_plots() -> None:
    """Figure 2–4: Width, height, file-size histograms per class."""
    if not AUDIT_CSV.exists():
        print("audit_report.csv not found.")
        return

    df = pd.read_csv(AUDIT_CSV)
    df = df[~df["corrupt"]]
    classes = sorted(df["cls"].unique())

    for metric, xlabel, title_sfx, fname in [
        ("width",      "Width (px)",       "Image Width Distribution",       "hist_width"),
        ("height",     "Height (px)",      "Image Height Distribution",      "hist_height"),
        ("size_bytes", "File Size (bytes)", "File Size Distribution",        "hist_filesize"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls in classes:
            vals = df[df["cls"] == cls][metric].dropna()
            ax.hist(vals, bins=50, alpha=0.6, label=cls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{title_sfx} by Class")
        ax.legend()
        plt.tight_layout()
        out = FIGURES_DIR / f"{fname}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Histogram → {out}")

        # Textual interpretation is written in the report
        print(f"  {metric}: "
              + "  ".join(f"{cls}: μ={df[df['cls']==cls][metric].mean():.0f}"
                          for cls in classes))


if __name__ == "__main__":
    print("Generating dataset figures …")
    sample_image_grid()
    histogram_plots()
    print("Done.")


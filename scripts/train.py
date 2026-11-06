"""
scripts/train.py — Task 2.1 to 2.7, Task 3.3 to 3.5
=======================================================
Full VisionGuard training script. Covers:
  - Task 2.1: tf.data pipeline (decode, resize, augment, batch, prefetch)
  - Task 2.2: hand-designed small CNN (baseline)
  - Task 2.3: MobileNetV2 transfer learning (freeze → unfreeze last block)
  - Task 2.4: seeds, callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
  - Task 2.5: test-set evaluation (one-time), threshold selection on val
  - Task 2.6: misclassified image grid
  - Task 3.3: MLflow logging (manual callback + autolog comparison)
  - Task 3.4: artifact attachment
  - Task 3.5: runs with varying hyperparameters

Usage:
  python scripts/train.py --model mobilenet --run-name "run_lr1e-3" \\
         --lr 1e-3 --dropout 0.4 --image-size 224 --epochs 30 \\
         --max-per-class 2000 --batch-size 32 --augment

Hyperparameter sweep:
  python scripts/train.py --sweep   # runs the ≥6 MLflow comparison runs
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Seeds — must be set before importing TF (Task 2.4)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import mlflow
import mlflow.tensorflow
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
SPLITS_DIR    = DATA_DIR / "splits"
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("report/figures")
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.1 — tf.data pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _decode_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple:
    """
    Load a JPEG/PNG from disk, decode to uint8 RGB, resize to (image_size, image_size).

    Resize strategy: squash (resize without cropping) using bilinear interpolation.
    This distorts aspect ratio for non-square images, but preserves all pixel
    information. Chosen over centre-crop because AI-generation artefacts often
    appear at image edges; cropping risks discarding them.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    Augmentation applied to train split only.
    Random horizontal flip, brightness, contrast, hue jitter.
    Does NOT augment val/test — that would evaluate on out-of-distribution samples.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_hue(img, max_delta=0.08)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def _preprocess_mobilenet(img: tf.Tensor, label: tf.Tensor) -> tuple:
    """Apply MobileNetV2 preprocess_input: maps [0,255] → [-1,1]."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = preprocess_input(img)
    return img, label


def build_dataset(split_csv: Path, image_size: int, batch_size: int,
                  augment: bool, repeat: bool = True) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a split CSV file.

    Pipeline stages:
      1. Read CSV → list of (path, label) pairs
      2. from_tensor_slices → elements
      3. shuffle (train only)
      4. map: decode + resize  (parallel, up to AUTOTUNE workers)
      5. map: augment          (train only)
      6. map: preprocess_input (MobileNetV2 normalisation)
      7. batch
      8. prefetch(AUTOTUNE)   — overlap GPU compute with disk I/O

    Why prefetch: without it the pipeline stalls after each batch while
    waiting for the next one to load from disk. Prefetch fills a background
    buffer so the GPU always has a batch ready.
    """
    df = pd.read_csv(split_csv, comment="#")
    paths  = df["path"].tolist()
    labels = df["label"].tolist()

    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if augment:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=SEED)

    ds = ds.map(
        lambda p, l: _decode_resize(p, l, image_size),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.map(_preprocess_mobilenet, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(AUTOTUNE)
    return ds, len(paths)


def save_augmented_batch_figure(split_csv: Path, image_size: int) -> Path:
    """Save a grid of 8 augmented training images to report/figures/."""
    ds, _ = build_dataset(split_csv, image_size, batch_size=8, augment=True, repeat=False)
    batch_img, batch_lbl = next(iter(ds))

    # Denormalise from [-1,1] back to [0,1] for display
    imgs_disp = (batch_img.numpy() + 1.0) / 2.0
    labels    = batch_lbl.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Sample Augmented Training Batch", fontsize=14)
    for ax, img, lbl in zip(axes.flatten(), imgs_disp, labels):
        ax.imshow(img)
        ax.set_title("AI-generated" if lbl == 1 else "Real", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / "augmented_batch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Augmented batch figure → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.2 — Hand-designed small CNN
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_baseline(image_size: int, dropout: float) -> tf.keras.Model:
    """
    Small hand-designed CNN with binary classification head.

    Architecture rationale:
      Conv2D + BatchNorm + MaxPool: extract local features, normalise
        activations (speeds convergence), reduce spatial resolution.
      Three conv blocks (32→64→128 filters): progressively abstract features.
      Dropout(0.4) before dense: regularise against overfitting on small data.
      Dense(128, relu): combine spatial features.
      Dense(1, sigmoid): binary output — P(AI-generated).

    Loss: binary_crossentropy — correct loss for single-sigmoid binary classifiers.
    Optimizer: Adam — adaptive LR, works well for image tasks out of the box.
    Metrics: accuracy, AUC — accuracy for intuition; AUC threshold-independent.
    """
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name="input")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool2")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inp, out, name="VisionGuard_CNN_Baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.3 — MobileNetV2 transfer learning
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenet(image_size: int, dropout: float,
                    num_unfreeze_layers: int = 20) -> tuple[tf.keras.Model, int]:
    """
    MobileNetV2 transfer learning.

    Stage 1 — Feature extraction (all base layers frozen):
      Load MobileNetV2 pretrained on ImageNet (without classifier top).
      Add global average pool + dropout + single sigmoid head.
      Only the head is trainable. Fast to converge.

    Stage 2 — Fine-tuning (last N layers unfrozen):
      Unfreeze the last `num_unfreeze_layers` layers of the base.
      Retrain at LR/10 so we nudge, not destroy, pretrained weights.

    Why MobileNetV2:
      - 3.4 M parameters vs ResNet50's 25 M → fits in CPU RAM comfortably.
      - ImageNet-pretrained weights have already learned edges, textures,
        shapes — excellent starting point for detecting AI artefacts.
      - preprocess_input maps [0,255] → [-1,1], matching training distribution.

    Why preprocess_input must match pretraining:
      MobileNetV2 weights were computed assuming input in [-1,1]. Using /255.0
      instead shifts every pixel by ~0.5 — the network "sees" a completely
      different distribution and its learned filters no longer fire correctly.
      This is silent: no error is raised, only subtly worse predictions.

    Returns (model, freeze_count) — freeze_count for the report table.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Stage 1: freeze all base layers
    freeze_count = len(base.layers)

    inp = tf.keras.Input(shape=(image_size, image_size, 3), name="input")
    x   = base(inp, training=False)
    x   = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x   = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inp, out, name="VisionGuard_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, freeze_count


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int, lr: float) -> None:
    """
    Unfreeze the last `num_layers` of the base model for fine-tuning.
    Recompile at the lower learning rate.
    """
    base = model.layers[1]  # MobileNetV2 is the second layer after Input
    base.trainable = True
    # Re-freeze all but the last num_layers
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Unfrozen layers in base: {trainable}/{len(base.layers)}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.4 — Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(checkpoint_path: str) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


class MLflowCallback(tf.keras.callbacks.Callback):
    """Task 3.3 — Manual MLflow logging callback."""

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if logs:
            mlflow.log_metrics(
                {f"train_{k}" if not k.startswith("val_") else k: v
                 for k, v in logs.items()},
                step=epoch,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Task 2.5 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def choose_threshold_on_val(model: tf.keras.Model,
                             val_ds: tf.data.Dataset,
                             val_count: int,
                             batch_size: int) -> float:
    """
    Choose the decision threshold that maximises F1 on the validation set.

    Why not use 0.5:
      Class imbalance or asymmetric error costs may warrant a different threshold.
      We evaluate costs on VALIDATION data, not test data, to avoid leaking
      information from the test set into the threshold choice.

    Error cost argument (Task 2.5):
      False negative (predicts "Real" for an AI-generated image) → more harmful.
      A platform that labels a fake as real lets disinformation spread.
      Therefore we bias toward lower thresholds (higher recall for AI-generated).
    """
    val_ds_once, _ = build_dataset(SPLITS_DIR / "val.csv",
                                   model.input_shape[1],
                                   batch_size=batch_size,
                                   augment=False, repeat=False)
    y_true, y_prob = [], []
    for imgs, labels in val_ds_once:
        probs = model.predict(imgs, verbose=0)
        y_prob.extend(probs.flatten().tolist())
        y_true.extend(labels.numpy().tolist())

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.75, 0.05):
        preds = (np.array(y_prob) >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = round(float(t), 2)

    print(f"Best threshold on val: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t


def evaluate_on_test(model: tf.keras.Model, threshold: float,
                     model_label: str, batch_size: int,
                     image_size: int) -> dict:
    """
    Evaluate the model on the test set ONCE.
    Returns a metrics dict and saves confusion matrix + ROC curve figures.
    """
    test_ds, test_count = build_dataset(
        SPLITS_DIR / "test.csv", image_size,
        batch_size=batch_size, augment=False, repeat=False
    )

    y_true, y_prob, x_all = [], [], []
    for imgs, labels in test_ds:
        probs = model.predict(imgs, verbose=0)
        y_prob.extend(probs.flatten().tolist())
        y_true.extend(labels.numpy().tolist())
        x_all.extend(imgs.numpy())

    y_pred = (np.array(y_prob) >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    metrics = dict(
        accuracy=round(acc, 4), precision=round(prec, 4),
        recall=round(rec, 4), f1=round(f1, 4),
        roc_auc=round(auc, 4), threshold=threshold,
    )
    print(f"\n[{model_label}] Test metrics: {metrics}")

    # Confusion matrix figure
    _save_confusion_matrix(cm, model_label)
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    _save_roc_curve(fpr, tpr, auc, model_label)
    # Misclassified grid
    _save_misclassified_grid(x_all, y_true, y_pred, y_prob, model_label)

    return metrics, y_true, y_prob, x_all


def _save_confusion_matrix(cm: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "AI-gen"]); ax.set_yticklabels(["Real", "AI-gen"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = FIGURES_DIR / f"confusion_matrix_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Confusion matrix → {out}")


def _save_roc_curve(fpr, tpr, auc_val, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {label}")
    ax.legend(); plt.tight_layout()
    out = FIGURES_DIR / f"roc_curve_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"ROC curve → {out}")


def _save_misclassified_grid(x_all, y_true, y_pred, y_prob, label: str) -> None:
    """Task 2.6 — 10-image grid of misclassified test samples."""
    wrong_idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    if not wrong_idx:
        print("No misclassifications — skipping grid.")
        return

    sample = wrong_idx[:10]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(f"Misclassified Test Images — {label}", fontsize=13)
    for ax, idx in zip(axes.flatten(), sample):
        img = (np.array(x_all[idx]) + 1.0) / 2.0  # denormalise
        ax.imshow(img.clip(0, 1))
        actual = "AI" if y_true[idx] == 1 else "Real"
        pred   = "AI" if y_pred[idx] == 1 else "Real"
        conf   = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
        ax.set_title(f"True:{actual} Pred:{pred}\nConf:{conf:.2f}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flatten())[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / f"misclassified_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Misclassified grid → {out}")


def _save_training_curves(history, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Find best epoch (min val_loss)
    val_losses = history.history.get("val_loss", [])
    best_ep = int(np.argmin(val_losses)) if val_losses else 0

    for ax, metric, title in zip(
        axes,
        ["loss", "accuracy", "auc"],
        ["Loss", "Accuracy", "AUC"],
    ):
        train_key = metric
        val_key   = f"val_{metric}"
        if train_key not in history.history:
            continue
        ax.plot(history.history[train_key], label="Train")
        ax.plot(history.history[val_key],   label="Val")
        ax.axvline(best_ep, color="red", ls="--", label=f"Best ep {best_ep+1}")
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("Epoch"); ax.legend()

    plt.tight_layout()
    out = FIGURES_DIR / f"curves_{label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Training curves → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 3.3 — MLflow instrumentation
# ══════════════════════════════════════════════════════════════════════════════

def setup_mlflow() -> None:
    """Read tracking URI and credentials from environment only."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("VisionGuard")
    print(f"MLflow tracking URI: {tracking_uri}")


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dvc_data_version() -> str:
    dvc_file = Path("data/raw.dvc")
    if dvc_file.exists():
        import yaml
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return meta.get("outs", [{}])[0].get("md5", "unknown")
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model_type: str = "mobilenet",
    run_name: str = "default_run",
    lr: float = 1e-3,
    dropout: float = 0.4,
    image_size: int = 224,
    epochs: int = 30,
    unfreeze_epochs: int = 10,
    batch_size: int = 32,
    num_unfreeze_layers: int = 20,
    augment: bool = True,
    use_autolog: bool = False,
) -> dict:
    """Train one model configuration and log to MLflow."""

    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n{'='*60}")
        print(f"MLflow run: {run_name}  id={run_id}")
        print(f"{'='*60}")

        # -- Parameters (Task 3.3) ----------------------------------------
        params = dict(
            model_type=model_type, lr=lr, dropout=dropout,
            image_size=image_size, epochs=epochs, batch_size=batch_size,
            augment=augment, seed=SEED,
            num_unfreeze_layers=num_unfreeze_layers,
        )
        mlflow.log_params(params)

        # -- Tags (Task 3.3) -----------------------------------------------
        mlflow.set_tags({
            "git_commit": get_git_sha(),
            "dvc_data_version": get_dvc_data_version(),
            "seed": str(SEED),
            "project": "VisionGuard_23L-0578",
        })

        # -- Datasets ---------------------------------------------------------
        train_ds, train_count = build_dataset(
            SPLITS_DIR / "train.csv", image_size, batch_size, augment=augment
        )
        val_ds, val_count = build_dataset(
            SPLITS_DIR / "val.csv", image_size, batch_size, augment=False
        )
        steps_per_epoch  = max(1, train_count // batch_size)
        validation_steps = max(1, val_count   // batch_size)

        # Measure 1 epoch time first
        print("Measuring 1 epoch …")
        t0 = time.time()
        # (measurement happens during actual first epoch below)

        # -- Build model -------------------------------------------------------
        checkpoint_path = str(MODELS_DIR / f"{run_name}_best.keras")

        if model_type == "cnn":
            model = build_cnn_baseline(image_size, dropout)
            if use_autolog:
                mlflow.tensorflow.autolog(log_models=True)
            history = model.fit(
                train_ds, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=get_callbacks(checkpoint_path) + [MLflowCallback()],
                verbose=1,
            )
            best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        else:  # mobilenet — two-stage training
            model, freeze_count = build_mobilenet(image_size, dropout, num_unfreeze_layers)
            mlflow.log_param("frozen_base_layers", freeze_count)

            if use_autolog:
                mlflow.tensorflow.autolog(log_models=True)

            print("\n--- Stage 1: Feature extraction (base frozen) ---")
            history1 = model.fit(
                train_ds, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=get_callbacks(checkpoint_path) + [MLflowCallback()],
                verbose=1,
            )
            epoch1_time = time.time() - t0
            print(f"Stage 1 complete. Approx time/epoch: {epoch1_time/max(1,len(history1.history['loss'])):.0f}s")

            # Unfreeze top layers and fine-tune
            print(f"\n--- Stage 2: Fine-tuning (top {num_unfreeze_layers} layers unfrozen) ---")
            unfreeze_top_layers(model, num_unfreeze_layers, lr / 10)
            history2 = model.fit(
                train_ds, epochs=unfreeze_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=validation_steps,
                callbacks=get_callbacks(checkpoint_path) + [MLflowCallback()],
                verbose=1,
            )

            # Merge histories for plotting
            class MergedHistory:
                def __init__(self, h1, h2):
                    self.history = {
                        k: h1.history[k] + h2.history.get(k, [])
                        for k in h1.history
                    }
            history = MergedHistory(history1, history2)
            best_epoch = int(np.argmin(history.history["val_loss"])) + 1

            # Save stage curves separately (Task 2.3)
            _save_training_curves(history1, f"{run_name}_stage1_freeze")
            _save_training_curves(history2, f"{run_name}_stage2_unfreeze")

        mlflow.log_param("best_epoch", best_epoch)

        # -- Load best checkpoint -----------------------------------------------
        if Path(checkpoint_path).exists():
            model = tf.keras.models.load_model(checkpoint_path)

        # -- Threshold selection on val (Task 2.5) --------------------------------
        threshold = choose_threshold_on_val(model, val_ds, val_count, batch_size)
        mlflow.log_param("val_threshold", threshold)

        # -- Test evaluation (Task 2.5) — one time only ---------------------------
        metrics, y_true, y_prob, x_all = evaluate_on_test(
            model, threshold, run_name, batch_size, image_size
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # -- Save curves (Task 2.3 / 2.4) -----------------------------------------
        _save_training_curves(history, run_name)

        # -- Attach artifacts (Task 3.4) -------------------------------------------
        for fig_path in FIGURES_DIR.glob(f"*{run_name}*"):
            mlflow.log_artifact(str(fig_path), artifact_path="figures")
        for csv_path in (SPLITS_DIR / "train.csv", SPLITS_DIR / "val.csv", SPLITS_DIR / "test.csv"):
            if csv_path.exists():
                mlflow.log_artifact(str(csv_path), artifact_path="splits")
        mlflow.log_artifact("requirements.txt")

        # Confusion matrix + ROC (already saved above)
        cm_path = FIGURES_DIR / f"confusion_matrix_{run_name.replace(' ','_')}.png"
        roc_path = FIGURES_DIR / f"roc_curve_{run_name.replace(' ','_')}.png"
        for p in [cm_path, roc_path]:
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="eval_figures")

        # Log the trained model (Task 3.4)
        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"\nRun '{run_name}' complete. Run ID: {run_id}")
        print(f"Test metrics: {metrics}")

        return {"run_id": run_id, "metrics": metrics, "threshold": threshold,
                "checkpoint": checkpoint_path}


# ══════════════════════════════════════════════════════════════════════════════
# Sweep: ≥6 runs (Task 3.5)
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_CONFIGS = [
    # run_name,                   model,       lr,    drop, aug,  unfreeze_n
    ("run1_mobilenet_base",       "mobilenet", 1e-3,  0.4,  True,  20),
    ("run2_mobilenet_lr_low",     "mobilenet", 1e-4,  0.4,  True,  20),
    ("run3_mobilenet_lr_high",    "mobilenet", 5e-3,  0.4,  True,  20),
    ("run4_mobilenet_no_aug",     "mobilenet", 1e-3,  0.4,  False, 20),
    ("run5_mobilenet_drop_high",  "mobilenet", 1e-3,  0.6,  True,  20),
    ("run6_mobilenet_unfreeze_5", "mobilenet", 1e-3,  0.4,  True,   5),
    ("run7_cnn_baseline",         "cnn",       1e-3,  0.4,  True,   0),
]


def run_sweep(epochs: int, batch_size: int, image_size: int, max_per_class: int | None) -> None:
    results = []
    for cfg in SWEEP_CONFIGS:
        name, mtype, lr, drop, aug, unf = cfg
        r = train(
            model_type=mtype, run_name=name,
            lr=lr, dropout=drop, augment=aug,
            image_size=image_size, epochs=epochs,
            batch_size=batch_size, num_unfreeze_layers=unf,
        )
        results.append({"run": name, **r["metrics"]})

    df = pd.DataFrame(results)
    out = FIGURES_DIR / "sweep_comparison.csv"
    df.to_csv(out, index=False)
    print("\n=== SWEEP RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard training script")
    parser.add_argument("--model",            default="mobilenet", choices=["cnn", "mobilenet"])
    parser.add_argument("--run-name",         default="default_run")
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--dropout",          type=float, default=0.4)
    parser.add_argument("--image-size",       type=int,   default=224)
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--unfreeze-epochs",  type=int,   default=10)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--num-unfreeze",     type=int,   default=20)
    parser.add_argument("--augment",          action="store_true")
    parser.add_argument("--autolog",          action="store_true")
    parser.add_argument("--sweep",            action="store_true",
                        help="Run the full ≥6 run hyperparameter sweep")
    parser.add_argument("--max-per-class",    type=int, default=None)
    args = parser.parse_args()

    if not (SPLITS_DIR / "train.csv").exists():
        print("ERROR: Split files not found. Run scripts/split_dataset.py first.")
        sys.exit(1)

    if args.sweep:
        run_sweep(args.epochs, args.batch_size, args.image_size, args.max_per_class)
    else:
        # Save augmented batch figure first
        save_augmented_batch_figure(SPLITS_DIR / "train.csv", args.image_size)
        train(
            model_type=args.model,
            run_name=args.run_name,
            lr=args.lr,
            dropout=args.dropout,
            image_size=args.image_size,
            epochs=args.epochs,
            unfreeze_epochs=args.unfreeze_epochs,
            batch_size=args.batch_size,
            num_unfreeze_layers=args.num_unfreeze,
            augment=args.augment,
            use_autolog=args.autolog,
        )


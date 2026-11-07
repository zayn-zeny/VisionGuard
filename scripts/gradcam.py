"""
scripts/gradcam.py — Bonus: Grad-CAM on correct and incorrect predictions
=========================================================================
Produces an 8-panel figure (4 correct + 4 wrong) showing where the champion
model looks when it decides "AI-generated" vs "Real".

Usage:
  python scripts/gradcam.py --model models/champion.keras

Output:
  report/figures/gradcam_grid.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

IMAGE_SIZE = (224, 224)
SPLIT_CSV = Path("data/splits/test.csv")
OUT = Path("report/figures/gradcam_grid.png")


def preprocess(path: str) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return preprocess_input(arr)


def raw_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def find_last_conv(model: tf.keras.Model) -> tuple[tf.keras.Model, str]:
    """Build a model that exposes the last conv activation and the prediction.

    Keras 3 refuses to graft a nested layer's symbolic output (``last_conv.output``)
    onto a different graph, so the gradient model is rebuilt from scratch:
    input -> base (its output IS the final conv feature map for MobileNetV2)
    -> the head layers that followed the base in the trained model.
    """
    base = next(layer for layer in model.layers if hasattr(layer, "layers"))
    convs = [layer for layer in base.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    last_conv = convs[-1]
    if last_conv.output.shape[1:] != base.output.shape[1:]:
        raise ValueError(
            f"Last conv output {last_conv.output.shape} is not the base output "
            f"{base.output.shape}; this Grad-CAM variant expects MobileNetV2."
        )

    tail = model.layers[model.layers.index(base) + 1:]
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    conv_out = base(inputs, training=False)
    x = conv_out
    for layer in tail:
        x = layer(x)
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])
    return grad_model, last_conv.name


def gradcam(grad_model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """Return a [0,1] heatmap for the predicted class score."""
    tensor = tf.convert_to_tensor(image[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tensor)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heat = tf.maximum(heat, 0)
    max_heat = tf.reduce_max(heat)
    heat = heat / (max_heat + 1e-8)
    return heat.numpy()


def overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat_img = Image.fromarray(np.uint8(255 * heat)).resize(IMAGE_SIZE)
    heat_arr = np.array(heat_img) / 255.0
    cmap = matplotlib.colormaps["jet"]
    coloured = cmap(heat_arr)[..., :3]
    return np.clip(0.55 * image + 0.45 * coloured, 0, 1)


def main(model_path: str) -> None:
    model = tf.keras.models.load_model(model_path)
    grad_model, conv_name = find_last_conv(model)
    print(f"Grad-CAM layer: {conv_name}")

    df = pd.read_csv(SPLIT_CSV)
    paths = df["path"].tolist()
    labels = df["label"].tolist()

    correct_ai, correct_real, wrong = [], [], []
    for idx, (path, label) in enumerate(zip(paths, labels)):
        prob = float(model.predict(preprocess(path)[None], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        if pred == label:
            (correct_ai if label == 1 else correct_real).append((idx, prob))
        else:
            wrong.append((idx, prob))

    print(f"correct AI: {len(correct_ai)}  correct Real: {len(correct_real)}  "
          f"wrong: {len(wrong)}")

    picks = (
        [(i, p, True) for i, p in correct_ai[:2]]
        + [(i, p, True) for i, p in correct_real[:2]]
        + [(i, p, False) for i, p in wrong[:4]]
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Grad-CAM — what the model looks at (top: correct, bottom: wrong)",
                 fontsize=13)
    for ax, (idx, prob, is_correct) in zip(axes.flatten(), picks):
        path = paths[idx]
        img = raw_image(path)
        heat = gradcam(grad_model, preprocess(path))
        ax.imshow(overlay(img, heat))
        actual = "AI" if labels[idx] == 1 else "Real"
        verdict = "correct" if is_correct else "WRONG"
        ax.set_title(f"{verdict}: true={actual}\np={prob:.2f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "models/champion.keras"),
    )
    args = parser.parse_args()
    main(args.model)

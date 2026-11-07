"""
scripts/tune.py — Task 2.7 (Apply): Keras Tuner search for VisionGuard
=====================================================================
Searches learning rate, dropout and the number of unfrozen MobileNetV2
layers with a random search, then writes the trial table used in the report.

Usage:
  python scripts/tune.py --trials 8 --epochs 2

Outputs:
  report/figures/tuner_trials.csv   one row per trial (hyperparameters + val AUC)
  report/figures/tuner_best.txt     best configuration + best validation AUC
  tuner_logs/                       Keras Tuner working directory (git-ignored)

Design note — why the best validation score can overfit the validation set:
  Every trial is judged on the SAME validation split. Choosing the maximum of
  eight noisy estimates on that split selects a configuration that is partly
  lucky on those 146 images. The reported score is therefore optimistic; the
  honest number is the single test-set evaluation of the champion, which is
  never used during the search. This is why the sweep/tuner only shortlists
  candidates and the final decision is made once on the test set.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Reuse the exact pipeline/model used by the training script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as vg  # noqa: E402  (seeds/env set above, same as train.py)

import keras_tuner as kt  # noqa: E402
import tensorflow as tf  # noqa: E402

TUNER_DIR = Path("tuner_logs")
FIGURES = Path("report/figures")


def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    """Build a MobileNetV2 with a tuned head and tuned amount of fine-tuning."""
    image_size = 224
    lr = hp.Choice("lr", [1e-4, 5e-4, 1e-3, 3e-3])
    dropout = hp.Float("dropout", 0.2, 0.6, step=0.1)
    n_unfreeze = hp.Choice("unfreeze_layers", [5, 10, 20, 40])

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False

    inp = tf.keras.Input(shape=(image_size, image_size, 3), name="input")
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inp, out, name="VisionGuard_Tuned")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main(trials: int, epochs: int, batch_size: int, resume: bool = False) -> None:
    train_ds, train_count = vg.build_dataset(
        vg.SPLITS_DIR / "train.csv", 224, batch_size, augment=True
    )
    val_ds, val_count = vg.build_dataset(
        vg.SPLITS_DIR / "val.csv", 224, batch_size, augment=False
    )

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=trials,
        executions_per_trial=1,
        overwrite=not resume,
        directory=str(TUNER_DIR),
        project_name="visionguard_tuning",
        seed=vg.SEED,
    )

    tuner.search(
        train_ds,
        epochs=epochs,
        steps_per_epoch=max(1, train_count // batch_size),
        validation_data=val_ds,
        validation_steps=max(1, val_count // batch_size),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
        ],
        verbose=1,
    )

    # ── Trial table for the report ────────────────────────────────────────
    rows = []
    for trial_id, trial in tuner.oracle.trials.items():
        score = trial.score
        rows.append({
            "trial": trial_id,
            "lr": trial.hyperparameters.get("lr"),
            "dropout": trial.hyperparameters.get("dropout"),
            "unfreeze_layers": trial.hyperparameters.get("unfreeze_layers"),
            "val_auc": None if score is None else round(float(score), 4),
            "status": trial.status,
        })

    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    df.to_csv(FIGURES / "tuner_trials.csv", index=False)

    best = tuner.get_best_hyperparameters(1)[0]
    best_auc = float(np.nanmax(df["val_auc"].to_numpy(dtype=float)))
    lines = [
        "Keras Tuner — VisionGuard MobileNetV2",
        f"Search: RandomSearch, {trials} trials, {epochs} epochs/trial",
        f"Best validation AUC: {best_auc:.4f}",
        f"  lr              = {best.get('lr')}",
        f"  dropout         = {best.get('dropout')}",
        f"  unfreeze_layers = {best.get('unfreeze_layers')}",
        "",
        df.to_string(index=False),
    ]
    (FIGURES / "tuner_best.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nTrial table → {FIGURES / 'tuner_trials.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras Tuner study for VisionGuard")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep completed trials in tuner_logs/ and only run the missing ones",
    )
    args = parser.parse_args()
    main(args.trials, args.epochs, args.batch_size, args.resume)

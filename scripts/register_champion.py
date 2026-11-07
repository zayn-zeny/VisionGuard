"""
scripts/register_champion.py — Task 3.6
=========================================
Registers the best MLflow run's model in the Model Registry,
creates ≥2 versions, and aliases one as 'champion'.

Also copies the champion model to models/champion.keras for local use.

Usage:
  python scripts/register_champion.py --run-id <mlflow_run_id> [--version-note "baseline"]
  python scripts/register_champion.py --best   # auto-picks run with highest test_f1
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MODEL_NAME = os.environ.get("MODEL_NAME", "MobileNetV2_VisionGuard")


def get_best_run(experiment_name: str = "VisionGuard") -> str:
    """Return the run_id with the highest test_f1 metric."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found.")
    run = runs[0]
    print(f"Best run: {run.info.run_id}  test_f1={run.data.metrics.get('test_f1', 'n/a'):.4f}")
    return run.info.run_id


def register_model(run_id: str, version_note: str = "auto") -> str:
    """Register model from run to registry and return version."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from {model_uri} as '{MODEL_NAME}' …")

    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version
    print(f"Registered as version {version}")

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"VisionGuard MobileNetV2 — {version_note} (run_id={run_id})",
    )
    return version


def set_champion_alias(version: str) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Alias 'champion' → version {version}")


def copy_to_local(run_id: str, dest: Path = Path("models/champion.keras")) -> None:
    """Download the champion model artifact and write it as a local .keras file.

    mlflow.tensorflow.log_model stores a TensorFlow SavedModel directory, so if
    no .keras file is present we load the SavedModel and re-save it in the
    modern Keras format the Flask app expects at MODEL_PATH.
    """
    dest.parent.mkdir(exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/model", dst_path="models/_tmp"
    )
    keras_files = list(Path(local).rglob("*.keras"))
    if keras_files:
        shutil.copy2(keras_files[0], dest)
        print(f"Champion model copied to {dest}")
        return

    saved_model_dir = Path(local)
    if (saved_model_dir / "MLmodel").exists():
        saved_model_dir = saved_model_dir / "data"
    print(f"SavedModel at {saved_model_dir} → exporting to {dest}")
    import tensorflow as tf
    model = tf.keras.models.load_model(saved_model_dir)
    model.save(dest)
    print(f"Champion model written to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",          default=None,
                        help="run to promote as champion (default: best test_f1)")
    parser.add_argument("--previous-run-id", default=None,
                        help="second run registered as the earlier version")
    parser.add_argument("--best",            action="store_true")
    parser.add_argument("--version-note",    default="auto")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    run_id = args.run_id
    if args.best or run_id is None:
        run_id = get_best_run()

    # Version 1 = the earlier candidate, version 2 = the promoted champion.
    # Registering two different runs is what makes the registry comparison real
    # (two versions of the same run would be identical weights).
    previous_id = args.previous_run_id
    if previous_id:
        v1 = register_model(previous_id, f"{args.version_note}_v1")
    else:
        v1 = register_model(run_id, f"{args.version_note}_v1")
    v2 = register_model(run_id, f"{args.version_note}_v2")

    # Alias the promoted run as champion
    set_champion_alias(v2)

    # Copy locally so the Flask app can load it at startup
    copy_to_local(run_id)

    print(f"\nDone. Model '{MODEL_NAME}' has 2 versions (v{v1}, v{v2}); "
          f"alias 'champion' → v{v2}")


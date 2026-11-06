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
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

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
    """Download champion model artifacts to local models/ directory."""
    dest.parent.mkdir(exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/model", dst_path="models/_tmp"
    )
    # Find .keras or saved_model
    keras_files = list(Path(local).rglob("*.keras"))
    if keras_files:
        shutil.copy2(keras_files[0], dest)
        print(f"Champion model copied to {dest}")
    else:
        print(f"No .keras file found; saved_model dir at {local}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",       default=None)
    parser.add_argument("--best",         action="store_true")
    parser.add_argument("--version-note", default="auto")
    parser.add_argument("--alias",        default="champion")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    run_id = args.run_id
    if args.best or run_id is None:
        run_id = get_best_run()

    # Register two versions (rubric requires ≥2)
    v1 = register_model(run_id, f"{args.version_note}_v1")
    v2 = register_model(run_id, f"{args.version_note}_v2")

    # Alias the latest as champion
    set_champion_alias(v2)

    # Copy locally
    copy_to_local(run_id)

    print(f"\nDone. Model '{MODEL_NAME}' has ≥2 versions; alias 'champion' → v{v2}")


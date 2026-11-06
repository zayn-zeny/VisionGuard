"""
scripts/predict_by_alias.py — Task 3.6
========================================
Resolves the champion model by its MLflow Registry alias and classifies one image.

Usage:
  python scripts/predict_by_alias.py --image path/to/image.jpg [--alias champion]

This demonstrates that the model can be fetched by alias from the registry
without hardcoding a version number, making champion promotion automatic.

How the champion reaches the server without per-request DagsHub calls:
  At container build time (or startup), this script (or an equivalent
  `mlflow artifacts download`) pulls the model once and saves it locally.
  The Flask app then loads from the local path at startup. This means:
  - Zero latency from registry calls per request
  - No dependency on DagsHub being reachable at request time
  - The model is baked into the deployment artifact, versioned by its hash
  See: Dockerfile COPY or docker-compose volume mount.
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

IMAGE_SIZE = (224, 224)
MODEL_NAME = os.environ.get("MODEL_NAME", "MobileNetV2_VisionGuard")


def download_champion(alias: str = "champion") -> str:
    """
    Resolve model by alias from MLflow Model Registry and return local path.
    Downloads to models/ directory if not already present.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    print(f"Resolving alias '{alias}' for model '{MODEL_NAME}' …")
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    version = mv.version
    print(f"  → Version {version}  (run_id={mv.run_id})")

    local_path = Path("models") / f"{MODEL_NAME}_v{version}.keras"
    if local_path.exists():
        print(f"  → Already cached at {local_path}")
        return str(local_path)

    # Download from registry
    print(f"  → Downloading model artifacts …")
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    local_dir = mlflow.artifacts.download_artifacts(model_uri, dst_path="models/")
    # Find the .keras file
    keras_files = list(Path(local_dir).rglob("*.keras"))
    if not keras_files:
        # Fallback: saved_model directory
        return local_dir
    return str(keras_files[0])


def preprocess(image_path: str) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def classify_image(model_path: str, image_path: str, threshold: float) -> dict:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    arr = preprocess(image_path)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "AI-generated" if prob >= threshold else "Real"
    return {"label": label, "confidence": round(prob, 4), "threshold": threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image using the champion model")
    parser.add_argument("--image",     required=True, help="Path to image file")
    parser.add_argument("--alias",     default="champion", help="Model registry alias")
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("THRESHOLD", "0.5")))
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        model_path = download_champion(args.alias)
    except Exception as e:
        # Fallback: use local MODEL_PATH if registry is unavailable
        model_path = os.environ.get("MODEL_PATH", "models/champion.keras")
        print(f"WARNING: Registry lookup failed ({e}). Using local path: {model_path}")

    result = classify_image(model_path, args.image, args.threshold)
    print(f"\n{'='*50}")
    print(f"Image     : {args.image}")
    print(f"Label     : {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Threshold : {result['threshold']}")
    print(f"{'='*50}")


"""
app/inference.py — Model inference layer for VisionGuard.

Supports two modes, selected by INFERENCE_MODE env var:
  - "local"  : loads the champion Keras model at app startup, runs inference
                in-process. Used in Tasks 4-6 (full Compose stack).
  - "remote" : sends a multipart POST request to the cloud inference API.
                Used in Tasks 7-8 (Flask on localhost, model on cloud).

Design rationale:
  Both modes expose the same function signature: classify(image_bytes) -> dict
  so routes.py never needs to know which backend is active. The swap is
  purely a configuration choice (INFERENCE_MODE in .env), not a code change.

Why load once at startup:
  Loading a Keras model (MobileNetV2 weights ~14 MB on disk, several hundred
  MB in RAM after layer initialisation) takes ~2-5 seconds. Doing that inside
  each request handler would impose that latency on every user and cause heavy
  memory churn under concurrent load. Loading once and storing the reference
  in module state means the model is warm for every subsequent request.

Why preprocessing must match training exactly:
  MobileNetV2 was pretrained on images normalised to [-1, 1] via its own
  preprocess_input function. If we serve with a different normalisation
  (e.g. /255.0) the input distribution silently shifts and predictions
  degrade—a form of training-serving skew that produces no error, only bad
  results. The same preprocess_input call used during training is therefore
  reproduced verbatim here.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ── Module-level model cache (local mode) ────────────────────────────────────
_model = None
_model_path: Optional[str] = None

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)  # MobileNetV2 native input size


def load_model_once(model_path: str) -> None:
    """Load the Keras model into module state.  Called once at app startup."""
    global _model, _model_path
    if _model is not None and _model_path == model_path:
        logger.info("Model already loaded from %s — skipping reload.", model_path)
        return

    # Import here so the module can be imported (for tests) even without TF
    import tensorflow as tf  # noqa: F401 — registers backend
    from tensorflow import keras

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Champion model not found at '{model_path}'. "
            "Run scripts/train.py and register the model, or set MODEL_PATH."
        )

    logger.info("Loading champion model from %s …", model_path)
    t0 = time.time()
    _model = keras.models.load_model(model_path)
    _model_path = model_path
    logger.info("Model loaded in %.2f s.", time.time() - t0)


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes, resize to IMAGE_SIZE, apply MobileNetV2
    preprocess_input, and return a (1, H, W, 3) float32 array.

    Resize strategy: squash (resize without cropping).  This distorts
    aspect ratio for non-square images but keeps all pixels in frame,
    avoiding accidental cropping of discriminative regions.  Chosen over
    centre-crop because the dataset contains full-frame AI artefacts that
    concentrate at edges; a crop risks discarding them.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)        # (H, W, 3) in [0, 255]
    arr = preprocess_input(arr)                  # → [-1, 1]
    return np.expand_dims(arr, axis=0)           # (1, H, W, 3)


def _image_meta(image_bytes: bytes) -> dict:
    """Return width, height, sha256 for a raw image byte string."""
    sha = hashlib.sha256(image_bytes).hexdigest()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
    except Exception:
        w, h = None, None
    return {"sha256": sha, "width": w, "height": h}


# ── Public interface ──────────────────────────────────────────────────────────

def classify(image_bytes: bytes, threshold: float, mode: str,
             remote_url: str = "", remote_key: str = "") -> dict:
    """
    Classify image_bytes as AI-generated or real.

    Parameters
    ----------
    image_bytes : raw upload bytes
    threshold   : decision threshold (applied to sigmoid output)
    mode        : "local" | "remote"
    remote_url  : cloud API endpoint (remote mode only)
    remote_key  : API key header value (remote mode only)

    Returns
    -------
    dict with keys: label, confidence, latency_ms, mode, error (if any)
    """
    meta = _image_meta(image_bytes)
    t0 = time.time()

    if mode == "local":
        result = _classify_local(image_bytes, threshold)
    elif mode == "remote":
        result = _classify_remote(image_bytes, threshold, remote_url, remote_key)
    else:
        raise ValueError(f"Unknown INFERENCE_MODE: {mode!r}")

    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    result["mode"] = mode
    result["meta"] = meta
    return result


def _classify_local(image_bytes: bytes, threshold: float) -> dict:
    """Run inference with the in-process Keras model."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_once() at startup.")

    arr = _preprocess_image(image_bytes)
    prob = float(_model.predict(arr, verbose=0)[0][0])  # sigmoid output
    label = "AI-generated" if prob >= threshold else "Real"
    return {"label": label, "confidence": round(prob, 4), "error": None}


def _classify_remote(image_bytes: bytes, threshold: float,
                     remote_url: str, remote_key: str) -> dict:
    """POST image bytes to the cloud inference API and parse response."""
    if not remote_url:
        raise ValueError("REMOTE_API_URL is not set.")

    headers = {"X-API-Key": remote_key}
    files = {"image": ("upload.jpg", io.BytesIO(image_bytes), "image/jpeg")}

    try:
        resp = requests.post(remote_url, headers=headers, files=files, timeout=30)
    except requests.exceptions.Timeout:
        return {"label": "Error", "confidence": 0.0, "error": "timeout"}
    except requests.exceptions.ConnectionError as exc:
        return {"label": "Error", "confidence": 0.0, "error": f"connection: {exc}"}

    if resp.status_code == 401:
        return {"label": "Error", "confidence": 0.0, "error": "unauthorized (401)"}
    if resp.status_code >= 500:
        return {"label": "Error", "confidence": 0.0,
                "error": f"server error ({resp.status_code})"}

    try:
        data = resp.json()
        prob = float(data["confidence"])
        label = "AI-generated" if prob >= threshold else "Real"
        return {"label": label, "confidence": round(prob, 4), "error": None}
    except (KeyError, ValueError) as exc:
        return {"label": "Error", "confidence": 0.0,
                "error": f"bad response format: {exc}"}


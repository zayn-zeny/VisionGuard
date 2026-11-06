"""
cloud_api/main.py — VisionGuard Cloud Inference API (Task 7)
==============================================================
A standalone FastAPI (or Flask) app deployed to the cloud.
Accepts an image POST, runs the champion model, returns JSON.

This is the REMOTE INFERENCE ENDPOINT that the Flask web app calls
when INFERENCE_MODE=remote.

Authentication: X-API-Key header checked against API_KEY env var.

Request:
  POST /predict
  Header: X-API-Key: <secret>
  Body: multipart/form-data with field "image" (JPEG/PNG/WebP, ≤10 MB)

Response 200:
  {"label": "AI-generated"|"Real", "confidence": 0.8532, "model": "MobileNetV2_VisionGuard"}

Error responses:
  401: missing/invalid API key
  415: unsupported file type
  422: corrupt image
  500: inference error
"""
from __future__ import annotations

import hashlib
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, UnidentifiedImageError

# Use FastAPI for async support and automatic OpenAPI docs
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY     = os.environ.get("REMOTE_API_KEY", "")
MODEL_PATH  = os.environ.get("MODEL_PATH", "models/champion.keras")
IMAGE_SIZE  = (224, 224)
MAX_SIZE_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "10"))

_model = None


def _load_model() -> None:
    global _model
    import tensorflow as tf
    from tensorflow import keras
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH} …")
    t0 = time.time()
    _model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded in {time.time()-t0:.2f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="VisionGuard Inference API",
    description="Classify images as AI-generated or Real. Roll: 23L-0578.",
    version="1.0.0",
    lifespan=lifespan,
)


def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return  # no key configured — open (dev mode only)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _preprocess(image_bytes: bytes) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    # Auth
    _check_api_key(x_api_key)

    # Size check
    contents = await image.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    # Content type check
    ct = image.content_type or ""
    if ct and ct not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {ct}")

    # Decode check
    try:
        img_check = Image.open(io.BytesIO(contents))
        img_check.verify()
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    # Inference
    try:
        arr  = _preprocess(contents)
        prob = float(_model.predict(arr, verbose=0)[0][0])
        label = "AI-generated" if prob >= 0.5 else "Real"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "label":      label,
        "confidence": round(prob, 4),
        "model":      os.environ.get("MODEL_NAME", "MobileNetV2_VisionGuard"),
        "sha256":     hashlib.sha256(contents).hexdigest()[:16],
    }


"""
vercel_api/api/index.py — VisionGuard inference API on Vercel (Task 7).

A FastAPI app run by Vercel's Python runtime as one serverless function.
It serves the champion model exported to TFLite (scripts/export_tflite.py):
full TensorFlow (~1.4 GB) cannot fit Vercel's ~250 MB function limit, while
the LiteRT interpreter plus a 9 MB float32 model fits easily and gives the
same probabilities (parity checked on the whole test split).

Endpoints
    GET  /         service info
    GET  /health   liveness + whether the model is loaded
    POST /predict  multipart field "image", header X-API-Key
                   -> {"label", "confidence", "threshold", "model", "sha256"}

Config (Vercel project environment variables; never in the repo)
    REMOTE_API_KEY  shared secret expected in X-API-Key (required)
    THRESHOLD       decision threshold, default 0.4
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

try:  # LiteRT on Vercel (Linux)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # local Windows test: same kernels shipped inside TensorFlow
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "champion.tflite"
MODEL_NAME = "MobileNetV2_VisionGuard"
IMAGE_SIZE = (224, 224)
# Vercel rejects request bodies above ~4.5 MB before they reach the function;
# the Flask client therefore sends a 224x224 PNG (see app/inference.py).
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "4.4")) * 1024 * 1024)
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="VisionGuard Inference API (23L-0578_A2_Ops)", version="2.0.0")

# Loaded once per warm function instance, never per request.
_interp = Interpreter(model_path=str(MODEL_PATH))
_interp.allocate_tensors()
_IN = _interp.get_input_details()[0]["index"]
_OUT = _interp.get_output_details()[0]["index"]


def _threshold() -> float:
    return float(os.environ.get("THRESHOLD", "0.4"))


def _preprocess(data: bytes) -> np.ndarray:
    """Identical to training/serving: RGB -> 224x224 bilinear squash ->
    mobilenet_v2.preprocess_input, which is x / 127.5 - 1."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr[None, ...]


@app.get("/")
def root():
    return {"service": "VisionGuard Inference API", "project": "23L-0578_A2_Ops",
            "endpoints": {"GET /health": "liveness", "POST /predict": "image + X-API-Key"}}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "model": MODEL_NAME,
            "runtime": "LiteRT (TFLite)", "threshold": _threshold()}


@app.post("/predict")
async def predict(image: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("REMOTE_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the upload limit.")
    if image.content_type and image.content_type not in ALLOWED:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported content type: {image.content_type}")
    try:
        Image.open(io.BytesIO(data)).verify()
        x = _preprocess(data)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    _interp.set_tensor(_IN, x)
    _interp.invoke()
    prob = float(_interp.get_tensor(_OUT)[0][0])
    thr = _threshold()
    return {"label": "AI-generated" if prob >= thr else "Real",
            "confidence": round(prob, 4), "threshold": thr, "model": MODEL_NAME,
            "sha256": hashlib.sha256(data).hexdigest()[:16]}

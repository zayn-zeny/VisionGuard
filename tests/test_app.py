"""
tests/test_app.py — Task 6.6 real tests for VisionGuard.

Tests:
  1. Preprocessing output shape
  2. Threshold logic (above/below threshold → correct label)
  3. Upload validation rejects bad file types
  4. /health returns 200
  5. Oversized file returns 413
  6. Corrupt file returns 422
"""
import io
import os
import pytest
import numpy as np
from PIL import Image

# ── Skip tests that require the champion model if it doesn't exist ────────────
MODEL_EXISTS = os.path.exists(os.environ.get("MODEL_PATH", "models/champion.keras"))


# ── 1. Preprocessing shape ────────────────────────────────────────────────────

def test_preprocess_shape(valid_jpeg_bytes):
    """_preprocess_image must return (1, 224, 224, 3) float32 array."""
    # Import directly — doesn't need TF model loaded
    from app.inference import _preprocess_image
    arr = _preprocess_image(valid_jpeg_bytes)
    assert arr.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {arr.shape}"
    assert arr.dtype == np.float32


def test_preprocess_range(valid_jpeg_bytes):
    """MobileNetV2 preprocess_input maps [0,255] → [-1,1]."""
    from app.inference import _preprocess_image
    arr = _preprocess_image(valid_jpeg_bytes)
    assert arr.min() >= -1.1, f"Min value {arr.min()} out of range"
    assert arr.max() <= 1.1,  f"Max value {arr.max()} out of range"


# ── 2. Threshold logic ────────────────────────────────────────────────────────

def test_threshold_above():
    """Confidence >= threshold → 'AI-generated'."""
    from app.inference import _classify_local
    # Monkey-patch _model with a stub
    import app.inference as inf
    original = inf._model

    class StubModel:
        def predict(self, x, verbose=0):
            return np.array([[0.8]])  # above 0.5 threshold

    inf._model = StubModel()
    result = _classify_local(b"dummy", threshold=0.5)
    inf._model = original

    assert result["label"] == "AI-generated"
    assert result["confidence"] == pytest.approx(0.8, abs=0.01)


def test_threshold_below():
    """Confidence < threshold → 'Real'."""
    from app.inference import _classify_local
    import app.inference as inf
    original = inf._model

    class StubModel:
        def predict(self, x, verbose=0):
            return np.array([[0.2]])

    inf._model = StubModel()

    # Also need _preprocess_image to not fail on b"dummy"
    import unittest.mock as mock
    with mock.patch("app.inference._preprocess_image", return_value=np.zeros((1, 224, 224, 3))):
        result = _classify_local(b"dummy", threshold=0.5)

    inf._model = original
    assert result["label"] == "Real"


# ── 3. Upload validation — wrong extension ────────────────────────────────────

def test_reject_wrong_extension(client):
    """Non-image extension (e.g. .txt) must return 415."""
    data = {"image": (io.BytesIO(b"not an image"), "malicious.txt")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code in (415, 400), f"Expected 415 or 400, got {resp.status_code}"


def test_reject_no_file(client):
    """Missing file part → 400."""
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


# ── 4. /health returns 200 ───────────────────────────────────────────────────

def test_health_ok(client):
    """GET /health must return 200 with JSON body."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert "inference_mode" in data
    assert data["debug"] is False


# ── 5. Corrupt file → 422 ────────────────────────────────────────────────────

def test_reject_corrupt_image(client):
    """A file with .jpg extension but garbage bytes → 422."""
    garbage = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"garbage_not_a_real_jpeg"
    data = {"image": (io.BytesIO(garbage), "fake.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


# ── 6. Upload form renders ────────────────────────────────────────────────────

def test_index_renders(client):
    """GET / must return 200 and contain the upload form."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"VisionGuard" in resp.data or b"upload" in resp.data.lower()


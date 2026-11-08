"""tests/test_app.py — Task 6.6 real tests for VisionGuard.

Tests:
  1. Preprocessing output shape / value range
  2. Threshold logic (above/below threshold → correct label)
  3. Upload validation rejects bad file types
  4. /health returns 200
  5. Corrupt file returns 422
  6. Successful upload persists a row and renders a verdict
"""
import io
import unittest.mock as mock

import numpy as np
import pytest


# ── 1. Preprocessing shape ────────────────────────────────────────────────────

def test_preprocess_shape(valid_jpeg_bytes):
    """_preprocess_image must return (1, 224, 224, 3) float32 array."""
    from app.inference import _preprocess_image
    arr = _preprocess_image(valid_jpeg_bytes)
    assert arr.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {arr.shape}"
    assert arr.dtype == np.float32


def test_preprocess_range(valid_jpeg_bytes):
    """MobileNetV2 preprocess_input maps [0,255] → [-1,1]."""
    from app.inference import _preprocess_image
    arr = _preprocess_image(valid_jpeg_bytes)
    assert arr.min() >= -1.1, f"Min value {arr.min()} out of range"
    assert arr.max() <= 1.1, f"Max value {arr.max()} out of range"


# ── 2. Threshold logic ────────────────────────────────────────────────────────

class _StubModel:
    def __init__(self, probability):
        self.probability = probability

    def predict(self, x, verbose=0):
        return np.array([[self.probability]])


def _classify_with_stub(probability):
    import app.inference as inf
    original = inf._model
    inf._model = _StubModel(probability)
    try:
        with mock.patch("app.inference._preprocess_image",
                        return_value=np.zeros((1, 224, 224, 3), dtype=np.float32)):
            return inf._classify_local(b"dummy", threshold=0.5)
    finally:
        inf._model = original


def test_threshold_above():
    """Confidence >= threshold → 'AI-generated'."""
    result = _classify_with_stub(0.8)
    assert result["label"] == "AI-generated"
    assert result["confidence"] == pytest.approx(0.8, abs=0.01)


def test_threshold_below():
    """Confidence < threshold → 'Real'."""
    result = _classify_with_stub(0.2)
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


# ── 7. Happy path writes the database trail (Task 5) ─────────────────────────

def test_successful_upload_persists_row(app, client, valid_jpeg_bytes):
    """A valid upload returns a verdict and writes exactly one predictions row."""
    from app.database import Prediction

    data = {"image": (io.BytesIO(valid_jpeg_bytes), "portrait.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302, f"Expected redirect to /result, got {resp.status_code}"
    assert "/result" in resp.headers["Location"]

    result_page = client.get("/result")
    assert result_page.status_code == 200
    assert b"AI-generated" in result_page.data or b"Real" in result_page.data

    with app.app_context():
        assert Prediction.query.count() == 1
        row = Prediction.query.first()
        assert row.filename_sanitized == "portrait.jpg"
        assert row.label in {"AI-generated", "Real"}
        assert 0.0 <= row.confidence <= 1.0
        assert len(row.image_sha256) == 64

    history = client.get("/history")
    assert history.status_code == 200
    assert b"portrait.jpg" in history.data


def test_postgres_url_pins_psycopg2_driver():
    """Regression (Task 6.5): SQLAlchemy 2.1 defaults postgresql:// to psycopg v3,
    which is not installed; the app must pin the psycopg2 driver explicitly."""
    from app import _normalise_db_url
    assert _normalise_db_url("postgresql://u:p@db:5432/x") == "postgresql+psycopg2://u:p@db:5432/x"
    assert _normalise_db_url("postgres://u:p@db/x") == "postgresql+psycopg2://u:p@db/x"
    assert _normalise_db_url("sqlite:///a.db") == "sqlite:///a.db"


def test_remote_payload_preserves_model_input():
    """Remote mode sends a 224x224 PNG (Vercel body limit ~4.5 MB). The model
    input built from that PNG must equal the one built from the original."""
    import io
    import numpy as np
    from PIL import Image
    from app.inference import _preprocess_image, _resized_png

    rng = np.random.default_rng(0)
    big = Image.fromarray(rng.integers(0, 255, (900, 1300, 3), dtype=np.uint8))
    buf = io.BytesIO()
    big.save(buf, format="JPEG", quality=95)
    original = buf.getvalue()

    payload = _resized_png(original)
    assert Image.open(io.BytesIO(payload)).size == (224, 224)
    assert np.array_equal(_preprocess_image(original), _preprocess_image(payload))

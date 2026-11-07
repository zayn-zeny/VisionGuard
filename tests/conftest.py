"""tests/conftest.py — pytest fixtures for VisionGuard tests."""
import io
import os

import numpy as np
import pytest
from PIL import Image

# Point at a test SQLite DB so tests never touch production
os.environ.setdefault("DATABASE_URL", "sqlite:///test_visionguard.db")
os.environ.setdefault("INFERENCE_MODE", "local")
os.environ.setdefault("MODEL_PATH", "models/champion.keras")
os.environ.setdefault("THRESHOLD", "0.5")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")
os.environ.setdefault("MODEL_NAME", "MobileNetV2_VisionGuard")
os.environ.setdefault("MODEL_VERSION", "champion")
os.environ.setdefault("MAX_CONTENT_LENGTH_MB", "10")


class _StubModel:
    """Deterministic stand-in used when the champion .keras file is absent (CI)."""

    def predict(self, x, verbose=0):
        return np.array([[0.8]])


@pytest.fixture
def app(monkeypatch):
    """Create Flask app in test mode (local inference, SQLite).

    The champion model file is git-ignored, so CI has no copy of it. When the
    file is missing we substitute a stub model: the tests then verify routing,
    validation, persistence and response shape without depending on 19 MB of
    weights being present in the repository.
    """
    import app.inference as inference

    if not os.path.exists(os.environ.get("MODEL_PATH", "models/champion.keras")):
        monkeypatch.setattr(inference, "load_model_once", lambda path: None)
        monkeypatch.setattr(inference, "_model", _StubModel())

    from app import create_app
    application = create_app(config_override={
        "TESTING": True,
        "INFERENCE_MODE": "local",
        "DATABASE_URL": "sqlite:///:memory:",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate a minimal valid JPEG image in memory."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

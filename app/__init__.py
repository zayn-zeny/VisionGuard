"""
app/__init__.py — Application factory for VisionGuard Flask app.
"""
import os
from flask import Flask
from dotenv import load_dotenv
from .database import db, init_db


def _normalise_db_url(url: str) -> str:
    """Pin the Postgres driver explicitly.

    SQLAlchemy 2.1 changed the default driver for ``postgresql://`` URLs
    from psycopg2 to psycopg (v3). The image ships psycopg2-binary, so a
    bare ``postgresql://`` URL crashed the workers at boot with
    ``ModuleNotFoundError: No module named 'psycopg'``.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def create_app(config_override: dict = None) -> Flask:
    """Create and configure the Flask application."""
    load_dotenv()

    app = Flask(__name__)

    # ── Core settings from environment ────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    app.config["DEBUG"] = False  # always off; task 4.4 requirement
    app.config["SQLALCHEMY_DATABASE_URI"] = _normalise_db_url(
        os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # Test each pooled connection before use. Without this, restarting the
    # db container left dead connections in the pool and the next request
    # failed with "server closed the connection unexpectedly" (Task 6.5).
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("MAX_CONTENT_LENGTH_MB", "10")
    ) * 1024 * 1024  # bytes

    # ── Inference settings ────────────────────────────────────────────
    app.config["INFERENCE_MODE"] = os.environ.get("INFERENCE_MODE", "local")
    app.config["MODEL_PATH"] = os.environ.get("MODEL_PATH", "models/champion.keras")
    app.config["THRESHOLD"] = float(os.environ.get("THRESHOLD", "0.5"))
    app.config["REMOTE_API_URL"] = os.environ.get("REMOTE_API_URL", "")
    app.config["REMOTE_API_KEY"] = os.environ.get("REMOTE_API_KEY", "")
    app.config["MODEL_NAME"] = os.environ.get("MODEL_NAME", "MobileNetV2_VisionGuard")
    app.config["MODEL_VERSION"] = os.environ.get("MODEL_VERSION", "champion")

    if config_override:
        app.config.update(config_override)

    # ── Database ─────────────────────────────────────────────────────
    db.init_app(app)
    with app.app_context():
        init_db(app)

    # ── Load model at startup (local mode only) ───────────────────────
    if app.config["INFERENCE_MODE"] == "local":
        from .inference import load_model_once
        load_model_once(app.config["MODEL_PATH"])

    # ── Register routes ───────────────────────────────────────────────
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app


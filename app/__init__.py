"""
app/__init__.py — Application factory for VisionGuard Flask app.
"""
import os
from flask import Flask
from dotenv import load_dotenv
from .database import db, init_db


def create_app(config_override: dict = None) -> Flask:
    """Create and configure the Flask application."""
    load_dotenv()

    app = Flask(__name__)

    # ── Core settings from environment ────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
    app.config["DEBUG"] = False  # always off; task 4.4 requirement
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///visionguard.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
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


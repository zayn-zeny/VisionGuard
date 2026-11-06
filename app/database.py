"""
app/database.py — SQLAlchemy model for prediction history.

Every prediction written to this table, regardless of inference mode.
Schema captures enough for Task 5 and the report's history discussion.
"""
from __future__ import annotations
import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

db = SQLAlchemy()


class Prediction(db.Model):
    """One row per classify request."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)

    # -- When ----------------------------------------------------------------
    timestamp_utc = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
    )

    # -- Decision ------------------------------------------------------------
    label = db.Column(db.String(16), nullable=False)      # "AI-generated" | "Real"
    confidence = db.Column(db.Float, nullable=False)       # raw sigmoid output [0,1]
    threshold = db.Column(db.Float, nullable=False)        # threshold used this request

    # -- Model provenance ----------------------------------------------------
    model_name = db.Column(db.String(128), nullable=False)
    model_version = db.Column(db.String(64), nullable=False)  # e.g. "champion"
    inference_mode = db.Column(db.String(16), nullable=False)  # "local" | "remote"

    # -- Image fingerprint ---------------------------------------------------
    image_sha256 = db.Column(db.String(64), nullable=False)
    image_size_bytes = db.Column(db.Integer, nullable=False)
    image_width = db.Column(db.Integer, nullable=True)
    image_height = db.Column(db.Integer, nullable=True)
    filename_sanitized = db.Column(db.String(256), nullable=False)

    # -- Performance ---------------------------------------------------------
    request_latency_ms = db.Column(db.Float, nullable=True)

    # -- Client info ---------------------------------------------------------
    client_ip = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} label={self.label!r} "
            f"conf={self.confidence:.3f} ts={self.timestamp_utc}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inference_mode": self.inference_mode,
            "image_sha256": self.image_sha256,
            "image_size_bytes": self.image_size_bytes,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "filename_sanitized": self.filename_sanitized,
            "request_latency_ms": self.request_latency_ms,
        }


def init_db(app: Flask) -> None:
    """Create tables if they don't exist. Called from app factory."""
    db.create_all()


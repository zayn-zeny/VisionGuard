"""
app/routes.py — Flask route handlers for VisionGuard.

Routes:
  GET  /          → upload form
  POST /predict   → classify uploaded image, write DB row, redirect to result
  GET  /result    → show verdict + confidence for last prediction
  GET  /history   → paginated prediction log
  GET  /health    → JSON health check (used by Docker HEALTHCHECK)

Security notes (Task 4.2):
  - Allowed extensions: .jpg .jpeg .png .webp
  - MIME validated against file content, not just extension
  - MAX_CONTENT_LENGTH enforced by Flask (raises 413)
  - Filenames sanitized via Werkzeug secure_filename
  - Files never written to disk — processed in memory
  - Friendly error pages for all failure cases
"""
from __future__ import annotations

import datetime
import hashlib
import io
import logging
import time

from flask import (
    Blueprint, current_app, flash, jsonify, redirect,
    render_template, request, session, url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from .database import Prediction, db
from .inference import classify

logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _validate_image(file_bytes: bytes) -> tuple[bool, str]:
    """
    Attempt to open file_bytes as a PIL image.
    Returns (ok, error_message).
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()           # raises if file is truncated or corrupt
        return True, ""
    except UnidentifiedImageError:
        return False, "File appears to be an image but could not be decoded."
    except Exception as exc:
        return False, f"Image validation failed: {exc}"


def _get_image_dimensions(file_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return img.size  # (width, height)
    except Exception:
        return None, None


# ── Routes ───────────────────────────────────────────────────────────────────

@main_bp.route("/", methods=["GET"])
def index():
    """Upload form."""
    return render_template("upload.html")


@main_bp.route("/predict", methods=["POST"])
def predict():
    """Accept image upload, run classifier, persist row, redirect to result."""
    cfg = current_app.config

    # 1. File present?
    if "image" not in request.files:
        flash("No file part in request.", "error")
        return render_template("error.html",
                               error_code=400,
                               message="No file was attached to the request."), 400

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.", "error")
        return render_template("error.html",
                               error_code=400,
                               message="No file was selected."), 400

    # 2. Extension check
    safe_name = secure_filename(file.filename)
    extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "?"
    if not _allowed_file(safe_name):
        return render_template(
            "error.html",
            error_code=415,
            message=(
                f"Unsupported file type: '{extension}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        ), 415

    # 3. MIME check
    mime = file.content_type or ""
    if mime and mime not in ALLOWED_MIMES:
        return render_template(
            "error.html",
            error_code=415,
            message=f"Unsupported MIME type: '{mime}'. Upload a JPEG, PNG, or WebP image.",
        ), 415

    # 4. Read into memory (Flask already enforces MAX_CONTENT_LENGTH → 413)
    file_bytes = file.read()

    # 5. Image decode check
    ok, decode_error = _validate_image(file_bytes)
    if not ok:
        return render_template(
            "error.html",
            error_code=422,
            message=decode_error,
        ), 422

    # 6. Run classifier
    t_start = time.time()
    result = classify(
        image_bytes=file_bytes,
        threshold=cfg["THRESHOLD"],
        mode=cfg["INFERENCE_MODE"],
        remote_url=cfg.get("REMOTE_API_URL", ""),
        remote_key=cfg.get("REMOTE_API_KEY", ""),
    )
    latency_ms = round((time.time() - t_start) * 1000, 1)

    if result.get("error"):
        return render_template(
            "error.html",
            error_code=502,
            message=f"Inference error: {result['error']}",
        ), 502

    # 7. Image metadata
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    width, height = _get_image_dimensions(file_bytes)

    # 8. Persist prediction row
    row = Prediction(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
        label=result["label"],
        confidence=result["confidence"],
        threshold=cfg["THRESHOLD"],
        model_name=cfg["MODEL_NAME"],
        model_version=cfg["MODEL_VERSION"],
        inference_mode=cfg["INFERENCE_MODE"],
        image_sha256=sha256,
        image_size_bytes=len(file_bytes),
        image_width=width,
        image_height=height,
        filename_sanitized=safe_name,
        request_latency_ms=result.get("latency_ms", latency_ms),
        client_ip=request.remote_addr,
        user_agent=request.user_agent.string[:512] if request.user_agent else None,
    )
    db.session.add(row)
    db.session.commit()

    # 9. Store result ID in session for /result
    session["last_prediction_id"] = row.id
    return redirect(url_for("main.result"))


@main_bp.route("/result", methods=["GET"])
def result():
    """Display the latest prediction result."""
    pred_id = session.get("last_prediction_id")
    if pred_id is None:
        return redirect(url_for("main.index"))

    row = db.session.get(Prediction, pred_id)
    if row is None:
        return redirect(url_for("main.index"))

    return render_template("result.html", pred=row)


@main_bp.route("/history", methods=["GET"])
def history():
    """Paginated prediction history, newest first."""
    page = request.args.get("page", 1, type=int)
    per_page = 20
    pagination = (
        Prediction.query.order_by(Prediction.timestamp_utc.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return render_template("history.html", pagination=pagination)


@main_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint consumed by Docker HEALTHCHECK and monitoring.

    Returns 200 with JSON body when the app is up and the DB is reachable.
    Returns 503 if the DB query fails.

    Why debug=False matters: Flask's built-in Werkzeug debugger exposes an
    interactive Python console in the browser, protected only by a PIN.
    With debug=True in production that PIN can be brute-forced or leaked via
    logs, giving an attacker arbitrary code execution on the server.  It also
    prints full tracebacks in HTTP responses, leaking stack traces, library
    versions, and file paths — information an attacker can use to tailor
    exploits.  Setting debug=False (enforced in app/__init__.py) eliminates
    both risks.
    """
    try:
        count = Prediction.query.count()
        db_status = "ok"
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return jsonify({"status": "unhealthy", "db": str(exc)}), 503

    return jsonify({
        "status": "healthy",
        "inference_mode": current_app.config["INFERENCE_MODE"],
        "model_name": current_app.config["MODEL_NAME"],
        "model_version": current_app.config["MODEL_VERSION"],
        "db": db_status,
        "prediction_count": count,
        "debug": current_app.debug,
    }), 200


# ── 413 / 500 error handlers ─────────────────────────────────────────────────

@main_bp.app_errorhandler(413)
def file_too_large(e):
    limit_mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return render_template(
        "error.html",
        error_code=413,
        message=f"File too large. Maximum upload size is {limit_mb} MB.",
    ), 413


@main_bp.app_errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500 error")
    return render_template(
        "error.html",
        error_code=500,
        message="An unexpected error occurred. Please try again.",
    ), 500


# ── VisionGuard Cloud Inference API (Task 7) ────────────────────────────────
# Deployed as a Docker Space on Hugging Face (pathway C, "any other cloud").
#
# Build context = this folder. The deploy script (scripts/deploy_hf_space.py)
# stages main.py, requirements.txt, this Dockerfile and models/champion.keras
# into the Space repository, so the model is BUILT INTO the image: the API
# never calls DagsHub at request time.
#
# Hugging Face Spaces routes public HTTPS traffic to port 7860 and runs the
# container as UID 1000, so the non-root user is created with that UID.

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home appuser

WORKDIR /app

# Dependencies first so code edits reuse the TensorFlow layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_PATH=models/champion.keras \
    THRESHOLD=0.4

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

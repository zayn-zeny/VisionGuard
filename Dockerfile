# ── VisionGuard Dockerfile ──────────────────────────────────────────────────
#
# Design decisions (Task 6.2):
#
# Base image: python:3.11-slim-bookworm
#   - "slim" strips development headers, documentation and non-essential locale
#     data, saving ~200 MB vs the full image.
#   - Python 3.11 is used here (not 3.13) because tensorflow-cpu 2.21 produces
#     smaller wheels for 3.11; both are compatible.
#
# tensorflow-cpu not tensorflow:
#   - The GPU variant bundles CUDA libraries (~2 GB). Our server has no GPU
#     (confirmed in Task 1.1) so shipping CUDA wastes disk and RAM and adds
#     unnecessary attack surface. tensorflow-cpu is functionally identical
#     for inference on a CPU host.
#
# Layer cache order (requirements before source):
#   - Docker layers are cached by content hash. If we copy source first,
#     any code edit invalidates the pip install layer and re-downloads ~1 GB
#     of packages on every build. Copying requirements.txt first means the
#     install layer is rebuilt ONLY when dependencies change, not when
#     application code changes. This turns a 10-minute rebuild into a 10-second one.
#
# Non-root user:
#   - Running as root inside a container means a container escape or file-write
#     vulnerability grants the attacker root on the host (with --privileged) or
#     the ability to overwrite host-mounted files. The non-root user 'appuser'
#     limits blast radius to the app directory only.
#
# HEALTHCHECK:
#   - Docker Compose uses this to signal when the container is ready.
#     The web service waits on condition: service_healthy for the db service,
#     and the db's own HEALTHCHECK. Without a HEALTHCHECK, Compose would start
#     web before the DB is accepting connections.

FROM python:3.11-slim-bookworm

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ─────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Install dependencies BEFORE copying source (layer cache) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/        ./app/
COPY scripts/    ./scripts/
COPY models/     ./models/
COPY .env.example .env.example

# ── Environment defaults (overridden by docker-compose .env) ─────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

# ── Switch to non-root user ───────────────────────────────────────────────────
RUN chown -R appuser:appgroup /app
USER appuser

# ── Health check (used by Docker Compose condition: service_healthy) ──────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

# ── Start Gunicorn (not Flask dev server — debug=False enforced in app) ───────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", \
     "--timeout", "120", "--access-logfile", "-", "app.wsgi:application"]


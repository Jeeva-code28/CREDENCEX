# ============================================================
#  CREDENCEX — Production Dockerfile for Render Deployment
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

# System deps for OpenCV, scikit-image, and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

# Runtime system libs for OpenCV / image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Use the lighter render config (en_core_web_sm, reduced workers/batch)
ENV CONFIG_PATH=configs/render_config.yaml

# Ensure HuggingFace cache dir exists (Render persistent disk mounts here)
RUN mkdir -p /opt/render/.cache/huggingface

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/render
USER appuser

# Render injects $PORT at runtime (default 10000)
ENV PORT=10000
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Start with gunicorn (uvicorn worker) — Render sends SIGTERM for graceful shutdown
CMD ["sh", "-c", "gunicorn src.api.app:app --bind 0.0.0.0:${PORT} --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120 --graceful-timeout 30"]

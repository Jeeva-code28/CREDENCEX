# ============================================================
#  CREDENCEX — Dockerfile for HuggingFace Spaces (FREE)
# ============================================================

FROM python:3.12-slim

# System deps for OpenCV, scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces runs as user with uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Download spaCy model at build time
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Use cloud-optimized config
ENV CONFIG_PATH=configs/render_config.yaml
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# HuggingFace Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

RUN chown -R user:user /app /tmp
USER user

CMD ["sh", "-c", "gunicorn src.api.app:app --bind 0.0.0.0:7860 --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120"]

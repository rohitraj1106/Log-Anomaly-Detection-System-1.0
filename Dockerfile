# ==============================================================================
# Multi-stage Dockerfile — Scalable Log Anomaly Detection Platform
# ==============================================================================
# Stage 1: Builder — install dependencies
# Stage 2: Runtime — slim production image
# ==============================================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="ML Engineering Team"
LABEL description="Scalable Log Anomaly Detection Platform"
LABEL version="1.1.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs models/artifacts \
    features/store experiments/runs experiments/evaluations \
    monitoring/metrics \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs \
    MODEL_ARTIFACTS_DIR=/app/models/artifacts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose ports
# 8000 = FastAPI
# 8501 = Streamlit Dashboard
# 9090 = Metrics
EXPOSE 8000 8501 9090

# Default: run the training pipeline then start the API
# PORT env var is injected by cloud platforms (Render, Railway, Fly.io)
CMD ["sh", "-c", "python main.py --mode train --logs 5000 --no-tune && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

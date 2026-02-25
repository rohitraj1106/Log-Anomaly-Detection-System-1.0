"""
FastAPI Application — Model Serving REST API.

Production-grade inference API with:
- POST /predict-log — single log prediction
- POST /predict-batch — batch prediction
- GET /health — health check
- GET /model-info — model metadata
- GET /metrics — Prometheus-compatible monitoring metrics

Features:
- Model and vectorizer preloading
- Async request handling with executor offloading
- Input validation via Pydantic
- Structured JSON responses
- Request latency tracking
- Rate limiting (SlowAPI)
- API key authentication
- Request ID correlation for distributed tracing
- CORS with restricted origins
- Error handling middleware
"""

import asyncio
import os
import sys
import time
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader

from api.schemas import (
    LogEntry,
    BatchLogRequest,
    LogPredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    PredictionResult,
    ErrorResponse,
)
from pipelines.ingestion import LogIngestionEngine
from pipelines.preprocessing import LogPreprocessor
from features.engineering import FeatureEngineer
from models.trainer import ModelTrainer
from monitoring.metrics import MetricsCollector
from utils.logger import get_logger

logger = get_logger(__name__)

# Thread pool for CPU-bound inference (prevents blocking the async event loop)
_executor = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state for model serving."""

    def __init__(self) -> None:
        self.model_trainer: Optional[ModelTrainer] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.ingestion_engine: Optional[LogIngestionEngine] = None
        self.preprocessor: Optional[LogPreprocessor] = None
        self.metrics_collector: MetricsCollector = MetricsCollector()
        self.start_time: float = time.time()
        self.total_predictions: int = 0
        self.anomaly_count: int = 0
        self.error_count: int = 0
        self.latencies: list = []
        self.model_loaded: bool = False


state = AppState()

# =============================================================================
# Settings
# =============================================================================

# Load settings with graceful fallback
try:
    from utils.settings import settings
    ALLOWED_ORIGINS = settings.allowed_origins_list
    API_KEY = settings.api_key
    RATE_LIMIT = settings.rate_limit
except Exception:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]
    API_KEY = os.getenv("LADP_API_KEY")
    RATE_LIMIT = os.getenv("LADP_RATE_LIMIT", "100/minute")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Log Anomaly Detection API",
    description=(
        "Production-grade REST API for detecting anomalies in distributed system logs. "
        "Supports single and batch prediction with real-time scoring."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — restricted to explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting (graceful fallback if slowapi not installed)
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": str(exc)},
        )
except ImportError:
    limiter = None
    logger.warning("slowapi not installed — rate limiting disabled")


# =============================================================================
# API Key Authentication
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Verify API key if configured. Passes through if no key is set."""
    if not API_KEY:
        return None  # No auth required
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# =============================================================================
# PII Masking
# =============================================================================

import re

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}


def mask_pii(text: str) -> str:
    """Mask PII patterns in log messages for security."""
    for name, pattern in _PII_PATTERNS.items():
        if name == "email":
            text = pattern.sub("[EMAIL_REDACTED]", text)
        elif name == "ip_address":
            text = pattern.sub("[IP_REDACTED]", text)
        elif name == "ssn":
            text = pattern.sub("[SSN_REDACTED]", text)
        elif name == "credit_card":
            text = pattern.sub("[CC_REDACTED]", text)
    return text


# =============================================================================
# Startup / Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Load model and initialize components on startup."""
    logger.info("🚀 Starting Log Anomaly Detection API v1.1.0...")

    state.ingestion_engine = LogIngestionEngine()
    state.preprocessor = LogPreprocessor()

    # Try to load saved model
    artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts")
    try:
        state.model_trainer = ModelTrainer.load_model(artifacts_dir=artifacts_dir)
        state.model_loaded = True
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "⚠️ No saved model found. Run the training pipeline first. "
            "API will return errors for prediction endpoints."
        )
        state.model_loaded = False

    # Try to load feature engineer (with fitted vectorizer)
    vectorizer_path = Path(artifacts_dir) / "latest" / "feature_engineer.pkl"
    if vectorizer_path.exists():
        with open(vectorizer_path, "rb") as f:
            state.feature_engineer = pickle.load(f)
        logger.info("✅ Feature engineer loaded")
    else:
        state.feature_engineer = FeatureEngineer()
        logger.warning("⚠️ No saved feature engineer found, using fresh instance")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    _executor.shutdown(wait=False)
    logger.info("Shutting down API...")


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add request ID for distributed tracing and log all requests."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    # Attach request ID for downstream use
    request.state.request_id = request_id

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

    logger.info(
        f"{request.method} {request.url.path} — {response.status_code} ({duration_ms:.1f}ms)",
        extra={"correlation_id": request_id, "duration_ms": duration_ms},
    )
    return response


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    state.error_count += 1
    state.metrics_collector.record_error()
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception (request_id={request_id}): {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _process_log_entry(entry: LogEntry) -> pd.DataFrame:
    """Convert a log entry to a preprocessed DataFrame."""
    log_dict = {
        "timestamp": entry.timestamp or datetime.now(timezone.utc).isoformat(),
        "level": entry.level or "INFO",
        "service": entry.service or "unknown",
        "source_ip": entry.source_ip or "",
        "message": entry.message,
    }
    df = state.ingestion_engine.ingest_dict(log_dict)
    df = state.preprocessor.preprocess(df)
    return df


def _predict_dataframe_sync(df: pd.DataFrame) -> tuple:
    """Run prediction on a preprocessed DataFrame (sync, for executor)."""
    if not state.model_loaded or state.model_trainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    # Feature engineering
    if state.feature_engineer and state.feature_engineer._is_fitted:
        features = state.feature_engineer.transform(df)
    else:
        features = state.feature_engineer.fit_transform(df)[0]

    labels = state.model_trainer.predict(features)
    scores = state.model_trainer.predict_proba(features)
    return labels, scores


async def _predict_dataframe(df: pd.DataFrame) -> tuple:
    """Run prediction off the event loop using a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _predict_dataframe_sync, df)


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/predict-log", response_model=LogPredictionResponse)
async def predict_single_log(
    entry: LogEntry,
    _api_key: Optional[str] = Depends(verify_api_key),
) -> LogPredictionResponse:
    """
    Predict whether a single log entry is anomalous.

    Returns the anomaly score, label, and confidence.
    """
    start = time.perf_counter()

    try:
        df = _process_log_entry(entry)
        labels, scores = await _predict_dataframe(df)

        label = int(labels[0])
        score = float(scores[0])
        is_anomaly = label == -1

        result = PredictionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_label=label,
            confidence=abs(score - 0.5) * 2,
        )

        inference_time = (time.perf_counter() - start) * 1000
        state.total_predictions += 1
        if is_anomaly:
            state.anomaly_count += 1
        state.latencies.append(inference_time)

        # Record to metrics collector
        state.metrics_collector.record_prediction(score, is_anomaly, inference_time)

        return LogPredictionResponse(
            prediction=result,
            log_entry=entry.model_dump(),
            model_version=state.model_trainer.version or "unknown",
            inference_time_ms=round(inference_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        state.error_count += 1
        state.metrics_collector.record_error()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchLogRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
) -> BatchPredictionResponse:
    """
    Predict anomalies for a batch of log entries.

    Efficient batch processing for high-throughput scenarios.
    """
    start = time.perf_counter()

    try:
        # Process all logs
        dfs = [_process_log_entry(entry) for entry in request.logs]
        combined_df = pd.concat(dfs, ignore_index=True)

        labels, scores = await _predict_dataframe(combined_df)

        predictions = []
        total_anomalies = 0
        for i in range(len(labels)):
            label = int(labels[i])
            score = float(scores[i])
            is_anomaly = label == -1
            if is_anomaly:
                total_anomalies += 1

            predictions.append(PredictionResult(
                is_anomaly=is_anomaly,
                anomaly_score=score,
                anomaly_label=label,
                confidence=abs(score - 0.5) * 2,
            ))

        inference_time = (time.perf_counter() - start) * 1000
        state.total_predictions += len(labels)
        state.anomaly_count += total_anomalies
        state.latencies.append(inference_time)

        # Record to metrics collector
        for i in range(len(labels)):
            state.metrics_collector.record_prediction(
                float(scores[i]), labels[i] == -1, inference_time / len(labels)
            )

        return BatchPredictionResponse(
            predictions=predictions,
            total_logs=len(labels),
            total_anomalies=total_anomalies,
            anomaly_rate=total_anomalies / max(len(labels), 1),
            model_version=state.model_trainer.version or "unknown",
            inference_time_ms=round(inference_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        state.error_count += 1
        state.metrics_collector.record_error()
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and orchestration."""
    return HealthResponse(
        status="healthy" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        model_version=(
            state.model_trainer.version if state.model_trainer else None
        ),
        uptime_seconds=round(time.time() - state.start_time, 1),
        total_predictions=state.total_predictions,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    _api_key: Optional[str] = Depends(verify_api_key),
) -> ModelInfoResponse:
    """Get metadata about the currently loaded model."""
    if not state.model_loaded or not state.model_trainer:
        raise HTTPException(status_code=503, detail="No model loaded")

    meta = state.model_trainer.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_version=meta.get("version", "unknown"),
        training_samples=meta.get("training_samples", 0),
        feature_count=meta.get("feature_count", 0),
        trained_at=meta.get("trained_at", "unknown"),
        params=meta.get("params", {}),
    )


@app.get("/metrics")
async def get_metrics(request: Request) -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in the Prometheus text exposition format for scraping
    by Prometheus server or other monitoring tools.
    """
    prometheus_text = state.metrics_collector.get_prometheus_metrics()

    # Also inject API-level counters
    extra = "\n".join([
        "",
        "# HELP ladp_api_uptime_seconds API uptime in seconds",
        "# TYPE ladp_api_uptime_seconds gauge",
        f"ladp_api_uptime_seconds {time.time() - state.start_time:.1f}",
        "",
        "# HELP ladp_model_loaded Whether the model is loaded (1=yes, 0=no)",
        "# TYPE ladp_model_loaded gauge",
        f"ladp_model_loaded {1 if state.model_loaded else 0}",
    ])

    return PlainTextResponse(
        content=prometheus_text + extra,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Log Anomaly Detection API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /predict-log",
            "POST /predict-batch",
            "GET /health",
            "GET /model-info",
            "GET /metrics",
        ],
    }

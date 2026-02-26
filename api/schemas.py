"""
Pydantic Schemas for the Log Anomaly Detection API.

Provides input validation, serialization, and OpenAPI documentation
for all API endpoints.
"""

from typing import Any

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """Single log entry for prediction."""

    timestamp: str | None = Field(
        None,
        description="Log timestamp in ISO format or common formats",
        example="2024-01-15 10:30:15",
    )
    level: str | None = Field(
        "INFO",
        description="Log level",
        example="ERROR",
    )
    service: str | None = Field(
        "unknown",
        description="Source service name",
        example="auth-service",
    )
    source_ip: str | None = Field(
        None,
        description="Source IP address",
        example="192.168.1.100",
    )
    message: str = Field(
        ...,
        description="Log message content",
        example="User login failed: invalid credentials",
        min_length=1,
        max_length=10000,
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "timestamp": "2024-01-15 10:30:15",
                    "level": "ERROR",
                    "service": "auth-service",
                    "source_ip": "45.33.32.156",
                    "message": "ALERT: Brute force attack detected — 500 failed attempts",
                }
            ]
        }


class BatchLogRequest(BaseModel):
    """Batch of log entries for prediction."""

    logs: list[LogEntry] = Field(
        ...,
        description="List of log entries",
        min_length=1,
        max_length=10000,
    )


class PredictionResult(BaseModel):
    """Single prediction result."""

    is_anomaly: bool = Field(..., description="Whether the log is anomalous")
    anomaly_score: float = Field(
        ...,
        description="Anomaly probability score (0-1, higher=more anomalous)",
    )
    anomaly_label: int = Field(
        ...,
        description="1=normal, -1=anomaly",
    )
    confidence: float = Field(
        ...,
        description="Confidence of the prediction (0-1)",
    )


class LogPredictionResponse(BaseModel):
    """Response for single log prediction."""

    prediction: PredictionResult
    log_entry: dict[str, Any]
    model_version: str
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for batch log prediction."""

    predictions: list[PredictionResult]
    total_logs: int
    total_anomalies: int
    anomaly_rate: float
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    model_version: str | None = None
    uptime_seconds: float = 0.0
    total_predictions: int = 0


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    model_version: str
    training_samples: int
    feature_count: int
    trained_at: str
    params: dict[str, Any]


class MetricsResponse(BaseModel):
    """Monitoring metrics response."""

    total_predictions: int
    anomaly_count: int
    anomaly_rate: float
    avg_latency_ms: float
    p99_latency_ms: float
    error_count: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    status_code: int

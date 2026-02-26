"""
Production Monitoring & Observability Module.

Implements:
- Prediction count tracking
- Anomaly rate monitoring
- Data drift detection
- Model latency tracking
- Alert thresholding
- Metrics export (Prometheus-style)
- Simulated dashboard metrics

In production, integrate with:
- Prometheus + Grafana
- Datadog / New Relic
- CloudWatch / Stackdriver
"""

import json
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np

from utils.helpers import ensure_directory, safe_divide
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Thread-safe metrics collector for production monitoring.

    Tracks:
    - Prediction counts and rates
    - Anomaly counts and rates
    - Latency percentiles
    - Error counts
    - Data drift signals
    """

    def __init__(
        self,
        window_size: int = 10000,
        export_dir: str = "monitoring/metrics",
    ) -> None:
        self._lock = threading.Lock()
        self._window_size = window_size
        self._export_dir = ensure_directory(export_dir)

        # Rolling windows
        self._prediction_times: deque[float] = deque(maxlen=window_size)
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._anomaly_flags: deque[bool] = deque(maxlen=window_size)
        self._scores: deque[float] = deque(maxlen=window_size)

        # Counters
        self._total_predictions: int = 0
        self._total_anomalies: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()

        # Drift tracking
        self._reference_scores: np.ndarray | None = None
        self._drift_alerts: list[dict[str, Any]] = []

        # Alert history
        self._alerts: list[dict[str, Any]] = []

        logger.info("MetricsCollector initialized")

    def record_prediction(
        self,
        anomaly_score: float,
        is_anomaly: bool,
        latency_ms: float,
    ) -> None:
        """Record a single prediction for monitoring."""
        with self._lock:
            now = time.time()
            self._prediction_times.append(now)
            self._latencies.append(latency_ms)
            self._anomaly_flags.append(is_anomaly)
            self._scores.append(anomaly_score)

            self._total_predictions += 1
            if is_anomaly:
                self._total_anomalies += 1

    def record_error(self) -> None:
        """Record a prediction error."""
        with self._lock:
            self._total_errors += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current monitoring metrics snapshot."""
        with self._lock:
            latencies = list(self._latencies)
            anomaly_flags = list(self._anomaly_flags)
            scores = list(self._scores)

        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "predictions": {
                "total": self._total_predictions,
                "window_size": len(latencies),
                "rate_per_second": self._compute_rate(),
            },
            "anomalies": {
                "total": self._total_anomalies,
                "window_count": sum(anomaly_flags) if anomaly_flags else 0,
                "window_rate": safe_divide(sum(anomaly_flags), len(anomaly_flags))
                if anomaly_flags
                else 0.0,
                "overall_rate": safe_divide(self._total_anomalies, self._total_predictions),
            },
            "latency": self._compute_latency_stats(latencies),
            "scores": self._compute_score_stats(scores),
            "errors": {
                "total": self._total_errors,
                "error_rate": safe_divide(
                    self._total_errors,
                    self._total_predictions + self._total_errors,
                ),
            },
            "drift": {
                "alerts": len(self._drift_alerts),
                "latest_alert": self._drift_alerts[-1] if self._drift_alerts else None,
            },
        }
        return metrics

    def check_alerts(
        self,
        anomaly_rate_warning: float = 0.1,
        anomaly_rate_critical: float = 0.25,
        latency_p99_threshold_ms: float = 500.0,
        error_rate_threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Check current metrics against alert thresholds."""
        alerts = []
        metrics = self.get_metrics()

        anomaly_rate = metrics["anomalies"]["window_rate"]
        if anomaly_rate > anomaly_rate_critical:
            alert = {
                "severity": "CRITICAL",
                "type": "anomaly_rate",
                "message": f"Anomaly rate {anomaly_rate:.1%} exceeds critical threshold {anomaly_rate_critical:.1%}",
                "timestamp": datetime.now(UTC).isoformat(),
                "value": anomaly_rate,
            }
            alerts.append(alert)
            self._alerts.append(alert)
        elif anomaly_rate > anomaly_rate_warning:
            alert = {
                "severity": "WARNING",
                "type": "anomaly_rate",
                "message": f"Anomaly rate {anomaly_rate:.1%} exceeds warning threshold {anomaly_rate_warning:.1%}",
                "timestamp": datetime.now(UTC).isoformat(),
                "value": anomaly_rate,
            }
            alerts.append(alert)
            self._alerts.append(alert)

        p99_lat = metrics["latency"].get("p99", 0)
        if p99_lat > latency_p99_threshold_ms:
            alert = {
                "severity": "WARNING",
                "type": "latency",
                "message": f"P99 latency {p99_lat:.0f}ms exceeds threshold {latency_p99_threshold_ms:.0f}ms",
                "timestamp": datetime.now(UTC).isoformat(),
                "value": p99_lat,
            }
            alerts.append(alert)
            self._alerts.append(alert)

        error_rate = metrics["errors"]["error_rate"]
        if error_rate > error_rate_threshold:
            alert = {
                "severity": "CRITICAL",
                "type": "error_rate",
                "message": f"Error rate {error_rate:.1%} exceeds threshold {error_rate_threshold:.1%}",
                "timestamp": datetime.now(UTC).isoformat(),
                "value": error_rate,
            }
            alerts.append(alert)
            self._alerts.append(alert)

        return alerts

    def set_reference_scores(self, scores: np.ndarray) -> None:
        """Set reference score distribution for drift detection."""
        self._reference_scores = scores
        logger.info(f"Reference scores set: {len(scores)} samples")

    def check_drift(self, threshold: float = 0.05) -> dict[str, Any] | None:
        """Check for data drift against reference distribution."""
        if self._reference_scores is None:
            return None

        current_scores = np.array(list(self._scores))
        if len(current_scores) < 100:
            return None

        try:
            from scipy import stats as scipy_stats

            statistic, p_value = scipy_stats.ks_2samp(self._reference_scores, current_scores)

            drift_detected = p_value < threshold
            result = {
                "drift_detected": drift_detected,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "threshold": threshold,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if drift_detected:
                self._drift_alerts.append(result)
                logger.warning(f"⚠️ Data drift detected: p-value={p_value:.6f}")

            return result
        except ImportError:
            return None

    def export_metrics(self, filename: str | None = None) -> str:
        """Export current metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = self._export_dir / filename
        metrics = self.get_metrics()
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        return str(path)

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format."""
        metrics = self.get_metrics()
        lines = [
            "# HELP ladp_predictions_total Total number of predictions",
            "# TYPE ladp_predictions_total counter",
            f"ladp_predictions_total {metrics['predictions']['total']}",
            "",
            "# HELP ladp_anomalies_total Total anomalies detected",
            "# TYPE ladp_anomalies_total counter",
            f"ladp_anomalies_total {metrics['anomalies']['total']}",
            "",
            "# HELP ladp_anomaly_rate Current anomaly rate",
            "# TYPE ladp_anomaly_rate gauge",
            f"ladp_anomaly_rate {metrics['anomalies']['window_rate']:.6f}",
            "",
            "# HELP ladp_latency_avg_ms Average prediction latency",
            "# TYPE ladp_latency_avg_ms gauge",
            f"ladp_latency_avg_ms {metrics['latency'].get('mean', 0):.2f}",
            "",
            "# HELP ladp_latency_p99_ms P99 prediction latency",
            "# TYPE ladp_latency_p99_ms gauge",
            f"ladp_latency_p99_ms {metrics['latency'].get('p99', 0):.2f}",
            "",
            "# HELP ladp_errors_total Total prediction errors",
            "# TYPE ladp_errors_total counter",
            f"ladp_errors_total {metrics['errors']['total']}",
        ]
        return "\n".join(lines)

    def _compute_rate(self) -> float:
        """Compute predictions per second over the last minute."""
        now = time.time()
        with self._lock:
            recent = [t for t in self._prediction_times if now - t < 60]
        return len(recent) / 60.0 if recent else 0.0

    @staticmethod
    def _compute_latency_stats(latencies: list) -> dict[str, float]:
        """Compute latency percentile statistics."""
        if not latencies:
            return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

        arr = np.array(latencies)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _compute_score_stats(scores: list) -> dict[str, float]:
        """Compute anomaly score statistics."""
        if not scores:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @property
    def alert_history(self) -> list[dict[str, Any]]:
        return list(self._alerts)

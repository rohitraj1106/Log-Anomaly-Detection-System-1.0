"""
Unit Tests — Monitoring Metrics Collector.

Tests metrics recording, alerting, drift detection, and Prometheus export.
"""

import pytest
import numpy as np

from monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def setup_method(self):
        self.collector = MetricsCollector(export_dir="monitoring/metrics")

    @pytest.mark.unit
    def test_initialization(self):
        """Collector initializes with zero counters."""
        metrics = self.collector.get_metrics()
        assert metrics["predictions"]["total"] == 0
        assert metrics["anomalies"]["total"] == 0
        assert metrics["errors"]["total"] == 0

    @pytest.mark.unit
    def test_record_prediction(self):
        """Recording a prediction increments the counter."""
        self.collector.record_prediction(
            anomaly_score=0.3, is_anomaly=False, latency_ms=10.5
        )
        metrics = self.collector.get_metrics()
        assert metrics["predictions"]["total"] == 1
        assert metrics["anomalies"]["total"] == 0

    @pytest.mark.unit
    def test_record_anomaly(self):
        """Recording an anomalous prediction increments both counters."""
        self.collector.record_prediction(
            anomaly_score=0.9, is_anomaly=True, latency_ms=15.0
        )
        metrics = self.collector.get_metrics()
        assert metrics["predictions"]["total"] == 1
        assert metrics["anomalies"]["total"] == 1

    @pytest.mark.unit
    def test_record_error(self):
        """Error counter increments correctly."""
        self.collector.record_error()
        self.collector.record_error()
        metrics = self.collector.get_metrics()
        assert metrics["errors"]["total"] == 2

    @pytest.mark.unit
    def test_latency_statistics(self):
        """Latency stats are computed correctly."""
        for latency in [10, 20, 30, 40, 50]:
            self.collector.record_prediction(0.3, False, latency)

        metrics = self.collector.get_metrics()
        assert metrics["latency"]["mean"] == pytest.approx(30.0, rel=0.01)
        assert metrics["latency"]["p99"] >= 10

    @pytest.mark.unit
    def test_alert_anomaly_rate_warning(self):
        """Alert triggered when anomaly rate exceeds warning threshold."""
        for _ in range(80):
            self.collector.record_prediction(0.3, False, 10)
        for _ in range(20):
            self.collector.record_prediction(0.9, True, 10)

        alerts = self.collector.check_alerts(anomaly_rate_warning=0.1)
        anomaly_alerts = [a for a in alerts if a["type"] == "anomaly_rate"]
        assert len(anomaly_alerts) > 0

    @pytest.mark.unit
    def test_no_alert_below_threshold(self):
        """No alert when metrics are within bounds."""
        for _ in range(100):
            self.collector.record_prediction(0.3, False, 10)

        alerts = self.collector.check_alerts()
        anomaly_alerts = [a for a in alerts if a["type"] == "anomaly_rate"]
        assert len(anomaly_alerts) == 0

    @pytest.mark.unit
    def test_prometheus_export_format(self):
        """Prometheus export produces valid text format."""
        self.collector.record_prediction(0.3, False, 10)
        output = self.collector.get_prometheus_metrics()

        assert "ladp_predictions_total" in output
        assert "ladp_anomalies_total" in output
        assert "# HELP" in output
        assert "# TYPE" in output

    @pytest.mark.unit
    def test_drift_detection_no_reference(self):
        """Drift check returns None when no reference is set."""
        result = self.collector.check_drift()
        assert result is None

    @pytest.mark.unit
    def test_drift_detection_with_reference(self):
        """Drift check returns a result when reference scores exist."""
        # Set reference scores
        rng = np.random.RandomState(42)
        self.collector.set_reference_scores(rng.randn(500))

        # Add enough current scores (from different distribution for drift)
        for _ in range(150):
            self.collector.record_prediction(
                anomaly_score=float(rng.randn() + 5), is_anomaly=False, latency_ms=10
            )

        result = self.collector.check_drift()
        assert result is not None
        assert "drift_detected" in result
        assert "p_value" in result

    @pytest.mark.unit
    def test_metrics_export_to_file(self, tmp_path):
        """Metrics can be exported to a JSON file."""
        collector = MetricsCollector(export_dir=str(tmp_path))
        collector.record_prediction(0.3, False, 10)
        path = collector.export_metrics("test_metrics.json")
        assert "test_metrics.json" in path

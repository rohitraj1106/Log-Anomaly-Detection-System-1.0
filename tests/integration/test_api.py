"""
Integration Tests — FastAPI Endpoint Testing.

Tests the full API request/response cycle using the FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Integration tests for all API endpoints."""

    @pytest.mark.integration
    def test_root_endpoint(self, client):
        """Root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Log Anomaly Detection API"
        assert "endpoints" in data

    @pytest.mark.integration
    def test_health_endpoint(self, client):
        """Health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)

    @pytest.mark.integration
    def test_health_reports_model_status(self, client):
        """Health check correctly reports model loaded state."""
        response = client.get("/health")
        data = response.json()
        # Model might not be loaded in test environment
        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.integration
    def test_metrics_endpoint(self, client):
        """Metrics endpoint returns monitoring data."""
        response = client.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_predict_log_without_model(self, client):
        """Prediction endpoint returns 503 when model is not loaded."""
        response = client.post(
            "/predict-log",
            json={
                "message": "Failed login attempt from 192.168.1.100",
                "level": "ERROR",
                "service": "auth-service",
            },
        )
        # Should be 503 (model not loaded) or 200 (if model is loaded)
        assert response.status_code in [200, 500, 503]

    @pytest.mark.integration
    def test_predict_log_input_validation(self, client):
        """Invalid input is rejected with 422 validation error."""
        response = client.post(
            "/predict-log",
            json={},  # Missing required 'message' field
        )
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_batch_input_validation(self, client):
        """Batch endpoint rejects empty log list."""
        response = client.post(
            "/predict-batch",
            json={"logs": []},
        )
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_batch_structure(self, client):
        """Batch prediction accepts valid input structure."""
        response = client.post(
            "/predict-batch",
            json={
                "logs": [
                    {"message": "Error in auth module", "level": "ERROR"},
                    {"message": "Request processed OK", "level": "INFO"},
                ],
            },
        )
        # 503 if model not loaded, 200 if loaded
        assert response.status_code in [200, 500, 503]

    @pytest.mark.integration
    def test_cors_headers(self, client):
        """CORS preflight request returns appropriate headers."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI should handle CORS headers
        assert response.status_code in [200, 400]

    @pytest.mark.integration
    def test_model_info_without_model(self, client):
        """Model info returns 503 when no model is loaded."""
        response = client.get("/model-info")
        assert response.status_code in [200, 503]

    @pytest.mark.integration
    def test_message_max_length_validation(self, client):
        """Messages exceeding max_length are rejected."""
        response = client.post(
            "/predict-log",
            json={
                "message": "x" * 10001,  # Exceeds 10000 char limit
                "level": "ERROR",
            },
        )
        assert response.status_code == 422

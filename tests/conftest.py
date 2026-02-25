"""
Shared test fixtures for the Log Anomaly Detection Platform.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_log_lines():
    """Sample raw log lines in various formats."""
    return [
        '2024-01-15 10:30:15 [ERROR] auth-service: User login failed: invalid credentials',
        '2024-01-15 10:30:16 [INFO] api-gateway: Request processed successfully',
        '2024-01-15 10:30:17 [WARNING] db-service: Connection pool running low',
        '2024-01-15 10:30:18 [DEBUG] cache-service: Cache hit for key user:1234',
        '2024-01-15 10:30:19 [CRITICAL] payment-service: Transaction timeout after 30s',
    ]


@pytest.fixture
def sample_log_df():
    """Pre-parsed log DataFrame matching the ingestion schema."""
    return pd.DataFrame({
        "timestamp": [
            "2024-01-15 10:30:15",
            "2024-01-15 10:30:16",
            "2024-01-15 10:30:17",
            "2024-01-15 10:30:18",
            "2024-01-15 10:30:19",
        ],
        "level": ["ERROR", "INFO", "WARNING", "DEBUG", "CRITICAL"],
        "service": ["auth-service", "api-gateway", "db-service", "cache-service", "payment-service"],
        "source_ip": ["192.168.1.100", "", "10.0.0.5", "", "172.16.0.1"],
        "message": [
            "User login failed: invalid credentials",
            "Request processed successfully",
            "Connection pool running low",
            "Cache hit for key user:1234",
            "Transaction timeout after 30s",
        ],
        "raw_log": ["raw1", "raw2", "raw3", "raw4", "raw5"],
        "log_hash": ["hash1", "hash2", "hash3", "hash4", "hash5"],
        "parse_method": ["app_log"] * 5,
        "is_anomaly": [None, None, None, None, None],
    })


@pytest.fixture
def sample_feature_matrix():
    """Small random feature matrix for model testing."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 20).astype(np.float32)


@pytest.fixture
def sample_feature_matrix_with_anomalies():
    """Feature matrix with embedded anomalies (outliers)."""
    rng = np.random.RandomState(42)
    normal = rng.randn(95, 20).astype(np.float32)
    anomalies = rng.randn(5, 20).astype(np.float32) * 10 + 5  # Outliers
    return np.vstack([normal, anomalies])


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file for ingestion testing."""
    log_content = "\n".join([
        "2024-01-15 10:30:15 [ERROR] auth-service: Login failed for user admin",
        "2024-01-15 10:30:16 [INFO] api-gateway: GET /api/users 200 OK",
        "2024-01-15 10:30:17 [WARNING] db-service: Slow query detected (2.5s)",
    ])
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content, encoding="utf-8")
    return str(log_file)


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV log file for ingestion testing."""
    csv_content = (
        "timestamp,level,service,source_ip,message,is_anomaly\n"
        "2024-01-15 10:30:15,ERROR,auth-service,192.168.1.1,Login failed,1\n"
        "2024-01-15 10:30:16,INFO,api-gateway,,Request OK,0\n"
        "2024-01-15 10:30:17,WARNING,db-service,10.0.0.5,Slow query,0\n"
    )
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return str(csv_file)

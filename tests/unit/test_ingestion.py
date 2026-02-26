"""
Unit Tests — Log Ingestion Engine.

Tests parsing, format handling, IP/timestamp extraction, and error resilience.
"""

import pandas as pd
import pytest

from pipelines.ingestion import LogIngestionEngine


class TestLogIngestionEngine:
    """Tests for LogIngestionEngine."""

    def setup_method(self):
        self.engine = LogIngestionEngine()

    # ----- Text Ingestion -----

    @pytest.mark.unit
    def test_ingest_text_app_log_format(self, sample_log_lines):
        """Ingesting app-format log lines produces valid DataFrame."""
        raw_text = "\n".join(sample_log_lines)
        df = self.engine.ingest_text(raw_text)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "level" in df.columns
        assert "message" in df.columns
        assert "log_hash" in df.columns

    @pytest.mark.unit
    def test_ingest_text_extracts_log_levels(self, sample_log_lines):
        """Log levels are correctly extracted from app log format."""
        raw_text = "\n".join(sample_log_lines)
        df = self.engine.ingest_text(raw_text)

        levels = df["level"].tolist()
        assert "ERROR" in levels
        assert "INFO" in levels
        assert "WARNING" in levels

    @pytest.mark.unit
    def test_ingest_text_extracts_services(self, sample_log_lines):
        """Service names are extracted from structured log lines."""
        raw_text = "\n".join(sample_log_lines)
        df = self.engine.ingest_text(raw_text)

        services = df["service"].tolist()
        assert "auth-service" in services
        assert "api-gateway" in services

    @pytest.mark.unit
    def test_ingest_text_empty_input(self):
        """Empty input returns DataFrame with correct schema (may contain fallback record)."""
        df = self.engine.ingest_text("")
        assert isinstance(df, pd.DataFrame)
        # Empty string may produce a fallback-parsed record; schema must be correct
        assert set(self.engine.SCHEMA).issubset(set(df.columns))

    @pytest.mark.unit
    def test_ingest_text_single_line(self):
        """Single log line is ingested successfully."""
        line = "2024-01-15 10:30:15 [ERROR] auth-service: Login failed"
        df = self.engine.ingest_text(line)
        assert len(df) == 1
        assert df.iloc[0]["level"] == "ERROR"

    # ----- File Ingestion -----

    @pytest.mark.unit
    def test_ingest_file_log(self, temp_log_file):
        """Ingesting a .log file produces valid DataFrame."""
        df = self.engine.ingest_file(temp_log_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    @pytest.mark.unit
    def test_ingest_file_csv(self, temp_csv_file):
        """Ingesting a .csv file produces valid DataFrame."""
        df = self.engine.ingest_file(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    @pytest.mark.unit
    def test_ingest_file_not_found(self):
        """FileNotFoundError raised for missing files."""
        with pytest.raises(FileNotFoundError):
            self.engine.ingest_file("/nonexistent/path/test.log")

    # ----- Dict Ingestion -----

    @pytest.mark.unit
    def test_ingest_dict(self):
        """Ingesting a structured dict produces a single-row DataFrame."""
        entry = {
            "timestamp": "2024-01-15 10:30:15",
            "level": "ERROR",
            "service": "auth-service",
            "message": "Login failed",
        }
        df = self.engine.ingest_dict(entry)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["level"] == "ERROR"

    @pytest.mark.unit
    def test_ingest_dict_missing_fields(self):
        """Dict with only 'message' field uses sensible defaults."""
        entry = {"message": "Something happened"}
        df = self.engine.ingest_dict(entry)
        assert len(df) == 1
        assert df.iloc[0]["level"] == "INFO"  # Default level

    # ----- IP Extraction -----

    @pytest.mark.unit
    def test_extract_ip_from_text(self):
        """IP addresses are correctly extracted."""
        assert self.engine._extract_ip("Request from 192.168.1.100") == "192.168.1.100"
        assert self.engine._extract_ip("No IP here") == ""

    # ----- Hash Deduplication -----

    @pytest.mark.unit
    def test_log_hash_uniqueness(self):
        """Different log lines produce different hashes."""
        line1 = "2024-01-15 10:30:15 [ERROR] svc: Error A"
        line2 = "2024-01-15 10:30:15 [ERROR] svc: Error B"
        df = self.engine.ingest_text(f"{line1}\n{line2}")
        hashes = df["log_hash"].tolist()
        assert hashes[0] != hashes[1]

    @pytest.mark.unit
    def test_identical_logs_same_hash(self):
        """Identical log lines produce the same hash."""
        line = "2024-01-15 10:30:15 [ERROR] svc: Error A"
        df = self.engine.ingest_text(f"{line}\n{line}")
        hashes = df["log_hash"].tolist()
        assert hashes[0] == hashes[1]

    # ----- Stats -----

    @pytest.mark.unit
    def test_ingestion_stats(self):
        """Stats counter tracks ingestion count."""
        self.engine.ingest_text("2024-01-15 10:30:15 [INFO] svc: Hello")
        stats = self.engine.stats
        assert stats["total_ingested"] >= 1

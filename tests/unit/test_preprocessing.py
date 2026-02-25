"""
Unit Tests — Data Preprocessing Pipeline.

Tests deduplication, normalization, derived features, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np

from pipelines.preprocessing import LogPreprocessor


class TestLogPreprocessor:
    """Tests for LogPreprocessor."""

    def setup_method(self):
        self.preprocessor = LogPreprocessor()

    @pytest.mark.unit
    def test_preprocess_returns_dataframe(self, sample_log_df):
        """Preprocessing returns a valid DataFrame."""
        result = self.preprocessor.preprocess(sample_log_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.unit
    def test_preprocess_adds_derived_features(self, sample_log_df):
        """Preprocessing adds message_length and level_numeric columns."""
        result = self.preprocessor.preprocess(sample_log_df.copy())
        assert "message_length" in result.columns
        assert "level_numeric" in result.columns

    @pytest.mark.unit
    def test_preprocess_handles_empty_df(self):
        """Preprocessing handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=[
            "timestamp", "level", "service", "source_ip",
            "message", "raw_log", "log_hash", "parse_method", "is_anomaly",
        ])
        result = self.preprocessor.preprocess(empty_df)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_deduplication_removes_exact_duplicates(self, sample_log_df):
        """Hash-based deduplication removes exact duplicate rows."""
        # Create duplicates
        df = pd.concat([sample_log_df, sample_log_df], ignore_index=True)
        result = self.preprocessor.preprocess(df)
        assert len(result) <= len(sample_log_df)

    @pytest.mark.unit
    def test_level_numeric_mapping(self, sample_log_df):
        """Log levels are mapped to correct numeric values."""
        result = self.preprocessor.preprocess(sample_log_df.copy())
        if "level_numeric" in result.columns:
            # CRITICAL should have highest numeric value
            critical_rows = result[result["level"] == "CRITICAL"]
            info_rows = result[result["level"] == "INFO"]
            if len(critical_rows) > 0 and len(info_rows) > 0:
                assert critical_rows["level_numeric"].iloc[0] > info_rows["level_numeric"].iloc[0]

    @pytest.mark.unit
    def test_message_length_calculation(self, sample_log_df):
        """message_length column reflects actual message length."""
        result = self.preprocessor.preprocess(sample_log_df.copy())
        if "message_length" in result.columns:
            for _, row in result.iterrows():
                assert row["message_length"] == len(str(row["message"]))

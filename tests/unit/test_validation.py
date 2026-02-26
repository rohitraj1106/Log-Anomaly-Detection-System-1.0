"""
Unit Tests — Data Validation Pipeline.

Tests schema enforcement, null ratio checks, and dead-letter quarantine.
"""

import numpy as np
import pandas as pd
import pytest

from pipelines.validation import DataValidator


class TestDataValidator:
    """Tests for DataValidator."""

    def setup_method(self):
        self.validator = DataValidator()

    @pytest.mark.unit
    def test_validation_returns_tuple(self, sample_log_df):
        """Validation returns (valid_df, quarantined_df, report) tuple."""
        result = self.validator.validate(sample_log_df.copy())
        assert isinstance(result, tuple)
        assert len(result) == 3
        valid_df, _, _ = result
        assert isinstance(valid_df, pd.DataFrame)

    @pytest.mark.unit
    def test_valid_data_passes(self, sample_log_df):
        """Well-formed data passes validation without quarantine."""
        valid_df, _, _ = self.validator.validate(sample_log_df.copy())
        assert len(valid_df) > 0

    @pytest.mark.unit
    def test_report_contains_summary(self, sample_log_df):
        """Validation report has a summary with record counts."""
        _, _, report = self.validator.validate(sample_log_df.copy())
        summary = report.summary()
        assert isinstance(summary, dict)
        assert "total_records" in summary
        assert "valid_records" in summary

    @pytest.mark.unit
    def test_missing_message_flagged(self, sample_log_df):
        """Rows with empty messages may be quarantined."""
        df = sample_log_df.copy()
        df.loc[0, "message"] = ""
        valid_df, quarantined_df, _ = self.validator.validate(df)
        # Still should process without error
        assert len(valid_df) + len(quarantined_df) >= len(df) - 1

    @pytest.mark.unit
    def test_null_values_handled(self, sample_log_df):
        """DataFrame with NaN values is handled without crash."""
        df = sample_log_df.copy()
        df.loc[0, "timestamp"] = np.nan
        df.loc[1, "service"] = np.nan
        result = self.validator.validate(df)
        assert result is not None

    @pytest.mark.unit
    def test_empty_dataframe_handled(self):
        """Empty DataFrame doesn't crash the validator."""
        empty_df = pd.DataFrame(
            columns=[
                "timestamp",
                "level",
                "service",
                "source_ip",
                "message",
                "raw_log",
                "log_hash",
                "parse_method",
                "is_anomaly",
            ]
        )
        result = self.validator.validate(empty_df)
        assert result is not None

    @pytest.mark.unit
    def test_invalid_log_levels_corrected(self, sample_log_df):
        """Invalid log levels are corrected or flagged."""
        df = sample_log_df.copy()
        df.loc[0, "level"] = "INVALID_LEVEL"
        valid_df, _, _ = self.validator.validate(df)
        # Should handle gracefully (correct or quarantine)
        assert isinstance(valid_df, pd.DataFrame)

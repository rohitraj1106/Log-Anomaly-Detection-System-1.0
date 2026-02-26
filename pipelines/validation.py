"""
Data Validation Module — Schema Enforcement and Quality Checks.

Implements production-grade data validation:
- Schema enforcement (required fields, data types)
- Null/missing value handling
- Timestamp range validation
- Message length validation
- Data quality reporting
- Configurable validation rules

Design: Each validator is a composable function that returns
(valid_df, invalid_df, report) — enabling dead-letter queue patterns.
"""

from typing import Any

import pandas as pd

from utils.config_loader import ConfigLoader
from utils.helpers import timer
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationReport:
    """Tracks validation results and generates quality reports."""

    def __init__(self) -> None:
        self.total_records: int = 0
        self.valid_records: int = 0
        self.invalid_records: int = 0
        self.checks: list[dict[str, Any]] = []

    def add_check(self, name: str, passed: int, failed: int, details: str = "") -> None:
        self.checks.append(
            {
                "check": name,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 1.0,
                "details": details,
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "overall_quality": (
                self.valid_records / self.total_records if self.total_records > 0 else 1.0
            ),
            "checks": self.checks,
        }


class DataValidator:
    """
    Production data validator with configurable quality checks.

    Validates ingested log DataFrames against schema and business rules.
    Returns cleaned data + quarantined invalid records.
    """

    REQUIRED_COLUMNS = ["timestamp", "level", "service", "message"]
    VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}

    def __init__(self, config_path: str = "configs/pipeline_config.yaml") -> None:
        try:
            self._config = ConfigLoader.load(config_path)
            self._max_null_ratio = self._config.get("validation.max_null_ratio", 0.3)
            self._max_msg_length = self._config.get("validation.max_message_length", 10000)
            self._min_year = self._config.get("validation.min_timestamp_year", 2020)
        except (FileNotFoundError, Exception):
            self._config = None
            self._max_null_ratio = 0.3
            self._max_msg_length = 10000
            self._min_year = 2020
        logger.info("DataValidator initialized")

    @timer
    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
        """
        Run all validation checks on the DataFrame.

        Args:
            df: Input DataFrame from ingestion.

        Returns:
            Tuple of (valid_df, quarantined_df, validation_report).
        """
        report = ValidationReport()
        report.total_records = len(df)

        if df.empty:
            logger.warning("Empty DataFrame received for validation")
            report.valid_records = 0
            return df, pd.DataFrame(), report

        # Track valid mask
        valid_mask = pd.Series(True, index=df.index)

        # 1. Schema check
        valid_mask, report = self._check_schema(df, valid_mask, report)

        # 2. Null ratio check
        valid_mask, report = self._check_nulls(df, valid_mask, report)

        # 3. Log level check
        valid_mask, report = self._check_log_levels(df, valid_mask, report)

        # 4. Timestamp check
        valid_mask, report = self._check_timestamps(df, valid_mask, report)

        # 5. Message length check
        valid_mask, report = self._check_message_length(df, valid_mask, report)

        # Split into valid and quarantined
        valid_df = df[valid_mask].copy().reset_index(drop=True)
        quarantined_df = df[~valid_mask].copy().reset_index(drop=True)

        report.valid_records = len(valid_df)
        report.invalid_records = len(quarantined_df)

        logger.info(
            f"Validation complete: {report.valid_records}/{report.total_records} valid "
            f"({report.invalid_records} quarantined)"
        )
        return valid_df, quarantined_df, report

    def _check_schema(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.Series, ValidationReport]:
        """Verify required columns exist."""
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            report.add_check(
                "schema",
                passed=0,
                failed=len(df),
                details=f"Missing columns: {missing_cols}",
            )
            return pd.Series(False, index=df.index), report

        report.add_check("schema", passed=len(df), failed=0, details="All columns present")
        return mask, report

    def _check_nulls(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.Series, ValidationReport]:
        """Check null ratios for critical fields."""
        null_mask = pd.Series(True, index=df.index)
        for col in ["timestamp", "message"]:
            if col in df.columns:
                col_nulls = df[col].isna() | (df[col].astype(str).str.strip() == "")
                null_mask &= ~col_nulls

        failed = (~null_mask).sum()
        report.add_check(
            "null_check",
            passed=int(null_mask.sum()),
            failed=int(failed),
            details=f"Records with null timestamp/message: {failed}",
        )
        return mask & null_mask, report

    def _check_log_levels(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.Series, ValidationReport]:
        """Validate log levels against known set."""
        if "level" not in df.columns:
            return mask, report

        level_mask = df["level"].astype(str).str.upper().isin(self.VALID_LEVELS)
        # Auto-fix: default unknown levels to INFO
        df.loc[~level_mask, "level"] = "INFO"

        failed = (~level_mask).sum()
        report.add_check(
            "log_level",
            passed=int(level_mask.sum()),
            failed=int(failed),
            details=f"Unknown levels defaulted to INFO: {failed}",
        )
        return mask, report  # Don't quarantine, just fix

    def _check_timestamps(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.Series, ValidationReport]:
        """Validate timestamp format and range."""
        if "timestamp" not in df.columns:
            return mask, report

        ts_valid = pd.Series(True, index=df.index)
        for idx, ts in df["timestamp"].items():
            try:
                ts_str = str(ts).strip()
                if not ts_str:
                    ts_valid.at[idx] = False
                    continue
                # Try parsing
                parsed = pd.to_datetime(ts_str, errors="coerce")
                if pd.isna(parsed) or (hasattr(parsed, "year") and parsed.year < self._min_year):
                    ts_valid.at[idx] = False
            except Exception:
                ts_valid.at[idx] = False

        failed = (~ts_valid).sum()
        report.add_check(
            "timestamp",
            passed=int(ts_valid.sum()),
            failed=int(failed),
            details=f"Invalid timestamps: {failed}",
        )
        return mask & ts_valid, report

    def _check_message_length(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: ValidationReport,
    ) -> tuple[pd.Series, ValidationReport]:
        """Check message length doesn't exceed limit."""
        if "message" not in df.columns:
            return mask, report

        length_mask = df["message"].astype(str).str.len() <= self._max_msg_length
        failed = (~length_mask).sum()
        report.add_check(
            "message_length",
            passed=int(length_mask.sum()),
            failed=int(failed),
            details=f"Oversized messages: {failed} (max={self._max_msg_length})",
        )
        return mask & length_mask, report

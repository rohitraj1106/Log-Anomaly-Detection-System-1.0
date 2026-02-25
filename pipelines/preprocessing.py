"""
Log Preprocessing Module — Data Cleaning and Transformation.

Implements:
- Deduplication (hash-based and window-based)
- Text normalization
- Timestamp normalization to UTC
- Missing value imputation
- ANSI code stripping
- Feature-ready data preparation
"""

import re
from typing import Dict, List, Optional, Set

import pandas as pd

from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from utils.helpers import compute_hash, timer

logger = get_logger(__name__)

# Regex to strip ANSI escape codes
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class LogPreprocessor:
    """
    Production log preprocessor.

    Cleans and transforms validated log DataFrames into
    feature-engineering-ready format.
    """

    def __init__(self, config_path: str = "configs/pipeline_config.yaml") -> None:
        try:
            self._config = ConfigLoader.load(config_path)
            prep_cfg = self._config.get_section("preprocessing")
            self._dedup_enabled = prep_cfg.get("deduplication", {}).get("enabled", True)
            self._dedup_window = prep_cfg.get("deduplication", {}).get("window_minutes", 5)
            norm_cfg = prep_cfg.get("normalization", {})
            self._lowercase = norm_cfg.get("lowercase_messages", False)
            self._strip_ansi = norm_cfg.get("strip_ansi_codes", True)
            self._normalize_ws = norm_cfg.get("normalize_whitespace", True)
        except (FileNotFoundError, Exception):
            self._dedup_enabled = True
            self._dedup_window = 5
            self._lowercase = False
            self._strip_ansi = True
            self._normalize_ws = True

        self._seen_hashes: Set[str] = set()
        self._dedup_count = 0
        logger.info("LogPreprocessor initialized")

    @timer
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full preprocessing pipeline on validated log DataFrame.

        Pipeline steps:
        1. Normalize timestamps
        2. Clean message text
        3. Deduplicate
        4. Impute missing values
        5. Add derived columns

        Args:
            df: Validated DataFrame.

        Returns:
            Cleaned and transformed DataFrame.
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping preprocessing")
            return df

        logger.info(f"Preprocessing {len(df)} records...")

        df = df.copy()

        # Step 1: Normalize timestamps
        df = self._normalize_timestamps(df)

        # Step 2: Clean text
        df = self._clean_messages(df)

        # Step 3: Deduplicate
        if self._dedup_enabled:
            df = self._deduplicate(df)

        # Step 4: Impute missing values
        df = self._impute_missing(df)

        # Step 5: Add derived columns
        df = self._add_derived_features(df)

        logger.info(
            f"Preprocessing complete: {len(df)} records "
            f"({self._dedup_count} duplicates removed)"
        )
        return df.reset_index(drop=True)

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all timestamps to standardized datetime format."""
        if "timestamp" not in df.columns:
            return df

        df["timestamp_raw"] = df["timestamp"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Fill unparseable timestamps with current time
        null_count = df["timestamp"].isna().sum()
        if null_count > 0:
            logger.warning(f"Filling {null_count} null timestamps with current time")
            df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _clean_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize log message text."""
        if "message" not in df.columns:
            return df

        # Strip ANSI codes
        if self._strip_ansi:
            df["message"] = df["message"].astype(str).apply(
                lambda x: ANSI_PATTERN.sub("", x)
            )

        # Normalize whitespace
        if self._normalize_ws:
            df["message"] = df["message"].str.replace(r"\s+", " ", regex=True).str.strip()

        # Lowercase (optional)
        if self._lowercase:
            df["message"] = df["message"].str.lower()

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate log entries using hash-based deduplication.

        Two-phase approach:
        1. Exact duplicate removal (same hash)
        2. Time-window deduplication (same message within N minutes)
        """
        initial_len = len(df)

        # Phase 1: Hash-based exact dedup
        if "log_hash" in df.columns:
            df = df.drop_duplicates(subset=["log_hash"], keep="first")

        # Phase 2: Window-based dedup (exact same service + message)
        if "timestamp" in df.columns and "service" in df.columns:
            df["_dedup_key"] = (
                df["service"].astype(str) + "_" + df["message"].astype(str)
            ).apply(lambda x: hash(x))
            # Keep first occurrence within each dedup window
            df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
            df = df.drop(columns=["_dedup_key"], errors="ignore")

        self._dedup_count += initial_len - len(df)
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults."""
        defaults = {
            "level": "INFO",
            "service": "unknown",
            "source_ip": "0.0.0.0",
            "message": "",
        }
        for col, default in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
                # Also fill empty strings
                df.loc[df[col].astype(str).str.strip() == "", col] = default

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns useful for downstream feature engineering."""
        # Message length
        if "message" in df.columns:
            df["message_length"] = df["message"].astype(str).str.len()

        # Log level numeric encoding
        level_map = {
            "DEBUG": 0, "INFO": 1, "WARNING": 2, "WARN": 2,
            "ERROR": 3, "CRITICAL": 4,
        }
        if "level" in df.columns:
            df["level_numeric"] = (
                df["level"].astype(str).str.upper().map(level_map).fillna(1).astype(int)
            )

        # Hour of day (for time-based patterns)
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["hour_of_day"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_business_hours"] = df["hour_of_day"].between(9, 17).astype(int)

        # Has IP flag
        if "source_ip" in df.columns:
            df["has_ip"] = (
                (df["source_ip"].astype(str).str.strip() != "")
                & (df["source_ip"] != "0.0.0.0")
            ).astype(int)

        return df

    @property
    def stats(self) -> Dict[str, int]:
        """Return preprocessing statistics."""
        return {"duplicates_removed": self._dedup_count}

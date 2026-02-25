"""
Log Ingestion Engine — Multi-Source Data Ingestion Layer.

Supports:
- Static log files (.log, .csv, .json, .jsonl)
- Streaming ingestion (via internal queue simulation)
- API-based ingestion (programmatic interface)

Implements:
- Regex-based log parsing
- Timestamp normalization
- Log level extraction
- IP address extraction
- Service/source tagging

Production Design:
- Configurable via YAML
- Extensible parser registry
- Robust error handling
- Metrics tracking for ingestion throughput
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from utils.helpers import compute_hash, parse_timestamp, timer

logger = get_logger(__name__)


class LogIngestionEngine:
    """
    Multi-source log ingestion engine.

    Parses raw logs from files, streams, and API inputs into structured
    DataFrames with normalized fields.
    """

    # Compiled regex patterns for log parsing
    DEFAULT_PATTERNS = {
        "app_log": re.compile(
            r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
            r"\[(?P<level>\w+)\]\s+"
            r"(?:(?P<service>\S+?):\s+)?"
            r"(?P<message>.*)"
        ),
        "syslog": re.compile(
            r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
            r"(?P<host>\S+)\s+"
            r"(?P<service>\S+?)(?:\[(?P<pid>\d+)\])?:\s+"
            r"(?P<message>.*)"
        ),
        "access_log": re.compile(
            r"(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+-\s+-\s+"
            r"\[(?P<timestamp>[^\]]+)\]\s+"
            r'"(?P<method>\w+)\s+(?P<path>\S+)\s+\S+"\s+'
            r"(?P<status>\d{3})\s+(?P<bytes>\d+)"
        ),
    }

    IP_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    SCHEMA = [
        "timestamp", "level", "service", "source_ip",
        "message", "raw_log", "log_hash", "parse_method", "is_anomaly",
    ]

    def __init__(self, config_path: str = "configs/pipeline_config.yaml") -> None:
        try:
            self._config = ConfigLoader.load(config_path)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            self._config = None
        self._ingestion_count = 0
        self._parse_errors = 0
        logger.info("LogIngestionEngine initialized")

    @timer
    def ingest_file(self, file_path: str) -> pd.DataFrame:
        """
        Ingest logs from a static file.

        Args:
            file_path: Path to the log file (.log, .csv, .json, .jsonl).

        Returns:
            DataFrame with structured log records.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")

        suffix = path.suffix.lower()
        logger.info(f"Ingesting file: {file_path} (format: {suffix})")

        if suffix == ".csv":
            return self._ingest_csv(path)
        elif suffix in (".json", ".jsonl"):
            return self._ingest_jsonl(path)
        elif suffix == ".log":
            return self._ingest_log(path)
        else:
            logger.warning(f"Unsupported format {suffix}, attempting raw log parse")
            return self._ingest_log(path)

    def ingest_text(self, raw_text: str, source: str = "api") -> pd.DataFrame:
        """
        Ingest a raw log string (for API/streaming ingestion).

        Args:
            raw_text: Raw log text (single or multi-line).
            source: Source identifier.

        Returns:
            DataFrame with structured log records.
        """
        lines = raw_text.strip().split("\n")
        records = []
        for line in lines:
            record = self._parse_log_line(line.strip(), source=source)
            if record:
                records.append(record)

        df = pd.DataFrame(records, columns=self.SCHEMA)
        self._ingestion_count += len(df)
        logger.info(f"Ingested {len(df)} records from text input (source={source})")
        return df

    def ingest_dict(self, log_entry: Dict[str, Any]) -> pd.DataFrame:
        """Ingest a single structured log entry (for JSON API)."""
        record = self._normalize_dict_entry(log_entry)
        df = pd.DataFrame([record], columns=self.SCHEMA)
        self._ingestion_count += 1
        return df

    def _ingest_csv(self, path: Path) -> pd.DataFrame:
        """Ingest from CSV with schema mapping."""
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            logger.error(f"CSV read failed: {e}")
            raise

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Map to standard schema
        records = []
        for _, row in df.iterrows():
            record = {
                "timestamp": str(row.get("timestamp", "")),
                "level": str(row.get("level", "INFO")).upper(),
                "service": str(row.get("service", "unknown")),
                "source_ip": self._extract_ip(str(row.get("source_ip", row.get("ip", "")))),
                "message": str(row.get("message", "")),
                "raw_log": str(row.to_dict()),
                "log_hash": compute_hash(str(row.to_dict())),
                "parse_method": "csv",
                "is_anomaly": int(row.get("is_anomaly", 0)) if "is_anomaly" in row.index else None,
            }
            records.append(record)

        result = pd.DataFrame(records, columns=self.SCHEMA)
        self._ingestion_count += len(result)
        logger.info(f"CSV ingestion complete: {len(result)} records from {path.name}")
        return result

    def _ingest_jsonl(self, path: Path) -> pd.DataFrame:
        """Ingest from JSON Lines format."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    record = self._normalize_dict_entry(entry)
                    records.append(record)
                except json.JSONDecodeError:
                    self._parse_errors += 1
                    if self._parse_errors % 100 == 0:
                        logger.warning(f"JSON parse errors: {self._parse_errors}")

        result = pd.DataFrame(records, columns=self.SCHEMA)
        self._ingestion_count += len(result)
        logger.info(f"JSONL ingestion complete: {len(result)} records from {path.name}")
        return result

    def _ingest_log(self, path: Path) -> pd.DataFrame:
        """Ingest from raw log files using regex parsing."""
        records = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = self._parse_log_line(line)
                if record:
                    records.append(record)

        result = pd.DataFrame(records, columns=self.SCHEMA)
        self._ingestion_count += len(result)
        logger.info(f"Log file ingestion complete: {len(result)} records from {path.name}")
        return result

    def _parse_log_line(
        self, line: str, source: str = "file"
    ) -> Optional[Dict[str, str]]:
        """
        Parse a single log line using the pattern registry.

        Tries each regex pattern and returns the first match.
        Falls back to generic parsing if no pattern matches.
        """
        for pattern_name, pattern in self.DEFAULT_PATTERNS.items():
            match = pattern.match(line)
            if match:
                groups = match.groupdict()
                return {
                    "timestamp": groups.get("timestamp", ""),
                    "level": groups.get("level", "INFO").upper(),
                    "service": groups.get("service", groups.get("host", "unknown")),
                    "source_ip": self._extract_ip(line),
                    "message": groups.get("message", line),
                    "raw_log": line,
                    "log_hash": compute_hash(line),
                    "parse_method": pattern_name,
                    "is_anomaly": None,
                }

        # Fallback: generic parsing
        return {
            "timestamp": self._extract_timestamp_fallback(line),
            "level": self._extract_level_fallback(line),
            "service": "unknown",
            "source_ip": self._extract_ip(line),
            "message": line,
            "raw_log": line,
            "log_hash": compute_hash(line),
            "parse_method": "fallback",
            "is_anomaly": None,
        }

    def _normalize_dict_entry(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """Normalize a dictionary log entry to standard schema."""
        raw = json.dumps(entry, default=str)
        return {
            "timestamp": str(entry.get("timestamp", "")),
            "level": str(entry.get("level", entry.get("severity", "INFO"))).upper(),
            "service": str(entry.get("service", entry.get("source", "unknown"))),
            "source_ip": self._extract_ip(
                str(entry.get("source_ip", entry.get("ip", entry.get("remote_addr", ""))))
            ),
            "message": str(entry.get("message", entry.get("msg", ""))),
            "raw_log": raw,
            "log_hash": compute_hash(raw),
            "parse_method": "json",
            "is_anomaly": entry.get("is_anomaly"),
        }

    def _extract_ip(self, text: str) -> str:
        """Extract the first IP address from text."""
        match = self.IP_PATTERN.search(text)
        return match.group() if match else ""

    def _extract_timestamp_fallback(self, text: str) -> str:
        """Attempt to extract timestamp using common patterns."""
        ts_patterns = [
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",
        ]
        for pattern in ts_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return ""

    def _extract_level_fallback(self, text: str) -> str:
        """Extract log level from text using keyword matching."""
        text_upper = text.upper()
        for level in ["CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG"]:
            if level in text_upper:
                return level
        return "INFO"

    @property
    def stats(self) -> Dict[str, int]:
        """Return ingestion statistics."""
        return {
            "total_ingested": self._ingestion_count,
            "parse_errors": self._parse_errors,
        }

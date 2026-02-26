"""
Helper Utilities for the Log Anomaly Detection Platform.

Common functions used across modules:
- Timing decorators
- Data type conversions
- File I/O helpers
- Hashing utilities
"""

import functools
import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{func.__qualname__} completed in {elapsed_ms:.2f}ms",
            extra={"duration_ms": elapsed_ms},
        )
        return result

    return wrapper


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of a text string for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return the Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(UTC)


def parse_timestamp(
    ts_string: str,
    formats: list[str] | None = None,
) -> datetime | None:
    """
    Parse a timestamp string trying multiple formats.

    Args:
        ts_string: Raw timestamp string.
        formats: List of datetime format strings to try.

    Returns:
        Parsed datetime object or None if no format matched.
    """
    if formats is None:
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%d/%b/%Y:%H:%M:%S %z",
            "%b %d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%m/%d/%Y %H:%M:%S",
        ]

    for fmt in formats:
        try:
            return datetime.strptime(ts_string.strip(), fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse timestamp: '{ts_string}'")
    return None


def chunk_list(data: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into dot-notation keys."""
    items: list[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def truncate_string(text: str, max_length: int = 500) -> str:
    """Truncate string to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

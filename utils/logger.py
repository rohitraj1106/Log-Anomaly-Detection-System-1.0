"""
Centralized Logging System for the Log Anomaly Detection Platform.

Provides structured logging with:
- Console and file handlers
- Rotating file logs
- JSON-structured log formatting
- Configurable log levels per module
- Correlation IDs for request tracing

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing batch", extra={"batch_size": 1000})
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production observability."""

    # Standard LogRecord attributes to exclude from extra fields
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Dynamically add ALL extra fields (request_id, correlation_id, etc.)
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                if key not in log_entry:
                    log_entry[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = (
            f"{color}[{timestamp}] "
            f"[{record.levelname:8s}] "
            f"[{record.name}] "
            f"{record.getMessage()}{self.RESET}"
        )
        if record.exc_info and record.exc_info[0] is not None:
            formatted += f"\n{self.formatException(record.exc_info)}"
        return formatted


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Override log level.
        log_to_file: Whether to enable file logging.
        log_file: Custom log file path.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, level or DEFAULT_LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    # Console handler with human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # File handler with JSON-structured format
    if log_to_file:
        file_path = log_file or str(LOG_DIR / "platform.log")
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=MAX_LOG_SIZE_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


def get_audit_logger() -> logging.Logger:
    """Get a dedicated audit logger for security-sensitive operations."""
    logger = logging.getLogger("audit")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    audit_file = str(LOG_DIR / "audit.log")
    handler = logging.handlers.RotatingFileHandler(
        audit_file,
        maxBytes=MAX_LOG_SIZE_BYTES,
        backupCount=10,
        encoding="utf-8",
    )
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    return logger

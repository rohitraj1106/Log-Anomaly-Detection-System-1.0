"""
Configuration Loader for the Log Anomaly Detection Platform.

Supports:
- YAML configuration files
- Environment variable overrides
- Nested config access via dot notation
- Config validation
- Multiple environment profiles (dev, staging, prod)

Usage:
    config = ConfigLoader.load("configs/pipeline_config.yaml")
    batch_size = config.get("pipeline.batch_size", default=1000)
"""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Defer yaml import to handle missing dependency gracefully
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


class ConfigLoader:
    """Singleton-pattern configuration loader with environment override support."""

    _instances: Dict[str, "ConfigLoader"] = {}

    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()

    @classmethod
    def load(cls, config_path: str) -> "ConfigLoader":
        """Load or retrieve cached configuration."""
        abs_path = str(Path(config_path).resolve())
        if abs_path not in cls._instances:
            cls._instances[abs_path] = cls(config_path)
            logger.info(f"Configuration loaded from: {config_path}")
        return cls._instances[abs_path]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached configurations."""
        cls._instances.clear()

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        if yaml is None:
            raise ImportError(
                "PyYAML is required. Install with: pip install pyyaml"
            )

        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """
        Override config values with environment variables.
        Convention: LADP_SECTION__KEY (double underscore for nesting).
        Example: LADP_PIPELINE__BATCH_SIZE=2000
        """
        prefix = "LADP_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().replace("__", ".")
                self._set_nested(config_path, self._cast_value(value))
                logger.debug(f"Config override from env: {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Args:
            key: Dot-separated path (e.g., 'pipeline.batch_size').
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section as a dictionary."""
        result = self.get(section, {})
        if isinstance(result, dict):
            return copy.deepcopy(result)
        return {}

    def _set_nested(self, key: str, value: Any) -> None:
        """Set a nested config value."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    @staticmethod
    def _cast_value(value: str) -> Any:
        """Auto-cast string values to appropriate Python types."""
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @property
    def raw(self) -> Dict[str, Any]:
        """Return a deep copy of the raw configuration dictionary."""
        return copy.deepcopy(self._config)

    def __repr__(self) -> str:
        return f"ConfigLoader(path='{self._config_path}', keys={list(self._config.keys())})"

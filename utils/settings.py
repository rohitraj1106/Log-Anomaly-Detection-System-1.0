"""
Centralized Application Settings — Pydantic Settings with Validation.

Validates ALL environment variables at startup using Pydantic, replacing
scattered os.getenv() calls. Provides type safety, default values, and
clear documentation of every configurable parameter.

Usage:
    from utils.settings import settings
    print(settings.api_port)       # 8000
    print(settings.log_level)      # "INFO"
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    model_config = SettingsConfigDict(
        env_prefix="LADP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    env: str = Field(default="development", description="Environment: development | staging | production")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- API Server ---
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")
    api_workers: int = Field(default=1, description="Uvicorn worker count")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Comma-separated CORS origins",
    )

    # --- Security ---
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    rate_limit: str = Field(default="100/minute", description="Rate limit per client IP")

    # --- Model ---
    model_artifacts_dir: Path = Field(default=Path("models/artifacts"))
    active_model: str = Field(default="isolation_forest")

    # --- Data ---
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    log_dir: Path = Field(default=Path("logs"))

    # --- Monitoring ---
    enable_prometheus: bool = Field(default=True)
    metrics_port: int = Field(default=9090)

    # --- Dashboard ---
    dashboard_port: int = Field(default=8501)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return upper

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        valid = {"development", "staging", "production", "test", "ci"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid env: {v}. Must be one of {valid}")
        return v.lower()

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# Singleton instance — import this everywhere
settings = Settings()

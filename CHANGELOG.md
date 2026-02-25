# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-25

### Added
- **Testing**: Comprehensive pytest suite with unit and integration tests
  - Unit tests for ingestion, preprocessing, validation, models, helpers, and monitoring
  - Integration tests for all API endpoints
  - Shared fixtures in `conftest.py`
- **CI/CD**: GitHub Actions pipeline (`.github/workflows/ci.yml`)
  - Lint → Test → Train/Evaluate → Docker Build stages
  - Multi-version Python testing matrix (3.11, 3.12)
- **Security**:
  - API key authentication support
  - Rate limiting via SlowAPI
  - Request ID correlation for distributed tracing
  - PII masking utility for log messages
  - `.dockerignore` to prevent secret leakage
  - Restricted CORS origins (replaced `allow_origins=["*"]`)
- **Infrastructure**:
  - `docker-compose.yml` with separate API, Dashboard, Prometheus, and Grafana services
  - `pyproject.toml` with ruff, mypy, and pytest configuration
  - `Makefile` for common dev commands
  - `.pre-commit-config.yaml` for automated code quality hooks
  - `.env.example` documenting all environment variables
  - Prometheus scrape configuration (`monitoring/prometheus.yml`)
- **Configuration**: Pydantic Settings (`utils/settings.py`) replacing scattered `os.getenv()` calls
- **Documentation**: `CONTRIBUTING.md`, `CHANGELOG.md`

### Changed
- API now uses validated Pydantic Settings for all configuration
- CORS origins restricted to explicit whitelist instead of `*`
- Updated `requirements.txt` with security and monitoring dependencies

## [1.0.0] - 2026-02-22

### Added
- Initial release with complete ML pipeline
- Multi-source log ingestion engine (file, text, dict, streaming)
- Data validation with dead-letter queue
- Preprocessing with hash-based deduplication
- Feature engineering (TF-IDF, templates, N-grams, time windows)
- Isolation Forest, One-Class SVM, and Autoencoder models
- Unified model trainer with versioning
- FastAPI REST API with batch prediction
- Streamlit visualization dashboard
- Custom experiment tracker
- Production monitoring with drift detection
- Docker multi-stage build

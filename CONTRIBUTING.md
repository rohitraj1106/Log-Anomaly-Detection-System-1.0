# Contributing to Log Anomaly Detection Platform

Thank you for your interest in contributing! This document provides guidelines and workflow for contributing.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/your-org/log-anomaly-detection-platform.git
cd log-anomaly-detection-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

1. **Create a feature branch** from `develop`:
   ```bash
   git checkout -b feature/your-feature develop
   ```

2. **Make changes** following the code style guidelines below.

3. **Run tests** before committing:
   ```bash
   make test        # All tests
   make lint        # Ruff linter
   make typecheck   # MyPy
   ```

4. **Submit a pull request** targeting `develop`.

## Code Style

- **Linting**: [Ruff](https://docs.astral.sh/ruff/) configured in `pyproject.toml`
- **Type checking**: [MyPy](https://mypy.readthedocs.io/) with gradual typing
- **Formatting**: Ruff formatter (100 char line width)
- **Docstrings**: Google style

## Testing Guidelines

- Write tests for every new feature or bug fix
- Place unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Use `@pytest.mark.unit` or `@pytest.mark.integration` markers
- Target **80%+ code coverage**

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new model evaluation metric
fix: handle null timestamps in ingestion
docs: update API documentation
test: add ingestion edge case tests
refactor: extract feature engineering helpers
ci: add Python 3.12 to test matrix
```

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

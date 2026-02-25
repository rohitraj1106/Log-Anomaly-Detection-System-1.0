# ==============================================================================
# Makefile — Scalable Log Anomaly Detection Platform
# ==============================================================================
# Standard development commands. Usage: make <target>
# ==============================================================================

.PHONY: help install install-dev lint format typecheck test test-unit test-integration
.PHONY: train api dashboard stream docker-build docker-up docker-down clean

.DEFAULT_GOAL := help

# --- Help ---
help: ## Show this help message
	@echo.
	@echo  Scalable Log Anomaly Detection Platform
	@echo  ========================================
	@echo.
	@echo  Usage: make [target]
	@echo.
	@echo  Targets:
	@echo    install          Install production dependencies
	@echo    install-dev      Install all dependencies (dev + prod)
	@echo    lint             Run ruff linter
	@echo    format           Run ruff formatter
	@echo    typecheck        Run mypy type checker
	@echo    test             Run all tests with coverage
	@echo    test-unit        Run unit tests only
	@echo    test-integration Run integration tests only
	@echo    train            Run the full training pipeline
	@echo    api              Start the FastAPI server
	@echo    dashboard        Start the Streamlit dashboard
	@echo    stream           Run the streaming demo
	@echo    docker-build     Build Docker image
	@echo    docker-up        Start all services via docker-compose
	@echo    docker-down      Stop all docker-compose services
	@echo    clean            Remove generated artifacts
	@echo.

# --- Installation ---
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install all dependencies including dev tools
	pip install -e ".[dev]"
	pre-commit install

# --- Code Quality ---
lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

typecheck: ## Run mypy type checker
	mypy . --ignore-missing-imports

# --- Testing ---
test: ## Run all tests with coverage
	pytest tests/ -v --cov --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

# --- Application ---
train: ## Run the full training pipeline
	python main.py --mode train

api: ## Start the FastAPI server
	python main.py --mode api

dashboard: ## Start the Streamlit dashboard
	python main.py --mode dashboard

stream: ## Run the streaming demo
	python main.py --mode stream

# --- Docker ---
docker-build: ## Build Docker image
	docker build -t log-anomaly-platform .

docker-up: ## Start all services via docker-compose
	docker-compose up -d

docker-down: ## Stop all docker-compose services
	docker-compose down

# --- Cleanup ---
clean: ## Remove generated artifacts
	if exist htmlcov rmdir /s /q htmlcov
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .mypy_cache rmdir /s /q .mypy_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist .coverage del .coverage
	for /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

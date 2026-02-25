# 🔍 Scalable Log Anomaly Detection Platform

> **Production-grade ML system for detecting anomalies in large-scale distributed system logs.**

A FAANG-level Machine Learning Engineering project that reflects real practices used at Google, Meta, Amazon, and Netflix — designed for observability, security monitoring, infrastructure reliability, fraud detection, and large-scale log analytics.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALABLE LOG ANOMALY DETECTION PLATFORM                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │  Log Files   │  │  Streaming  │  │  REST API   │   DATA SOURCES         │
│  │ .log / .csv  │  │   (Kafka)   │  │  Ingestion  │                        │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘                        │
│         │                 │                 │                               │
│         └─────────────────┼─────────────────┘                               │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │              INGESTION LAYER                           │                 │
│  │  • Regex-based log parsing                             │                 │
│  │  • Timestamp normalization                             │                 │
│  │  • IP extraction & service tagging                     │                 │
│  └────────────────────────┬───────────────────────────────┘                 │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │              DATA PIPELINE                             │                 │
│  │  • Schema validation    • Deduplication                │                 │
│  │  • Null handling        • Data quality reporting       │                 │
│  └────────────────────────┬───────────────────────────────┘                 │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │              FEATURE ENGINEERING                       │                 │
│  │  • TF-IDF embeddings    • Log template extraction      │                 │
│  │  • N-gram features      • Time-window aggregation      │                 │
│  │  • Frequency indicators • Feature Store                │                 │
│  └────────────────────────┬───────────────────────────────┘                 │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │              MODELING LAYER                            │                 │
│  │  ┌─────────────┐ ┌──────────┐ ┌────────────────┐      │                 │
│  │  │  Isolation   │ │ One-Class│ │    LSTM         │      │                 │
│  │  │   Forest ⭐  │ │   SVM   │ │  Autoencoder    │      │                 │
│  │  └──────────────┘ └──────────┘ └────────────────┘      │                 │
│  │  • Hyperparameter tuning  • Cross-validation           │                 │
│  │  • Model versioning       • Config-driven selection    │                 │
│  └────────────────────────┬───────────────────────────────┘                 │
│                           ▼                                                 │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐              │
│  │   REST API       │  │  Streaming   │  │   Dashboard      │              │
│  │   (FastAPI)      │  │  Pipeline    │  │   (Streamlit)    │              │
│  │  /predict-log    │  │  Real-time   │  │  Visualization   │              │
│  │  /predict-batch  │  │  scoring     │  │  & Monitoring    │              │
│  └──────────────────┘  └──────────────┘  └──────────────────┘              │
│                                                                             │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │              OBSERVABILITY                             │                 │
│  │  • Experiment tracking   • Metrics collection          │                 │
│  │  • Data drift detection  • Latency monitoring          │                 │
│  │  • Alert thresholds      • Prometheus-style export     │                 │
│  └────────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
scalable-log-anomaly-platform/
│
├── main.py                      # 🚀 Main entry point — CLI for all modes
│
├── configs/
│   ├── pipeline_config.yaml     # Ingestion, validation, preprocessing config
│   ├── model_config.yaml        # Model hyperparameters, feature settings
│   └── api_config.yaml          # API, streaming, monitoring settings
│
├── data/
│   ├── generate_logs.py         # Synthetic log generator (6 microservices)
│   ├── raw/                     # Raw ingested log files
│   └── processed/               # Cleaned, validated data
│
├── pipelines/
│   ├── ingestion.py             # Multi-source log ingestion engine
│   ├── validation.py            # Schema enforcement & data quality
│   ├── preprocessing.py         # Deduplication, normalization, cleaning
│   └── orchestrator.py          # DAG-based pipeline orchestration
│
├── features/
│   ├── engineering.py           # TF-IDF, templates, N-grams, time features
│   └── store.py                 # Versioned feature store simulation
│
├── models/
│   ├── isolation_forest.py      # Primary: Isolation Forest detector
│   ├── one_class_svm.py         # Alternative: One-Class SVM
│   ├── autoencoder.py           # Advanced: Dense Autoencoder
│   ├── trainer.py               # Unified training interface
│   ├── evaluator.py             # Labeled + unlabeled evaluation
│   └── artifacts/               # Serialized models & metadata
│
├── experiments/
│   ├── tracker.py               # Custom experiment tracking (MLflow-style)
│   └── runs/                    # Persisted experiment runs
│
├── api/
│   ├── main.py                  # FastAPI (rate limiting, API keys, tracing)
│   └── schemas.py               # Pydantic request/response schemas
│
├── streaming/
│   └── processor.py             # Real-time streaming simulator
│
├── dashboard/
│   └── app.py                   # Streamlit visualization dashboard
│
├── monitoring/
│   ├── metrics.py               # Production monitoring & observability
│   └── prometheus.yml           # Prometheus scrape configuration
│
├── utils/
│   ├── logger.py                # Structured JSON logging with correlation IDs
│   ├── settings.py              # 🆕 Pydantic Settings (validated env vars)
│   ├── config_loader.py         # YAML config with env overrides
│   └── helpers.py               # Timing, hashing, parsing utilities
│
├── tests/                       # 🆕 Comprehensive test suite
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_ingestion.py    # Ingestion engine tests
│   │   ├── test_preprocessing.py# Preprocessing tests
│   │   ├── test_validation.py   # Validation pipeline tests
│   │   ├── test_models.py       # ML model tests
│   │   ├── test_helpers.py      # Utility tests
│   │   └── test_monitoring.py   # Monitoring tests
│   └── integration/
│       └── test_api.py          # FastAPI endpoint tests
│
├── .github/workflows/ci.yml     # 🆕 GitHub Actions CI/CD pipeline
├── Dockerfile                   # Multi-stage production Docker image
├── docker-compose.yml           # 🆕 Multi-service deployment
├── pyproject.toml               # 🆕 Modern project config (ruff, mypy, pytest)
├── Makefile                     # 🆕 Developer convenience commands
├── .pre-commit-config.yaml      # 🆕 Pre-commit hooks
├── .env.example                 # 🆕 Environment variable documentation
├── .dockerignore                # 🆕 Docker build exclusions
├── .gitignore                   # 🆕 Git exclusions
├── CONTRIBUTING.md              # 🆕 Contribution guidelines
├── CHANGELOG.md                 # 🆕 Change history
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Training Pipeline

```bash
python main.py --mode train
```

This single command executes the **complete 8-step pipeline**:

```
Generate → Ingest → Validate → Preprocess → Feature Engineer → Train → Evaluate → Track
```

### 3. Start the API Server

```bash
python main.py --mode api
```

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Launch the Dashboard

```bash
python main.py --mode dashboard
```

- **Dashboard**: http://localhost:8501

### 5. Run Streaming Demo

```bash
python main.py --mode stream
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `train`, `api`, `stream`, `dashboard`, `generate` | `train` |
| `--logs` | Number of synthetic logs to generate | `50000` |
| `--anomaly-ratio` | Fraction of anomalous logs | `0.05` |
| `--no-tune` | Skip hyperparameter tuning (faster) | `False` |

---

## 🐳 Docker Deployment

### Single Container

```bash
# Build the image
docker build -t log-anomaly-platform .

# Run training + API
docker run -p 8000:8000 -p 8501:8501 log-anomaly-platform

# Run training only
docker run log-anomaly-platform python main.py --mode train

# Run API only (with pre-trained model volume)
docker run -p 8000:8000 -v ./models:/app/models log-anomaly-platform python main.py --mode api
```

### Multi-Service (Docker Compose)

```bash
# Start all services (API + Dashboard + Prometheus + Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

---

## 🧪 Testing

**87 tests** covering unit and integration testing:

```bash
# Run all tests with coverage
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Ingestion Engine | 14 tests | Parsing, formats, IP extraction, hashing |
| Preprocessing | 6 tests | Dedup, normalization, derived features |
| Validation | 7 tests | Schema, nulls, quarantine, edge cases |
| Models | 12 tests | Fit, predict, save/load, versioning |
| Monitoring | 11 tests | Metrics, alerts, drift, Prometheus export |
| Helpers | 17 tests | Hash, divide, timestamp, chunk, flatten |
| API Endpoints | 11 tests | Health, predict, validation, CORS |

---

## 🛠️ Development

### Setup

```bash
# Install dev dependencies (includes pytest, ruff, mypy)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Makefile Commands

```bash
make lint          # Run ruff linter
make format        # Run ruff formatter
make typecheck     # Run mypy type checker
make test          # Run all tests with coverage
make train         # Run training pipeline
make api           # Start FastAPI server
make docker-up     # Start all services
make clean         # Remove generated artifacts
```

### CI/CD Pipeline

Every push and PR triggers a **GitHub Actions pipeline** (`.github/workflows/ci.yml`):

```
Git Push → Lint & Type Check → Test (Python 3.11 + 3.12) → Train & Evaluate → Docker Build
```

---

## 🔐 Security

| Feature | Implementation |
|---------|---------------|
| **API Authentication** | Optional API key via `X-API-Key` header |
| **Rate Limiting** | SlowAPI with configurable limits per IP |
| **CORS** | Restricted to explicit origin whitelist |
| **PII Masking** | Emails, IPs, SSNs, credit cards redacted from logs |
| **Request Tracing** | UUID correlation IDs on every request (`X-Request-ID`) |
| **Non-root Container** | Docker runs as `appuser`, not root |
| **Secret Isolation** | `.dockerignore` prevents `.env` from entering images |

## 📊 Data Pipeline Design

### Ingestion Layer
The `LogIngestionEngine` supports three ingestion modes:

| Mode | Format | Use Case |
|------|--------|----------|
| File ingestion | `.log`, `.csv`, `.json`, `.jsonl` | Batch processing historical logs |
| Text ingestion | Raw multi-line strings | API endpoint ingestion |
| Dict ingestion | Structured JSON objects | Streaming / programmatic input |

**Log parsing** uses a registry of compiled regex patterns that handles:
- **App logs**: `2024-01-15 10:30:15 [ERROR] auth-service: message`
- **Syslog**: `Feb 15 10:30:15 server01 sshd[1234]: message`
- **Access logs**: `192.168.1.1 - - [15/Jan/2024:10:30:15] "GET /api" 200 1234`
- **JSON structured logs**: Parsed directly

For each log, the pipeline extracts: **timestamp**, **log level**, **service/source**, **IP address**, **message content**, and a **SHA-256 hash** for deduplication.

### Validation Layer
Production data quality gates:
- **Schema enforcement**: Required fields validation
- **Null ratio checks**: Configurable threshold per column
- **Log level validation**: Auto-correction to known levels
- **Timestamp validation**: Range and format checking
- **Message length validation**: Truncation protection
- **Dead-letter queue**: Invalid records quarantined separately

### Preprocessing Layer
- **Hash-based deduplication**: Exact duplicate removal
- **Window-based deduplication**: Same message within N minutes
- **ANSI code stripping**: Clean terminal escape sequences
- **Whitespace normalization**: Consistent spacing
- **Timestamp normalization**: All timestamps to `datetime64`
- **Missing value imputation**: Sensible defaults per column
- **Derived features**: `message_length`, `level_numeric`, `hour_of_day`, `day_of_week`

---

## 🧠 Model Design Decisions

### Why Isolation Forest (Primary)?

| Criterion | Isolation Forest | One-Class SVM | Autoencoder |
|-----------|-----------------|---------------|-------------|
| **Unsupervised** | ✅ No labels needed | ✅ | ✅ |
| **High-dimensional data** | ✅ Excellent | ❌ Poor | ⚠️ Moderate |
| **Training speed** | ✅ O(n·t·log(ψ)) | ❌ O(n²) | ⚠️ O(epochs·n) |
| **Scalability** | ✅ Linear | ❌ Quadratic | ⚠️ GPU preferred |
| **Sparse features (TF-IDF)** | ✅ Native | ❌ Needs scaling | ⚠️ Dense required |
| **Interpretability** | ⚠️ Score-based | ❌ Kernel space | ❌ Black box |
| **Contamination tuning** | ✅ Built-in | ⚠️ Via nu | ⚠️ Via threshold |

**Isolation Forest** is the optimal choice for production log anomaly detection because:
1. **No label requirement** — critical since most production logs are unlabeled
2. **Handles sparse TF-IDF matrices natively** — no dense conversion needed
3. **Linear training complexity** — scales to millions of logs per day
4. **Built-in contamination parameter** — directly controls anomaly sensitivity
5. **Parallelizable** (`n_jobs=-1`) — leverages multi-core systems

### Feature Engineering Strategy

```
Raw Log → [TF-IDF (5000d)] + [Template ID + Rarity] + [Statistical (8d)]
        + [Time Windows (5d)] + [Frequency (4d)]
        = ~5020-dimensional feature vector
```

| Feature Type | Dimensions | Captures |
|-------------|------------|----------|
| TF-IDF | ~5000 | Semantic patterns, rare words, attack signatures |
| Templates | 2 | Structural patterns, template rarity |
| Statistical | 8 | Length, level, time, IP presence, word count |
| Time Windows | 5 | Burst patterns, rate anomalies |
| Frequency | 4 | Service rarity, level rarity, IP frequency |

---

## 📈 Scaling Strategy

### Horizontal Scaling (Kubernetes)

```yaml
# Conceptual K8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-api
spec:
  replicas: 3                    # Scale API horizontally
  template:
    spec:
      containers:
      - name: api
        image: log-anomaly-platform:latest
        command: ["python", "main.py", "--mode", "api"]
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-retraining
spec:
  schedule: "0 2 * * *"         # Retrain daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: log-anomaly-platform:latest
            command: ["python", "main.py", "--mode", "train", "--no-tune"]
```

### Production Scaling Approach

| Component | Scaling Strategy | Tool |
|-----------|-----------------|------|
| **Log Ingestion** | Kafka partitioning, parallel consumers | Kafka + Flink |
| **Feature Store** | Distributed storage, caching | Feast + Redis |
| **Model Training** | Distributed training, GPU | Spark ML / Ray |
| **API Serving** | Horizontal pod autoscaling | K8s HPA |
| **Streaming** | Micro-batch parallelism | Flink / Spark Streaming |
| **Dashboard** | CDN + caching | CloudFront + Redis |

### Throughput Targets

| Metric | Target | Current Simulation |
|--------|--------|--------------------|
| Ingestion | 1M logs/day | 50K logs (demo) |
| Inference latency | <50ms p99 | ~10-30ms |
| Batch processing | 10K logs/sec | ~5K logs/sec |
| Model retraining | <1 hour | ~30 seconds |

---

## 🔍 Monitoring Strategy

### Metrics Collected

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| `predictions_total` | Counter | — |
| `anomaly_rate` | Gauge | >10% warning, >25% critical |
| `latency_p99` | Gauge | >500ms |
| `error_rate` | Gauge | >5% |
| `drift_p_value` | Gauge | <0.05 (KS test) |

### Data Drift Detection
Uses the **Kolmogorov-Smirnov test** to compare the current anomaly score distribution against a reference baseline. When `p-value < 0.05`, the system triggers a drift alert indicating the model may need retraining.

### Prometheus Integration
The `/metrics` endpoint exports Prometheus-compatible metrics:
```
ladp_predictions_total 15234
ladp_anomalies_total 762
ladp_anomaly_rate 0.050000
ladp_latency_avg_ms 12.45
ladp_latency_p99_ms 48.23
ladp_errors_total 3
```

---

## 🛡️ Failure Handling

| Failure Mode | Detection | Recovery |
|-------------|-----------|----------|
| **Model not loaded** | Health check returns "degraded" | Auto-retry on startup; fallback to latest version |
| **Data quality drop** | Validation report null_ratio > threshold | Quarantine bad records; alert on-call |
| **Drift detected** | KS test p-value < 0.05 | Trigger automated retraining pipeline |
| **Ingestion failure** | Retry with exponential backoff | Dead-letter queue for failed records |
| **API timeout** | Latency p99 > threshold | Circuit breaker; horizontal scaling |
| **Pipeline task failure** | Task status = FAILED | Retry with backoff; skip dependent tasks |
| **Feature store corruption** | Version mismatch | Rollback to previous version |

---

## 🔬 Evaluation

### Labeled Data (Supervised)
When ground truth labels are available:
- **Precision**: How many detected anomalies are real?
- **Recall**: How many real anomalies did we catch?
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish anomalies

### Unlabeled Data (Production)
When no labels are available:
- **Score Distribution Analysis**: Statistical properties of anomaly scores
- **Percentile Thresholds**: Sensitivity at P90, P95, P97, P99
- **Drift Detection**: KS test against reference distribution
- **Anomaly Rate Monitoring**: Track rate over sliding windows

---

## 🔄 CI/CD Integration (Conceptual)

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Git Push │───>│  Lint &  │───>│  Train & │───>│  Deploy  │
│           │    │  Test    │    │  Evaluate│    │  to K8s  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │               │               │
                     ▼               ▼               ▼
               Unit Tests       Model metrics    Canary rollout
               Type checks      > threshold?     Blue/green deploy
               Lint (ruff)      Version model    Health check gate
```

### Pipeline Stages:
1. **Build**: Install deps, run linting (ruff/mypy)
2. **Test**: Unit tests, integration tests
3. **Train**: Run training pipeline on CI dataset
4. **Evaluate**: Assert metrics above minimum thresholds
5. **Register**: Version model in artifact registry
6. **Deploy**: Canary deployment → gradual rollout → full deploy
7. **Monitor**: Post-deploy drift and latency monitoring

---

## 🔮 Future Improvements

| Area | Improvement | Priority |
|------|-------------|----------|
| **Models** | Transformer-based log embedding (LogBERT) | High |
| **Models** | Online learning for continuous adaptation | High |
| **Infrastructure** | Real Kafka/Pub-Sub integration | High |
| **Infrastructure** | Distributed training with Ray/Spark | Medium |
| **Feature Store** | Feast integration with Redis cache | Medium |
| **Monitoring** | Grafana dashboards with Prometheus | Medium |
| **Security** | Log anonymization (PII masking) | High |
| **Evaluation** | A/B testing framework for model comparison | Medium |
| **Pipeline** | Apache Airflow DAG integration | Medium |
| **API** | gRPC endpoint for lower latency | Low |
| **Storage** | Time-series DB (InfluxDB/TimescaleDB) | Medium |

---

## 🏗️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core implementation |
| ML | scikit-learn, NumPy, SciPy | Models, features, evaluation |
| API | FastAPI, Pydantic, Uvicorn | Model serving |
| Security | SlowAPI, API Key Auth, PII masking | Rate limiting, authentication, privacy |
| Dashboard | Streamlit, Plotly | Visualization |
| Config | YAML, Pydantic Settings | Configuration management |
| Logging | stdlib logging (JSON) + correlation IDs | Structured observability |
| Testing | pytest, pytest-cov, httpx | Unit + integration tests (87 tests) |
| Linting | Ruff, MyPy, pre-commit | Code quality & type safety |
| CI/CD | GitHub Actions | Automated pipeline |
| Container | Docker (multi-stage), Docker Compose | Deployment |
| Monitoring | Prometheus, Grafana | Metrics & dashboards |
| Orchestration | Custom DAG engine | Pipeline management |

---

## 📝 License

This project is built for educational and portfolio purposes, demonstrating production ML engineering practices.

---

<p align="center">
  Built with ❤️ for Machine Learning Engineering excellence.
</p>

"""
Scalable Log Anomaly Detection Platform — Main Entry Point.

Orchestrates the end-to-end ML pipeline:
1. Data Generation (if needed)
2. Ingestion
3. Validation
4. Preprocessing
5. Feature Engineering
6. Model Training (with optional tuning)
7. Evaluation
8. Model Serialization
9. Experiment Tracking

Usage:
    python main.py                    # Run full pipeline
    python main.py --mode train       # Training pipeline only
    python main.py --mode evaluate    # Evaluation only
    python main.py --mode stream      # Streaming demo
    python main.py --mode api         # Start API server
    python main.py --mode dashboard   # Start Streamlit dashboard
    python main.py --mode generate    # Generate sample data
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from utils.helpers import ensure_directory, timer
from pipelines.ingestion import LogIngestionEngine
from pipelines.validation import DataValidator
from pipelines.preprocessing import LogPreprocessor
from pipelines.orchestrator import PipelineOrchestrator
from features.engineering import FeatureEngineer
from features.store import FeatureStore
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from experiments.tracker import ExperimentTracker
from monitoring.metrics import MetricsCollector

logger = get_logger(__name__)

# =============================================================================
# Pipeline Constants
# =============================================================================
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
CONFIG_PIPELINE = "configs/pipeline_config.yaml"
CONFIG_MODEL = "configs/model_config.yaml"
CONFIG_API = "configs/api_config.yaml"


# =============================================================================
# Pipeline Step Functions
# =============================================================================

def step_generate_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic log data for training."""
    from data.generate_logs import generate_logs

    num_logs = context.get("num_logs", 50000)
    anomaly_ratio = context.get("anomaly_ratio", 0.05)

    logger.info(f"Generating {num_logs} synthetic logs (anomaly_ratio={anomaly_ratio})...")
    log_file = generate_logs(
        num_logs=num_logs,
        anomaly_ratio=anomaly_ratio,
        output_dir=DATA_RAW_DIR,
    )
    return {"log_file": log_file, "num_logs": num_logs}


def step_ingest(context: Dict[str, Any]) -> pd.DataFrame:
    """Ingest raw log data."""
    engine = LogIngestionEngine(config_path=CONFIG_PIPELINE)

    # Prefer CSV (has labels)
    csv_path = Path(DATA_RAW_DIR) / "system_logs.csv"
    log_path = Path(DATA_RAW_DIR) / "system_logs.log"

    if csv_path.exists():
        df = engine.ingest_file(str(csv_path))
        logger.info(f"Ingested {len(df)} records from CSV")
    elif log_path.exists():
        df = engine.ingest_file(str(log_path))
        logger.info(f"Ingested {len(df)} records from log file")
    else:
        raise FileNotFoundError(
            f"No log files found in {DATA_RAW_DIR}. "
            "Run with --mode generate first."
        )

    context["raw_df"] = df
    return df


def step_validate(context: Dict[str, Any]) -> pd.DataFrame:
    """Validate ingested data."""
    df = context.get("ingest_output")
    if df is None:
        df = context.get("raw_df")
    if df is None:
        raise ValueError("No data available for validation")

    validator = DataValidator(config_path=CONFIG_PIPELINE)
    valid_df, quarantined_df, report = validator.validate(df)

    logger.info(f"Validation report: {json.dumps(report.summary(), indent=2, default=str)}")

    # Save quarantined records
    if not quarantined_df.empty:
        q_path = ensure_directory(DATA_PROCESSED_DIR) / "quarantined.csv"
        quarantined_df.to_csv(q_path, index=False)
        logger.info(f"Quarantined {len(quarantined_df)} records → {q_path}")

    context["valid_df"] = valid_df
    context["validation_report"] = report.summary()
    return valid_df


def step_preprocess(context: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess validated data."""
    df = context.get("validate_output")
    if df is None:
        df = context.get("valid_df")
    if df is None:
        raise ValueError("No validated data available")

    preprocessor = LogPreprocessor(config_path=CONFIG_PIPELINE)
    processed_df = preprocessor.preprocess(df)

    # Save processed data
    processed_path = ensure_directory(DATA_PROCESSED_DIR) / "processed_logs.csv"
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved: {processed_path} ({len(processed_df)} records)")

    context["processed_df"] = processed_df
    return processed_df


def step_feature_engineering(context: Dict[str, Any]) -> np.ndarray:
    """Extract features from preprocessed data."""
    df = context.get("preprocess_output")
    if df is None:
        df = context.get("processed_df")
    if df is None:
        raise ValueError("No preprocessed data available")

    engineer = FeatureEngineer(config_path=CONFIG_MODEL)
    feature_matrix, feature_names = engineer.fit_transform(df)

    # Save to feature store
    store = FeatureStore()
    version = store.save_features(
        features=feature_matrix,
        feature_names=feature_names,
        dataset_name="training",
        extra_metadata={"num_records": len(df)},
    )

    # Save the fitted feature engineer for inference
    artifacts_dir = ensure_directory("models/artifacts/latest")
    with open(artifacts_dir / "feature_engineer.pkl", "wb") as f:
        pickle.dump(engineer, f)

    context["feature_matrix"] = feature_matrix
    context["feature_names"] = feature_names
    context["feature_engineer"] = engineer
    context["feature_version"] = version

    logger.info(f"Features: {feature_matrix.shape} (version: {version})")
    return feature_matrix


def step_train(context: Dict[str, Any]) -> Dict[str, Any]:
    """Train the anomaly detection model."""
    X = context.get("feature_engineering_output")
    if X is None:
        X = context.get("feature_matrix")
    if X is None:
        raise ValueError("No feature matrix available")

    trainer = ModelTrainer(config_path=CONFIG_MODEL)
    tune = context.get("tune_hyperparams", True)

    training_result = trainer.train(X, tune_hyperparams=tune)

    # Save model
    save_path = trainer.save_model()

    context["model_trainer"] = trainer
    context["training_result"] = training_result
    context["model_save_path"] = save_path

    logger.info(f"Model trained and saved: {save_path}")
    return training_result


def step_evaluate(context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the trained model."""
    trainer = context.get("model_trainer")
    X = context.get("feature_matrix")
    df = context.get("processed_df")

    if trainer is None or X is None:
        raise ValueError("Model or features not available for evaluation")

    evaluator = ModelEvaluator()

    # Get predictions
    predictions = trainer.predict(X)
    scores = trainer.predict_proba(X)

    results = {}

    # Labeled evaluation (if labels exist)
    if df is not None and "is_anomaly" in df.columns:
        y_true = df["is_anomaly"].values
        labeled_results = evaluator.evaluate_labeled(
            y_true=y_true,
            y_pred=predictions,
            y_scores=scores,
            model_name=trainer.metadata.get("model_name", "model"),
        )
        results["labeled"] = labeled_results
        evaluator.save_report(labeled_results, "labeled_evaluation.json")

    # Unlabeled evaluation
    raw_scores = trainer.score_samples(X)
    unlabeled_results = evaluator.evaluate_unlabeled(
        scores=scores,
        model_name=trainer.metadata.get("model_name", "model"),
    )
    results["unlabeled"] = unlabeled_results
    evaluator.save_report(unlabeled_results, "unlabeled_evaluation.json")

    # Drift detection baseline
    context["reference_scores"] = raw_scores

    context["evaluation_results"] = results
    return results


def step_experiment_tracking(context: Dict[str, Any]) -> Dict[str, Any]:
    """Log the experiment run."""
    tracker = ExperimentTracker("log_anomaly_detection")

    training_result = context.get("training_result", {})
    eval_results = context.get("evaluation_results", {})

    run = tracker.start_run(
        run_name=f"{training_result.get('model_name', 'model')}_{training_result.get('version', 'v0')}",
        tags={"pipeline": "full", "model": training_result.get("model_name", "unknown")},
    )

    # Log parameters
    run.log_params(training_result.get("params", {}))
    run.log_param("training_samples", training_result.get("training_samples", 0))
    run.log_param("feature_count", training_result.get("feature_count", 0))

    # Log metrics
    labeled = eval_results.get("labeled", {})
    if labeled:
        run.log_metrics({
            "precision": labeled.get("precision", 0),
            "recall": labeled.get("recall", 0),
            "f1_score": labeled.get("f1_score", 0),
            "roc_auc": labeled.get("roc_auc", 0) or 0,
        })

    # Log artifacts
    model_path = context.get("model_save_path", "")
    if model_path:
        run.log_artifact(model_path)

    tracker.end_run("completed")

    # Print comparison
    comparison = tracker.list_runs(sort_by="f1_score")
    logger.info(f"Experiment logged. Total runs: {len(comparison)}")

    return {"run_id": run.run_id, "run_name": run.run_name}


# =============================================================================
# Pipeline Runners
# =============================================================================

@timer
def run_full_pipeline(
    num_logs: int = 50000,
    anomaly_ratio: float = 0.05,
    tune_hyperparams: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete end-to-end ML pipeline.

    Steps: Generate → Ingest → Validate → Preprocess → Features → Train → Evaluate → Track
    """
    print("\n" + "=" * 80)
    print("  🚀 SCALABLE LOG ANOMALY DETECTION PLATFORM — Full Pipeline")
    print("=" * 80 + "\n")

    orchestrator = PipelineOrchestrator("full_training_pipeline")

    # Build DAG
    orchestrator.add_task(
        "generate", step_generate_data,
        description="Generate synthetic log data",
    )
    orchestrator.add_task(
        "ingest", step_ingest,
        depends_on=["generate"],
        description="Ingest raw log files",
    )
    orchestrator.add_task(
        "validate", step_validate,
        depends_on=["ingest"],
        description="Validate data quality",
    )
    orchestrator.add_task(
        "preprocess", step_preprocess,
        depends_on=["validate"],
        description="Clean and preprocess logs",
    )
    orchestrator.add_task(
        "feature_engineering", step_feature_engineering,
        depends_on=["preprocess"],
        description="Extract ML features",
    )
    orchestrator.add_task(
        "train", step_train,
        depends_on=["feature_engineering"],
        description="Train anomaly detection model",
    )
    orchestrator.add_task(
        "evaluate", step_evaluate,
        depends_on=["train"],
        description="Evaluate model performance",
    )
    orchestrator.add_task(
        "experiment_tracking", step_experiment_tracking,
        depends_on=["evaluate"],
        description="Log experiment results",
    )

    # Run pipeline
    context = {
        "num_logs": num_logs,
        "anomaly_ratio": anomaly_ratio,
        "tune_hyperparams": tune_hyperparams,
    }
    results = orchestrator.run(context=context)

    # Print summary
    summary = orchestrator.get_summary()
    print("\n" + "=" * 80)
    print("  📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for task_name, task_info in summary["tasks"].items():
        status_icon = "✅" if task_info["status"] == "success" else "❌"
        print(f"  {status_icon} {task_name:30s} — {task_info['status']:10s} ({task_info['duration_ms']:.0f}ms)")

    # Print evaluation results
    eval_results = context.get("evaluation_results", {})
    labeled = eval_results.get("labeled", {})
    if labeled:
        print("\n" + "-" * 80)
        print("  🎯 MODEL EVALUATION RESULTS")
        print("-" * 80)
        print(f"  Precision:  {labeled.get('precision', 0):.4f}")
        print(f"  Recall:     {labeled.get('recall', 0):.4f}")
        print(f"  F1 Score:   {labeled.get('f1_score', 0):.4f}")
        print(f"  ROC-AUC:    {labeled.get('roc_auc', 'N/A')}")
        print(f"  Anomalies:  {labeled.get('predicted_anomalies', 0)} detected "
              f"/ {labeled.get('true_anomalies', 0)} actual")

    print("\n" + "=" * 80)
    print("  ✅ Pipeline completed successfully!")
    print("=" * 80 + "\n")

    return summary


def run_api_server() -> None:
    """Start the FastAPI model serving API."""
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("\n🚀 Starting Log Anomaly Detection API...")
    print("   📍 API Docs: http://localhost:8000/docs")
    print("   📍 Health:   http://localhost:8000/health\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )


def run_streaming_demo() -> None:
    """Run the real-time streaming anomaly detection demo."""
    from streaming.processor import run_streaming_demo
    print("\n⚡ Starting Streaming Anomaly Detection Demo...")
    print("   Press Ctrl+C to stop.\n")
    run_streaming_demo(duration_seconds=60)


def run_dashboard() -> None:
    """Start the Streamlit visualization dashboard."""
    import subprocess
    print("\n📊 Starting Streamlit Dashboard...")
    print("   📍 Dashboard: http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=PROJECT_ROOT,
    )


def run_generate_only(num_logs: int = 50000) -> None:
    """Generate sample data only."""
    from data.generate_logs import generate_logs
    print(f"\n📝 Generating {num_logs} synthetic log entries...")
    generate_logs(num_logs=num_logs, anomaly_ratio=0.05)
    print("✅ Data generation complete!\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalable Log Anomaly Detection Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Run full pipeline
  python main.py --mode train          Training pipeline
  python main.py --mode api            Start API server
  python main.py --mode stream         Streaming demo
  python main.py --mode dashboard      Start dashboard
  python main.py --mode generate       Generate sample data
  python main.py --logs 100000         Custom dataset size
  python main.py --no-tune             Skip hyperparameter tuning
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "api", "stream", "dashboard", "generate"],
        help="Pipeline mode to run (default: train)",
    )
    parser.add_argument(
        "--logs", type=int, default=50000,
        help="Number of log entries to generate (default: 50000)",
    )
    parser.add_argument(
        "--anomaly-ratio", type=float, default=0.05,
        help="Fraction of anomalous logs (default: 0.05)",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter tuning",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_full_pipeline(
            num_logs=args.logs,
            anomaly_ratio=args.anomaly_ratio,
            tune_hyperparams=not args.no_tune,
        )
    elif args.mode == "api":
        run_api_server()
    elif args.mode == "stream":
        run_streaming_demo()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "generate":
        run_generate_only(num_logs=args.logs)


if __name__ == "__main__":
    main()

"""
Experiment Tracker — ML Experiment Logging and Comparison.

Custom lightweight experiment tracking system that simulates
MLflow/Weights & Biases functionality:
- Run tracking with unique IDs
- Parameter logging
- Metric logging with step tracking
- Model artifact tracking
- Run comparison
- Experiment history persistence

In production, integrate with MLflow, W&B, or Vertex AI Experiments.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.helpers import ensure_directory

logger = get_logger(__name__)


class ExperimentRun:
    """Represents a single experiment run."""

    def __init__(
        self,
        run_id: str,
        experiment_name: str,
        run_name: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{run_id[:8]}"
        self.status = "running"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.artifacts: List[str] = []
        self.tags: Dict[str, str] = {}
        self.notes: str = ""

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.params.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value, optionally with a step number."""
        if key not in self.metrics:
            self.metrics[key] = []
        entry = {"value": value, "timestamp": datetime.now(timezone.utc).isoformat()}
        if step is not None:
            entry["step"] = step
        self.metrics[key].append(entry)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str) -> None:
        """Register an artifact path."""
        self.artifacts.append(artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        self.tags[key] = value

    def end(self, status: str = "completed") -> None:
        """Mark the run as completed."""
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def get_best_metric(self, key: str, mode: str = "max") -> Optional[float]:
        """Get the best value of a metric."""
        if key not in self.metrics:
            return None
        values = [entry["value"] for entry in self.metrics[key]]
        if mode == "max":
            return max(values)
        return min(values)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize run to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "params": self.params,
            "metrics": {
                k: [entry["value"] for entry in v]
                for k, v in self.metrics.items()
            },
            "best_metrics": {
                k: self.get_best_metric(k) for k in self.metrics
            },
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
        }


class ExperimentTracker:
    """
    Lightweight experiment tracker.

    Provides MLflow-style API for tracking ML experiments:
    - Create and manage experiment runs
    - Log parameters, metrics, and artifacts
    - Compare runs
    - Persist experiment history

    Usage:
        tracker = ExperimentTracker("anomaly_detection")
        with tracker.start_run("isolation_forest_v1") as run:
            run.log_params({"n_estimators": 200, "contamination": 0.05})
            # ... train model ...
            run.log_metrics({"precision": 0.85, "recall": 0.72, "f1": 0.78})
            run.log_artifact("models/artifacts/v1/model.pkl")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_dir: str = "experiments/runs",
    ) -> None:
        self.experiment_name = experiment_name
        self._tracking_dir = ensure_directory(Path(tracking_dir) / experiment_name)
        self._runs: Dict[str, ExperimentRun] = {}
        self._active_run: Optional[ExperimentRun] = None

        # Load existing runs
        self._load_history()
        logger.info(
            f"ExperimentTracker '{experiment_name}' initialized "
            f"({len(self._runs)} existing runs)"
        )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            run_name: Human-readable run name.
            tags: Optional tags for the run.

        Returns:
            ExperimentRun instance (also works as context manager).
        """
        run_id = str(uuid.uuid4())
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            run_name=run_name,
        )
        if tags:
            for k, v in tags.items():
                run.set_tag(k, v)

        self._runs[run_id] = run
        self._active_run = run

        logger.info(f"Started run: {run.run_name} ({run_id[:8]}...)")
        return run

    def end_run(self, status: str = "completed") -> None:
        """End the active run and persist it."""
        if self._active_run:
            self._active_run.end(status)
            self._save_run(self._active_run)
            logger.info(
                f"Run '{self._active_run.run_name}' ended ({status})"
            )
            self._active_run = None

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a specific run by ID."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all runs, optionally sorted by a metric.

        Args:
            sort_by: Metric name to sort by.
            ascending: Sort order.

        Returns:
            List of run summaries.
        """
        runs = [run.to_dict() for run in self._runs.values()]

        if sort_by:
            runs.sort(
                key=lambda r: r.get("best_metrics", {}).get(sort_by, float("-inf")),
                reverse=not ascending,
            )

        return runs

    def compare_runs(
        self,
        run_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare metrics across runs.

        Args:
            run_ids: Specific runs to compare (all if None).
            metrics: Specific metrics to compare.

        Returns:
            Comparison table.
        """
        runs_to_compare = (
            [self._runs[rid] for rid in run_ids if rid in self._runs]
            if run_ids
            else list(self._runs.values())
        )

        comparison = {
            "runs": [],
            "best_by_metric": {},
        }

        for run in runs_to_compare:
            run_summary = {
                "run_name": run.run_name,
                "run_id": run.run_id[:8],
                "params": run.params,
                "best_metrics": {
                    k: run.get_best_metric(k) for k in run.metrics
                },
            }
            comparison["runs"].append(run_summary)

        # Find best run per metric
        all_metric_keys = set()
        for run in runs_to_compare:
            all_metric_keys.update(run.metrics.keys())

        if metrics:
            all_metric_keys &= set(metrics)

        for metric_key in all_metric_keys:
            best_run = None
            best_value = float("-inf")
            for run in runs_to_compare:
                val = run.get_best_metric(metric_key)
                if val is not None and val > best_value:
                    best_value = val
                    best_run = run.run_name

            comparison["best_by_metric"][metric_key] = {
                "run": best_run,
                "value": best_value,
            }

        return comparison

    def get_best_run(self, metric: str = "f1_score") -> Optional[Dict[str, Any]]:
        """Get the run with the best value for a given metric."""
        best_run = None
        best_value = float("-inf")

        for run in self._runs.values():
            val = run.get_best_metric(metric)
            if val is not None and val > best_value:
                best_value = val
                best_run = run

        if best_run:
            return best_run.to_dict()
        return None

    def _save_run(self, run: ExperimentRun) -> None:
        """Persist a run to disk."""
        run_file = self._tracking_dir / f"{run.run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def _load_history(self) -> None:
        """Load existing runs from disk."""
        for run_file in self._tracking_dir.glob("*.json"):
            try:
                with open(run_file, "r") as f:
                    data = json.load(f)
                run = ExperimentRun(
                    run_id=data["run_id"],
                    experiment_name=data["experiment_name"],
                    run_name=data.get("run_name"),
                )
                run.status = data.get("status", "completed")
                run.started_at = data.get("started_at", "")
                run.completed_at = data.get("completed_at")
                run.params = data.get("params", {})
                run.artifacts = data.get("artifacts", [])
                run.tags = data.get("tags", {})
                self._runs[run.run_id] = run
            except Exception as e:
                logger.warning(f"Failed to load run from {run_file}: {e}")

    @property
    def active_run(self) -> Optional[ExperimentRun]:
        return self._active_run

"""
Model Evaluator — Comprehensive Evaluation for Labeled and Unlabeled Scenarios.

Supports:
A) Labeled evaluation: Precision, Recall, F1, ROC-AUC
B) Unlabeled evaluation: Score distributions, percentile thresholds, drift detection

Also provides:
- Cross-validation
- Evaluation report generation
- Comparison across models
"""

import json
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from utils.helpers import ensure_directory, safe_divide, timer
from utils.logger import get_logger

logger = get_logger(__name__)

# Avoid hard dependency on sklearn.metrics at import time
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive model evaluator.

    Handles both labeled (supervised) and unlabeled (unsupervised) evaluation
    scenarios common in anomaly detection.
    """

    def __init__(self, output_dir: str = "experiments/evaluations") -> None:
        self._output_dir = ensure_directory(output_dir)
        logger.info("ModelEvaluator initialized")

    @timer
    def evaluate_labeled(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray | None = None,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """
        Evaluate model with ground truth labels.

        Args:
            y_true: True labels (1=normal, -1=anomaly OR 0=normal, 1=anomaly).
            y_pred: Predicted labels.
            y_scores: Anomaly probability scores (for ROC-AUC).
            model_name: Name for the evaluation report.

        Returns:
            Dictionary of evaluation metrics.
        """
        if not SKLEARN_METRICS_AVAILABLE:
            logger.warning("sklearn.metrics not available, using basic evaluation")
            return self._basic_evaluation(y_true, y_pred)

        # Normalize labels to binary: 0=normal, 1=anomaly
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # y_true: if {0,1} → keep as-is; if {-1,1} → map -1→1, 1→0
        unique_true = set(np.unique(y_true_arr))
        if unique_true.issubset({0, 1}):
            y_true_binary = y_true_arr.astype(int)
        else:
            y_true_binary = np.where(y_true_arr == -1, 1, 0)

        # y_pred: Isolation Forest outputs -1=anomaly, 1=normal → map -1→1, 1→0
        unique_pred = set(np.unique(y_pred_arr))
        if unique_pred.issubset({0, 1}):
            y_pred_binary = y_pred_arr.astype(int)
        else:
            y_pred_binary = np.where(y_pred_arr == -1, 1, 0)

        metrics = {
            "model_name": model_name,
            "total_samples": len(y_true),
            "true_anomalies": int(y_true_binary.sum()),
            "predicted_anomalies": int(y_pred_binary.sum()),
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        }

        # ROC-AUC (requires probability scores)
        if y_scores is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_scores))
            except ValueError as e:
                logger.warning(f"ROC-AUC computation failed: {e}")
                metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics["confusion_matrix"] = {
            "true_negatives": int(cm[0][0]) if cm.shape[0] > 0 else 0,
            "false_positives": int(cm[0][1]) if cm.shape[1] > 1 else 0,
            "false_negatives": int(cm[1][0]) if cm.shape[0] > 1 else 0,
            "true_positives": int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
        }

        # Classification report
        metrics["classification_report"] = classification_report(
            y_true_binary,
            y_pred_binary,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
            zero_division=0,
        )

        logger.info(
            f"[{model_name}] Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
        )
        return metrics

    @timer
    def evaluate_unlabeled(
        self,
        scores: np.ndarray,
        model_name: str = "model",
        percentile_thresholds: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate model on unlabeled data (production scenario).

        Args:
            scores: Anomaly scores from the model.
            model_name: Name for the report.
            percentile_thresholds: Percentiles for threshold analysis.

        Returns:
            Unlabeled evaluation metrics.
        """
        if percentile_thresholds is None:
            percentile_thresholds = [90, 95, 97, 99, 99.5]

        metrics = {
            "model_name": model_name,
            "total_samples": len(scores),
            "score_statistics": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                "skewness": float(scipy_stats.skew(scores)),
                "kurtosis": float(scipy_stats.kurtosis(scores)),
            },
            "threshold_analysis": {},
        }

        # Percentile threshold analysis
        for pct in percentile_thresholds:
            threshold = np.percentile(scores, pct)
            n_anomalies = (scores >= threshold).sum()
            anomaly_rate = n_anomalies / len(scores)
            metrics["threshold_analysis"][f"p{pct}"] = {
                "threshold": float(threshold),
                "anomalies_detected": int(n_anomalies),
                "anomaly_rate": float(anomaly_rate),
            }

        # Score distribution bins (for histogram)
        hist, bin_edges = np.histogram(scores, bins=50)
        metrics["score_distribution"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

        logger.info(
            f"[{model_name}] Unlabeled eval: "
            f"mean={metrics['score_statistics']['mean']:.4f}, "
            f"std={metrics['score_statistics']['std']:.4f}"
        )
        return metrics

    @timer
    def detect_drift(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
        threshold: float = 0.05,
        method: str = "ks_test",
    ) -> dict[str, Any]:
        """
        Detect data/model drift by comparing score distributions.

        Args:
            reference_scores: Baseline anomaly scores.
            current_scores: Recent anomaly scores.
            threshold: P-value threshold for drift detection.
            method: Statistical test method.

        Returns:
            Drift detection results.
        """
        if method == "ks_test":
            statistic, p_value = scipy_stats.ks_2samp(reference_scores, current_scores)
        else:
            # Default to KS test
            statistic, p_value = scipy_stats.ks_2samp(reference_scores, current_scores)

        drift_detected = p_value < threshold

        result = {
            "method": method,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": threshold,
            "drift_detected": drift_detected,
            "reference_mean": float(np.mean(reference_scores)),
            "current_mean": float(np.mean(current_scores)),
            "mean_shift": float(np.mean(current_scores) - np.mean(reference_scores)),
        }

        if drift_detected:
            logger.warning(f"⚠️ Drift detected! p-value={p_value:.6f} < {threshold}")
        else:
            logger.info(f"No drift detected. p-value={p_value:.6f}")

        return result

    def compare_models(self, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare multiple model evaluations."""
        comparison = {
            "models": [],
            "best_model": None,
            "best_f1": 0.0,
        }

        for eval_result in evaluations:
            model_summary = {
                "name": eval_result.get("model_name", "unknown"),
                "precision": eval_result.get("precision", 0),
                "recall": eval_result.get("recall", 0),
                "f1_score": eval_result.get("f1_score", 0),
                "roc_auc": eval_result.get("roc_auc"),
            }
            comparison["models"].append(model_summary)

            f1 = eval_result.get("f1_score", 0)
            if f1 > comparison["best_f1"]:
                comparison["best_f1"] = f1
                comparison["best_model"] = model_summary["name"]

        logger.info(f"Best model: {comparison['best_model']} (F1={comparison['best_f1']:.4f})")
        return comparison

    def save_report(self, metrics: dict[str, Any], filename: str = "evaluation_report.json") -> str:
        """Save evaluation report to file."""
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Evaluation report saved: {path}")
        return str(path)

    @staticmethod
    def _basic_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Fallback evaluation without sklearn."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = ((y_pred == -1) & (y_true == -1)).sum()
        fp = ((y_pred == -1) & (y_true == 1)).sum()
        fn = ((y_pred == 1) & (y_true == -1)).sum()
        tn = ((y_pred == 1) & (y_true == 1)).sum()

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

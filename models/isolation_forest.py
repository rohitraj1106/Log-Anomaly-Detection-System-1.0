"""
Isolation Forest Anomaly Detector.

Primary model for log anomaly detection:
- Hyperparameter tuning via grid search
- Contamination optimization
- Cross-validation scoring
- Anomaly score calibration

Isolation Forest is ideal for log anomaly detection because:
1. Works well with high-dimensional sparse data (TF-IDF)
2. Doesn't require labeled anomalies
3. Linear time complexity — scales to millions
4. Robust to irrelevant features
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid

from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Production Isolation Forest anomaly detector.

    Wraps sklearn's IsolationForest with:
    - Configurable hyperparameters
    - Grid search tuning
    - Anomaly score calibration
    - Threshold optimization
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: Any = "auto",
        max_samples: Any = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self._model: Optional[IsolationForest] = None
        self._threshold: float = 0.0
        self._is_fitted: bool = False
        logger.info(f"IsolationForestDetector initialized: {self.params}")

    @timer
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on training data.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            self for method chaining.
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples, {X.shape[1]} features")
        self._model = IsolationForest(**self.params)
        self._model.fit(X)
        self._is_fitted = True

        # Compute threshold from training scores
        scores = self._model.decision_function(X)
        self._threshold = np.percentile(scores, 5)  # Bottom 5%

        logger.info(
            f"Model fitted. Score range: [{scores.min():.4f}, {scores.max():.4f}], "
            f"threshold: {self._threshold:.4f}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns:
            Array of 1 (normal) and -1 (anomaly).
        """
        self._check_fitted()
        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).

        Returns:
            Array of anomaly scores.
        """
        self._check_fitted()
        return self._model.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get calibrated anomaly probabilities.

        Normalizes decision function scores to [0, 1] range
        where 1 = most anomalous.

        Returns:
            Array of anomaly probabilities.
        """
        self._check_fitted()
        scores = self._model.decision_function(X)
        # Normalize: lower scores = more anomalous → invert and scale
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.full(len(scores), 0.5)
        proba = 1.0 - (scores - min_score) / (max_score - min_score)
        return proba

    @timer
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        scoring_percentile: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Grid search for optimal hyperparameters.

        Uses the stability of anomaly scores as the optimization criterion
        (since we typically don't have labels).

        Args:
            X: Feature matrix.
            param_grid: Parameter combinations to try.
            scoring_percentile: Percentile for anomaly threshold.

        Returns:
            Best parameters and scores.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "contamination": [0.01, 0.05, 0.1],
                "max_samples": [0.5, 0.75, 1.0],
            }

        logger.info(f"Hyperparameter tuning with {len(list(ParameterGrid(param_grid)))} combinations")

        best_score = float("-inf")
        best_params: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        for params in ParameterGrid(param_grid):
            try:
                model = IsolationForest(
                    random_state=self.params["random_state"],
                    n_jobs=self.params["n_jobs"],
                    **params,
                )
                model.fit(X)
                scores = model.decision_function(X)

                # Score stability metric: higher negative scores separation = better
                anomaly_scores = scores[scores < np.percentile(scores, scoring_percentile)]
                normal_scores = scores[scores >= np.percentile(scores, scoring_percentile)]

                if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                    separation = normal_scores.mean() - anomaly_scores.mean()
                else:
                    separation = 0.0

                result = {
                    "params": params,
                    "separation_score": separation,
                    "score_std": scores.std(),
                    "anomaly_ratio": (scores < np.percentile(scores, scoring_percentile)).mean(),
                }
                results.append(result)

                if separation > best_score:
                    best_score = separation
                    best_params = params

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")

        logger.info(f"Best params: {best_params} (separation={best_score:.4f})")

        # Refit with best params
        self.params.update(best_params)
        self.fit(X)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results,
        }

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before prediction")

    @property
    def model(self) -> Optional[IsolationForest]:
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

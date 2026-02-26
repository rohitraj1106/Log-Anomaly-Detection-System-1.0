"""
One-Class SVM Anomaly Detector.

Alternative model for comparison:
- Kernel-based boundary learning
- Works well when anomaly boundary is non-linear
- Better for smaller, denser feature sets
"""

from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from utils.helpers import timer
from utils.logger import get_logger

logger = get_logger(__name__)


class OneClassSVMDetector:
    """
    One-Class SVM anomaly detector.

    Wraps sklearn's OneClassSVM with preprocessing,
    hyperparameter tuning, and calibrated scoring.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05,
        max_iter: int = -1,
    ) -> None:
        self.params = {
            "kernel": kernel,
            "gamma": gamma,
            "nu": nu,
            "max_iter": max_iter,
        }
        self._model: OneClassSVM | None = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        logger.info(f"OneClassSVMDetector initialized: {self.params}")

    @timer
    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """Fit the One-Class SVM on training data."""
        logger.info(f"Training One-Class SVM on {X.shape[0]} samples")

        # Scale features (important for SVM)
        X_scaled = self._scaler.fit_transform(X)

        self._model = OneClassSVM(**self.params)
        self._model.fit(X_scaled)
        self._is_fitted = True

        scores = self._model.decision_function(X_scaled)
        logger.info(f"SVM fitted. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1=normal, -1=anomaly)."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated anomaly probabilities [0, 1]."""
        scores = self.score_samples(X)
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.full(len(scores), 0.5)
        return 1.0 - (scores - min_s) / (max_s - min_s)

    @timer
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        param_grid: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Grid search for optimal SVM hyperparameters."""
        if param_grid is None:
            param_grid = {
                "kernel": ["rbf", "poly"],
                "nu": [0.01, 0.05, 0.1],
                "gamma": ["scale", "auto"],
            }

        logger.info(f"SVM hyperparameter tuning: {len(list(ParameterGrid(param_grid)))} combos")

        best_score = float("-inf")
        best_params = {}

        X_scaled = self._scaler.fit_transform(X)

        for params in ParameterGrid(param_grid):
            try:
                model = OneClassSVM(**params)
                model.fit(X_scaled)
                scores = model.decision_function(X_scaled)
                separation = scores.std()  # Higher variance = better separation

                if separation > best_score:
                    best_score = separation
                    best_params = params

            except Exception as e:
                logger.warning(f"SVM tuning failed for {params}: {e}")

        self.params.update(best_params)
        self.fit(X)

        logger.info(f"Best SVM params: {best_params}")
        return {"best_params": best_params, "best_score": best_score}

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("OneClassSVM must be fitted first")

    @property
    def model(self) -> OneClassSVM | None:
        return self._model

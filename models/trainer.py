"""
Model Trainer — Unified Training Interface.

Provides a single interface for:
- Model selection via configuration
- Training with configurable parameters
- Model serialization and versioning
- Cross-validation
- Experiment tracking integration

Supports: Isolation Forest, One-Class SVM, Autoencoder.
"""

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.isolation_forest import IsolationForestDetector
from models.one_class_svm import OneClassSVMDetector
from models.autoencoder import AutoencoderDetector
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from utils.helpers import ensure_directory, timer

logger = get_logger(__name__)

# Model registry
MODEL_REGISTRY = {
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "autoencoder": AutoencoderDetector,
}


class ModelTrainer:
    """
    Unified model training and management interface.

    Handles model lifecycle:
    1. Initialization from config
    2. Training (with optional tuning)
    3. Serialization/versioning
    4. Loading for inference
    """

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        try:
            self._config = ConfigLoader.load(config_path)
            self._active_model_name = self._config.get("model.active_model", "isolation_forest")
            self._artifacts_dir = self._config.get("model.artifacts_dir", "models/artifacts")
        except (FileNotFoundError, Exception):
            self._config = None
            self._active_model_name = "isolation_forest"
            self._artifacts_dir = "models/artifacts"

        self._model: Optional[Any] = None
        self._model_version: Optional[str] = None
        self._training_metadata: Dict[str, Any] = {}

        ensure_directory(self._artifacts_dir)
        logger.info(f"ModelTrainer initialized (active: {self._active_model_name})")

    @timer
    def train(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None,
        tune_hyperparams: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train the selected model.

        Args:
            X: Feature matrix.
            model_name: Override the model to train.
            tune_hyperparams: Whether to run hyperparameter tuning.
            extra_params: Additional model parameters.

        Returns:
            Dictionary with training results and metadata.
        """
        name = model_name or self._active_model_name

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        # Get model-specific config
        model_params = self._get_model_params(name)
        if extra_params:
            model_params.update(extra_params)

        logger.info(f"Training model '{name}' with params: {model_params}")

        # Initialize model
        model_class = MODEL_REGISTRY[name]
        self._model = model_class(**model_params)

        # Optional hyperparameter tuning
        tuning_results = None
        if tune_hyperparams:
            logger.info(f"Running hyperparameter tuning for {name}...")
            tuning_results = self._model.tune_hyperparameters(X)

        # Train
        self._model.fit(X)

        # Generate version
        self._model_version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        # Store metadata
        self._training_metadata = {
            "model_name": name,
            "version": self._model_version,
            "training_samples": X.shape[0],
            "feature_count": X.shape[1],
            "params": model_params,
            "tuning_results": tuning_results,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Training complete: {name} {self._model_version}")
        return self._training_metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1=anomaly, 1=normal)."""
        self._check_model()
        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get raw anomaly scores."""
        self._check_model()
        return self._model.score_samples(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated anomaly probabilities."""
        self._check_model()
        return self._model.predict_proba(X)

    @timer
    def save_model(self, version: Optional[str] = None) -> str:
        """
        Save model artifacts to disk with versioning.

        Saves:
        - Serialized model (pickle)
        - Model metadata (JSON)
        - Model config snapshot

        Args:
            version: Override version string.

        Returns:
            Path to saved artifacts directory.
        """
        self._check_model()
        v = version or self._model_version or "v_latest"

        save_dir = ensure_directory(Path(self._artifacts_dir) / v)

        # Save model
        model_path = save_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._training_metadata, f, indent=2, default=str)

        # Save a "latest" symlink/copy
        latest_dir = Path(self._artifacts_dir) / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(save_dir, latest_dir)

        logger.info(f"Model saved: {save_dir}")
        return str(save_dir)

    @classmethod
    def load_model(
        cls,
        artifacts_dir: str = "models/artifacts",
        version: str = "latest",
    ) -> "ModelTrainer":
        """
        Load a trained model from disk.

        Args:
            artifacts_dir: Base artifacts directory.
            version: Version to load (or "latest").

        Returns:
            ModelTrainer instance with loaded model.
        """
        model_dir = Path(artifacts_dir) / version

        if not model_dir.exists():
            raise FileNotFoundError(f"Model version not found: {model_dir}")

        model_path = model_dir / "model.pkl"
        meta_path = model_dir / "metadata.json"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        trainer = cls.__new__(cls)
        trainer._model = model
        trainer._model_version = metadata.get("version", version)
        trainer._training_metadata = metadata
        trainer._active_model_name = metadata.get("model_name", "unknown")
        trainer._artifacts_dir = artifacts_dir
        trainer._config = None

        logger.info(
            f"Model loaded: {trainer._active_model_name} {trainer._model_version}"
        )
        return trainer

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all saved model versions."""
        artifacts = Path(self._artifacts_dir)
        versions = []
        if artifacts.exists():
            for d in sorted(artifacts.iterdir()):
                if d.is_dir() and d.name != "latest":
                    meta_path = d / "metadata.json"
                    meta = {}
                    if meta_path.exists():
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                    versions.append({
                        "version": d.name,
                        "model": meta.get("model_name", "unknown"),
                        "trained_at": meta.get("trained_at", ""),
                        "samples": meta.get("training_samples", 0),
                    })
        return versions

    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific parameters from config."""
        if self._config is None:
            return {}

        params = self._config.get_section(f"model.{model_name}")
        # Remove non-model keys
        params.pop("tuning", None)
        return params

    def _check_model(self) -> None:
        if self._model is None:
            raise RuntimeError("No model trained or loaded. Call train() or load_model().")

    @property
    def model(self) -> Any:
        return self._model

    @property
    def version(self) -> Optional[str]:
        return self._model_version

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._training_metadata

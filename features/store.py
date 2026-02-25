"""
Feature Store — Persistent Feature Storage and Retrieval.

Simulates a production feature store:
- Save/load feature matrices
- Version features
- Track feature metadata
- Support Parquet and NumPy formats

In production, replace with Feast, Tecton, or a custom solution.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.helpers import ensure_directory

logger = get_logger(__name__)


class FeatureStore:
    """
    Local feature store simulation.

    Stores feature matrices, metadata, and versioning info
    for reproducibility and reuse.
    """

    def __init__(self, store_path: str = "features/store") -> None:
        self._store_path = ensure_directory(store_path)
        self._metadata_path = self._store_path / "metadata.json"
        self._metadata = self._load_metadata()
        logger.info(f"FeatureStore initialized at: {self._store_path}")

    def save_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        dataset_name: str = "training",
        version: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save feature matrix with versioning.

        Args:
            features: NumPy feature matrix.
            feature_names: List of feature column names.
            dataset_name: Logical name for the dataset.
            version: Version string (auto-generated if None).
            extra_metadata: Additional metadata to store.

        Returns:
            Version string for retrieval.
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")

        version_dir = ensure_directory(self._store_path / dataset_name / version)

        # Save feature matrix as NPY
        features_path = version_dir / "features.npy"
        np.save(str(features_path), features)

        # Save feature names
        names_path = version_dir / "feature_names.json"
        with open(names_path, "w") as f:
            json.dump(feature_names, f, indent=2)

        # Update metadata
        entry = {
            "dataset": dataset_name,
            "version": version,
            "shape": list(features.shape),
            "num_features": len(feature_names),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "features_path": str(features_path),
            "names_path": str(names_path),
        }
        if extra_metadata:
            entry["extra"] = extra_metadata

        key = f"{dataset_name}/{version}"
        self._metadata[key] = entry
        self._save_metadata()

        logger.info(
            f"Features saved: {dataset_name}/{version} "
            f"({features.shape[0]} samples × {features.shape[1]} features)"
        )
        return version

    def load_features(
        self,
        dataset_name: str = "training",
        version: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load feature matrix by dataset name and version.

        Args:
            dataset_name: Logical dataset name.
            version: Specific version (latest if None).

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        if version is None:
            version = self._get_latest_version(dataset_name)
            if version is None:
                raise FileNotFoundError(f"No features found for dataset: {dataset_name}")

        key = f"{dataset_name}/{version}"
        if key not in self._metadata:
            raise FileNotFoundError(f"Feature version not found: {key}")

        entry = self._metadata[key]

        features = np.load(entry["features_path"], allow_pickle=False)
        with open(entry["names_path"], "r") as f:
            feature_names = json.load(f)

        logger.info(f"Features loaded: {key} ({features.shape})")
        return features, feature_names

    def list_versions(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all stored feature versions."""
        versions = []
        for key, entry in self._metadata.items():
            if dataset_name is None or entry.get("dataset") == dataset_name:
                versions.append({
                    "key": key,
                    "dataset": entry.get("dataset"),
                    "version": entry.get("version"),
                    "shape": entry.get("shape"),
                    "created_at": entry.get("created_at"),
                })
        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    def _get_latest_version(self, dataset_name: str) -> Optional[str]:
        """Get the latest version for a dataset."""
        versions = [
            entry["version"]
            for key, entry in self._metadata.items()
            if entry.get("dataset") == dataset_name
        ]
        if not versions:
            return None
        return sorted(versions)[-1]

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Persist metadata to disk."""
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

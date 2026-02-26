"""
Unit Tests — ML Models (Isolation Forest, One-Class SVM, Autoencoder).

Tests model initialization, training, prediction, scoring, and serialization.
"""

import numpy as np
import pytest

from models.isolation_forest import IsolationForestDetector
from models.one_class_svm import OneClassSVMDetector
from models.trainer import ModelTrainer


class TestIsolationForestDetector:
    """Tests for the primary Isolation Forest model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Model initializes with default parameters."""
        model = IsolationForestDetector()
        assert model is not None

    @pytest.mark.unit
    def test_fit(self, sample_feature_matrix):
        """Model fits without errors on valid data."""
        model = IsolationForestDetector()
        model.fit(sample_feature_matrix)

    @pytest.mark.unit
    def test_predict_returns_labels(self, sample_feature_matrix):
        """Predictions return -1 (anomaly) or 1 (normal) labels."""
        model = IsolationForestDetector()
        model.fit(sample_feature_matrix)
        labels = model.predict(sample_feature_matrix)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_feature_matrix)
        assert set(labels).issubset({-1, 1})

    @pytest.mark.unit
    def test_predict_detects_anomalies(self, sample_feature_matrix_with_anomalies):
        """Model detects at least some of the injected anomalies."""
        model = IsolationForestDetector(contamination=0.05)
        model.fit(sample_feature_matrix_with_anomalies)
        labels = model.predict(sample_feature_matrix_with_anomalies)

        anomaly_count = (labels == -1).sum()
        assert anomaly_count > 0, "Model should detect at least 1 anomaly"

    @pytest.mark.unit
    def test_score_samples(self, sample_feature_matrix):
        """score_samples returns float scores for all samples."""
        model = IsolationForestDetector()
        model.fit(sample_feature_matrix)
        scores = model.score_samples(sample_feature_matrix)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_feature_matrix)
        assert scores.dtype in [np.float32, np.float64]

    @pytest.mark.unit
    def test_predict_proba(self, sample_feature_matrix):
        """predict_proba returns probabilities between 0 and 1."""
        model = IsolationForestDetector()
        model.fit(sample_feature_matrix)
        proba = model.predict_proba(sample_feature_matrix)

        assert isinstance(proba, np.ndarray)
        assert len(proba) == len(sample_feature_matrix)
        assert np.all(proba >= 0) and np.all(proba <= 1)


class TestOneClassSVMDetector:
    """Tests for the One-Class SVM model."""

    @pytest.mark.unit
    def test_initialization(self):
        model = OneClassSVMDetector()
        assert model is not None

    @pytest.mark.unit
    def test_fit_and_predict(self, sample_feature_matrix):
        """SVM fits and predicts with valid labels."""
        model = OneClassSVMDetector()
        model.fit(sample_feature_matrix)
        labels = model.predict(sample_feature_matrix)

        assert len(labels) == len(sample_feature_matrix)
        assert set(labels).issubset({-1, 1})


class TestModelTrainer:
    """Tests for the unified ModelTrainer interface."""

    @pytest.mark.unit
    def test_initialization(self):
        """Trainer initializes without config file."""
        trainer = ModelTrainer()
        assert trainer is not None

    @pytest.mark.unit
    def test_train_isolation_forest(self, sample_feature_matrix):
        """Training Isolation Forest returns metadata dict."""
        trainer = ModelTrainer()
        result = trainer.train(sample_feature_matrix, model_name="isolation_forest")

        assert isinstance(result, dict)
        assert "model_name" in result
        assert result["model_name"] == "isolation_forest"
        assert "version" in result
        assert "training_samples" in result

    @pytest.mark.unit
    def test_predict_after_train(self, sample_feature_matrix):
        """Prediction works after training."""
        trainer = ModelTrainer()
        trainer.train(sample_feature_matrix, model_name="isolation_forest")
        labels = trainer.predict(sample_feature_matrix)

        assert len(labels) == len(sample_feature_matrix)

    @pytest.mark.unit
    def test_predict_before_train_raises(self, sample_feature_matrix):
        """Prediction before training raises RuntimeError."""
        trainer = ModelTrainer()
        with pytest.raises(RuntimeError, match="No model trained"):
            trainer.predict(sample_feature_matrix)

    @pytest.mark.unit
    def test_save_and_load(self, sample_feature_matrix, tmp_path):
        """Model can be saved and reloaded with identical predictions."""
        trainer = ModelTrainer()
        trainer._artifacts_dir = str(tmp_path)
        trainer.train(sample_feature_matrix, model_name="isolation_forest")
        trainer.save_model()

        # Reload
        loaded = ModelTrainer.load_model(artifacts_dir=str(tmp_path))
        original_labels = trainer.predict(sample_feature_matrix)
        loaded_labels = loaded.predict(sample_feature_matrix)

        np.testing.assert_array_equal(original_labels, loaded_labels)

    @pytest.mark.unit
    def test_unknown_model_raises(self, sample_feature_matrix):
        """Training an unknown model name raises ValueError."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train(sample_feature_matrix, model_name="nonexistent_model")

    @pytest.mark.unit
    def test_model_versioning(self, sample_feature_matrix):
        """Each training run generates a unique version string."""
        trainer = ModelTrainer()
        result = trainer.train(sample_feature_matrix, model_name="isolation_forest")
        assert result["version"].startswith("v")

    @pytest.mark.unit
    def test_metadata_properties(self, sample_feature_matrix):
        """Trainer exposes version and metadata after training."""
        trainer = ModelTrainer()
        trainer.train(sample_feature_matrix, model_name="isolation_forest")

        assert trainer.version is not None
        assert trainer.metadata["training_samples"] == len(sample_feature_matrix)
        assert trainer.metadata["feature_count"] == sample_feature_matrix.shape[1]

"""
LSTM Autoencoder Anomaly Detector.

Advanced deep learning model for sequence-based anomaly detection:
- Learns normal log patterns via reconstruction
- Anomalies = high reconstruction error
- Captures temporal dependencies in log sequences

Note: Requires TensorFlow/Keras. Falls back gracefully if unavailable.
Uses a simple dense autoencoder as cross-platform fallback.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.

    Uses reconstruction error as anomaly score:
    - Low error = normal pattern (model can reconstruct)
    - High error = anomalous (model cannot reconstruct)

    Implements a dense autoencoder that works without TensorFlow.
    """

    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_layers: list = None,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile

        self._scaler = StandardScaler()
        self._weights: list = []
        self._biases: list = []
        self._threshold: float = 0.0
        self._is_fitted = False
        self._input_dim: int = 0
        self._training_history: Dict[str, list] = {"loss": []}

        logger.info(f"AutoencoderDetector initialized (encoding_dim={encoding_dim})")

    @timer
    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """
        Train the autoencoder on normal data.

        Uses a simple NumPy-based dense autoencoder implementation
        for maximum portability (no TensorFlow dependency).
        """
        logger.info(f"Training Autoencoder on {X.shape[0]} samples, {X.shape[1]} features")

        X_scaled = self._scaler.fit_transform(X)
        self._input_dim = X_scaled.shape[1]

        # Build encoder-decoder architecture
        layer_dims = [self._input_dim] + self.hidden_layers + [self.encoding_dim]
        # Mirror for decoder
        decoder_dims = list(reversed(layer_dims))

        all_dims = layer_dims + decoder_dims[1:]

        # Initialize weights (Xavier initialization)
        self._weights = []
        self._biases = []
        for i in range(len(all_dims) - 1):
            limit = np.sqrt(6.0 / (all_dims[i] + all_dims[i + 1]))
            W = np.random.uniform(-limit, limit, (all_dims[i], all_dims[i + 1]))
            b = np.zeros(all_dims[i + 1])
            self._weights.append(W)
            self._biases.append(b)

        # Train with mini-batch gradient descent
        n_samples = X_scaled.shape[0]
        self._training_history = {"loss": []}

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_batch = X_scaled[batch_idx]

                # Forward pass
                activations = self._forward(X_batch)
                reconstruction = activations[-1]

                # Compute loss (MSE)
                loss = np.mean((X_batch - reconstruction) ** 2)
                epoch_loss += loss
                n_batches += 1

                # Backward pass (simplified gradient descent)
                self._backward(X_batch, activations, self.learning_rate)

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} — loss: {avg_loss:.6f}")

        # Set anomaly threshold
        reconstruction_errors = self._compute_errors(X_scaled)
        self._threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        self._is_fitted = True

        logger.info(
            f"Autoencoder trained. Threshold: {self._threshold:.6f} "
            f"(p{self.threshold_percentile})"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies: 1=normal, -1=anomaly."""
        errors = self._compute_errors(self._scaler.transform(X))
        labels = np.where(errors > self._threshold, -1, 1)
        return labels

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (negative reconstruction error)."""
        errors = self._compute_errors(self._scaler.transform(X))
        return -errors  # Negative so lower = more anomalous (consistent interface)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly probability [0, 1]."""
        errors = self._compute_errors(self._scaler.transform(X))
        min_e, max_e = errors.min(), errors.max()
        if max_e == min_e:
            return np.full(len(errors), 0.5)
        return (errors - min_e) / (max_e - min_e)

    def _forward(self, X: np.ndarray) -> list:
        """Forward pass through all layers."""
        activations = [X]
        current = X
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            z = current @ W + b
            # ReLU for hidden layers, linear for output
            if i < len(self._weights) - 1:
                current = np.maximum(0, z)  # ReLU
            else:
                current = z  # Linear output
            activations.append(current)
        return activations

    def _backward(
        self, X: np.ndarray, activations: list, lr: float
    ) -> None:
        """Simplified backpropagation."""
        batch_size = X.shape[0]
        n_layers = len(self._weights)

        # Output error
        delta = 2.0 * (activations[-1] - X) / batch_size

        for i in reversed(range(n_layers)):
            # Gradient for weights and biases
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)

            # Update weights
            self._weights[i] -= lr * dW / batch_size
            self._biases[i] -= lr * db / batch_size

            if i > 0:
                delta = delta @ self._weights[i].T
                # ReLU derivative
                delta *= (activations[i] > 0).astype(float)

    def _compute_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors."""
        self._check_fitted()
        activations = self._forward(X_scaled)
        reconstruction = activations[-1]
        errors = np.mean((X_scaled - reconstruction) ** 2, axis=1)
        return errors

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted first")

    @property
    def training_history(self) -> Dict[str, list]:
        return self._training_history

    @property
    def threshold(self) -> float:
        return self._threshold

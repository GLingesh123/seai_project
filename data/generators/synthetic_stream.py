"""
Synthetic Streaming Data Generator for SEAI

Features:
- chunked streaming batches
- reproducible random seed
- binary classification labels
- sudden / gradual / recurring drift
- feature distribution shift
- class boundary shift
- noise injection
"""

import numpy as np
from typing import Tuple, Optional


class SyntheticStream:
    """
    Streaming synthetic dataset with controllable concept drift.
    """

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        total_samples: int = 20000,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.total_samples = total_samples
        self.generated = 0

        self.rng = np.random.default_rng(seed)

        # base distribution params
        self.base_mean = 0.0
        self.base_std = 1.0

        # drift state
        self.drift_offset = 0.0
        self.boundary_shift = 0.0
        self.noise_level = 0.0

    # -------------------------
    # Drift Controls
    # -------------------------

    def set_sudden_drift(
        self,
        feature_shift: float = 2.0,
        boundary_shift: float = 1.5,
        noise: float = 0.2
    ):
        """Apply sudden distribution change"""
        self.drift_offset = feature_shift
        self.boundary_shift = boundary_shift
        self.noise_level = noise

    def set_gradual_drift(
        self,
        progress: float,
        max_feature_shift: float = 2.0,
        max_boundary_shift: float = 1.5,
        max_noise: float = 0.3
    ):
        """
        Gradually increase drift based on progress (0 → 1)
        """
        self.drift_offset = max_feature_shift * progress
        self.boundary_shift = max_boundary_shift * progress
        self.noise_level = max_noise * progress

    def reset_drift(self):
        """Return to original distribution"""
        self.drift_offset = 0.0
        self.boundary_shift = 0.0
        self.noise_level = 0.0

    # -------------------------
    # Core Generation
    # -------------------------

    def _generate_features(self) -> np.ndarray:
        X = self.rng.normal(
            loc=self.base_mean + self.drift_offset,
            scale=self.base_std,
            size=(self.chunk_size, self.input_dim)
        )

        if self.noise_level > 0:
            noise = self.rng.normal(
                0,
                self.noise_level,
                size=X.shape
            )
            X = X + noise

        return X

    def _generate_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Linear boundary classifier with shiftable threshold
        """
        scores = X.sum(axis=1)
        y = (scores > self.boundary_shift).astype(int)
        return y

    # -------------------------
    # Public API
    # -------------------------

    def next_chunk(
        self,
        drift_mode: Optional[str] = None,
        gradual_progress: float = 0.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get next stream chunk.

        drift_mode:
            None
            "sudden"
            "gradual"
            "reset"

        gradual_progress:
            value between 0 and 1 for gradual drift
        """

        if self.generated >= self.total_samples:
            return None

        # apply drift mode
        if drift_mode == "sudden":
            self.set_sudden_drift()

        elif drift_mode == "gradual":
            self.set_gradual_drift(gradual_progress)

        elif drift_mode == "reset":
            self.reset_drift()

        X = self._generate_features()
        y = self._generate_labels(X)

        self.generated += self.chunk_size
        return X, y

    # -------------------------
    # Diagnostics
    # -------------------------

    def current_state(self) -> dict:
        return {
            "generated": self.generated,
            "drift_offset": self.drift_offset,
            "boundary_shift": self.boundary_shift,
            "noise_level": self.noise_level
        }

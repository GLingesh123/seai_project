import numpy as np
from typing import Tuple, Optional


class SyntheticStream:

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

        # base distribution
        self.base_mean = 0.0
        self.base_std = 1.0

        # -------- TRUE CONCEPT DRIFT SETUP (SOFTENED) --------
        self.w_base = self.rng.normal(size=input_dim)

        # 🔧 reduced drift strength
        self.w_drift = self.rng.normal(size=input_dim) * 1.8

        self.current_w = self.w_base.copy()

        # covariate drift params (softened)
        self.drift_offset = 0.0
        self.noise_level = 0.0

    # -------------------------
    # Drift Controls
    # -------------------------

    def set_sudden_drift(self):
        # 🔧 softer sudden drift
        self.current_w = self.w_drift
        self.drift_offset = 1.2
        self.noise_level = 0.25

    def set_gradual_drift(self, progress: float):
        self.current_w = (
            (1 - progress) * self.w_base +
            progress * self.w_drift
        )

        # 🔧 softer gradual drift
        self.drift_offset = 1.0 * progress
        self.noise_level = 0.2 * progress

    def reset_drift(self):
        self.current_w = self.w_base.copy()
        self.drift_offset = 0.0
        self.noise_level = 0.0

    # -------------------------
    # Feature Generation
    # -------------------------

    def _generate_features(self) -> np.ndarray:

        X = self.rng.normal(
            loc=self.base_mean + self.drift_offset,
            scale=self.base_std,
            size=(self.chunk_size, self.input_dim)
        )

        if self.noise_level > 0:
            X += self.rng.normal(
                0,
                self.noise_level,
                size=X.shape
            )

        return X

    # -------------------------
    # TRUE Concept Labels
    # -------------------------

    def _generate_labels(self, X: np.ndarray) -> np.ndarray:

        scores = X @ self.current_w

        # 🔧 reduced label noise
        scores += self.rng.normal(0, 0.25, size=len(scores))

        return (scores > 0).astype(int)

    # -------------------------
    # Public API
    # -------------------------

    def next_chunk(
        self,
        drift_mode: Optional[str] = None,
        gradual_progress: float = 0.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:

        if self.generated >= self.total_samples:
            return None

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

    def current_state(self):
        return {
            "generated": self.generated,
            "noise": self.noise_level,
            "offset": self.drift_offset
        }

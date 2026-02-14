"""
Unified Stream Loader for SEAI

Combines:
- SyntheticStream (data generation)
- DriftInjector (drift timing control)

Provides a single clean interface for trainers and experiments.
"""

from typing import Optional, Tuple, Dict

from data.generators.synthetic_stream import SyntheticStream
from data.loaders.drift_injector import DriftInjector
from config import (
    INPUT_DIM,
    STREAM_CHUNK_SIZE,
    STREAM_TOTAL_SAMPLES,
    SEED
)


class StreamLoader:
    """
    Unified streaming interface.
    """

    def __init__(
        self,
        scenario: Optional[Dict] = None,
        input_dim: int = INPUT_DIM,
        chunk_size: int = STREAM_CHUNK_SIZE,
        total_samples: int = STREAM_TOTAL_SAMPLES,
        seed: int = SEED
    ):
        # data generator
        self.stream = SyntheticStream(
            input_dim=input_dim,
            chunk_size=chunk_size,
            total_samples=total_samples,
            seed=seed
        )

        # drift controller
        self.injector = DriftInjector(scenario or {"type": "none"})

        self.step = 0

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def next_batch(self) -> Optional[Tuple]:
        """
        Returns:
            X, y, info_dict
        or None if stream finished
        """

        drift_mode = self.injector.get_drift_mode(self.step)
        progress = self.injector.get_gradual_progress(self.step)

        batch = self.stream.next_chunk(
            drift_mode=drift_mode,
            gradual_progress=progress
        )

        if batch is None:
            return None

        X, y = batch

        info = {
            "step": self.step,
            "drift_mode": drift_mode,
            "gradual_progress": round(progress, 3),
            "stream_state": self.stream.current_state()
        }

        self.step += 1
        return X, y, info

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def reset(self):
        """Restart stream and step counter"""
        self.stream.generated = 0
        self.step = 0

    def describe(self) -> str:
        return f"StreamLoader(step={self.step}, injector={self.injector.describe()})"

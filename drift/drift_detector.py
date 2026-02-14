"""
SEAI Drift Detector Wrapper

Consumes error signal stream (1 - accuracy)
and detects concept drift.

Default: ADWIN (river)

Supports:
- drift detection
- warning state (for detectors that support it)
- reset
- step tracking
"""

from typing import Optional, Dict

from river.drift import ADWIN

from config import DRIFT_DELTA


class DriftDetector:
    """
    Unified drift detector wrapper.
    """

    def __init__(self, delta: float = DRIFT_DELTA):
        self.delta = delta
        self.detector = ADWIN(delta=delta)

        self.step = 0
        self.last_drift_step: Optional[int] = None

    # -------------------------------------------------
    # Update
    # -------------------------------------------------

    def update(self, error_value: float) -> Dict:
        """
        Update detector with error signal.

        error_value:
            typically = 1 - accuracy
            range [0,1]
        """

        self.step += 1
        self.detector.update(error_value)

        drift = self.detector.drift_detected

        if drift:
            self.last_drift_step = self.step

        return {
            "step": self.step,
            "error": float(error_value),
            "drift": bool(drift),
            "last_drift_step": self.last_drift_step
        }

    # -------------------------------------------------
    # State
    # -------------------------------------------------

    def reset(self):
        """Reset detector state"""
        self.detector = ADWIN(delta=self.delta)
        self.step = 0
        self.last_drift_step = None

    # -------------------------------------------------
    # Info
    # -------------------------------------------------

    def status(self) -> Dict:
        return {
            "step": self.step,
            "delta": self.delta,
            "last_drift_step": self.last_drift_step
        }

    def __repr__(self):
        return (
            f"DriftDetector(delta={self.delta}, "
            f"step={self.step}, "
            f"last_drift={self.last_drift_step})"
        )

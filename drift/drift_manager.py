from typing import Dict, Optional, List

from config import DRIFT_DELTA, DEBUG_DRIFT

from .detectors.adwin_detector import ADWINDetector
from .detectors.page_hinkley_detector import PageHinkleyDetector
from .detectors.feature_drift import FeatureDriftDetector
from .drift_event import DriftEvent


class DriftManager:
    """
    SEAI Advanced Drift Manager

    Features:
    - multi-detector voting
    - error + feature drift
    - configurable vote threshold
    - severity scoring
    - structured event tracking
    """

    # -------------------------------------------------

    def __init__(
        self,
        delta: float = DRIFT_DELTA,
        min_votes: int = 2
    ):

        self.delta = delta
        self.min_votes = min_votes

        # ---------- detectors ----------
        self.detectors = [
            ADWINDetector(delta),
            PageHinkleyDetector(),
        ]

        self.feature_detector = FeatureDriftDetector()

        self.total_detectors = len(self.detectors) + (
            1 if self.feature_detector else 0
        )

        # ---------- state ----------
        self.step = 0
        self.last_drift_step: Optional[int] = None
        self.events: List[DriftEvent] = []

    # -------------------------------------------------

    def update(
        self,
        error_value: float,
        features=None
    ) -> Dict:

        self.step += 1
        fired: List[str] = []

        # ---------- error detectors ----------
        for d in self.detectors:
            try:
                if d.update(error_value):
                    fired.append(d.name)
            except Exception as e:
                if DEBUG_DRIFT:
                    print(f"[DRIFT][{d.name}] error:", e)

        # ---------- feature detector ----------
        if self.feature_detector and features is not None:
            try:
                if self.feature_detector.update(features):
                    fired.append(self.feature_detector.name)
            except Exception as e:
                if DEBUG_DRIFT:
                    print("[DRIFT][feature] error:", e)

        # ---------- voting ----------
        drift = len(fired) >= self.min_votes

        # ---------- severity ----------
        vote_strength = len(fired) / max(1, self.total_detectors)

        # balanced SEAI severity score
        severity = float(
            0.6 * vote_strength +
            0.4 * min(1.0, error_value)
        )

        # ---------- event ----------
        if drift:

            self.last_drift_step = self.step

            event = DriftEvent.create(
                step=self.step,
                detectors=fired,
                severity=severity,
                error_value=error_value,
                total_detectors=self.total_detectors
            )

            self.events.append(event)

            if DEBUG_DRIFT:
                print(
                    f"[DRIFT] step={self.step} "
                    f"detectors={fired} "
                    f"votes={len(fired)}/{self.total_detectors} "
                    f"severity={severity:.3f}"
                )

        else:
            event = None

        # ---------- output ----------
        return {
            "step": self.step,
            "drift": drift,
            "detectors": fired,
            "severity": severity,
            "event": event
        }

    # -------------------------------------------------

    def reset(self):
        """
        Reset detectors after drift.
        Keeps event history for evaluation.
        """

        for d in self.detectors:
            if hasattr(d, "reset"):
                d.reset(self.delta)

        if self.feature_detector and hasattr(
            self.feature_detector, "reset"
        ):
            self.feature_detector.reset()

        self.last_drift_step = None

    # -------------------------------------------------

    def summary(self):

        return {
            "total_steps": self.step,
            "drift_events": len(self.events),
            "last_drift_step": self.last_drift_step,
            "min_votes": self.min_votes,
            "total_detectors": self.total_detectors
        }

from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class DriftEvent:
    """
    Structured drift record for SEAI experiments.

    Stored by DriftManager and used by:
    - metrics
    - plots
    - adaptation latency eval
    - drift precision eval
    """

    step: int
    detectors: List[str]

    # strength indicators
    severity: float
    error_value: float
    vote_count: int
    total_detectors: int

    # timing
    timestamp: float

    # adaptation linkage (filled later by loop if needed)
    adaptation_triggered: bool = False
    notes: Optional[str] = None

    # -------------------------------------------------

    @classmethod
    def create(
        cls,
        step: int,
        detectors: List[str],
        severity: float,
        error_value: float,
        total_detectors: Optional[int] = None
    ):

        vote_count = len(detectors)

        if total_detectors is None:
            total_detectors = vote_count

        return cls(
            step=step,
            detectors=list(detectors),
            severity=float(severity),
            error_value=float(error_value),
            vote_count=vote_count,
            total_detectors=total_detectors,
            timestamp=time.time(),
        )

    # -------------------------------------------------

    def vote_ratio(self) -> float:
        if self.total_detectors == 0:
            return 0.0
        return self.vote_count / self.total_detectors

    # -------------------------------------------------

    def mark_adaptation(self, note: Optional[str] = None):
        self.adaptation_triggered = True
        if note:
            self.notes = note

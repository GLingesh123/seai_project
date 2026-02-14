"""
Drift Injector for SEAI Streaming Pipeline

Controls WHEN drift happens during streaming.
Works with SyntheticStream generator.

Supports:
- sudden drift
- gradual drift windows
- recurring drift
- reset phases
- scenario-driven configuration
"""

from typing import Dict, List, Optional


class DriftInjector:
    """
    Controls drift timing for a streaming source.
    """

    def __init__(self, scenario: Dict):
        """
        scenario format examples:

        sudden:
            {
                "type": "sudden",
                "steps": [40]
            }

        gradual:
            {
                "type": "gradual",
                "start": 40,
                "end": 70
            }

        recurring:
            {
                "type": "recurring",
                "steps": [30, 80, 130]
            }
        """

        self.scenario = scenario
        self.dtype = scenario.get("type", "none")

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def get_drift_mode(self, step: int) -> Optional[str]:
        """
        Returns drift_mode string for stream.next_chunk()

        Returns:
            None | "sudden" | "gradual" | "reset"
        """

        if self.dtype == "none":
            return None

        if self.dtype == "sudden":
            if step in self.scenario.get("steps", []):
                return "sudden"
            return None

        if self.dtype == "recurring":
            if step in self.scenario.get("steps", []):
                return "sudden"
            return None

        if self.dtype == "gradual":
            start = self.scenario["start"]
            end = self.scenario["end"]

            if start <= step <= end:
                return "gradual"

            if step == end + 1:
                return "reset"

            return None

        return None

    # -------------------------------------------------
    # Gradual Drift Progress
    # -------------------------------------------------

    def get_gradual_progress(self, step: int) -> float:
        """
        Returns progress value (0 → 1) for gradual drift window.
        """

        if self.dtype != "gradual":
            return 0.0

        start = self.scenario["start"]
        end = self.scenario["end"]

        if step < start:
            return 0.0

        if step > end:
            return 1.0

        return (step - start) / float(end - start)

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    def describe(self) -> str:
        return f"DriftInjector(type={self.dtype}, config={self.scenario})"

"""
SEAI Experiment Logger

Logs streaming training metrics and drift events.

Features:
- step metrics logging
- drift event tracking
- JSON + CSV export
- auto directory creation
- append-safe
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config import LOG_DIR, RESULTS_DIR


class ExperimentLogger:
    """
    Experiment metrics logger.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_id = f"{experiment_name}_{self.start_time}"

        self.metrics: List[Dict] = []
        self.drift_events: List[Dict] = []

        self._prepare_dirs()

    # -------------------------------------------------
    # Directory Setup
    # -------------------------------------------------

    def _prepare_dirs(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(f"{RESULTS_DIR}/json", exist_ok=True)
        os.makedirs(f"{RESULTS_DIR}/csv", exist_ok=True)

    # -------------------------------------------------
    # Metric Logging
    # -------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        accuracy: float,
        drift: bool = False,
        extra: Optional[Dict] = None
    ):
        record = {
            "step": step,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "drift": bool(drift)
        }

        if extra:
            record.update(extra)

        self.metrics.append(record)

    # -------------------------------------------------
    # Drift Event Logging
    # -------------------------------------------------

    def log_drift_event(self, step: int, detector_info: Dict):
        event = {
            "step": step,
            "time": datetime.now().isoformat(),
            "info": detector_info
        }
        self.drift_events.append(event)

    # -------------------------------------------------
    # Save Methods
    # -------------------------------------------------

    def save_json(self):
        path = f"{RESULTS_DIR}/json/{self.run_id}.json"

        payload = {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "metrics": self.metrics,
            "drift_events": self.drift_events
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved JSON log → {path}")

    def save_csv(self):
        if not self.metrics:
            print("No metrics to save.")
            return

        df = pd.DataFrame(self.metrics)
        path = f"{RESULTS_DIR}/csv/{self.run_id}.csv"
        df.to_csv(path, index=False)

        print(f"Saved CSV log → {path}")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self) -> Dict:
        if not self.metrics:
            return {}

        df = pd.DataFrame(self.metrics)

        return {
            "run_id": self.run_id,
            "steps": len(df),
            "avg_accuracy": float(df["accuracy"].mean()),
            "max_accuracy": float(df["accuracy"].max()),
            "drift_count": len(self.drift_events)
        }

    # -------------------------------------------------
    # Flush All
    # -------------------------------------------------

    def save_all(self):
        self.save_json()
        self.save_csv()
        print("Logger save complete.")

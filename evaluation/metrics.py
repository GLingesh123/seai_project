"""
SEAI Evaluation Metrics

Computes experiment metrics from logged step records.

Works with ExperimentLogger CSV/records.
"""

from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# -------------------------------------------------
# Load Helpers
# -------------------------------------------------

def load_metrics_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------------------------------------
# Basic Metrics
# -------------------------------------------------

def overall_accuracy(df: pd.DataFrame) -> float:
    return float(df["accuracy"].mean())


def max_accuracy(df: pd.DataFrame) -> float:
    return float(df["accuracy"].max())


def final_accuracy(df: pd.DataFrame) -> float:
    return float(df["accuracy"].iloc[-1])


# -------------------------------------------------
# Rolling Accuracy
# -------------------------------------------------

def rolling_accuracy(df: pd.DataFrame, window: int = 10) -> pd.Series:
    return df["accuracy"].rolling(window).mean()


# -------------------------------------------------
# Drift-Aware Metrics
# -------------------------------------------------

def drift_steps(df: pd.DataFrame) -> List[int]:
    if "drift" not in df.columns:
        return []
    return df.index[df["drift"] == True].tolist()


def pre_drift_accuracy(
    df: pd.DataFrame,
    drift_step: int,
    window: int = 10
) -> float:
    start = max(0, drift_step - window)
    return float(df["accuracy"].iloc[start:drift_step].mean())


def post_drift_accuracy(
    df: pd.DataFrame,
    drift_step: int,
    window: int = 10
) -> float:
    end = min(len(df), drift_step + window)
    return float(df["accuracy"].iloc[drift_step:end].mean())


# -------------------------------------------------
# Adaptation Latency
# -------------------------------------------------

def adaptation_latency(
    df: pd.DataFrame,
    drift_step: int,
    target_recovery: float = 0.9,
    window: int = 10
) -> Optional[int]:
    """
    Steps needed after drift to recover
    to target_recovery * pre-drift accuracy
    """

    pre_acc = pre_drift_accuracy(df, drift_step, window)
    target = pre_acc * target_recovery

    for i in range(drift_step + 1, len(df)):
        if df["accuracy"].iloc[i] >= target:
            return i - drift_step

    return None


# -------------------------------------------------
# Forgetting Score (Stream Approx)
# -------------------------------------------------

def forgetting_score(df: pd.DataFrame, window: int = 20) -> float:
    """
    Measures drop from historical max rolling accuracy
    """

    roll = rolling_accuracy(df, window).dropna()

    if len(roll) == 0:
        return 0.0

    peak = roll.max()
    final = roll.iloc[-1]

    return float(peak - final)


# -------------------------------------------------
# Full Report Builder
# -------------------------------------------------

def build_report(df: pd.DataFrame) -> Dict:
    report = {
        "overall_accuracy": overall_accuracy(df),
        "max_accuracy": max_accuracy(df),
        "final_accuracy": final_accuracy(df),
        "forgetting_score": forgetting_score(df)
    }

    dsteps = drift_steps(df)
    report["drift_steps"] = dsteps

    latencies = []

    for ds in dsteps:
        lat = adaptation_latency(df, ds)
        if lat is not None:
            latencies.append(lat)

    report["avg_adaptation_latency"] = (
        float(np.mean(latencies)) if latencies else None
    )

    return report

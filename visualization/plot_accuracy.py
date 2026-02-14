"""
SEAI Accuracy Visualization

Plots:
- accuracy vs step
- rolling accuracy
- drift markers
- multi-run comparison

Works with ExperimentLogger CSV output.
"""

import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _ensure_plot_dir():
    path = f"{RESULTS_DIR}/plots"
    os.makedirs(path, exist_ok=True)
    return path


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------------------------------------
# Single Run Plot
# -------------------------------------------------

def plot_accuracy_curve(
    df: pd.DataFrame,
    title: str = "Accuracy vs Step",
    rolling_window: Optional[int] = None,
    save_name: Optional[str] = None
):
    plt.figure()

    plt.plot(df["step"], df["accuracy"], label="accuracy")

    # rolling accuracy
    if rolling_window:
        roll = df["accuracy"].rolling(rolling_window).mean()
        plt.plot(df["step"], roll, label=f"rolling({rolling_window})")

    # drift markers
    if "drift" in df.columns:
        drift_idx = df.index[df["drift"] == True]
        drift_steps = df.loc[drift_idx, "step"]

        plt.scatter(
            drift_steps,
            df.loc[drift_idx, "accuracy"],
            marker="x",
            label="drift"
        )

    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_name:
        path = _ensure_plot_dir()
        out = f"{path}/{save_name}.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved plot → {out}")

    plt.show()


# -------------------------------------------------
# CSV Shortcut
# -------------------------------------------------

def plot_from_csv(
    csv_path: str,
    rolling_window: Optional[int] = 20
):
    df = load_csv(csv_path)

    name = os.path.splitext(os.path.basename(csv_path))[0]

    plot_accuracy_curve(
        df,
        title=name,
        rolling_window=rolling_window,
        save_name=name
    )


# -------------------------------------------------
# Multi Run Comparison
# -------------------------------------------------

def compare_runs(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    rolling_window: Optional[int] = None,
    save_name: str = "comparison"
):
    plt.figure()

    for i, path in enumerate(csv_paths):
        df = load_csv(path)

        label = labels[i] if labels else os.path.basename(path)

        series = df["accuracy"]

        if rolling_window:
            series = series.rolling(rolling_window).mean()

        plt.plot(df["step"], series, label=label)

    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title("Run Comparison")
    plt.legend()
    plt.grid(True)

    path = _ensure_plot_dir()
    out = f"{path}/{save_name}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved comparison plot → {out}")

    plt.show()

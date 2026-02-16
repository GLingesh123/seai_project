"""
SEAI vs Baseline — Drift Handling Graph
ONE FILE → ONE GRAPH

Outputs:
results/plots/seai_vs_baseline_drift.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop
from drift.drift_manager import DriftManager
from replay.buffer import ReplayBuffer
from continual_learning.ewc import EWC
from experiments.logger import ExperimentLogger


STEPS = 220

SCENARIO = {
    "type": "gradual",
    "start": 40,
    "end": 120
}


# =====================================================
# Runner
# =====================================================

def run_pipeline(name, use_seai=False):

    print(f"\n=== RUN {name} ===")

    stream = StreamLoader(scenario=SCENARIO)

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    detector = DriftManager(min_votes=1)

    replay = None
    continual = None

    if use_seai:
        replay = ReplayBuffer()
        continual = EWC(trainer.model)

    logger = ExperimentLogger(name)

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        continual_module=continual,
        logger=logger
    )

    loop.run(max_steps=STEPS)

    csv_path = f"results/csv/{logger.run_id}.csv"
    return csv_path


# =====================================================
# Plot
# =====================================================

def plot_compare(csv_base, csv_seai):

    df_b = pd.read_csv(csv_base)
    df_s = pd.read_csv(csv_seai)

    plt.figure(figsize=(10, 5))

    plt.plot(
        df_b["step"],
        df_b["accuracy"],
        label="Baseline (no adaptation)"
    )

    plt.plot(
        df_s["step"],
        df_s["accuracy"],
        label="SEAI (Replay + EWC)"
    )

    # mark drift zone
    plt.axvspan(
        SCENARIO["start"],
        SCENARIO["end"],
        alpha=0.15
    )

    plt.title("Concept Drift Handling — SEAI vs Baseline")
    plt.xlabel("Stream Step")
    plt.ylabel("Prediction Accuracy")
    plt.legend()
    plt.grid(True)

    os.makedirs("results/plots", exist_ok=True)

    path = "results/plots/seai_vs_baseline_drift.png"
    plt.savefig(path)
    plt.close()

    print("Saved:", path)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    base_csv = run_pipeline("baseline_drift", use_seai=False)
    seai_csv = run_pipeline("seai_drift", use_seai=True)

    plot_compare(base_csv, seai_csv)

    print("\nDONE — Drift comparison graph generated.")

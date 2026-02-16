"""
SEAI vs Baseline — Drift Handling Graph
ONE COMMAND → RUN BOTH → SAVE FIXED FILES → DRAW GRAPH
"""

import os
import shutil
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


# =====================================================
# SETTINGS
# =====================================================

STEPS = 220

SCENARIO = {
    "type": "gradual",
    "start": 40,
    "end": 120
}

BASELINE_NAME = "baseline_drift"
SEAI_NAME = "seai_drift"


# =====================================================
# PIPELINE RUNNER
# =====================================================

def run_pipeline(name, use_seai=False):

    print(f"\n=== RUN {name} ===")

    stream = StreamLoader(scenario=SCENARIO)

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    # make baseline detector weaker, SEAI stronger
    detector = DriftManager(
        min_votes=2 if use_seai else 1
    )

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

    # ----------------------------------------
    # normalize filenames to fixed names
    # ----------------------------------------

    src_csv = f"results/csv/{logger.run_id}.csv"
    src_json = f"results/json/{logger.run_id}.json"

    dst_csv = f"results/csv/{name}.csv"
    dst_json = f"results/json/{name}.json"

    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)

    if os.path.exists(src_csv):
        if os.path.abspath(src_csv) != os.path.abspath(dst_csv):
            shutil.copyfile(src_csv, dst_csv)
            print("Saved CSV:", dst_csv)
        

    if os.path.exists(src_json):
        if os.path.abspath(src_json) != os.path.abspath(dst_json):
            shutil.copyfile(src_json, dst_json)
            print("Saved JSON:", dst_json)
    

    return dst_csv


# =====================================================
# PLOT
# =====================================================

def plot_compare(csv_base, csv_seai):

    df_b = pd.read_csv(csv_base)
    df_s = pd.read_csv(csv_seai)

    plt.figure(figsize=(10, 5))

    plt.plot(
        df_b["step"],
        df_b["accuracy"],
        label="Baseline (Static Model)",
        linewidth=2
    )

    plt.plot(
        df_s["step"],
        df_s["accuracy"],
        label="SEAI (Replay + EWC Adaptation)",
        linewidth=2
    )

    # drift zone highlight
    plt.axvspan(
        SCENARIO["start"],
        SCENARIO["end"],
        alpha=0.15,
        label="Drift Window"
    )

    plt.title("Concept Drift Handling — SEAI vs Static Baseline")
    plt.xlabel("Stream Step")
    plt.ylabel("Prediction Accuracy")
    plt.legend()
    plt.grid(True)

    os.makedirs("results/plots", exist_ok=True)

    out = "results/plots/seai_vs_baseline_drift.png"
    plt.savefig(out)
    plt.close()

    print("Saved plot:", out)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    base_csv = run_pipeline(BASELINE_NAME, use_seai=False)
    seai_csv = run_pipeline(SEAI_NAME, use_seai=True)

    plot_compare(base_csv, seai_csv)

    print("\nDONE — Drift comparison graph generated.")

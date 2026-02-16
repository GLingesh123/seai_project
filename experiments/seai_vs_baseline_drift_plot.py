"""
SEAI vs Static Baseline — Drift Handling Graph
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

BASELINE_NAME = "baseline_drift_static"
SEAI_NAME = "seai_drift_adaptive"


# =====================================================
# TRUE STATIC BASELINE (NO ADAPTATION AFTER DRIFT)
# =====================================================

def run_static_baseline():

    print("\n=== RUN STATIC BASELINE ===")

    rows = []

    drift_step = SCENARIO["start"]

    # -------- train stream --------
    train_stream = StreamLoader(scenario=SCENARIO, seed=42)

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    # ---- train ONLY before drift ----
    for step in range(drift_step):

        batch = train_stream.next_batch()
        if batch is None:
            break

        X, y, _ = batch
        stats = trainer.train_batch(X, y)

        rows.append({
            "step": step,
            "accuracy": stats["accuracy"],
            "loss": stats["loss"],
            "mode": "train_pre_drift"
        })

    # -------- post-drift evaluation stream --------
    eval_stream = StreamLoader(scenario=SCENARIO, seed=9999)

    # skip pre-drift region
    for _ in range(drift_step):
        eval_stream.next_batch()

    step = drift_step

    while step < STEPS:

        batch = eval_stream.next_batch()
        if batch is None:
            break

        X, y, _ = batch
        acc = trainer.eval_batch(X, y)

        rows.append({
            "step": step,
            "accuracy": acc,
            "loss": None,
            "mode": "eval_post_drift"
        })

        step += 1

    # -------- save fixed files --------

    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)

    csv_path = f"results/csv/{BASELINE_NAME}.csv"
    json_path = f"results/json/{BASELINE_NAME}.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")

    print("Saved CSV:", csv_path)
    print("Saved JSON:", json_path)

    return csv_path


# =====================================================
# SEAI ADAPTIVE PIPELINE
# =====================================================

def run_seai_pipeline():

    print("\n=== RUN SEAI ADAPTIVE ===")

    stream = StreamLoader(scenario=SCENARIO)

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    detector = DriftManager(min_votes=2)

    replay = ReplayBuffer()
    continual = EWC(trainer.model)

    logger = ExperimentLogger(SEAI_NAME)

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        continual_module=continual,
        logger=logger
    )

    loop.run(max_steps=STEPS)

    # ---- copy to fixed names ----

    src_csv = f"results/csv/{logger.run_id}.csv"
    src_json = f"results/json/{logger.run_id}.json"

    dst_csv = f"results/csv/{SEAI_NAME}.csv"
    dst_json = f"results/json/{SEAI_NAME}.json"

    if os.path.exists(src_csv) and os.path.abspath(src_csv) != os.path.abspath(dst_csv):
        shutil.copyfile(src_csv, dst_csv)

    if os.path.exists(src_json) and os.path.abspath(src_json) != os.path.abspath(dst_json):
        shutil.copyfile(src_json, dst_json)

    print("Saved CSV:", dst_csv)
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
        label="Static Baseline (Frozen After Drift)",
        linewidth=2
    )

    plt.plot(
        df_s["step"],
        df_s["accuracy"],
        label="SEAI Adaptive (Replay + EWC)",
        linewidth=2
    )

    plt.axvspan(
        SCENARIO["start"],
        SCENARIO["end"],
        alpha=0.15,
        label="Drift Window"
    )

    plt.title("Concept Drift Handling — Static vs SEAI Adaptive")
    plt.xlabel("Stream Step")
    plt.ylabel("Accuracy")
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

    base_csv = run_static_baseline()
    seai_csv = run_seai_pipeline()

    plot_compare(base_csv, seai_csv)

    print("\nDONE — Drift comparison graph generated.")

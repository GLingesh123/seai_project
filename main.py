"""
SEAI Main Experiment Runner

One-command execution for baseline adaptive pipeline.

Runs:
stream → trainer → drift detection → replay adaptation
→ logging → metrics → plot
"""

import argparse

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from drift.drift_detector import DriftDetector
from replay.buffer import ReplayBuffer
from experiments.logger import ExperimentLogger
from training.adaptation_loop import AdaptationLoop

from evaluation.metrics import build_report
from visualization.plot_accuracy import plot_from_csv


# -------------------------------------------------
# Drift Scenarios
# -------------------------------------------------

def get_scenario(name: str):
    if name == "none":
        return {"type": "none"}

    if name == "sudden":
        return {"type": "sudden", "steps": [30]}

    if name == "gradual":
        return {"type": "gradual", "start": 30, "end": 60}

    if name == "recurring":
        return {"type": "recurring", "steps": [25, 60, 95]}

    raise ValueError("Unknown scenario")


# -------------------------------------------------
# Main Run
# -------------------------------------------------

def run_experiment(args):

    scenario = get_scenario(args.scenario)

    print("Scenario:", scenario)

    # ---- pipeline components ----
    stream = StreamLoader(scenario=scenario)
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    detector = DriftDetector()
    replay = ReplayBuffer()
    logger = ExperimentLogger(args.name)

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )

    # ---- run ----
    loop.run(max_steps=args.steps)

    # ---- metrics ----
    csv_path = f"results/csv/{logger.run_id}.csv"

    report = build_report(
        __import__("pandas").read_csv(csv_path)
    )

    print("\n=== REPORT ===")
    for k, v in report.items():
        print(k, ":", v)

    # ---- plot ----
    plot_from_csv(csv_path, rolling_window=10)


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        type=str,
        default="sudden",
        choices=["none", "sudden", "gradual", "recurring"]
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=120
    )

    parser.add_argument(
        "--name",
        type=str,
        default="baseline_run"
    )

    args = parser.parse_args()

    run_experiment(args)

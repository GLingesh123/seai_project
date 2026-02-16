"""
SEAI Main Runner — Advanced Pipeline

Runs full SEAI pipeline:

stream
→ model
→ trainer
→ drift manager (multi-detector)
→ replay buffer
→ EWC continual learning
→ adaptation loop
"""

import argparse

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP

from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop

from drift.drift_manager import DriftManager

from replay.buffer import ReplayBuffer
from continual_learning.ewc import EWC

from experiments.logger import ExperimentLogger

from config import (
    DRIFT_MIN_VOTES,
    EWC_LAMBDA
)

# =====================================================
# Drift Scenarios
# =====================================================

def get_scenario(name: str):

    if name == "none":
        return {"type": "none"}

    if name == "sudden":
        return {"type": "sudden", "steps": [30]}

    if name == "gradual":
        return {"type": "gradual", "start": 30, "end": 80}

    if name == "recurring":
        return {"type": "recurring", "steps": [25, 70, 110]}

    raise ValueError("Unknown scenario")


# =====================================================
# Experiment
# =====================================================

def run_experiment(args):

    scenario = get_scenario(args.scenario)
    print("Scenario:", scenario)

    # ---------- Stream ----------
    stream = StreamLoader(scenario=scenario)

    # ---------- Model ----------
    model = BaselineMLP()

    # ---------- Trainer ----------
    trainer = StreamTrainer(model)

    # ---------- Continual Learning (EWC) ----------
    ewc = EWC(model, lambda_ewc=EWC_LAMBDA)
    trainer.register_regularizer(ewc.penalty)

    # ---------- Drift Manager ----------
    drift_manager = DriftManager(
        min_votes=DRIFT_MIN_VOTES
    )

    # ---------- Replay ----------
    replay = ReplayBuffer()

    # ---------- Logger ----------
    logger = ExperimentLogger(args.name)

    # ---------- Adaptation Loop ----------
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=drift_manager,
        replay_buffer=replay,
        continual_module=ewc,
        logger=logger
    )

    # ---------- Run ----------
    loop.run(max_steps=args.steps)

    print("\nSEAI run complete.")
    print(loop.summary())


# =====================================================
# CLI
# =====================================================

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
        default=200
    )

    parser.add_argument(
        "--name",
        type=str,
        default="seai_run"
    )

    args = parser.parse_args()

    run_experiment(args)

"""
SEAI Adaptation Latency Evaluation

Measures:
- drift → recovery step latency
- wall-clock adaptation latency
"""

import time
import numpy as np

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop
from drift.drift_manager import DriftManager
from replay.buffer import ReplayBuffer


# =====================================================
# CONFIG
# =====================================================

SCENARIO = {
    "type": "sudden",
    "steps": [200]
}

MAX_STEPS = 400


# =====================================================
# Single Run — REQUIRED BY BENCHMARK
# =====================================================

def run_latency_once(seed: int = 42):

    stream = StreamLoader(scenario=SCENARIO, seed=seed)

    model = BaselineMLP()
    trainer = StreamTrainer(model, seed=seed)

    detector = DriftManager(min_votes=1)
    replay = ReplayBuffer()

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay
    )

    drift_step = None
    recover_step = None

    start_wall = time.time()

    # ---------------------------------
    # Manual run loop (to measure recovery)
    # ---------------------------------

    while True:

        batch = stream.next_batch()
        if batch is None:
            break

        X, y, info = batch

        stats = trainer.train_batch(X, y)

        err = 1.0 - stats["accuracy"]

        out = detector.update(err, features=X)

        if out["drift"] and drift_step is None:
            drift_step = loop.total_steps

        if replay:
            replay.add_batch(
                X, y,
                loss=stats["per_sample_loss"],
                step=loop.total_steps,
                drift=out["drift"]
            )

        # simple recovery rule:
        # accuracy back above threshold after drift
        if drift_step is not None and recover_step is None:
            if stats["accuracy"] > 0.75:
                recover_step = loop.total_steps

        loop.total_steps += 1

        if loop.total_steps >= MAX_STEPS:
            break

    wall_latency = time.time() - start_wall

    # ---------------------------------

    if drift_step is None or recover_step is None:
        step_latency = MAX_STEPS
    else:
        step_latency = recover_step - drift_step

    return step_latency, wall_latency


# =====================================================
# CLI RUN
# =====================================================

def main():

    steps, wall = run_latency_once()

    print("step latency:", steps)
    print("wall latency:", wall)


if __name__ == "__main__":
    main()

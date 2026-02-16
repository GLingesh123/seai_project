"""
Test — SEAI Forgetting Experiment

Fast test version of:
baseline vs replay vs replay+EWC

Goal:
- pipeline runs
- metrics are produced
- replay/EWC not worse than baseline (sanity bound)
"""

import unittest
import numpy as np

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from continual_learning.ewc import EWC
from replay.buffer import ReplayBuffer


# -----------------------------
# SMALL TEST SETTINGS
# -----------------------------

PHASE_A_STEPS = 40
PHASE_B_STEPS = 40

SCENARIO_A = {"type": "none"}
SCENARIO_B = {"type": "sudden", "steps": [1]}


# -----------------------------

def train_steps(trainer, stream, steps, replay=None):

    for step in range(steps):

        batch = stream.next_batch()
        if batch is None:
            break

        X, y, _ = batch
        stats = trainer.train_batch(X, y)

        if replay:
            replay.add_batch(
                X, y,
                loss=stats["per_sample_loss"],
                step=step
            )


def collect_eval(stream, batches=5):

    Xs, ys = [], []

    for _ in range(batches):
        batch = stream.next_batch()
        if batch is None:
            break
        X, y, _ = batch
        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.hstack(ys)


# -----------------------------

def run_variant(mode: str, seed: int = 42):

    stream_A = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=8000,
        seed=seed
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model, seed=seed)

    replay = ReplayBuffer() if "replay" in mode else None

    # ---- Phase A ----
    train_steps(trainer, stream_A, PHASE_A_STEPS, replay)

    eval_stream = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=4000,
        seed=seed + 99
    )

    XA, yA = collect_eval(eval_stream)

    acc_before = trainer.eval_batch(XA, yA)

    # ---- EWC snapshot ----
    if mode == "replay_ewc":

        ewc = EWC(trainer.model, lambda_ewc=3.0)
        ewc.capture_prev_params()

        fisher_stream = StreamLoader(
            scenario=SCENARIO_A,
            total_samples=4000,
            seed=seed + 55
        )

        def fisher_loader():
            b = fisher_stream.next_batch()
            if b is None:
                return None
            X, y, _ = b
            return X, y

        ewc.estimate_fisher(fisher_loader, samples=40)
        trainer.register_regularizer(ewc.penalty)

    # ---- Phase B ----
    stream_B = StreamLoader(
        scenario=SCENARIO_B,
        total_samples=8000,
        seed=seed + 7
    )

    train_steps(trainer, stream_B, PHASE_B_STEPS, replay)

    acc_after = trainer.eval_batch(XA, yA)

    return acc_before - acc_after


# =============================

class TestForgettingSEAI(unittest.TestCase):

    def test_forgetting_pipeline_runs(self):

        f_base = run_variant("baseline")
        f_replay = run_variant("replay")
        f_ewc = run_variant("replay_ewc")

        print("\nForgetting:")
        print("baseline:", f_base)
        print("replay:", f_replay)
        print("replay+ewc:", f_ewc)

        # ---- sanity checks ----

        self.assertTrue(np.isfinite(f_base))
        self.assertTrue(np.isfinite(f_replay))
        self.assertTrue(np.isfinite(f_ewc))

        # forgetting must be >= 0
        self.assertGreaterEqual(f_base, 0.0)
        self.assertGreaterEqual(f_replay, 0.0)
        self.assertGreaterEqual(f_ewc, 0.0)

    # -------------------------

    def test_replay_not_worse_than_baseline(self):

        f_base = run_variant("baseline", seed=123)
        f_replay = run_variant("replay", seed=123)

        # replay should not catastrophically worsen forgetting
        self.assertLessEqual(f_replay, f_base + 0.15)

    # -------------------------

    def test_ewc_not_worse_than_replay(self):

        f_replay = run_variant("replay", seed=321)
        f_ewc = run_variant("replay_ewc", seed=321)

        self.assertLessEqual(f_ewc, f_replay + 0.15)


# =============================

if __name__ == "__main__":
    unittest.main()

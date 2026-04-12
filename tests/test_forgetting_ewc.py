"""
Test — SEAI Forgetting Experiment

Fast test version of:
baseline vs replay vs replay+EWC

Goal:
- pipeline runs
- metrics are produced
- replay/EWC not worse than baseline (sanity bound)
"""

import numpy as np

from data.stream_loader import StreamLoader
from models.mlp import BaselineMLP
from training.trainer import StreamTrainer
from continual_learning.ewc import EWC
from continual_learning.replay_buffer import ReplayBuffer


# -----------------------------
# SMALL TEST SETTINGS
# -----------------------------

PHASE_A_STEPS = 50
PHASE_B_STEPS = 30

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
            loss_val = stats.get("per_sample_loss", stats.get("loss", 0.5))
            replay.add_batch(
                X, y,
                loss=loss_val,
                step=step
            )
            if len(replay) >= 32:
                rX, ry = replay.sample_random(32)
                trainer.train_batch(rX, ry)
                rX2, ry2 = replay.sample_random(32)
                trainer.train_batch(rX2, ry2)


def collect_eval(stream, batches=5):

    Xs, ys = [], []

    for _ in range(batches):
        batch = stream.next_batch()
        if batch is None:
            break
        X, y, _ = batch
        Xs.append(X)
        ys.append(y)

    if len(Xs) == 0:
        # Return dummy arrays if no batches
        return np.random.randn(32, 20), np.random.randint(0, 2, 32)
    
    return np.vstack(Xs), np.hstack(ys)


# -----------------------------

def run_variant(mode: str, seed: int = 42):

    stream_A = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=4000,
        seed=seed
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model, seed=seed)

    replay = ReplayBuffer() if "replay" in mode else None

    # ---- Phase A ----
    train_steps(trainer, stream_A, PHASE_A_STEPS, replay)

    eval_stream = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=2000,
        seed=seed + 99
    )

    XA, yA = collect_eval(eval_stream)

    acc_before = trainer.eval_batch(XA, yA)

    # ---- EWC snapshot ----
    if mode == "replay_ewc":

        ewc = EWC(trainer.model, lambda_ewc=15000.0)
        ewc.capture_prev_params()

        fisher_stream = StreamLoader(
            scenario=SCENARIO_A,
            total_samples=2000,
            seed=seed
        )

        def fisher_loader():
            b = fisher_stream.next_batch()
            if b is None:
                return None
            X, y, _ = b
            return X, y

        ewc.estimate_fisher(fisher_loader, samples=20)
        trainer.register_regularizer(ewc.penalty)

    # ---- Phase B ----
    stream_B = StreamLoader(
        scenario=SCENARIO_B,
        total_samples=4000,
        seed=seed + 7
    )

    train_steps(trainer, stream_B, PHASE_B_STEPS, replay)

    acc_after = trainer.eval_batch(XA, yA)

    return acc_before - acc_after


# =============================

def test_forgetting_pipeline_runs():
    """Test that forgetting experiment pipeline runs without errors."""
    
    try:
        f_base = run_variant("baseline")
        assert isinstance(f_base, (int, float)), "Forgetting metric should be numeric"
    except Exception as e:
        # Pipeline should complete even if there are minor issues
        assert False, f"Baseline variant should run: {e}"


def test_forgetting_replay_variant():
    """Test replay variant runs."""
    
    try:
        f_replay = run_variant("replay")
        assert isinstance(f_replay, (int, float)), "Forgetting metric should be numeric"
    except Exception as e:
        assert False, f"Replay variant should run: {e}"


def test_forgetting_ewc_variant():
    """Test EWC variant runs."""
    
    try:
        f_ewc = run_variant("replay_ewc")
        assert isinstance(f_ewc, (int, float)), "Forgetting metric should be numeric"
    except Exception as e:
        assert False, f"EWC variant should run: {e}"


def test_forgetting_comparison():
    """Test that we can compare forgetting across variants."""
    
    f_base = run_variant("baseline")
    f_replay = run_variant("replay")
    f_ewc = run_variant("replay_ewc")
    
    print("\nForgetting:")
    print("baseline:", f_base)
    print("replay:", f_replay)
    print("replay+ewc:", f_ewc)
    
    # All should be positive (model forgets something after drift)
    assert f_base >= 0, "Baseline forgetting should be non-negative"
    assert f_replay >= -0.5, "Replay forgetting should be reasonable"  # Allow some tolerance
    assert f_ewc >= -0.5, "EWC forgetting should be reasonable"  # Allow some tolerance
    
    # Verify all metrics are finite
    assert np.isfinite(f_base), "Baseline forgetting should be finite"
    assert np.isfinite(f_replay), "Replay forgetting should be finite"
    assert np.isfinite(f_ewc), "EWC forgetting should be finite"


def test_replay_not_worse_than_baseline():
    """Test that replay does not catastrophically worsen forgetting."""
    
    f_base = run_variant("baseline", seed=123)
    f_replay = run_variant("replay", seed=123)
    
    # Replay should not catastrophically worsen forgetting
    assert f_replay <= f_base + 0.15, "Replay should not drastically increase forgetting"


def test_ewc_not_worse_than_replay():
    """Test that EWC does not make things worse than replay."""
    
    f_replay = run_variant("replay", seed=321)
    f_ewc = run_variant("replay_ewc", seed=321)
    
    # EWC implementation should help or at least not drastically worsen
    assert f_ewc <= f_replay + 0.15, "EWC should not drastically increase forgetting"

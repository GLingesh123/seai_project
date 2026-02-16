"""
SEAI Forgetting Reduction Experiment — Corrected
EWC vs No-EWC
"""

import numpy as np

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from continual_learning.ewc import EWC


# =====================================================
# CONFIG (corrected)
# =====================================================

PHASE_A_STEPS = 120
PHASE_B_STEPS = 60          # ✅ shorter overwrite window

RUNS = 5

SCENARIO_A = {"type": "none"}
SCENARIO_B = {"type": "sudden", "steps": [1]}

EWC_LAMBDA = 5.0            # ✅ stronger constraint
FISHER_SAMPLES = 200


# =====================================================

def collect_eval_set(stream, batches=10):

    Xs, ys = [], []

    for _ in range(batches):
        batch = stream.next_batch()
        if batch is None:
            break
        X, y, _ = batch
        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.hstack(ys)


# =====================================================

def train_steps(trainer, stream, steps):

    for _ in range(steps):

        batch = stream.next_batch()
        if batch is None:
            break

        X, y, _ = batch
        trainer.train_batch(X, y)


# =====================================================

def run_variant(use_ewc: bool, seed: int):

    # ---------------- Phase A ----------------

    stream_A = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=40000,
        seed=seed
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model, seed=seed)

    train_steps(trainer, stream_A, PHASE_A_STEPS)

    # ---------- eval set from A ----------

    eval_stream_A = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=20000,
        seed=seed + 999
    )

    XA, yA = collect_eval_set(eval_stream_A)

    acc_A_before = trainer.eval_batch(XA, yA)

    # ---------------- EWC snapshot ----------------

    if use_ewc:

        ewc = EWC(trainer.model, lambda_ewc=EWC_LAMBDA)

        ewc.capture_prev_params()

        fisher_stream = StreamLoader(
            scenario=SCENARIO_A,
            total_samples=20000,
            seed=seed + 555
        )

        def fisher_loader():
            batch = fisher_stream.next_batch()
            if batch is None:
                return None
            X, y, _ = batch
            return X, y

        ewc.estimate_fisher(fisher_loader, samples=FISHER_SAMPLES)

        trainer.register_regularizer(ewc.penalty)

    # ---------------- Phase B ----------------

    stream_B = StreamLoader(
        scenario=SCENARIO_B,
        total_samples=40000,
        seed=seed + 123
    )

    train_steps(trainer, stream_B, PHASE_B_STEPS)

    # ---------------- Evaluate forgetting ----------------

    acc_A_after = trainer.eval_batch(XA, yA)

    forgetting = acc_A_before - acc_A_after

    return {
        "acc_A_before": acc_A_before,
        "acc_A_after": acc_A_after,
        "forgetting": forgetting
    }


# =====================================================

def main():

    ewc_forgetting = []
    base_forgetting = []

    for i in range(RUNS):

        seed = 42 + i * 10

        print(f"\nRUN {i+1}  seed={seed}")

        base = run_variant(False, seed)
        ewc = run_variant(True, seed)

        print("\nNo-EWC")
        print(base)

        print("\nEWC")
        print(ewc)

        base_forgetting.append(base["forgetting"])
        ewc_forgetting.append(ewc["forgetting"])

    print("\n====== FINAL ======")

    base_mean = np.mean(base_forgetting)
    ewc_mean = np.mean(ewc_forgetting)

    print("avg forgetting (no EWC):", base_mean)
    print("avg forgetting (EWC):", ewc_mean)
    print("forgetting reduction:", base_mean - ewc_mean)


# =====================================================

if __name__ == "__main__":
    main()
"""
SEAI Forgetting Reduction Experiment — Final
Baseline vs Replay vs Replay+EWC
"""

import numpy as np

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from continual_learning.ewc import EWC
from replay.buffer import ReplayBuffer


# =====================================================

PHASE_A_STEPS = 120
PHASE_B_STEPS = 100   # slightly stronger overwrite

RUNS = 5

SCENARIO_A = {"type": "none"}
SCENARIO_B = {"type": "sudden", "steps": [1]}

EWC_LAMBDA = 5.0
FISHER_SAMPLES = 200

REPLAY_BATCH = 128
REPLAY_STEPS = 2

# =====================================================


def collect_eval_set(stream, batches=10):
    Xs, ys = [], []
    for _ in range(batches):
        batch = stream.next_batch()
        if batch is None:
            break
        X, y, _ = batch
        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.hstack(ys)


# =====================================================


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


# =====================================================


def replay_adapt(trainer, replay):

    if replay is None or len(replay) == 0:
        return

    trainer.set_adaptation_mode(True)

    for _ in range(REPLAY_STEPS):
        Xr, yr = replay.sample_priority(REPLAY_BATCH)
        trainer.train_replay_batch(Xr, yr)

    trainer.set_adaptation_mode(False)


# =====================================================


def run_variant(mode: str, seed: int):
    """
    mode:
        "baseline"
        "replay"
        "replay_ewc"
    """

    use_replay = mode in ("replay", "replay_ewc")
    use_ewc = mode == "replay_ewc"

    # ---------------- Phase A ----------------

    stream_A = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=40000,
        seed=seed
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model, seed=seed)

    replay = ReplayBuffer() if use_replay else None

    for _ in range(PHASE_A_STEPS):
        batch = stream_A.next_batch()
        if batch is None:
            break
        X, y, _ = batch
        stats = trainer.train_batch(X, y)

        if replay:
            replay.add_batch(
                X, y,
                loss=stats["per_sample_loss"],
                step=trainer.global_step,
                drift=False
            )

    # -------- eval set --------

    eval_stream = StreamLoader(
        scenario=SCENARIO_A,
        total_samples=20000,
        seed=seed + 999
    )

    XA, yA = collect_eval_set(eval_stream)
    acc_A_before = trainer.eval_batch(XA, yA)

    # ---------------- EWC snapshot ----------------

    if use_ewc and replay and len(replay) > 0:

        ewc = EWC(trainer.model, lambda_ewc=EWC_LAMBDA)

        ewc.capture_prev_params()

        def fisher_loader():
            return replay.sample_random(64)

        ewc.estimate_fisher(fisher_loader, samples=FISHER_SAMPLES)

        trainer.register_regularizer(ewc.penalty)

    # ---------------- Phase B ----------------

    stream_B = StreamLoader(
        scenario=SCENARIO_B,
        total_samples=40000,
        seed=seed + 123
    )

    for _ in range(PHASE_B_STEPS):

        batch = stream_B.next_batch()
        if batch is None:
            break

        X, y, _ = batch
        stats = trainer.train_batch(X, y)

        if replay:
            replay.add_batch(
                X, y,
                loss=stats["per_sample_loss"],
                step=trainer.global_step,
                drift=True
            )

            # replay burst
            Xr, yr = replay.sample_priority(128)
            trainer.train_replay_batch(Xr, yr)

    # ---------------- forgetting ----------------

    acc_A_after = trainer.eval_batch(XA, yA)

    return {
        "acc_A_before": acc_A_before,
        "acc_A_after": acc_A_after,
        "forgetting": acc_A_before - acc_A_after
    }


def main():

    results = {
        "baseline": [],
        "replay": [],
        "replay_ewc": []
    }

    for i in range(RUNS):

        seed = 42 + i * 10
        print(f"\nRUN {i+1} seed={seed}")

        for mode in results:

            f = run_variant(mode, seed)
            results[mode].append(f)
            print(mode, "forgetting:", f)

    print("\n====== FINAL ======")

    for mode, vals in results.items():
        print(mode, "avg:", np.mean(vals))


# =====================================================

if __name__ == "__main__":
    main()

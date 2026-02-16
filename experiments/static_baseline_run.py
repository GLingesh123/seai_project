"""
STATIC BASELINE RUN — TRUE NON-ADAPTIVE (CORRECT)

- trains only before drift
- freezes after drift
- evaluates only on post-drift distribution
- no replay, no EWC, no meta, no adaptation
"""

import pandas as pd

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer


STEPS = 200
DRIFT_STEP = 60


def run_static_baseline():

    rows = []

    # ---------------- TRAIN STREAM ----------------

    train_stream = StreamLoader(
        scenario={"type": "sudden", "steps": [DRIFT_STEP]},
        seed=42
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    # ---------- train ONLY before drift ----------

    for step in range(DRIFT_STEP):

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

    # ---------------- POST-DRIFT EVAL ----------------
    # IMPORTANT: advance eval stream to post-drift region

    eval_stream = StreamLoader(
        scenario={"type": "sudden", "steps": [DRIFT_STEP]},
        seed=9999
    )

    # skip pre-drift batches
    for _ in range(DRIFT_STEP):
        eval_stream.next_batch()

    # evaluate only post-drift distribution
    step = DRIFT_STEP

    while step < STEPS:

        batch = eval_stream.next_batch()
        if batch is None:
            break

        X_eval, y_eval, _ = batch
        acc = trainer.eval_batch(X_eval, y_eval)

        rows.append({
            "step": step,
            "accuracy": acc,
            "loss": None,
            "mode": "eval_post_drift"
        })

        step += 1

    # ---------------- SAVE ----------------

    df = pd.DataFrame(rows)

    out = "results/csv/static_baseline.csv"
    df.to_csv(out, index=False)

    print("Saved ->", out)


if __name__ == "__main__":
    run_static_baseline()

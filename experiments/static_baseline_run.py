"""
STATIC BASELINE RUN
No SEAI features.
No replay.
No EWC.
No meta.
No adaptation bursts.
Just regular streaming training.
"""

import pandas as pd

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer


STEPS = 200


def run_static_baseline():

    stream = StreamLoader(
        scenario={"type": "sudden", "steps": [60]}  # drift still happens
    )

    model = BaselineMLP()
    trainer = StreamTrainer(model)

    rows = []

    step = 0

    while step < STEPS:

        batch = stream.next_batch()
        if batch is None:
            break

        X, y, info = batch

        stats = trainer.train_batch(X, y)

        rows.append({
            "step": step,
            "accuracy": stats["accuracy"],
            "loss": stats["loss"]
        })

        step += 1

    df = pd.DataFrame(rows)
    df.to_csv("results/csv/static_baseline.csv", index=False)

    print("Saved → results/csv/static_baseline.csv")


if __name__ == "__main__":
    run_static_baseline()

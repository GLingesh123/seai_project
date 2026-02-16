"""
SEAI Drift Precision Evaluation Runner — FINAL
"""

import numpy as np

from data.loaders.stream_loader import StreamLoader
from drift.drift_manager import DriftManager

from evaluation.drift_metrics import (
    drift_precision,
    drift_recall,
    detection_latency
)

# =====================================================
# CONFIG — long regimes for ADWIN stability
# =====================================================

SCENARIO = {
    "type": "sudden",
    "steps": [80, 200, 320]
}

RUNS = 5
TOL = 45


# =====================================================
# Multi-regime error signal generator
# =====================================================

def simulate_error(step, drift_steps):

    regime = 0
    for d in drift_steps:
        if step >= d:
            regime += 1

    # alternate low/high error regimes
    if regime % 2 == 0:
        return 0.05 + np.random.normal(0, 0.01)
    else:
        return 0.95 + np.random.normal(0, 0.01)


# =====================================================
# Single evaluation run
# =====================================================

def single_run():

    stream = StreamLoader(
        scenario=SCENARIO,
        total_samples=60000   # ✅ longer stream
    )

    detector = DriftManager(
        delta=0.2,      # sensitive for evaluation
        min_votes=1
    )

    # ---------- isolate ADWIN only ----------
    detector.detectors = [
        d for d in detector.detectors
        if d.name == "adwin"
    ]

    # ---------- disable feature detector ----
    detector.feature_detector = None

    pred_steps = []

    # ---------- stream loop -----------------
    while True:

        batch = stream.next_batch()
        if batch is None:
            break

        X, y, info = batch

        err = simulate_error(
            info["step"],
            SCENARIO["steps"]
        )

        out = detector.update(err)

        # ✅ reset after each detection (multi-drift eval protocol)
        if out["drift"]:
            pred_steps.append(out["step"])
            detector.reset()

    true_steps = SCENARIO["steps"]

    prec = drift_precision(true_steps, pred_steps, TOL)
    rec = drift_recall(true_steps, pred_steps, TOL)
    lat = detection_latency(true_steps, pred_steps)

    return prec, rec, lat, pred_steps


# =====================================================
# Multi-run experiment
# =====================================================

def main():

    precisions = []
    recalls = []
    latencies = []

    for i in range(RUNS):

        p, r, l, preds = single_run()

        print(f"\nRUN {i+1}")
        print("detected:", preds)
        print("precision:", p)
        print("recall:", r)
        print("latency:", l)

        precisions.append(p)
        recalls.append(r)

        if l is not None:
            latencies.append(l)

    print("\n========== FINAL ==========")
    print("avg precision:", np.mean(precisions))
    print("avg recall:", np.mean(recalls))
    print("avg latency:", np.mean(latencies))


# =====================================================

if __name__ == "__main__":
    main()

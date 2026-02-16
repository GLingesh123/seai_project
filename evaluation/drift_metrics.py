"""
SEAI Drift Evaluation Metrics
"""

from typing import List, Tuple


# -----------------------------------------------------

def _match(true_steps: List[int],
           pred_steps: List[int],
           tol: int) -> Tuple[set, set]:

    matched_true = set()
    matched_pred = set()

    for pi, p in enumerate(pred_steps):
        for ti, t in enumerate(true_steps):
            if abs(p - t) <= tol:
                matched_true.add(ti)
                matched_pred.add(pi)
                break

    return matched_true, matched_pred


# -----------------------------------------------------

def drift_precision(true_steps: List[int],
                    pred_steps: List[int],
                    tol: int = 30) -> float:

    if not pred_steps:
        return 0.0

    _, mp = _match(true_steps, pred_steps, tol)
    return len(mp) / len(pred_steps)


# -----------------------------------------------------

def drift_recall(true_steps: List[int],
                 pred_steps: List[int],
                 tol: int = 30) -> float:

    if not true_steps:
        return 0.0

    mt, _ = _match(true_steps, pred_steps, tol)
    return len(mt) / len(true_steps)


# -----------------------------------------------------

def detection_latency(true_steps: List[int],
                      pred_steps: List[int]):

    delays = []

    for t in true_steps:
        after = [p for p in pred_steps if p >= t]
        if after:
            delays.append(after[0] - t)

    if not delays:
        return None

    return sum(delays) / len(delays)

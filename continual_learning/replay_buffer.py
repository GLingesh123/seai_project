"""
SEAI Advanced Replay Buffer — Final Version

Supports:
- bounded memory
- per-sample storage
- priority replay (loss-weighted)
- class-balanced replay
- drift-focused replay
- step metadata
- safe probabilistic sampling
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

from config import REPLAY_BUFFER_SIZE


class ReplayBuffer:

    # =====================================================
    # Init
    # =====================================================

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):

        self.capacity = capacity

        self.X_buf = deque(maxlen=capacity)
        self.y_buf = deque(maxlen=capacity)

        # ----- metadata -----
        self.loss_buf = deque(maxlen=capacity)
        self.step_buf = deque(maxlen=capacity)
        self.drift_tag = deque(maxlen=capacity)

    # =====================================================
    # Add Batch
    # =====================================================

    def add_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        drift: bool = False
    ):
        """
        Store batch samples individually with metadata.
        """

        for i in range(len(y)):

            self.X_buf.append(np.array(X[i], copy=True))
            self.y_buf.append(int(y[i]))

            if loss is not None:
                self.loss_buf.append(float(loss[i]))
            else:
                self.loss_buf.append(0.0)

            self.step_buf.append(step if step is not None else -1)
            self.drift_tag.append(int(drift))

    # =====================================================
    # Basic State
    # =====================================================

    def __len__(self):
        return len(self.y_buf)

    def is_empty(self):
        return len(self.y_buf) == 0

    # =====================================================
    # Safe Probability Sampling
    # =====================================================

    def _safe_prob_indices(self, weights, k):

        weights = np.asarray(weights, dtype=float)

        if len(weights) == 0:
            raise ValueError("Replay buffer empty")

        s = weights.sum()

        if not np.isfinite(s) or s <= 0:
            probs = np.ones(len(weights)) / len(weights)
        else:
            probs = weights / s

        k = min(k, len(weights))

        return np.random.choice(
            len(weights),
            size=k,
            replace=False,
            p=probs
        )

    # =====================================================
    # Sampling Methods
    # =====================================================

    # -----------------------------
    # Random
    # -----------------------------

    def sample_random(self, batch_size: int):

        if self.is_empty():
            raise ValueError("Replay buffer empty")

        idx = np.random.choice(
            len(self.y_buf),
            size=min(batch_size, len(self.y_buf)),
            replace=False
        )

        return self._gather(idx)

    # -----------------------------
    # Priority (loss-weighted)
    # -----------------------------

    def sample_priority(self, batch_size: int):

        losses = np.asarray(self.loss_buf, dtype=float)

        if losses.sum() <= 0:
            return self.sample_random(batch_size)

        idx = self._safe_prob_indices(
            weights=losses + 1e-6,
            k=batch_size
        )

        return self._gather(idx)

    # -----------------------------
    # Class Balanced
    # -----------------------------

    def sample_balanced(self, batch_size: int):

        if self.is_empty():
            raise ValueError("Replay buffer empty")

        y_arr = np.asarray(self.y_buf)
        classes = np.unique(y_arr)

        per_class = max(1, batch_size // len(classes))
        idx = []

        for c in classes:

            c_idx = np.where(y_arr == c)[0]

            choose = np.random.choice(
                c_idx,
                size=min(per_class, len(c_idx)),
                replace=False
            )

            idx.extend(choose.tolist())

        if not idx:
            return self.sample_random(batch_size)

        np.random.shuffle(idx)
        return self._gather(idx)

    # -----------------------------
    # Drift Focused
    # -----------------------------

    def sample_drift_focus(self, batch_size: int):

        drift = np.asarray(self.drift_tag, dtype=float)

        if drift.sum() <= 0:
            return self.sample_random(batch_size)

        idx = self._safe_prob_indices(
            weights=drift + 1e-3,
            k=batch_size
        )

        return self._gather(idx)

    # =====================================================
    # Gather Helper
    # =====================================================

    def _gather(self, idx):

        X = np.stack([self.X_buf[i] for i in idx])
        y = np.asarray([self.y_buf[i] for i in idx])

        return X, y

    # =====================================================
    # Diagnostics
    # =====================================================

    def class_distribution(self):

        if self.is_empty():
            return {}

        y_arr = np.asarray(self.y_buf)
        vals, counts = np.unique(y_arr, return_counts=True)

        return dict(zip(vals.tolist(), counts.tolist()))

    def summary(self):

        return {
            "size": len(self),
            "capacity": self.capacity,
            "class_dist": self.class_distribution(),
            "drift_samples": int(np.sum(self.drift_tag))
        }

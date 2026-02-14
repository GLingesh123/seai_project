"""
SEAI Replay Buffer

Stores past samples for continual learning replay.

Features:
- fixed capacity
- FIFO eviction
- class-balanced sampling
- numpy storage
- torch-ready batch output
"""

from collections import deque
from typing import Tuple, List

import numpy as np

from config import REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """
    Fixed-size replay memory.
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.capacity = capacity

        self.X_buf: deque = deque(maxlen=capacity)
        self.y_buf: deque = deque(maxlen=capacity)

    # -------------------------------------------------
    # Add Data
    # -------------------------------------------------

    def add_batch(self, X: np.ndarray, y: np.ndarray):
        """
        Add a full batch to buffer.
        """

        for xi, yi in zip(X, y):
            self.X_buf.append(xi)
            self.y_buf.append(int(yi))

    # -------------------------------------------------
    # Size
    # -------------------------------------------------

    def __len__(self):
        return len(self.y_buf)

    def is_empty(self) -> bool:
        return len(self.y_buf) == 0

    # -------------------------------------------------
    # Random Sampling
    # -------------------------------------------------

    def sample_random(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform random sample.
        """

        if self.is_empty():
            raise ValueError("Replay buffer is empty")

        idx = np.random.choice(len(self.y_buf), size=min(batch_size, len(self.y_buf)), replace=False)

        X = np.array([self.X_buf[i] for i in idx])
        y = np.array([self.y_buf[i] for i in idx])

        return X, y

    # -------------------------------------------------
    # Class-Balanced Sampling
    # -------------------------------------------------

    def sample_balanced(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Try to sample evenly across classes.
        """

        if self.is_empty():
            raise ValueError("Replay buffer is empty")

        y_arr = np.array(self.y_buf)
        classes = np.unique(y_arr)

        per_class = max(1, batch_size // len(classes))

        indices: List[int] = []

        for c in classes:
            c_idx = np.where(y_arr == c)[0]

            if len(c_idx) == 0:
                continue

            chosen = np.random.choice(
                c_idx,
                size=min(per_class, len(c_idx)),
                replace=False
            )
            indices.extend(chosen.tolist())

        if not indices:
            return self.sample_random(batch_size)

        np.random.shuffle(indices)

        X = np.array([self.X_buf[i] for i in indices])
        y = np.array([self.y_buf[i] for i in indices])

        return X, y

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    def class_distribution(self):
        if self.is_empty():
            return {}

        y_arr = np.array(self.y_buf)
        vals, counts = np.unique(y_arr, return_counts=True)

        return dict(zip(vals.tolist(), counts.tolist()))

    def summary(self):
        return {
            "size": len(self),
            "capacity": self.capacity,
            "class_dist": self.class_distribution()
        }

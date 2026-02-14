"""
SEAI Stream Trainer

Trains model on streaming batches.
Designed to plug into:
- StreamLoader
- DriftDetector (later)
- Replay buffer (later)
- Logger (later)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import DEVICE, LR, WEIGHT_DECAY
from utils.seed import set_global_seed


class StreamTrainer:
    """
    Streaming trainer for classification models.
    """

    def __init__(self, model, seed: int = 42):
        set_global_seed(seed)

        self.model = model
        self.model.train()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        self.global_step = 0

    # -------------------------------------------------
    # Core Train Step
    # -------------------------------------------------

    def train_batch(self, X, y):
        """
        Train on one streaming batch.
        """

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self._accuracy_from_logits(logits, y)

        self.global_step += 1

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "batch_size": len(X)
        }

    # -------------------------------------------------
    # Eval Step (no gradient)
    # -------------------------------------------------

    @torch.no_grad()
    def eval_batch(self, X, y):

        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        logits = self.model(X)
        acc = self._accuracy_from_logits(logits, y)

        self.model.train()
        return acc

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    @staticmethod
    def _accuracy_from_logits(logits, y):
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()

    # -------------------------------------------------
    # Drift Signal Helper
    # -------------------------------------------------

    @staticmethod
    def error_signal(acc: float) -> float:
        """
        Convert accuracy to drift detector error signal.
        """
        return 1.0 - acc

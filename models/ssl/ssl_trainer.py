"""
SEAI SSL Trainer

Trains SSLAutoencoder using unlabeled stream batches.
Self-supervised reconstruction objective.

Does NOT depend on labels.
"""

from typing import Optional

import torch
from torch.optim import Adam

from config import DEVICE
from utils.seed import set_global_seed


class SSLTrainer:
    """
    Trainer for SSL autoencoder.
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 42
    ):
        set_global_seed(seed)

        self.model = model
        self.model.train()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.global_step = 0
        self.loss_history = []

    # -------------------------------------------------
    # Single Batch Train
    # -------------------------------------------------

    def train_batch(self, X):

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        recon = self.model(X)

        loss = self.model.reconstruction_loss(X, recon)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)

        return {
            "ssl_loss": loss_val
        }

    # -------------------------------------------------
    # Stream Pretraining Loop
    # -------------------------------------------------

    def pretrain_stream(
        self,
        stream_loader,
        steps: int = 200
    ):
        """
        Pretrain encoder from stream batches.
        Ignores labels.
        """

        print(f"[SSL] pretraining for {steps} steps")

        for _ in range(steps):

            batch = stream_loader.next_batch()
            if batch is None:
                break

            X, y, info = batch  # y ignored

            stats = self.train_batch(X)

            if self.global_step % 20 == 0:
                print(
                    f"[SSL] step {self.global_step} "
                    f"loss={stats['ssl_loss']:.4f}"
                )

        print("[SSL] pretraining complete")

    # -------------------------------------------------
    # Encode Helper
    # -------------------------------------------------

    @torch.no_grad()
    def encode_numpy(self, X):
        """
        Encode numpy batch → numpy latent
        """
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        z = self.model.encode(X)

        return z.cpu().numpy()

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self):
        if not self.loss_history:
            return {}

        return {
            "steps": self.global_step,
            "final_loss": self.loss_history[-1],
            "avg_loss": sum(self.loss_history) / len(self.loss_history)
        }

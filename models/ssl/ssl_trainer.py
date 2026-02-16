"""
SEAI SSL Trainer — Final Version

Trains SSL autoencoder on unlabeled stream batches.
"""

from typing import Optional

import torch
from torch.optim import Adam

from config import DEVICE
from utils.seed import set_global_seed


class SSLTrainer:

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        seed: int = 42
    ):
        set_global_seed(seed)

        self.model = model.to(DEVICE)
        self.model.train()

        self.grad_clip = grad_clip

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.global_step = 0
        self.loss_history = []

    # -------------------------------------------------

    def train_batch(self, X):

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=DEVICE
        )

        # ✅ use model SSL loss (denoising etc.)
        loss = self.model.loss(X)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

        self.optimizer.step()

        self.global_step += 1

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        return {"ssl_loss": loss_val}

    # -------------------------------------------------

    def pretrain_stream(
        self,
        stream_loader,
        steps: int = 200
    ):

        print(f"[SSL] pretraining for {steps} steps")

        self.model.train()

        for _ in range(steps):

            batch = stream_loader.next_batch()
            if batch is None:
                break

            X, _, _ = batch

            stats = self.train_batch(X)

            if self.global_step % 20 == 0:
                print(
                    f"[SSL] step {self.global_step} "
                    f"loss={stats['ssl_loss']:.4f}"
                )

        print("[SSL] pretraining complete")

    # -------------------------------------------------

    @torch.no_grad()
    def encode_numpy(self, X):

        self.model.eval()

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=DEVICE
        )

        z = self.model.encode(X)

        self.model.train()

        return z.cpu().numpy()

    # -------------------------------------------------

    def summary(self):

        if not self.loss_history:
            return {}

        return {
            "steps": self.global_step,
            "final_loss": self.loss_history[-1],
            "avg_loss": sum(self.loss_history) / len(self.loss_history)
        }

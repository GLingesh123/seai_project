"""
SEAI Elastic Weight Consolidation (EWC)

Reduces catastrophic forgetting by penalizing
changes to important parameters.

Implements diagonal Fisher approximation.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from config import DEVICE


class EWC:
    """
    Elastic Weight Consolidation helper.
    """

    def __init__(self, model, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    # -------------------------------------------------
    # Snapshot Parameters
    # -------------------------------------------------

    def capture_prev_params(self):
        """
        Save current model parameters.
        """
        self.prev_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # -------------------------------------------------
    # Fisher Estimation
    # -------------------------------------------------

    def estimate_fisher(
        self,
        data_loader_fn,
        samples: int = 50
    ):
        """
        Estimate diagonal Fisher using model gradients.

        data_loader_fn → function that returns (X,y) batches
        """

        fisher = {
            n: torch.zeros_like(p, device=DEVICE)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()

        for _ in range(samples):

            batch = data_loader_fn()
            if batch is None:
                break

            X, y = batch

            X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            y = torch.tensor(y, dtype=torch.long, device=DEVICE)

            self.model.zero_grad()

            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach() ** 2

        # average
        for n in fisher:
            fisher[n] /= samples

        self.fisher = fisher
        self.model.train()

    # -------------------------------------------------
    # EWC Penalty
    # -------------------------------------------------

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        """

        if not self.prev_params or not self.fisher:
            return torch.tensor(0.0, device=DEVICE)

        loss = 0.0

        for n, p in self.model.named_parameters():

            if n not in self.prev_params:
                continue

            prev = self.prev_params[n]
            fisher = self.fisher[n]

            loss += (fisher * (p - prev) ** 2).sum()

        return self.lambda_ewc * loss

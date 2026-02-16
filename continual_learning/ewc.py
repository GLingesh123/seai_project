from typing import Dict, Callable, Optional
import torch
import torch.nn.functional as F

from config import DEVICE


class EWC:
    """
    Elastic Weight Consolidation — Stable Version

    Improvements:
    - stable Fisher averaging
    - gradient safety
    - fisher clipping
    - device-safe snapshots
    - trainer-compatible regularizer
    """

    def __init__(
        self,
        model,
        lambda_ewc: float = 5.0,     # 🔥 stronger by default
        fisher_clip: float = 10.0,   # prevent explosion
        fisher_floor: float = 1e-6   # prevent zeros
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_clip = fisher_clip
        self.fisher_floor = fisher_floor

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    # -------------------------------------------------
    # Snapshot parameters
    # -------------------------------------------------

    def capture_prev_params(self):

        self.prev_params = {
            n: p.detach().clone().to(DEVICE)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # -------------------------------------------------
    # Fisher estimation
    # -------------------------------------------------

    def estimate_fisher(
        self,
        data_fn: Callable[[], Optional[tuple]],
        samples: int = 100
    ):

        fisher = {
            n: torch.zeros_like(p, device=DEVICE)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        used = 0

        for _ in range(samples):

            batch = data_fn()
            if batch is None:
                break

            X, y = batch

            X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
            y = torch.as_tensor(y, dtype=torch.long, device=DEVICE)

            self.model.zero_grad(set_to_none=True)

            logits = self.model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is None or not p.requires_grad:
                    continue

                g = p.grad.detach()

                fisher[n] += g * g

            used += 1

        if used == 0:
            self.fisher = fisher
            self.model.train()
            return

        # ---- average ----
        for n in fisher:
            fisher[n] /= used

            # ---- stabilize ----
            fisher[n] = torch.clamp(
                fisher[n],
                min=self.fisher_floor,
                max=self.fisher_clip
            )

        self.fisher = fisher
        self.model.train()

    # -------------------------------------------------
    # EWC penalty
    # -------------------------------------------------

    def penalty(self, model=None):

        if not self.prev_params or not self.fisher:
            return torch.tensor(0.0, device=DEVICE)

        loss = torch.tensor(0.0, device=DEVICE)

        for n, p in self.model.named_parameters():

            if n not in self.prev_params:
                continue

            prev = self.prev_params[n]
            fisher = self.fisher[n]

            loss = loss + (fisher * (p - prev) ** 2).sum()

        return self.lambda_ewc * loss

    # -------------------------------------------------
    # Trainer hook alias
    # -------------------------------------------------

    def regularizer(self, model=None):
        return self.penalty(model)

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    def fisher_stats(self):

        if not self.fisher:
            return {}

        vals = torch.cat([
            v.flatten()
            for v in self.fisher.values()
        ])

        return {
            "mean": vals.mean().item(),
            "max": vals.max().item(),
            "min": vals.min().item()
        }

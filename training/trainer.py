from typing import Callable, List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import DEVICE, LR, WEIGHT_DECAY
from utils.seed import set_global_seed


class StreamTrainer:

    def __init__(
        self,
        model,
        ssl_module: Optional[object] = None,
        ssl_weight: float = 0.1,
        grad_clip: float = 5.0,
        seed: int = 42
    ):
        set_global_seed(seed)

        self.model = model.to(DEVICE)
        self.model.train()

        self.ssl_module = ssl_module
        self.ssl_weight = ssl_weight
        self.grad_clip = grad_clip

        self.base_lr = LR
        self.adapt_lr = LR * 0.5

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=WEIGHT_DECAY
        )

        self.global_step = 0
        self.adaptation_mode = False
        self.regularizers: List[Callable] = []

    # -------------------------------------------------

    def set_adaptation_mode(self, flag: bool):

        self.adaptation_mode = flag
        lr = self.adapt_lr if flag else self.base_lr

        for g in self.optimizer.param_groups:
            g["lr"] = lr

    # -------------------------------------------------

    def register_regularizer(self, fn: Callable):
        self.regularizers.append(fn)

    # -------------------------------------------------

    def train_batch(self, X, y, replay: bool = False) -> Dict[str, Any]:

        X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(y, dtype=torch.long, device=DEVICE)

        logits = self.model(X)

        ce_loss = F.cross_entropy(logits, y)

        # ----- continual regularizers -----
        reg_loss = torch.tensor(0.0, device=DEVICE)
        if reg_loss.item() > 0:
            print("EWC penalty:", reg_loss.item())

        for reg in self.regularizers:
            val = reg(self.model)
            if not torch.is_tensor(val):
                val = torch.tensor(val, device=DEVICE)
            reg_loss = reg_loss + val

        # ----- SSL auxiliary -----
        ssl_loss = torch.tensor(0.0, device=DEVICE)

        if self.ssl_module is not None:
            if hasattr(self.ssl_module, "loss"):
                ssl_raw = self.ssl_module.loss(X)
                ssl_loss = self.ssl_weight * ssl_raw
            else:
                raise ValueError(
                    "SSL module must implement loss(X)"
                )

        loss = ce_loss + reg_loss + ssl_loss

        # ----- backward -----
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

        self.optimizer.step()

        acc = self._accuracy_from_logits(logits, y)

        with torch.no_grad():
            per_sample = F.cross_entropy(
                logits, y, reduction="none"
            ).detach().cpu().numpy()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "reg_loss": reg_loss.item(),
            "ssl_loss": ssl_loss.item(),
            "accuracy": acc,
            "batch_size": int(X.shape[0]),
            "per_sample_loss": per_sample,
            "replay": replay,
            "adapt_mode": self.adaptation_mode,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    # -------------------------------------------------

    def train_replay_batch(self, X, y):
        return self.train_batch(X, y, replay=True)

    # -------------------------------------------------

    @torch.no_grad()
    def eval_batch(self, X, y) -> float:

        self.model.eval()

        X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(y, dtype=torch.long, device=DEVICE)

        logits = self.model(X)
        acc = self._accuracy_from_logits(logits, y)

        self.model.train()
        return acc

    # -------------------------------------------------

    @torch.no_grad()
    def extract_features(self, X):

        if not hasattr(self.model, "extract_features"):
            return None

        X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        z = self.model.extract_features(X)

        if torch.is_tensor(z):
            return z.detach().cpu().numpy()

        return z

    # -------------------------------------------------

    @staticmethod
    def _accuracy_from_logits(logits, y):
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()

    @staticmethod
    def error_signal(acc: float) -> float:
        return 1.0 - acc

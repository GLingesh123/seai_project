"""
SEAI Meta Initialization Utilities
Contains:
- MetaPretrainer
- MetaWarmStart
"""

import copy
import torch
import torch.nn.functional as F

from data.loaders.stream_loader import StreamLoader
from config import DEVICE


# =====================================================
# META PRETRAINER
# =====================================================

class MetaPretrainer:
    """
    Multi-scenario meta-style pretraining
    """

    SCENARIOS = [
        {"type": "none"},
        {"type": "sudden", "steps": [20]},
        {"type": "gradual", "start": 15, "end": 40},
        {"type": "recurring", "steps": [15, 35]},
    ]

    def __init__(self, model, steps_per_scenario=30):
        self.base_model = model
        self.steps_per_scenario = steps_per_scenario

    def run(self):

        print("[META] pretraining start")

        for scenario in self.SCENARIOS:

            stream = StreamLoader(scenario=scenario)
            task_model = copy.deepcopy(self.base_model)

            opt = torch.optim.Adam(task_model.parameters(), lr=1e-3)

            for _ in range(self.steps_per_scenario):

                batch = stream.next_batch()
                if batch is None:
                    break

                X, y, _ = batch

                X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
                y = torch.tensor(y, dtype=torch.long, device=DEVICE)

                logits = task_model(X)
                loss = F.cross_entropy(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

            self._merge(task_model)

        print("[META] pretraining done")

    def _merge(self, task_model, alpha=0.5):

        with torch.no_grad():
            for p_base, p_task in zip(
                self.base_model.parameters(),
                task_model.parameters()
            ):
                p_base.data = (
                    (1 - alpha) * p_base.data +
                    alpha * p_task.data
                )

    def get_model(self):
        return self.base_model


# =====================================================
# META WARM START (FAST INNER ADAPT)
# =====================================================

class MetaWarmStart:

    def __init__(self, inner_lr=1e-3, inner_steps=3):
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def adapt(self, model, X, y):

        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        for _ in range(self.inner_steps):

            logits = model(X)
            loss = F.cross_entropy(logits, y)

            grads = torch.autograd.grad(
                loss,
                model.parameters()
            )

            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    p -= self.inner_lr * g

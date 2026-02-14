"""
Baseline MLP Classifier for SEAI

Purpose:
- first working model in pipeline
- control baseline for ablation studies
- used before transformer / SSL variants
"""

import torch
import torch.nn as nn

from config import (
    INPUT_DIM,
    NUM_CLASSES,
    MLP_HIDDEN_DIM,
    DEVICE
)


class BaselineMLP(nn.Module):
    """
    Simple but strong baseline MLP.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = MLP_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, num_classes)
        )

        self.to(DEVICE)

    # ---------------------------------
    # Forward
    # ---------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ---------------------------------
    # Prediction Helpers
    # ---------------------------------

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    # ---------------------------------
    # Info
    # ---------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

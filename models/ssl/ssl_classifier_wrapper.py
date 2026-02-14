"""
SSL Encoder + Classifier Wrapper

Wraps a pretrained SSL encoder with a classification head.

Drop-in replacement for BaselineMLP in StreamTrainer.
"""

import torch
import torch.nn as nn

from config import DEVICE, NUM_CLASSES


class SSLClassifier(nn.Module):
    """
    encoder → classifier head
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = NUM_CLASSES,
        freeze_encoder: bool = True
    ):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, num_classes)
        )

        self.to(DEVICE)

    # ---------------------------------
    # Forward
    # ---------------------------------

    def forward(self, x: torch.Tensor):

        # encoder may be from autoencoder model
        if hasattr(self.encoder, "encode_train"):
            z = self.encoder.encode_train(x)
        else:
            z = self.encoder(x)

        logits = self.head(z)
        return logits

    # ---------------------------------
    # Prediction Helpers
    # ---------------------------------

    @torch.no_grad()
    def predict_logits(self, x):
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def predict_classes(self, x):
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    # ---------------------------------
    # Control
    # ---------------------------------

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    # ---------------------------------
    # Info
    # ---------------------------------

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

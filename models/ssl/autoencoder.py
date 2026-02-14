"""
SEAI Self-Supervised Autoencoder

Learns feature representations from unlabeled stream data.

Usage later:
X → encoder → latent z → classifier head
"""

import torch
import torch.nn as nn

from config import INPUT_DIM, DEVICE


class SSLAutoencoder(nn.Module):
    """
    Tabular autoencoder for SSL representation learning.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = 32,
        hidden_dim: int = 128
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim)
        )

        self.to(DEVICE)

    # ---------------------------------
    # Forward
    # ---------------------------------

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    # ---------------------------------
    # Encode Only (Important for later)
    # ---------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        self.eval()
        return self.encoder(x)

    # ---------------------------------
    # Encode With Grad (for joint train)
    # ---------------------------------

    def encode_train(self, x: torch.Tensor):
        return self.encoder(x)

    # ---------------------------------
    # Reconstruction Loss
    # ---------------------------------

    @staticmethod
    def reconstruction_loss(x, recon):
        return nn.functional.mse_loss(recon, x)

    # ---------------------------------
    # Info
    # ---------------------------------

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

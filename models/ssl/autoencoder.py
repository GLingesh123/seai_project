"""
SEAI SSL Autoencoder — Trainer Compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_DIM, DEVICE


class SSLAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        noise_std: float = 0.05
    ):
        super().__init__()

        self.noise_std = noise_std

        # -------- encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # -------- decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim)
        )

        self.to(DEVICE)

    # ---------------------------------

    def forward(self, x):

        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    # ---------------------------------
    # SSL Loss (Trainer expects this)
    # ---------------------------------

    def loss(self, x):

        if not torch.is_tensor(x):
            x = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=DEVICE
            )

        # denoising corruption
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise

        recon = self.forward(x_noisy)

        return F.mse_loss(recon, x)

    # ---------------------------------

    @torch.no_grad()
    def encode(self, x):

        self.eval()

        if not torch.is_tensor(x):
            x = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=DEVICE
            )

        return self.encoder(x)

    # ---------------------------------

    def encode_train(self, x):
        return self.encoder(x)

    # ---------------------------------

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

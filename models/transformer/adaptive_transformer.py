"""
SEAI Adaptive Tabular Transformer

Transformer-based classifier for tabular feature vectors.

Design:
features → token embeddings → transformer encoder → pooled → classifier
"""

import torch
import torch.nn as nn

from config import INPUT_DIM, NUM_CLASSES, DEVICE


class AdaptiveTransformer(nn.Module):
    """
    Lightweight tabular transformer.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # ---------------------------------
        # Feature → token embedding
        # each scalar feature → vector
        # ---------------------------------
        self.feature_embed = nn.Linear(1, d_model)

        # positional embeddings for feature index
        self.pos_embed = nn.Parameter(
            torch.randn(input_dim, d_model)
        )

        # ---------------------------------
        # Transformer encoder
        # ---------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # ---------------------------------
        # Classifier head
        # ---------------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.to(DEVICE)

    # ---------------------------------
    # Forward
    # ---------------------------------

    def forward(self, x: torch.Tensor):
        """
        x: [B, F]
        """

        B, F = x.shape

        # reshape to tokens
        x = x.unsqueeze(-1)                 # [B, F, 1]
        tok = self.feature_embed(x)         # [B, F, d_model]

        # add positional embedding
        tok = tok + self.pos_embed.unsqueeze(0)

        # transformer encode
        z = self.encoder(tok)               # [B, F, d_model]

        # pool across feature tokens
        pooled = z.mean(dim=1)              # [B, d_model]

        logits = self.head(pooled)
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
    # Info
    # ---------------------------------

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

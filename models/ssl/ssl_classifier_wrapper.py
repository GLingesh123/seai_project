"""
SEAI SSL Encoder + Classifier Wrapper — Final
"""

import torch
import torch.nn as nn

from config import DEVICE, NUM_CLASSES


class SSLClassifier(nn.Module):

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
        self._encoder_frozen = freeze_encoder

        if freeze_encoder:
            self.freeze_encoder()
            self.encoder.eval()

        # ---- classifier head ----
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),   # ✅ stream-safe
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        self.to(DEVICE)

    # ---------------------------------

    def forward(self, x: torch.Tensor):

        if self._encoder_frozen:
            with torch.no_grad():
                z = self._encode(x)
        else:
            z = self._encode(x)

        logits = self.head(z)
        return logits

    # ---------------------------------

    def _encode(self, x):

        if hasattr(self.encoder, "encode_train"):
            return self.encoder.encode_train(x)
        elif hasattr(self.encoder, "encode"):
            return self.encoder.encode(x)
        else:
            return self.encoder(x)

    # ---------------------------------
    # Feature extraction (SEAI required)
    # ---------------------------------

    def extract_features(self, x: torch.Tensor):
        return self._encode(x)

    # ---------------------------------
    # Prediction helpers
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
    # Encoder control
    # ---------------------------------

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        self.encoder.train()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()

    # ---------------------------------

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

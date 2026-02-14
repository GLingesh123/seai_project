import torch

from models.ssl.autoencoder import SSLAutoencoder
from models.ssl.ssl_classifier_wrapper import SSLClassifier
from config import INPUT_DIM, DEVICE

ae = SSLAutoencoder()
model = SSLClassifier(ae.encoder, latent_dim=32)

x = torch.randn(32, INPUT_DIM).to(DEVICE)

out = model(x)

print("logits:", out.shape)
print("trainable params:", model.num_parameters())

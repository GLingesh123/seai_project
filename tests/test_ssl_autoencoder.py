import torch
from models.ssl.autoencoder import SSLAutoencoder
from config import INPUT_DIM, DEVICE

model = SSLAutoencoder()

x = torch.randn(64, INPUT_DIM).to(DEVICE)

recon = model(x)
z = model.encode(x)

print("input:", x.shape)
print("recon:", recon.shape)
print("latent:", z.shape)
print("params:", model.num_parameters())

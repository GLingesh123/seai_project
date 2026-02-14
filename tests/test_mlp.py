import torch
from models.baseline.mlp import BaselineMLP
from config import INPUT_DIM, DEVICE

model = BaselineMLP()

x = torch.randn(32, INPUT_DIM).to(DEVICE)

out = model(x)

print("Output shape:", out.shape)
print("Params:", model.num_parameters())

import torch
from models.transformer.adaptive_transformer import AdaptiveTransformer
from config import INPUT_DIM, DEVICE

model = AdaptiveTransformer()

x = torch.randn(32, INPUT_DIM).to(DEVICE)

out = model(x)

print("logits:", out.shape)
print("params:", model.num_parameters())

import torch
from models.mlp import BaselineMLP
from continual_learning.ewc import EWC
import numpy as np

def test_ewc_penalty():
    model = BaselineMLP()
    ewc = EWC(model)
    
    # capture params
    ewc.capture_prev_params()
    
    # fake batch provider
    def fake_batch():
        X = np.random.randn(32, 20)
        y = np.random.randint(0, 2, 32)
        return X, y
    
    ewc.estimate_fisher(fake_batch, samples=2)
    
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p += 0.01 * torch.randn_like(p)
    
    pen = ewc.penalty()
    assert pen.detach().item() >= 0

import torch
from models.mlp import BaselineMLP
from config import INPUT_DIM, DEVICE


def test_mlp_instantiation():
    """Test that MLP model can be instantiated."""
    print("\n" + "="*70)
    print("[TEST] test_mlp_instantiation - Starting MLP instantiation test")
    print(f"[INFO] Device: {DEVICE}")
    
    model = BaselineMLP()
    print(f"[SUCCESS] MLP model instantiated successfully")
    print(f"[INFO] Model type: {type(model).__name__}")
    assert model is not None
    print("[PASS] test_mlp_instantiation completed\n")


def test_mlp_forward_pass():
    """Test MLP forward pass with batch input."""
    print("\n" + "="*70)
    print("[TEST] test_mlp_forward_pass - Testing forward pass")
    print(f"[INFO] Input dimension: {INPUT_DIM}, Batch size: 32")
    
    model = BaselineMLP()
    x = torch.randn(32, INPUT_DIM).to(DEVICE)
    print(f"[INFO] Input tensor shape: {x.shape}, Device: {x.device}")
    
    out = model(x)
    print(f"[INFO] Output tensor shape: {out.shape}, dtype: {out.dtype}")
    
    assert out.shape == (32, 2), f"Expected output shape (32, 2), got {out.shape}"
    assert out.dtype == torch.float32
    print("[PASS] test_mlp_forward_pass completed\n")


def test_mlp_parameters():
    """Test that MLP has trainable parameters."""
    print("\n" + "="*70)
    print("[TEST] test_mlp_parameters - Checking model parameters")
    
    model = BaselineMLP()
    params = model.num_parameters()
    print(f"[INFO] Total parameters: {params}")
    
    assert params > 0, "Model should have parameters"
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {trainable_params}")
    assert trainable_params > 0, "Model should have trainable parameters"
    print("[PASS] test_mlp_parameters completed\n")


def test_mlp_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n" + "="*70)
    print("[TEST] test_mlp_gradient_flow - Testing gradient computation")
    
    model = BaselineMLP()
    x = torch.randn(32, INPUT_DIM).to(DEVICE)
    target = torch.randint(0, 2, (32,)).to(DEVICE)
    print(f"[INFO] Input shape: {x.shape}, Target shape: {target.shape}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = loss_fn(out, target)
    print(f"[INFO] Computed loss: {loss.item():.6f}")
    
    loss.backward()
    print("[INFO] Backward pass completed, computing gradients")
    
    grad_count = 0
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            assert p.grad is not None, "Gradients should be computed"
            assert not torch.all(p.grad == 0), "Not all gradients should be zero"
            grad_count += 1
    
    print(f"[INFO] Verified gradients for {grad_count} parameter tensors")
    print("[PASS] test_mlp_gradient_flow completed\n")

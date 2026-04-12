from data.stream_loader import StreamLoader
from models.mlp import BaselineMLP
from training.trainer import StreamTrainer


def test_trainer_creation():
    """Test StreamTrainer instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_trainer_creation - Initializing StreamTrainer")
    
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    print(f"[INFO] StreamTrainer created successfully")
    print(f"[INFO] Trainer type: {type(trainer).__name__}")
    print(f"[INFO] Model: {type(model).__name__}")
    
    assert trainer is not None
    print("[PASS] test_trainer_creation completed\n")


def test_trainer_train_batch():
    """Test training a single batch."""
    print("\n" + "="*70)
    print("[TEST] test_trainer_train_batch - Training single batch")
    
    loader = StreamLoader(scenario={"type": "none"})
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    X, y, info = loader.next_batch()
    print(f"[INFO] Loaded batch - X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Batch step: {info['step']}, drift_mode: {info['drift_mode']}")
    
    stats = trainer.train_batch(X, y)
    
    print(f"[INFO] Training completed - Stats: {stats}")
    print(f"[INFO] Loss: {stats.get('loss', 'N/A')}")
    
    assert stats is not None, "Should return training stats"
    assert isinstance(stats, dict), "Stats should be a dictionary"
    print("[PASS] test_trainer_train_batch completed\n")


def test_trainer_multiple_batches():
    """Test training multiple batches."""
    print("\n" + "="*70)
    print("[TEST] test_trainer_multiple_batches - Training 5 batches")
    
    loader = StreamLoader(scenario={"type": "none"})
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    batch_count = 0
    for i in range(5):
        X, y, info = loader.next_batch()
        stats = trainer.train_batch(X, y)
        batch_count += 1
        
        print(f"[INFO] Batch {i+1}/5 - Loss: {stats.get('loss', 'N/A')}")
        assert "loss" in stats or "per_sample_loss" in stats, "Stats should contain loss"
    
    print(f"[INFO] Completed training {batch_count} batches")
    assert batch_count == 5, f"Expected 5 batches processed, got {batch_count}"
    print("[PASS] test_trainer_multiple_batches completed\n")


def test_trainer_stats_fields():
    """Test that training stats contain expected fields."""
    print("\n" + "="*70)
    print("[TEST] test_trainer_stats_fields - Validating stats fields")
    
    loader = StreamLoader(scenario={"type": "none"})
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    X, y, info = loader.next_batch()
    stats = trainer.train_batch(X, y)
    
    print(f"[INFO] Training stats keys: {list(stats.keys())}")
    
    # Check for common training statistics
    assert "loss" in stats or "per_sample_loss" in stats, "Stats should contain loss information"
    print("[PASS] test_trainer_stats_fields completed\n")


def test_trainer_gradient_updates():
    """Test that trainer updates model parameters."""
    print("\n" + "="*70)
    print("[TEST] test_trainer_gradient_updates - Checking parameter updates")
    
    loader = StreamLoader(scenario={"type": "none"})
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    
    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]
    print(f"[INFO] Stored {len(initial_params)} initial parameter tensors")
    
    X, y, info = loader.next_batch()
    stats = trainer.train_batch(X, y)
    
    # Check that parameters have changed
    params_changed = False
    changed_count = 0
    for i, (old_p, new_p) in enumerate(zip(initial_params, model.parameters())):
        if not (old_p == new_p).all():
            params_changed = True
            changed_count += 1
    
    print(f"[INFO] Parameter updates - {changed_count}/{len(initial_params)} tensors modified")
    assert params_changed, "Model parameters should be updated after training"
    print("[PASS] test_trainer_gradient_updates completed\n")

import numpy as np
from continual_learning.replay_buffer import ReplayBuffer


def test_replay_buffer_creation():
    """Test ReplayBuffer instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_replay_buffer_creation - Initializing replay buffer")
    
    buf = ReplayBuffer(capacity=200)
    print(f"[INFO] ReplayBuffer created with capacity: 200")
    print(f"[INFO] Current buffer size: {len(buf)}")
    
    assert len(buf) == 0, "New buffer should be empty"
    print("[PASS] test_replay_buffer_creation completed\n")


def test_replay_buffer_add_batch():
    """Test adding batch to replay buffer."""
    print("\n" + "="*70)
    print("[TEST] test_replay_buffer_add_batch - Adding batch to buffer")
    
    buf = ReplayBuffer(capacity=200)
    
    X = np.random.randn(128, 5)
    y = np.random.randint(0, 2, size=128)
    print(f"[INFO] Batch shape - X: {X.shape}, y: {y.shape}")
    
    buf.add_batch(X, y)
    print(f"[INFO] Batch added successfully")
    print(f"[INFO] Buffer now contains: {len(buf)} samples")
    
    assert len(buf) == 128, f"Buffer should have 128 samples, got {len(buf)}"
    print("[PASS] test_replay_buffer_add_batch completed\n")


def test_replay_buffer_class_distribution():
    """Test class distribution tracking."""
    print("\n" + "="*70)
    print("[TEST] test_replay_buffer_class_distribution - Analyzing class distribution")
    
    buf = ReplayBuffer(capacity=200)
    
    X = np.random.randn(128, 5)
    y = np.random.randint(0, 2, size=128)
    
    buf.add_batch(X, y)
    dist = buf.class_distribution()
    
    print(f"[INFO] Class distribution: {dist}")
    
    assert dist is not None, "Should return class distribution"
    assert isinstance(dist, dict), "Distribution should be a dictionary"
    print("[PASS] test_replay_buffer_class_distribution completed\n")


def test_replay_buffer_random_sample():
    """Test random sampling from buffer."""
    print("\n" + "="*70)
    print("[TEST] test_replay_buffer_random_sample - Testing random sampling")
    
    buf = ReplayBuffer(capacity=200)
    
    X = np.random.randn(128, 5)
    y = np.random.randint(0, 2, size=128)
    
    buf.add_batch(X, y)
    print(f"[INFO] Buffer populated with 128 samples")
    
    Xs, ys = buf.sample_random(32)
    print(f"[INFO] Random sample drawn - X shape: {Xs.shape}, y shape: {ys.shape}")
    
    assert Xs.shape == (32, 5), f"Expected sample shape (32, 5), got {Xs.shape}"
    assert ys.shape == (32,), f"Expected label shape (32,), got {ys.shape}"
    print("[PASS] test_replay_buffer_random_sample completed\n")


def test_replay_buffer_balanced_sample():
    """Test balanced sampling from buffer."""
    print("\n" + "="*70)
    print("[TEST] test_replay_buffer_balanced_sample - Testing balanced sampling")
    
    buf = ReplayBuffer(capacity=200)
    
    # Create balanced data
    X = np.random.randn(128, 5)
    y = np.array([0] * 64 + [1] * 64)
    print(f"[INFO] Created balanced dataset: 64 samples per class (total: 128)")
    
    buf.add_batch(X, y)
    
    Xs, ys = buf.sample_balanced(32)
    print(f"[INFO] Balanced sample drawn - X shape: {Xs.shape}, y shape: {ys.shape}")
    
    assert Xs.shape == (32, 5), f"Expected sample shape (32, 5), got {Xs.shape}"
    assert ys.shape == (32,), f"Expected label shape (32,), got {ys.shape}"
    print("[PASS] test_replay_buffer_balanced_sample completed\n")

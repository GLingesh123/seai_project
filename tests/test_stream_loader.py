from data.stream_loader import StreamLoader


def test_stream_loader_creation():
    """Test StreamLoader instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_stream_loader_creation - Initializing StreamLoader")
    
    scenario = {
        "type": "gradual",
        "start": 5,
        "end": 10
    }
    
    loader = StreamLoader(scenario=scenario)
    print(f"[INFO] StreamLoader created with scenario: {scenario}")
    
    assert loader is not None
    print("[PASS] test_stream_loader_creation completed\n")


def test_stream_loader_next_batch():
    """Test getting batches from stream loader."""
    print("\n" + "="*70)
    print("[TEST] test_stream_loader_next_batch - Fetching batch from stream")
    
    scenario = {
        "type": "gradual",
        "start": 5,
        "end": 10
    }
    
    loader = StreamLoader(scenario=scenario)
    
    batch = loader.next_batch()
    assert batch is not None, "Should return a batch"
    
    X, y, info = batch
    print(f"[INFO] Batch retrieved - X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Batch step: {info['step']}, drift_mode: {info['drift_mode']}")
    
    assert X is not None, "X should not be None"
    assert y is not None, "y should not be None"
    assert info is not None, "info should not be None"
    print("[PASS] test_stream_loader_next_batch completed\n")


def test_stream_loader_batch_info():
    """Test batch information fields."""
    print("\n" + "="*70)
    print("[TEST] test_stream_loader_batch_info - Verifying batch metadata")
    
    scenario = {
        "type": "gradual",
        "start": 5,
        "end": 10
    }
    
    loader = StreamLoader(scenario=scenario)
    
    batch = loader.next_batch()
    X, y, info = batch
    
    print(f"[INFO] Batch info keys: {list(info.keys())}")
    
    assert "step" in info, "info should contain 'step'"
    assert "drift_mode" in info, "info should contain 'drift_mode'"
    assert "gradual_progress" in info, "info should contain 'gradual_progress'"
    print(f"[INFO] step={info['step']}, drift_mode={info['drift_mode']}, progress={info['gradual_progress']}")
    print("[PASS] test_stream_loader_batch_info completed\n")


def test_stream_loader_multiple_batches():
    """Test getting multiple batches."""
    print("\n" + "="*70)
    print("[TEST] test_stream_loader_multiple_batches - Fetching 15 consecutive batches")
    
    scenario = {
        "type": "none"
    }
    
    loader = StreamLoader(scenario=scenario)
    
    batches = []
    for i in range(15):
        batch = loader.next_batch()
        if batch is not None:
            batches.append(batch)
            print(f"[INFO] Batch {i+1} retrieved - Size: {batch[0].shape[0]} samples")
    
    print(f"[INFO] Total batches retrieved: {len(batches)}")
    assert len(batches) == 15, f"Expected 15 batches, got {len(batches)}"
    print("[PASS] test_stream_loader_multiple_batches completed\n")


def test_stream_loader_batch_shapes():
    """Test that batches have correct shapes."""
    print("\n" + "="*70)
    print("[TEST] test_stream_loader_batch_shapes - Validating batch dimensions")
    
    scenario = {
        "type": "gradual",
        "start": 5,
        "end": 10
    }
    
    loader = StreamLoader(scenario=scenario)
    
    X, y, info = loader.next_batch()
    
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Samples in batch: {X.shape[0]}, Features: {X.shape[1]}")
    
    assert X.shape[0] > 0, "X should have samples"
    assert y.shape[0] > 0, "y should have labels"
    assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
    print("[PASS] test_stream_loader_batch_shapes completed\n")

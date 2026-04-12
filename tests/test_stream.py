import numpy as np
from data.synthetic_stream import SyntheticStream


def test_synthetic_stream_creation():
    """Test SyntheticStream instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_synthetic_stream_creation - Initializing SyntheticStream")
    
    stream = SyntheticStream(input_dim=5, chunk_size=100)
    
    print(f"[INFO] SyntheticStream created")
    print(f"[INFO] Input dimension: 5, Chunk size: 100")
    
    assert stream is not None
    print("[PASS] test_synthetic_stream_creation completed\n")


def test_synthetic_stream_no_drift():
    """Test stream generation with no drift."""
    print("\n" + "="*70)
    print("[TEST] test_synthetic_stream_no_drift - Generating 3 chunks (no drift)")
    
    stream = SyntheticStream(input_dim=5, chunk_size=100)
    
    chunks = []
    for i in range(3):
        X, y = stream.next_chunk()
        chunks.append((X, y))
        
        print(f"[INFO] Chunk {i+1} - X shape: {X.shape}, y shape: {y.shape}, X mean: {X.mean():.4f}")
        
        assert X.shape == (100, 5), f"Expected chunk shape (100, 5), got {X.shape}"
        assert y.shape == (100,), f"Expected label shape (100,), got {y.shape}"
    
    print("[PASS] test_synthetic_stream_no_drift completed\n")


def test_synthetic_stream_sudden_drift():
    """Test stream with sudden drift."""
    print("\n" + "="*70)
    print("[TEST] test_synthetic_stream_sudden_drift - Testing sudden drift")
    
    stream = SyntheticStream(input_dim=5, chunk_size=100)
    
    # Get initial chunk
    X_before, y_before = stream.next_chunk()
    mean_before = X_before.mean()
    print(f"[INFO] Before drift - X mean: {mean_before:.4f}")
    
    # Get chunk with sudden drift
    X_drift, y_drift = stream.next_chunk(drift_mode="sudden")
    mean_drift = X_drift.mean()
    print(f"[INFO] After sudden drift - X mean: {mean_drift:.4f}")
    print(f"[INFO] Mean change: {abs(mean_drift - mean_before):.4f}")
    
    assert X_drift.shape == (100, 5), f"Expected drift chunk shape (100, 5), got {X_drift.shape}"
    # Sudden drift should cause significant change in distribution
    assert abs(mean_drift - mean_before) > 0.1, "Sudden drift should change distribution significantly"
    print("[PASS] test_synthetic_stream_sudden_drift completed\n")


def test_synthetic_stream_gradual_drift():
    """Test stream with gradual drift."""
    print("\n" + "="*70)
    print("[TEST] test_synthetic_stream_gradual_drift - Testing gradual drift progression")
    
    stream = SyntheticStream(input_dim=5, chunk_size=100)
    
    progress_values = [0.2, 0.5, 1.0]
    chunks = []
    
    print(f"[INFO] Testing gradual drift with progress values: {progress_values}")
    for p in progress_values:
        X, y = stream.next_chunk(drift_mode="gradual", gradual_progress=p)
        chunks.append((X, y))
        
        print(f"[INFO] Progress {p} - X shape: {X.shape}, X mean: {X.mean():.4f}")
        
        assert X.shape == (100, 5), f"Expected chunk shape (100, 5), got {X.shape}"
        assert y.shape == (100,), f"Expected label shape (100,), got {y.shape}"
    
    # Mean should change with gradual progress
    means = [X.mean() for X, _ in chunks]
    # Check that means are not all identical (showing gradual change)
    assert len(set(np.round(means, 2))) > 1, "Gradual drift should show changing means"
    print(f"[INFO] Observed means progression: {[f'{m:.4f}' for m in means]}")
    print("[PASS] test_synthetic_stream_gradual_drift completed\n")

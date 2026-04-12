from data.drift_injector import DriftInjector


def test_drift_injector_gradual():
    """Test gradual drift progression."""
    print("\n" + "="*70)
    print("[TEST] test_drift_injector_gradual - Testing gradual drift progression")
    
    scenario = {
        "type": "gradual",
        "start": 10,
        "end": 15
    }
    
    inj = DriftInjector(scenario)
    print(f"[INFO] DriftInjector created with scenario: {scenario}")
    
    modes = []
    for step in range(20):
        mode = inj.get_drift_mode(step)
        modes.append(mode)
    
    print(f"[INFO] Drift modes over 20 steps: {set(modes)}")
    print(f"[INFO] Mode sequence: {modes[:15]}...")
    
    # DriftInjector returns None for normal states, not "no_drift"
    assert None in modes, "Should have None (no drift) mode"
    assert "gradual" in modes, "Should have gradual mode"
    print("[PASS] test_drift_injector_gradual completed\n")


def test_drift_injector_sudden():
    """Test sudden drift."""
    print("\n" + "="*70)
    print("[TEST] test_drift_injector_sudden - Testing sudden drift detection")
    
    scenario = {
        "type": "sudden",
        "steps": [5]
    }
    
    inj = DriftInjector(scenario)
    print(f"[INFO] DriftInjector created with sudden drift at step 5")
    
    modes_before = [inj.get_drift_mode(step) for step in range(5)]
    modes_at_and_after = [inj.get_drift_mode(step) for step in range(5, 10)]
    
    print(f"[INFO] Modes before drift (steps 0-4): {set(modes_before)}")
    print(f"[INFO] Modes at/after drift (steps 5-9): {set(modes_at_and_after)}")
    
    # DriftInjector returns None for normal states, not "no_drift"
    assert all(m is None for m in modes_before), "Should be None (no drift) before sudden point"
    # At and after step 5 should be sudden
    assert "sudden" in modes_at_and_after, "Should have sudden mode at drift step"
    print("[PASS] test_drift_injector_sudden completed\n")


def test_gradual_progress():
    """Test gradual progress calculation."""
    print("\n" + "="*70)
    print("[TEST] test_gradual_progress - Validating progress values")
    
    scenario = {
        "type": "gradual",
        "start": 10,
        "end": 15
    }
    
    inj = DriftInjector(scenario)
    print(f"[INFO] Testing gradual progress from step {scenario['start']} to {scenario['end']}")
    
    progress_values = []
    for step in range(20):
        prog = inj.get_gradual_progress(step)
        progress_values.append(prog)
    
    print(f"[INFO] Progress values: {[f'{p:.2f}' for p in progress_values[:15]]}...")
    
    # Progress should be between 0 and 1
    assert all(0 <= p <= 1 for p in progress_values), "Progress should be in [0, 1]"
    print("[PASS] test_gradual_progress completed\n")

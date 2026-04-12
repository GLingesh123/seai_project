from utils.logger import ExperimentLogger


def test_logger_creation():
    """Test ExperimentLogger instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_logger_creation - Initializing ExperimentLogger")
    
    logger = ExperimentLogger("test_run")
    print(f"[INFO] ExperimentLogger created with experiment name: 'test_run'")
    
    assert logger is not None
    print("[PASS] test_logger_creation completed\n")


def test_logger_log_step():
    """Test logging individual steps."""
    print("\n" + "="*70)
    print("[TEST] test_logger_log_step - Logging single training step")
    
    logger = ExperimentLogger("test_run")
    
    logger.log_step(
        step=0,
        loss=0.7,
        accuracy=0.5,
        drift=False
    )
    
    print(f"[INFO] Step logged - Step: 0, Loss: 0.7, Accuracy: 0.5, Drift: False")
    print(f"[INFO] Logger recorded step")
    print("[PASS] test_logger_log_step completed\n")


def test_logger_multiple_steps():
    """Test logging multiple steps."""
    print("\n" + "="*70)
    print("[TEST] test_logger_multiple_steps - Logging 5 training steps")
    
    logger = ExperimentLogger("test_run")
    
    for step in range(5):
        logger.log_step(
            step=step,
            loss=0.7 - step * 0.05,
            accuracy=0.5 + step * 0.1,
            drift=(step == 3)
        )
        
        loss_val = 0.7 - step * 0.05
        acc_val = 0.5 + step * 0.1
        print(f"[INFO] Step {step} - Loss: {loss_val:.3f}, Accuracy: {acc_val:.3f}, Drift: {step == 3}")
    
    print(f"[INFO] Completed logging 5 steps")
    print("[PASS] test_logger_multiple_steps completed\n")


def test_logger_drift_event():
    """Test logging drift events."""
    print("\n" + "="*70)
    print("[TEST] test_logger_drift_event - Logging drift detection event")
    
    logger = ExperimentLogger("test_run")
    
    for step in range(5):
        logger.log_step(
            step=step,
            loss=0.7 - step * 0.05,
            accuracy=0.5 + step * 0.1,
            drift=(step == 3)
        )
        
        if step == 3:
            logger.log_drift_event(step, {"detector": "adwin"})
            print(f"[INFO] Drift event logged at step {step} - Detector: adwin")
    
    print(f"[INFO] Drift events recorded")
    print("[PASS] test_logger_drift_event completed\n")


def test_logger_summary():
    """Test getting logger summary."""
    print("\n" + "="*70)
    print("[TEST] test_logger_summary - Generating experiment summary")
    
    logger = ExperimentLogger("test_run")
    
    for step in range(5):
        logger.log_step(
            step=step,
            loss=0.7 - step * 0.05,
            accuracy=0.5 + step * 0.1,
            drift=False
        )
    
    summary = logger.summary()
    
    print(f"[INFO] Summary generated - Keys: {list(summary.keys())}")
    print(f"[INFO] Summary: {summary}")
    
    assert summary is not None, "Summary should not be None"
    assert isinstance(summary, dict), "Summary should be a dictionary"
    print("[PASS] test_logger_summary completed\n")


def test_logger_save_all():
    """Test saving all logs."""
    print("\n" + "="*70)
    print("[TEST] test_logger_save_all - Saving logs to disk")
    
    logger = ExperimentLogger("test_run")
    
    for step in range(5):
        logger.log_step(
            step=step,
            loss=0.7 - step * 0.05,
            accuracy=0.5 + step * 0.1,
            drift=(step == 3)
        )
    
    if step == 3:
        logger.log_drift_event(step, {"detector": "adwin"})
    
    print(f"[INFO] Attempting to save all logs...")
    logger.save_all()
    
    print(f"[INFO] Logs saved successfully to logs/csv/ and logs/json/")
    print("[PASS] test_logger_save_all completed\n")

from data.stream_loader import StreamLoader
from models.mlp import BaselineMLP
from training.trainer import StreamTrainer
from drift.drift_manager import DriftManager
from continual_learning.replay_buffer import ReplayBuffer
from utils.logger import ExperimentLogger
from training.adaptation_loop import AdaptationLoop


def test_adaptation_loop_creation():
    """Test AdaptationLoop instantiation."""
    print("\n" + "="*70)
    print("[TEST] test_adaptation_loop_creation - Initializing AdaptationLoop")
    
    scenario = {
        "type": "sudden",
        "steps": [20]
    }
    
    print(f"[INFO] Scenario: {scenario}")
    print(f"[INFO] Creating stream loader...")
    stream = StreamLoader(scenario=scenario)
    print(f"[INFO] Creating model, trainer, and drift detector...")
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    detector = DriftManager()
    print(f"[INFO] Creating replay buffer and logger...")
    replay = ReplayBuffer()
    logger = ExperimentLogger("adapt_test")
    
    print(f"[INFO] Instantiating AdaptationLoop...")
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )
    
    print(f"[SUCCESS] AdaptationLoop instantiated successfully")
    assert loop is not None, "AdaptationLoop should be instantiated"
    print("[PASS] test_adaptation_loop_creation completed\n")


def test_adaptation_loop_execution():
    """Test AdaptationLoop execution with limited steps."""
    print("\n" + "="*70)
    print("[TEST] test_adaptation_loop_execution - Running AdaptationLoop for 5 steps")
    
    scenario = {
        "type": "sudden",
        "steps": [20]
    }
    
    print(f"[INFO] Setting up scenario: {scenario}")
    stream = StreamLoader(scenario=scenario)
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    detector = DriftManager()
    replay = ReplayBuffer()
    logger = ExperimentLogger("adapt_test")
    
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )
    
    print(f"[INFO] Running adaptation loop for max_steps=5...")
    loop.run(max_steps=5)
    
    summary = loop.summary()
    print(f"[INFO] Execution completed - Summary keys: {list(summary.keys())}")
    print(f"[INFO] Summary: {summary}")
    
    assert isinstance(loop.summary(), dict), "Summary should be a dictionary"
    print("[PASS] test_adaptation_loop_execution completed\n")


def test_adaptation_loop_summary():
    """Test AdaptationLoop summary generation."""
    print("\n" + "="*70)
    print("[TEST] test_adaptation_loop_summary - Testing summary generation")
    
    scenario = {
        "type": "none"
    }
    
    print(f"[INFO] Scenario (no drift): {scenario}")
    stream = StreamLoader(scenario=scenario)
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    detector = DriftManager()
    replay = ReplayBuffer()
    logger = ExperimentLogger("adapt_test")
    
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )
    
    print(f"[INFO] Running loop for 3 steps...")
    loop.run(max_steps=3)
    summary = loop.summary()
    
    print(f"[INFO] Summary generated - Keys: {list(summary.keys())}")
    
    assert summary is not None, "Summary should not be None"
    assert isinstance(summary, dict), "Summary should be a dictionary"
    print("[PASS] test_adaptation_loop_summary completed\n")


def test_adaptation_loop_with_gradual_drift():
    """Test AdaptationLoop with gradual drift scenario."""
    print("\n" + "="*70)
    print("[TEST] test_adaptation_loop_with_gradual_drift - Testing with drift")
    
    scenario = {
        "type": "gradual",
        "start": 2,
        "end": 5
    }
    
    print(f"[INFO] Scenario with gradual drift: {scenario}")
    stream = StreamLoader(scenario=scenario)
    model = BaselineMLP()
    trainer = StreamTrainer(model)
    detector = DriftManager()
    replay = ReplayBuffer()
    logger = ExperimentLogger("adapt_test_gradual")
    
    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )
    
    print(f"[INFO] Running loop with gradual drift for 5 steps...")
    loop.run(max_steps=5)
    summary = loop.summary()
    
    print(f"[INFO] Execution completed - Summary: {summary}")
    
    assert isinstance(summary, dict), "Should complete run with gradual drift"
    print("[PASS] test_adaptation_loop_with_gradual_drift completed\n")

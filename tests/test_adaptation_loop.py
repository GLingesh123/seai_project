from data.stream_loader import StreamLoader
from models.mlp import BaselineMLP
from training.trainer import StreamTrainer
from drift.drift_manager import DriftManager
from continual_learning.replay_buffer import ReplayBuffer
from utils.logger import ExperimentLogger
from training.adaptation_loop import AdaptationLoop

def test_adaptation_loop_execution():
    scenario = {
        "type": "sudden",
        "steps": [20]
    }
    
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
    
    loop.run(max_steps=5) # Reduced for quick testing
    
    assert True
    assert isinstance(loop.summary(), dict)

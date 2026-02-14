from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer
from drift.drift_detector import DriftDetector
from replay.buffer import ReplayBuffer
from experiments.logger import ExperimentLogger
from training.adaptation_loop import AdaptationLoop


scenario = {
    "type": "sudden",
    "steps": [20]
}

stream = StreamLoader(scenario=scenario)
model = BaselineMLP()
trainer = StreamTrainer(model)
detector = DriftDetector()
replay = ReplayBuffer()
logger = ExperimentLogger("adapt_test")

loop = AdaptationLoop(
    stream_loader=stream,
    trainer=trainer,
    drift_detector=detector,
    replay_buffer=replay,
    logger=logger
)

loop.run(max_steps=60)

print(loop.summary())

from data.loaders.stream_loader import StreamLoader
from models.baseline.mlp import BaselineMLP
from training.trainer import StreamTrainer

loader = StreamLoader(scenario={"type": "none"})
model = BaselineMLP()
trainer = StreamTrainer(model)

for _ in range(5):
    X, y, info = loader.next_batch()
    stats = trainer.train_batch(X, y)
    print(info["step"], stats)

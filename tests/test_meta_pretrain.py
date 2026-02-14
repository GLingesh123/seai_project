from models.baseline.mlp import BaselineMLP
from models.meta.meta_pretrain import MetaPretrainer

model = BaselineMLP()

meta = MetaPretrainer(
    model,
    steps_per_scenario=10
)

meta.run()

print("Meta pretraining finished.")

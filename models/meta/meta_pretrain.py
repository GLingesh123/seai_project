"""
SEAI Meta Pretraining

Meta-style pretraining across multiple drift scenarios
to learn robust initial weights.

Not full MAML — practical multi-distribution pretraining.
"""

import copy
from typing import List, Dict

import torch

from data.loaders.stream_loader import StreamLoader
from training.trainer import StreamTrainer
from utils.seed import set_global_seed


# -------------------------------------------------
# Default Meta Scenarios
# -------------------------------------------------

META_SCENARIOS: List[Dict] = [
    {"type": "none"},
    {"type": "sudden", "steps": [20]},
    {"type": "gradual", "start": 15, "end": 40},
    {"type": "recurring", "steps": [15, 35, 60]},
]


# -------------------------------------------------
# Meta Pretrainer
# -------------------------------------------------

class MetaPretrainer:
    """
    Multi-scenario pretraining for robust initialization.
    """

    def __init__(
        self,
        model,
        scenarios: List[Dict] = META_SCENARIOS,
        steps_per_scenario: int = 40,
        seed: int = 42
    ):
        set_global_seed(seed)

        self.base_model = model
        self.scenarios = scenarios
        self.steps_per_scenario = steps_per_scenario

    # -------------------------------------------------
    # Run Meta Pretraining
    # -------------------------------------------------

    def run(self):

        print("\n[META] pretraining start")

        for i, scenario in enumerate(self.scenarios):

            print(f"\n[META] scenario {i+1}/{len(self.scenarios)} → {scenario}")

            # fresh stream per scenario
            stream = StreamLoader(scenario=scenario)

            # clone model weights for this task
            task_model = copy.deepcopy(self.base_model)

            trainer = StreamTrainer(task_model)

            step = 0

            while step < self.steps_per_scenario:

                batch = stream.next_batch()
                if batch is None:
                    break

                X, y, info = batch
                trainer.train_batch(X, y)

                step += 1

            # ---- merge task weights back (meta update) ----
            self._merge_weights(task_model)

        print("\n[META] pretraining complete")

    # -------------------------------------------------
    # Weight Merge (Simple Meta Update)
    # -------------------------------------------------

    def _merge_weights(self, task_model, alpha: float = 0.5):
        """
        Move base weights toward task-trained weights.
        """

        with torch.no_grad():
            for p_base, p_task in zip(
                self.base_model.parameters(),
                task_model.parameters()
            ):
                p_base.data = (
                    (1 - alpha) * p_base.data +
                    alpha * p_task.data
                )

    # -------------------------------------------------
    # Get Meta-Initialized Model
    # -------------------------------------------------

    def get_model(self):
        return self.base_model

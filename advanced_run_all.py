"""
SEAI Advanced Batch Runner — FIXED

Runs and compares:

- baseline
- SSL
- transformer
- meta-init
- EWC

Fixes:
- early drift injection
- longer runs
- safe EWC integration
- guaranteed drift metrics
"""

import glob
import pandas as pd

from data.loaders.stream_loader import StreamLoader
from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop
from drift.drift_detector import DriftDetector
from replay.buffer import ReplayBuffer
from experiments.logger import ExperimentLogger

from models.baseline.mlp import BaselineMLP
from models.ssl.autoencoder import SSLAutoencoder
from models.ssl.ssl_trainer import SSLTrainer
from models.ssl.ssl_classifier_wrapper import SSLClassifier
from models.transformer.adaptive_transformer import AdaptiveTransformer
from models.meta.meta_pretrain import MetaPretrainer

from continual_learning.ewc import EWC

from evaluation.compare_runs import compare_csv_runs
from visualization.plot_accuracy import compare_runs


# -------------------------------------------------
# Variant Runner
# -------------------------------------------------

def run_variant(name, model_builder, scenario, steps=220, use_ewc=False):

    print("\n" + "=" * 60)
    print("RUN:", name)
    print("=" * 60)

    stream = StreamLoader(scenario=scenario)
    model = model_builder()

    trainer = StreamTrainer(model)
    detector = DriftDetector()
    replay = ReplayBuffer()
    logger = ExperimentLogger(name)

    # -------- safe EWC integration --------
    if use_ewc:
        ewc = EWC(model)
        ewc.capture_prev_params()

        def batch_fn():
            b = stream.next_batch()
            if b is None:
                return None
            X, y, _ = b
            return X, y

        ewc.estimate_fisher(batch_fn, samples=40)

        orig_train = trainer.train_batch

        def train_with_ewc(X, y):
            stats = orig_train(X, y)
            pen = ewc.penalty()
            trainer.optimizer.zero_grad()
            pen.backward()
            trainer.optimizer.step()
            return stats

        trainer.train_batch = train_with_ewc

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        logger=logger
    )

    loop.run(max_steps=steps)

    files = sorted(glob.glob(f"results/csv/{name}_*.csv"))
    return files[-1]


# -------------------------------------------------
# Model Builders
# -------------------------------------------------

def build_baseline():
    return BaselineMLP()


def build_ssl_model():

    ae = SSLAutoencoder()
    ssl_trainer = SSLTrainer(ae)

    ssl_trainer.pretrain_stream(
        StreamLoader({"type": "none"}),
        steps=100
    )

    return SSLClassifier(ae.encoder, freeze_encoder=True)


def build_transformer():
    return AdaptiveTransformer()


def build_meta_model():

    model = BaselineMLP()
    meta = MetaPretrainer(model, steps_per_scenario=25)
    meta.run()
    return meta.get_model()


# -------------------------------------------------
# Main Batch — EARLY DRIFT FIX
# -------------------------------------------------

if __name__ == "__main__":

    # 🔥 EARLY drift so detector always sees it
    scenario = {"type": "gradual", "start": 10, "end": 40}

    csv_map = {}

    csv_map["adv_baseline"] = run_variant(
        "adv_baseline",
        build_baseline,
        scenario
    )

    csv_map["adv_ssl"] = run_variant(
        "adv_ssl",
        build_ssl_model,
        scenario
    )

    csv_map["adv_transformer"] = run_variant(
        "adv_transformer",
        build_transformer,
        scenario
    )

    csv_map["adv_meta"] = run_variant(
        "adv_meta",
        build_meta_model,
        scenario
    )

    csv_map["adv_ewc"] = run_variant(
        "adv_ewc",
        build_baseline,
        scenario,
        use_ewc=True
    )

    # -------- comparison table --------
    compare_csv_runs(csv_map, "advanced_comparison")

    # -------- comparison plot --------
    compare_runs(
        list(csv_map.values()),
        labels=list(csv_map.keys()),
        rolling_window=15,
        save_name="advanced_comparison_plot"
    )

    print("\n✅ Advanced comparison complete (with drift).")

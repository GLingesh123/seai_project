"""
SEAI Advanced Batch Runner — CLEAN + CORRECT

Compares:

- baseline
- SSL encoder model
- transformer
- meta-init
- EWC
- full SEAI

Aligned with:
drift + replay + EWC + meta warm start
"""

import glob
import pandas as pd

from data.loaders.stream_loader import StreamLoader
from training.trainer import StreamTrainer
from training.adaptation_loop import AdaptationLoop
from drift.drift_manager import DriftManager
from replay.buffer import ReplayBuffer
from experiments.logger import ExperimentLogger

from models.baseline.mlp import BaselineMLP
from models.ssl.autoencoder import SSLAutoencoder
from models.ssl.ssl_trainer import SSLTrainer
from models.ssl.ssl_classifier_wrapper import SSLClassifier

from models.transformer.adaptive_transformer import AdaptiveTransformer
from models.meta.meta_init import MetaPretrainer
from models.meta.meta_init import MetaWarmStart

from continual_learning.ewc import EWC

from evaluation.compare_runs import compare_csv_runs
from visualization.plot_accuracy import compare_runs


# =====================================================
# Variant Runner
# =====================================================

def run_variant(
    name,
    model_builder,
    scenario,
    steps=250,
    use_replay=True,
    use_ewc=False,
    use_meta=False
):

    print("\n" + "=" * 60)
    print("RUN:", name)
    print("=" * 60)

    stream = StreamLoader(scenario=scenario)

    model = model_builder()
    trainer = StreamTrainer(model)

    detector = DriftManager(min_votes=2)

    replay = ReplayBuffer() if use_replay else None
    continual = EWC(model) if use_ewc else None
    meta = MetaWarmStart() if use_meta else None

    logger = ExperimentLogger(name)

    loop = AdaptationLoop(
        stream_loader=stream,
        trainer=trainer,
        drift_detector=detector,
        replay_buffer=replay,
        continual_module=continual,
        meta_module=meta,
        logger=logger
    )

    loop.run(max_steps=steps)

    files = sorted(glob.glob(f"results/csv/{name}_*.csv"))
    return files[-1]


# =====================================================
# Model Builders
# =====================================================

def build_baseline():
    return BaselineMLP()


def build_ssl_model():

    ae = SSLAutoencoder()
    ssl_trainer = SSLTrainer(ae)

    print("[SSL] pretraining encoder...")
    ssl_trainer.pretrain_stream(
        StreamLoader({"type": "none"}),
        steps=120
    )

    return SSLClassifier(ae.encoder, freeze_encoder=True)


def build_transformer():
    return AdaptiveTransformer()


def build_meta_model():

    base = BaselineMLP()

    print("[META] pretraining...")
    meta = MetaPretrainer(base, steps_per_scenario=30)
    meta.run()

    return meta.get_model()


# =====================================================
# Main Batch Run
# =====================================================

if __name__ == "__main__":

    # early drift so detectors always trigger
    scenario = {"type": "gradual", "start": 15, "end": 60}

    csv_map = {}

    # ---------------- baseline ----------------
    csv_map["baseline"] = run_variant(
        "baseline",
        build_baseline,
        scenario,
        use_replay=False
    )

    # ---------------- SSL ----------------
    csv_map["ssl"] = run_variant(
        "ssl",
        build_ssl_model,
        scenario
    )

    # ---------------- transformer ----------------
    csv_map["transformer"] = run_variant(
        "transformer",
        build_transformer,
        scenario
    )

    # ---------------- meta init ----------------
    csv_map["meta_init"] = run_variant(
        "meta_init",
        build_meta_model,
        scenario
    )

    # ---------------- EWC only ----------------
    csv_map["ewc"] = run_variant(
        "ewc",
        build_baseline,
        scenario,
        use_ewc=True
    )

    # ---------------- FULL SEAI ----------------
    csv_map["seai"] = run_variant(
        "seai",
        build_baseline,
        scenario,
        use_replay=True,
        use_ewc=True,
        use_meta=True
    )

    # =====================================================
    # Comparison Table
    # =====================================================

    compare_csv_runs(csv_map, "advanced_comparison")

    # =====================================================
    # Comparison Plot
    # =====================================================

    compare_runs(
        list(csv_map.values()),
        labels=list(csv_map.keys()),
        rolling_window=15,
        save_name="advanced_comparison_plot"
    )

    print("\n✅ Advanced SEAI comparison complete.")

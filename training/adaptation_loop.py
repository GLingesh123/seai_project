from typing import Optional
import time
import numpy as np

from config import (
    REPLAY_BATCH_SIZE,
    REPLAY_AFTER_DRIFT_STEPS,
    EWC_FISHER_SAMPLES,
    EWC_FISHER_BATCH
)


class AdaptationLoop:
    """
    SEAI Online Adaptation Controller

    Pipeline:
    stream → train → drift detect →
    EWC snapshot → replay adapt → meta warm start
    """

    # -------------------------------------------------

    def __init__(
        self,
        stream_loader,
        trainer,
        drift_detector,
        replay_buffer=None,
        continual_module=None,   # EWC
        meta_module=None,        # meta warm start
        logger=None,
        replay_steps: int = REPLAY_AFTER_DRIFT_STEPS
    ):

        self.stream = stream_loader
        self.trainer = trainer
        self.detector = drift_detector
        self.replay = replay_buffer
        self.continual = continual_module
        self.meta = meta_module
        self.logger = logger

        self.replay_steps = replay_steps

        self.total_steps = 0
        self.drift_count = 0
        self.last_drift_time = None

        # register EWC penalty into trainer
        if self.continual and hasattr(self.continual, "penalty"):
            self.trainer.register_regularizer(
                self.continual.penalty
            )

    # -------------------------------------------------

    def run(self, max_steps: Optional[int] = None):

        while True:

            batch = self.stream.next_batch()
            if batch is None:
                break

            X, y, info = batch

            stats = self.trainer.train_batch(X, y)

            acc = stats["accuracy"]
            loss = stats["loss"]

            err_signal = 1.0 - acc

            # ---------- feature-aware drift ----------
            features = None
            if hasattr(self.trainer, "extract_features"):
                features = self.trainer.extract_features(X)

            try:
                drift_out = self.detector.update(
                    error_value=err_signal,
                    features=features
                )
            except TypeError:
                drift_out = self.detector.update(err_signal)

            drift_flag = drift_out["drift"]

            # ---------- replay store ----------
            if self.replay:
                try:
                    self.replay.add_batch(
                        X, y,
                        loss=stats.get("per_sample_loss"),
                        step=self.total_steps,
                        drift=drift_flag
                    )
                except TypeError:
                    # fallback for simple buffer
                    self.replay.add_batch(X, y)

            # ---------- logging ----------
            if self.logger:
                self.logger.log_step(
                    step=self.total_steps,
                    loss=loss,
                    accuracy=acc,
                    drift=drift_flag
                )

            # ---------- drift reaction ----------
            if drift_flag:
                self._handle_drift(drift_out)

            self.total_steps += 1

            if max_steps and self.total_steps >= max_steps:
                break

        if self.logger:
            self.logger.save_all()

    # -------------------------------------------------

    def _handle_drift(self, drift_info):

        self.drift_count += 1
        self.last_drift_time = time.time()

        severity = drift_info.get("severity", 1.0)

        print(
            f"[DRIFT] step={self.total_steps} "
            f"severity={severity:.3f}"
        )

        # mark event as adapted
        event = drift_info.get("event")
        if event:
            event.mark_adaptation("SEAI adaptation triggered")

        # ======================
        # 1️⃣ EWC snapshot
        # ======================

        if self.continual and self.replay and len(self.replay) > 0:

            print("[EWC] snapshot + fisher")

            self.continual.capture_prev_params()

            def fisher_sampler():
                return self.replay.sample_random(
                    EWC_FISHER_BATCH
                )

            self.continual.estimate_fisher(
                fisher_sampler,
                samples=150
            )
            print("[EWC] fisher stats:", self.continual.fisher_stats())
        # ======================
        # 2️⃣ Replay adaptation
        # ======================

        if self.replay and len(self.replay) > 0:
            self._replay_adaptation(severity)

       # ======================
        # 3️⃣ Meta warm start
        # ======================

        if self.meta is not None and self.replay is not None and len(self.replay) > 0:

            print("[META] warm start")

            try:

                # prefer priority sampling if available
                if hasattr(self.replay, "sample_priority"):
                    Xm, ym = self.replay.sample_priority(64)
                else:
                    Xm, ym = self.replay.sample_random(64)

                # safety check
                if Xm is not None and len(Xm) > 0:
                    self.meta.adapt(
                        self.trainer.model,
                        Xm,
                        ym
                    )

            except Exception as e:
                print("[META] warm start skipped:", e)


        # ======================
        # 4️⃣ Reset detectors
        # ======================

        if hasattr(self.detector, "reset"):
            self.detector.reset()

        latency = time.time() - self.last_drift_time
        print(f"[ADAPT] latency={latency:.3f}s")

    # -------------------------------------------------

    def _replay_adaptation(self, severity):

        steps = int(self.replay_steps * (1 + severity))

        self.trainer.set_adaptation_mode(True)

        print(f"[ADAPT] replay x{steps}")

        for _ in range(steps):

            if hasattr(self.replay, "sample_priority"):
                Xr, yr = self.replay.sample_priority(
                    REPLAY_BATCH_SIZE
                )
            else:
                Xr, yr = self.replay.sample_random(
                    REPLAY_BATCH_SIZE
                )

            self.trainer.train_replay_batch(Xr, yr)

        self.trainer.set_adaptation_mode(False)

    # -------------------------------------------------

    def summary(self):

        return {
            "total_steps": self.total_steps,
            "drift_events": self.drift_count,
            "last_drift_time": self.last_drift_time,
            "replay_size": len(self.replay) if self.replay else 0,
            "adapt_lr": getattr(
                self.trainer,
                "adapt_lr",
                None
            )
        }

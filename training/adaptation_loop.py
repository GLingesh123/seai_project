"""
SEAI Adaptation Loop

Coordinates:
- stream loader
- trainer
- drift detector
- replay buffer
- logger

Behavior:
detect drift → trigger replay training bursts → continue stream training
"""

from typing import Optional

from config import REPLAY_BATCH_SIZE, REPLAY_AFTER_DRIFT_STEPS


class AdaptationLoop:
    """
    Online adaptation controller.
    """

    def __init__(
        self,
        stream_loader,
        trainer,
        drift_detector,
        replay_buffer=None,
        logger=None,
        replay_steps: int = REPLAY_AFTER_DRIFT_STEPS
    ):
        self.stream = stream_loader
        self.trainer = trainer
        self.detector = drift_detector
        self.replay = replay_buffer
        self.logger = logger

        self.replay_steps = replay_steps

        self.total_steps = 0
        self.drift_count = 0

    # -------------------------------------------------
    # Main Run Loop
    # -------------------------------------------------

    def run(self, max_steps: Optional[int] = None):
        """
        Run streaming adaptation loop.
        """

        while True:

            batch = self.stream.next_batch()
            if batch is None:
                print("Stream finished.")
                break

            X, y, info = batch

            # ---- train on stream batch ----
            stats = self.trainer.train_batch(X, y)

            acc = stats["accuracy"]
            loss = stats["loss"]

            # ---- update replay memory ----
            if self.replay is not None:
                self.replay.add_batch(X, y)

            # ---- drift detection ----
            err_signal = self.trainer.error_signal(acc)
            drift_out = self.detector.update(err_signal)

            drift_flag = drift_out["drift"]

            # ---- logging ----
            if self.logger:
                self.logger.log_step(
                    step=self.total_steps,
                    loss=loss,
                    accuracy=acc,
                    drift=drift_flag,
                    extra={"stream_step": info["step"]}
                )

            # ---- handle drift ----
            if drift_flag:
                self._handle_drift(drift_out)

            self.total_steps += 1

            if max_steps and self.total_steps >= max_steps:
                break

        if self.logger:
            self.logger.save_all()

    # -------------------------------------------------
    # Drift Handling
    # -------------------------------------------------

    def _handle_drift(self, drift_info):
        """
        Called when detector signals drift.
        """

        self.drift_count += 1

        print(f"[DRIFT] detected at step {self.total_steps}")

        if self.logger:
            self.logger.log_drift_event(self.total_steps, drift_info)

        # ---- replay adaptation burst ----
        if self.replay and len(self.replay) > 0:
            self._replay_adaptation()

    # -------------------------------------------------
    # Replay Burst Training
    # -------------------------------------------------

    def _replay_adaptation(self):

        print("[ADAPT] replay burst start")

        for _ in range(self.replay_steps):
            Xr, yr = self.replay.sample_balanced(REPLAY_BATCH_SIZE)
            self.trainer.train_batch(Xr, yr)

        print("[ADAPT] replay burst end")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self):
        return {
            "total_steps": self.total_steps,
            "drift_events": self.drift_count
        }

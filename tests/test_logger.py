from experiments.logger import ExperimentLogger

logger = ExperimentLogger("test_run")

for step in range(5):
    logger.log_step(
        step=step,
        loss=0.7 - step * 0.05,
        accuracy=0.5 + step * 0.1,
        drift=(step == 3)
    )

    if step == 3:
        logger.log_drift_event(step, {"detector": "adwin"})

logger.save_all()

print(logger.summary())

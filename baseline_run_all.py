"""
SEAI Batch Experiment Runner

Runs multiple scenarios sequentially:
- control
- sudden
- gradual

Each run is independent with separate logs.
"""

from types import SimpleNamespace

from main import run_experiment


def make_args(scenario, steps, name):
    return SimpleNamespace(
        scenario=scenario,
        steps=steps,
        name=name
    )


def run_all():

    runs = [
        ("none", 120, "control_run"),
        ("sudden", 120, "sudden_run"),
        ("gradual", 150, "gradual_run"),
    ]

    for scenario, steps, name in runs:
        print("\n" + "="*50)
        print("RUN:", name)
        print("="*50)

        args = make_args(scenario, steps, name)
        run_experiment(args)


if __name__ == "__main__":
    run_all()

from data.loaders.drift_injector import DriftInjector

scenario = {
    "type": "gradual",
    "start": 10,
    "end": 15
}

inj = DriftInjector(scenario)

for step in range(20):
    mode = inj.get_drift_mode(step)
    prog = inj.get_gradual_progress(step)
    print(step, mode, round(prog, 2))

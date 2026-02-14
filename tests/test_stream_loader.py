from data.loaders.stream_loader import StreamLoader

scenario = {
    "type": "gradual",
    "start": 5,
    "end": 10
}

loader = StreamLoader(scenario=scenario)

for _ in range(15):
    batch = loader.next_batch()
    if batch is None:
        break

    X, y, info = batch

    print(
        info["step"],
        info["drift_mode"],
        info["gradual_progress"],
        round(X.mean(), 2),
        round(y.mean(), 2)
    )

from data.generators.synthetic_stream import SyntheticStream

stream = SyntheticStream(input_dim=5, chunk_size=100)

print("=== No Drift ===")
for i in range(3):
    X, y = stream.next_chunk()
    print(i, X.mean(), y.mean())

print("\n=== Sudden Drift ===")
X, y = stream.next_chunk(drift_mode="sudden")
print("drift", X.mean(), y.mean())

print("\n=== Gradual Drift ===")
for p in [0.2, 0.5, 1.0]:
    X, y = stream.next_chunk(drift_mode="gradual", gradual_progress=p)
    print("gradual", p, X.mean(), y.mean())

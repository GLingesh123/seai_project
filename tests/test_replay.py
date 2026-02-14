import numpy as np
from replay.buffer import ReplayBuffer

buf = ReplayBuffer(capacity=200)

# fake data
X = np.random.randn(128, 5)
y = np.random.randint(0, 2, size=128)

buf.add_batch(X, y)

print("Size:", len(buf))
print("Dist:", buf.class_distribution())

Xs, ys = buf.sample_random(32)
print("Random sample:", Xs.shape, ys.shape)

Xs, ys = buf.sample_balanced(32)
print("Balanced sample:", Xs.shape, ys.shape)

import numpy as np


class FeatureDriftDetector:
    """
    Window mean shift detector for feature vectors
    """

    def __init__(self, window=200, threshold=2.0):
        self.name = "feature_shift"
        self.window = window
        self.threshold = threshold
        self.buffer = []

    def update(self, x_vec):

        self.buffer.append(np.mean(x_vec))

        if len(self.buffer) < self.window:
            return False

        w = np.array(self.buffer[-self.window:])
        first = w[: self.window // 2]
        second = w[self.window // 2:]

        shift = abs(first.mean() - second.mean())

        return shift > self.threshold

    def reset(self):
        self.buffer = []

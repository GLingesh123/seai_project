from river.drift import ADWIN


class ADWINDetector:

    def __init__(self, delta):
        self.detector = ADWIN(delta=delta)
        self.name = "adwin"

    def update(self, x):
        self.detector.update(x)
        return self.detector.drift_detected

    def reset(self, delta):
        self.detector = ADWIN(delta=delta)

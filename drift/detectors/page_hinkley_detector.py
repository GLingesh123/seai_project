from river.drift import PageHinkley


class PageHinkleyDetector:

    def __init__(self):
        self.detector = PageHinkley()
        self.name = "page_hinkley"

    def update(self, x):
        self.detector.update(x)
        return self.detector.drift_detected

    def reset(self):
        self.detector = PageHinkley()

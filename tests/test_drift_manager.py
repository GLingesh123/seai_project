"""
SEAI DriftManager Tests
"""

import unittest
import numpy as np

from drift.drift_manager import DriftManager
from config import DRIFT_DELTA_TEST


class TestDriftManager(unittest.TestCase):

    # -----------------------------------------

    def setUp(self):
        self.detector = DriftManager(
            delta=DRIFT_DELTA_TEST,
            min_votes=1   # ✅ key line
        )


    # -----------------------------------------
    # Test: stable stream → no drift
    # -----------------------------------------

    def test_no_drift_stable(self):

        drifts = 0

        for _ in range(200):
            err = 0.1 + np.random.normal(0, 0.01)
            out = self.detector.update(error_value=err)

            if out["drift"]:
                drifts += 1

        self.assertEqual(drifts, 0)

    # -----------------------------------------
    # Test: sudden shift → drift expected
    # -----------------------------------------

    def test_detects_sudden_drift(self):

        drift_found = False

        # stable phase
        for _ in range(120):
            self.detector.update(error_value=0.1)

        # stronger + longer shift
        for _ in range(200):
            out = self.detector.update(error_value=0.9)

            if out["drift"]:
                drift_found = True
                break

        self.assertTrue(drift_found)

    # -----------------------------------------

    def test_output_fields(self):

        out = self.detector.update(error_value=0.2)

        self.assertIn("drift", out)
        self.assertIn("step", out)
        self.assertIn("severity", out)

    # -----------------------------------------

    def test_step_increments(self):

        s1 = self.detector.update(0.2)["step"]
        s2 = self.detector.update(0.2)["step"]

        self.assertEqual(s2, s1 + 1)

    # -----------------------------------------

    def test_feature_input(self):

        X = np.random.randn(32, 20)

        out = self.detector.update(
            error_value=0.3,
            features=X
        )

        self.assertIn("drift", out)


# -----------------------------------------

if __name__ == "__main__":
    unittest.main()

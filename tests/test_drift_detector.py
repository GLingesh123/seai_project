from drift.drift_detector import DriftDetector

det = DriftDetector()

print("=== stable ===")
for _ in range(100):
    det.update(0.05)

print("=== drift phase ===")
for i in range(200):
    out = det.update(0.95)
    if out["drift"]:
        print("DRIFT DETECTED AT STEP", out["step"])
        break

from evaluation.metrics import build_report
import pandas as pd

# fake logged data
data = {
    "step": list(range(50)),
    "accuracy": [0.5 + i*0.005 for i in range(50)],
    "loss": [1 - i*0.01 for i in range(50)],
    "drift": [False]*20 + [True] + [False]*29
}

df = pd.DataFrame(data)

report = build_report(df)
print(report)

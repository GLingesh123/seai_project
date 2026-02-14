"""
SEAI Run Comparison Utility
"""

import os
import pandas as pd

from evaluation.metrics import build_report
from config import RESULTS_DIR


def compare_csv_runs(name_to_csv: dict, save_name="comparison_summary"):

    rows = []

    for name, path in name_to_csv.items():

        df = pd.read_csv(path)
        report = build_report(df)

        report["run"] = name
        rows.append(report)

    comp_df = pd.DataFrame(rows)

    cols = [
        "run",
        "overall_accuracy",
        "max_accuracy",
        "final_accuracy",
        "forgetting_score",
        "avg_adaptation_latency",
        "drift_steps"
    ]

    comp_df = comp_df[cols]

    out_path = os.path.join(
        RESULTS_DIR,
        f"{save_name}.csv"
    )

    comp_df.to_csv(out_path, index=False)

    print("\n=== COMPARISON TABLE ===")
    print(comp_df.to_string(index=False))
    print("\nSaved →", out_path)

    return comp_df

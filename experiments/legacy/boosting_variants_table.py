"""Legacy v1 — pretty-print the boosting-variants results table.

Reads `results/legacy/boosting_variants/run_variants.json` (produced by
`boosting_variants.py`) and prints one summary table per dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_RESULTS = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "legacy"
    / "boosting_variants"
    / "run_variants.json"
)


def main(results_path: str | Path = _RESULTS) -> None:
    with open(results_path, "r") as f:
        results = json.load(f)

    rows = []
    for ds_name, ds_results in results["datasets"].items():
        for model_key, metrics in ds_results.items():
            rows.append({
                "dataset": ds_name,
                "model": model_key,
                "rmse_train": metrics.get("rmse_train"),
                "rmse_test": metrics.get("rmse_test"),
                "n_estimators": metrics.get("n_estimators_"),
                "best_iter": metrics.get("best_iter"),
                "best_rmse": metrics.get("best_rmse"),
                "n_leaves": metrics.get("n_leaves_"),
            })

    df = pd.DataFrame(rows)
    for ds in df["dataset"].unique():
        print(f"\n=== Results for {ds} ===")
        print(df[df["dataset"] == ds])


if __name__ == "__main__":
    main()

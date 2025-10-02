import json
import pandas as pd

# Load results
with open("src/numerical_illustrations/outputs/run_variants.json", "r") as f:
    results = json.load(f)

rows = []
for ds_name, ds_results in results["datasets"].items():
    for model_key, metrics in ds_results.items():
        rows.append(
            {
                "dataset": ds_name,
                "model": model_key,
                "rmse_train": metrics.get("rmse_train"),
                "rmse_test": metrics.get("rmse_test"),
                "n_estimators": metrics.get("n_estimators_"),
                # new fields
                "best_iter": metrics.get("best_iter"),
                "best_rmse": metrics.get("best_rmse"),
                "best_complexity": metrics.get("best_complexity"),
                "n_leaves": metrics.get("n_leaves_"),
            }
        )

df = pd.DataFrame(rows)

# Print one table per dataset
for ds in df["dataset"].unique():
    print(f"\n=== Results for {ds} ===")
    print(
        df[df["dataset"] == ds][
            [
                "model",
                "rmse_train",
                "rmse_test",
                "n_estimators",
                "best_iter",
                "best_rmse",
                "n_leaves",
                # "best_complexity",
            ]
        ]
    )

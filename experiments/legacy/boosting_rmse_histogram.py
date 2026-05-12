"""Legacy v1 — RMSE histogram across seeds for GBM and PSumBoosting.

Not part of the revised paper's Section 4. Kept available for reproducing
v1-era exploration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trees.data import load_dataset
from trees.models import GBM, PSumBoosting
from trees.stats import rmse


OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "legacy" / "boosting_rmse_histogram"


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-iterations", type=int, default=100)
    return p


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_dataset("linear_model", n=1000, p=5, sigma=1.0,
                           random_state=args.seed)
    rmses_gbm, rmses_psum = [], []
    for k in range(args.n_iterations):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=k)
        gbm = GBM(k_folds=1, max_estimators=200, max_depth=2,
                  learning_rate=0.1).fit(Xtr, ytr, random_state=k)
        rmses_gbm.append(rmse(yte, gbm.predict(Xte)))

        ps = PSumBoosting(learning_rate=0.1, max_estimators=200,
                          significance_level=0.05, max_depth=2,
                          min_samples_leaf=20).fit(Xtr, ytr, random_state=k)
        rmses_psum.append(rmse(yte, ps.predict(Xte)))

    pd.DataFrame({"gbm": rmses_gbm, "psum_boosting": rmses_psum}).to_csv(
        args.out_dir / "rmses.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmses_gbm, bins=20, alpha=0.6, label="GBM")
    ax.hist(rmses_psum, bins=20, alpha=0.6, label="PSumBoosting")
    ax.set_xlabel("test RMSE")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "rmse_histogram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote rmses.csv and rmse_histogram.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()

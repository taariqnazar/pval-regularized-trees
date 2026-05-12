"""Section 4.2.2 — Illustrating the randomness of tree construction via CV.

Reproduces:
- Fig 8: histogram of leaf counts when fitting an L^2-tree via 5-fold CV
  on n=400 training samples from a Gaussian linear model, with 500 resampled
  fold partitions.
- Table 3: test RMSE of the p-sum method for delta in {0.01, 0.05, 0.10}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trees.data import load_dataset
from trees.models import CCPCV, PSumTree
from trees.stats import rmse


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / Path(__file__).stem


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-iterations", type=int, default=500)
    p.add_argument("--quick", action="store_true")
    return p


def make_figure8_and_table(out_dir: Path, n_iter: int, seed: int):
    X, y, _ = load_dataset("linear_model", n=500, p=6, sigma=1.0, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed)
    # Fig 8: leaf counts from 500 resampled CV runs of the CCP path.
    leaf_counts = []
    for k in range(n_iter):
        m = CCPCV(min_samples_leaf=20, n_folds=5,
                  max_depth=8).fit(X_tr, y_tr, random_state=k)
        leaf_counts.append(m.n_leaves_)
    leaf_counts = np.asarray(leaf_counts)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(leaf_counts.min(), leaf_counts.max() + 2) - 0.5
    weights = np.ones_like(leaf_counts) / len(leaf_counts)
    ax.hist(leaf_counts, bins=bins, weights=weights, color="C0",
            edgecolor="black")
    ax.set_xlabel("number of leaves")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(out_dir / "figure8.pdf", bbox_inches="tight")
    plt.close(fig)

    # Table 3: p-sum performance at three deltas.
    rows = []
    for delta in (0.01, 0.05, 0.10):
        m = PSumTree(significance_level=delta, max_depth=8,
                     min_samples_leaf=20).fit(X_tr, y_tr, random_state=seed)
        rows.append({
            "delta": delta,
            "leaves": int(m.n_leaves_),
            "rmse": float(rmse(y_te, m.predict(X_te))),
        })
    pd.DataFrame(rows).to_csv(out_dir / "table3.csv", index=False)

    pd.DataFrame({"leaves": leaf_counts}).to_csv(
        out_dir / "figure8_raw.csv", index=False
    )


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_iter = 10 if args.quick else args.n_iterations
    make_figure8_and_table(args.out_dir, n_iter=n_iter, seed=args.seed)
    print(f"Wrote figure8 and table3 to {args.out_dir}")


if __name__ == "__main__":
    main()

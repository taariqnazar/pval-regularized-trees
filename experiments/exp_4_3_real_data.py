"""Section 4.3 — comparison on Boston Housing, Box Lunch Study, and California Housing.

Reproduces:
- Table 4: dataset summaries (n, d, response).
- Table 5: per-dataset mean RMSE +/- sd RMSE and mean |T| for
  CV (no delta) and p-sum / Cappelli at delta in {0.01, 0.05, 0.10}.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trees.data import load_dataset
from trees.models import PSumTree, CappelliSTP, CCPCV


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / Path(__file__).stem
DELTAS = (0.01, 0.05, 0.10)
MIN_SAMPLES_LEAF = 20
TEST_FRAC = 0.20
DATASETS = ("boston_housing", "box_lunch", "california_housing")
DATASET_LABELS = {
    "boston_housing": "Boston Housing",
    "box_lunch": "Box Lunch Study",
    "california_housing": "California Housing",
}
RESPONSE_NAMES = {
    "boston_housing": "MEDV ($1000s)",
    "box_lunch": "kcal24h0 (24-h kcal)",
    "california_housing": "MedHouseVal ($100k)",
}


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    return p


@dataclass
class RunResult:
    dataset: str
    n: int
    d: int
    rmses: Dict[str, List[float]]
    leaves: Dict[str, List[int]]


def _method_names() -> List[str]:
    out = ["CV"]
    for fam in ("p-sum", "Cappelli"):
        for d in DELTAS:
            out.append(f"{fam}@{d:.2f}")
    return out


def _fit_one_rep(X_tr, y_tr, X_te, y_te, seed) -> Dict[str, Tuple[float, int]]:
    out: Dict[str, Tuple[float, int]] = {}
    cv = CCPCV(min_samples_leaf=MIN_SAMPLES_LEAF, n_folds=5).fit(
        X_tr, y_tr, random_state=seed)
    out["CV"] = (float(np.sqrt(np.mean((cv.predict(X_te) - y_te) ** 2))),
                 cv.n_leaves_)

    for delta in DELTAS:
        ps = PSumTree(significance_level=delta,
                      min_samples_leaf=MIN_SAMPLES_LEAF
                      ).fit(X_tr, y_tr, random_state=seed)
        out[f"p-sum@{delta:.2f}"] = (
            float(np.sqrt(np.mean((ps.predict(X_te) - y_te) ** 2))),
            ps.n_leaves_,
        )
        cap = CappelliSTP(significance_level=delta,
                          min_samples_leaf=MIN_SAMPLES_LEAF
                          ).fit(X_tr, y_tr, random_state=seed)
        out[f"Cappelli@{delta:.2f}"] = (
            float(np.sqrt(np.mean((cap.predict(X_te) - y_te) ** 2))),
            cap.n_leaves_,
        )
    return out


def run_dataset(name: str, reps: int, quick: bool, seed: int) -> RunResult:
    X, y = load_dataset(name)
    if quick and name == "california_housing":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=2000, replace=False)
        X, y = X[idx], y[idx]
    methods = _method_names()
    rmses = {m: [] for m in methods}
    leaves = {m: [] for m in methods}
    for r in range(reps):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=r)
        rep = _fit_one_rep(X_tr, y_tr, X_te, y_te, seed=r)
        for m in methods:
            rmses[m].append(rep[m][0])
            leaves[m].append(rep[m][1])
    return RunResult(name, len(X), X.shape[1], rmses, leaves)


def write_table4(results: List[RunResult], out_dir: Path) -> None:
    rows = [
        {
            "Dataset": DATASET_LABELS[r.dataset],
            "n": r.n,
            "d": r.d,
            "response": RESPONSE_NAMES[r.dataset],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table4.csv", index=False)


def write_table5(results: List[RunResult], out_dir: Path) -> None:
    rows = []
    for r in results:
        for m in _method_names():
            rmses = np.asarray(r.rmses[m])
            leaves = np.asarray(r.leaves[m])
            rows.append({
                "dataset": DATASET_LABELS[r.dataset],
                "method": m,
                "mean_rmse": float(rmses.mean()),
                "sd_rmse": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                "mean_leaves": float(leaves.mean()),
            })
    pd.DataFrame(rows).to_csv(out_dir / "table5.csv", index=False)


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reps = 2 if args.quick else args.reps
    results = [run_dataset(name, reps=reps, quick=args.quick, seed=args.seed)
               for name in DATASETS]
    write_table4(results, args.out_dir)
    write_table5(results, args.out_dir)
    raw_rows = []
    for r in results:
        for rep_idx in range(reps):
            for m in _method_names():
                raw_rows.append({
                    "dataset": r.dataset, "rep": rep_idx, "method": m,
                    "rmse": r.rmses[m][rep_idx],
                    "leaves": r.leaves[m][rep_idx],
                })
    pd.DataFrame(raw_rows).to_csv(args.out_dir / "raw.csv", index=False)
    print(f"Wrote table4, table5, raw to {args.out_dir}")


if __name__ == "__main__":
    main()

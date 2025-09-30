#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import plot_tree  # noqa: F401 (kept for parity / optional plotting)

from single_tree import ccp_pval_regularised_tree, train_regression_tree
# or generate_neufeldt_data if you switch
from utils import generate_linear_data


# ------------------------------ Config dataclasses ------------------------------


@dataclass
class TreeParams:
    max_depth: int = 3
    min_samples_split: int = 20
    random_state: int = 0


@dataclass
class CVParams:
    k_folds: int = 5
    n_iterations: int = 500


# ------------------------------ Helpers ------------------------------


def evaluate_alpha_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tree_params: TreeParams,
) -> Tuple[List[float], List[Dict]]:
    """
    Fit an unpruned tree, get its cost-complexity path (ccp_alphas),
    prune at each alpha, and evaluate RMSE and leaf counts on the test set.
    """
    base_tree = train_regression_tree(
        X_train,
        y_train,
        max_depth=tree_params.max_depth,
        min_samples_split=tree_params.min_samples_split,
        random_state=tree_params.random_state,
    )

    path = base_tree.cost_complexity_pruning_path(X_train, y_train)
    path_alphas = path.ccp_alphas.tolist()

    per_alpha_metrics = []
    for alpha in path_alphas:
        t = train_regression_tree(
            X_train,
            y_train,
            max_depth=tree_params.max_depth,
            min_samples_split=tree_params.min_samples_split,
            ccp_alpha=alpha,
            random_state=tree_params.random_state,
        )
        rmse = float(np.sqrt(mean_squared_error(y_test, t.predict(X_test))))
        per_alpha_metrics.append(
            {"alpha": float(alpha), "rmse": rmse,
             "n_leaves": int(t.get_n_leaves())}
        )

    return path_alphas, per_alpha_metrics


def map_alpha_to_path(alpha: float, path_alphas: List[float]) -> float:
    """
    Map any alpha to the pruning-path grid by returning the largest path alpha <= alpha.
    """
    for a in reversed(path_alphas):
        if a <= alpha:
            return float(a)
    return float(path_alphas[0])


def run_pval_method(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    deltas: List[float],
    tree_params: TreeParams,
) -> List[Dict]:
    results = []
    for delta in deltas:
        t = ccp_pval_regularised_tree(
            X_train,
            y_train,
            significance_level=delta,
            max_depth=tree_params.max_depth,
            min_samples_split=tree_params.min_samples_split,
            random_state=tree_params.random_state,
        )
        rmse = float(np.sqrt(mean_squared_error(y_test, t.predict(X_test))))
        results.append(
            {
                "delta": float(delta),
                "rmse": rmse,
                "ccp_alpha": float(t.ccp_alpha),
                "n_leaves": int(t.get_n_leaves()),
            }
        )
    return results


def run_cv_alpha_selection_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_params: CVParams,
    tree_params: TreeParams,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Repeat: (1) KFold CV to pick alpha (mean-best over folds),
            (2) refit on full train with that alpha, (3) eval on test.
    Returns per-iteration records and a summary of the RMSE distribution.
    """
    per_iter_results = []
    rmses = []

    for i in range(cv_params.n_iterations):
        kf = KFold(n_splits=cv_params.k_folds, random_state=i, shuffle=True)
        fold_best_alphas = []

        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            # Train a base tree to get its alpha path on the fold's train split
            base = train_regression_tree(
                X_tr,
                y_tr,
                max_depth=tree_params.max_depth,
                min_samples_split=tree_params.min_samples_split,
                random_state=tree_params.random_state,
            )
            ccp_path = base.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas

            # Choose alpha by validation MSE along that path
            best_alpha = 0.0
            best_mse = float("inf")
            for alpha in ccp_path:
                t = train_regression_tree(
                    X_tr,
                    y_tr,
                    max_depth=tree_params.max_depth,
                    min_samples_split=tree_params.min_samples_split,
                    ccp_alpha=alpha,
                    random_state=tree_params.random_state,
                )
                mse = mean_squared_error(y_va, t.predict(X_va))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)
            fold_best_alphas.append(best_alpha)

        alpha_avg = float(np.mean(fold_best_alphas))

        # Refit on all training data with chosen alpha
        opt_tree = train_regression_tree(
            X_train,
            y_train,
            max_depth=tree_params.max_depth,
            min_samples_split=tree_params.min_samples_split,
            ccp_alpha=alpha_avg,
            random_state=tree_params.random_state,
        )
        rmse = float(np.sqrt(mean_squared_error(
            y_test, opt_tree.predict(X_test))))
        rmses.append(rmse)

        per_iter_results.append(
            {
                "iteration": int(i),
                "alpha_avg": alpha_avg,
                "realized_alpha": float(opt_tree.ccp_alpha),
                "rmse": rmse,
                "n_leaves": int(opt_tree.get_n_leaves()),
            }
        )

    rmses_arr = np.array(rmses, dtype=float)
    summary = {
        "count": int(rmses_arr.size),
        "mean": float(rmses_arr.mean()),
        "std": float(rmses_arr.std(ddof=0)),
        "min": float(rmses_arr.min()),
        "max": float(rmses_arr.max()),
    }
    return per_iter_results, summary


def _alpha_to_nleaves_map(per_alpha_metrics: List[Dict]) -> Dict[float, int]:
    """Map exact path alpha -> number of leaves using per_alpha_metrics."""
    d = {}
    for rec in per_alpha_metrics:
        d[float(rec["alpha"])] = int(rec["n_leaves"])
    return d


def build_cv_alpha_frequency_table(
    per_iter_results: List[Dict],
    path_alphas: List[float],
    per_alpha_metrics: List[Dict],
) -> List[Dict]:
    """
    Build a frequency/statistics table by mapped path alpha:
      alpha, n_leaves, count, share, rmse_mean, rmse_std, rmse_min, rmse_max
    """
    # Map realized alphas to path grid
    realized = np.array([r["realized_alpha"] for r in per_iter_results], dtype=float)
    mapped = np.array([map_alpha_to_path(a, path_alphas) for a in realized], dtype=float)
    rmses = np.array([r["rmse"] for r in per_iter_results], dtype=float)

    # Unique groups in ascending alpha
    groups = np.array(sorted({float(m) for m in mapped}))
    n_total = len(mapped)

    # n_leaves per alpha from path diagnostics
    a2leaf = _alpha_to_nleaves_map(per_alpha_metrics)

    rows: List[Dict] = []
    for a in groups:
        mask = (mapped == a)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        r = rmses[mask]
        rows.append(
            {
                "alpha": float(a),
                "n_leaves": int(a2leaf.get(float(a), -1)),
                "count": cnt,
                "share": float(cnt / n_total),
                "rmse_mean": float(r.mean()),
                "rmse_std": float(r.std(ddof=0)),
                "rmse_min": float(r.min()),
                "rmse_max": float(r.max()),
            }
        )

    # Sort by alpha (increasing), break ties by leaves (decreasing) just in case
    rows.sort(key=lambda x: (x["alpha"], -x["n_leaves"]))
    return rows


def make_cv_histogram_by_mapped_alpha(
    per_iter_results: List[Dict],
    path_alphas: List[float],
    per_alpha_metrics: List[Dict],
    n_bins: int = 12,
    rmse_min: float | None = None,
    rmse_max: float | None = None,
) -> Dict:
    """
    Build a contingency table: counts[bin, group] where group is the
    mapped pruning level (largest path alpha <= realized_alpha).
    Returns bin edges/centers, the group alpha values, counts, and
    human-readable group labels "alpha=<...>, leaves=<...>".
    """
    rmses = np.array([r["rmse"] for r in per_iter_results], dtype=float)
    realized = np.array([r["realized_alpha"] for r in per_iter_results], dtype=float)

    if rmse_min is None:
        rmse_min = float(rmses.min())
    if rmse_max is None:
        rmse_max = float(rmses.max())

    bin_edges = np.linspace(rmse_min, rmse_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    # Map to path
    mapped = np.array([map_alpha_to_path(a, path_alphas) for a in realized], dtype=float)
    groups = sorted(list({float(m) for m in mapped}))

    # Build counts
    counts = np.zeros((n_bins, len(groups)), dtype=int)
    bin_idx = np.digitize(rmses, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    g2col = {g: j for j, g in enumerate(groups)}
    for i in range(len(rmses)):
        b = bin_idx[i]
        j = g2col[mapped[i]]
        counts[b, j] += 1

    # Labels with leaves
    a2leaf = _alpha_to_nleaves_map(per_alpha_metrics)
    group_labels = [f"alpha={g:.10f}, leaves={a2leaf.get(g, -1)}" for g in groups]

    return {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "bin_centers": [float(x) for x in bin_centers.tolist()],
        "groups": [float(g) for g in groups],
        "group_labels": group_labels,
        "counts": counts.tolist(),  # n_bins x n_groups
    }


# ------------------------------ Main experiment ------------------------------


def main():
    # Global seed (recorded for reproducibility; sklearn/tree randomness is controlled too)
    np.random.seed(1)

    # ----- Data generation -----
    data_params = dict(n=500, sigma=1, p=6, random_state=42)
    X, y, beta = generate_linear_data(**data_params)  # <- your updated function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ----- Fixed params -----
    tree_params = TreeParams(max_depth=8, min_samples_split=20, random_state=0)

    print(tree_params.max_depth)
    cv_params = CVParams(k_folds=5, n_iterations=500)
    pval_deltas = [0.10, 0.05, 0.01]

    # ----- P-value method -----
    pval_results = run_pval_method(
        X_train, y_train, X_test, y_test, pval_deltas, tree_params
    )

    # ----- Pruning path diagnostics (on baseline split) -----
    path_alphas, per_alpha_metrics = evaluate_alpha_path(
        X_train, y_train, X_test, y_test, tree_params
    )

    # ----- CV experiments -----
    per_iter_results, cv_summary = run_cv_alpha_selection_experiments(
        X_train, y_train, X_test, y_test, cv_params, tree_params
    )

    # Map all iterations' realized alphas to the path grid
    mapped_set = sorted(
        {map_alpha_to_path(r["realized_alpha"], path_alphas)
         for r in per_iter_results}
    )
    unique_realized_alphas_mapped = [float(a) for a in mapped_set]

    # ----- Frequency table by mapped alpha (for TeX table #2) -----
    cv_alpha_freq_table = build_cv_alpha_frequency_table(
        per_iter_results=per_iter_results,
        path_alphas=path_alphas,
        per_alpha_metrics=per_alpha_metrics,
    )

    # ----- Histogram by mapped alpha (for TeX figure) -----
    cv_hist = make_cv_histogram_by_mapped_alpha(
        per_iter_results=per_iter_results,
        path_alphas=path_alphas,
        per_alpha_metrics=per_alpha_metrics,
        n_bins=12,  # tweak as you like
    )

    # ----- Assemble JSON payload -----
    payload = {
        "experiment": {
            "description": "Compare randomness between CV-pruned trees and p-value-regularised trees on synthetic data.",
            "numpy_global_seed": 1,
        },
        "data": {
            "generator": "generate_linear_data",
            "params": data_params,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "true_beta": [float(b) for b in beta.tolist()],
        },
        "modeling": {
            "tree_params": asdict(tree_params),
            "cv_params": asdict(cv_params),
            "pval_deltas": pval_deltas,
        },
        "results": {
            "pval_method": pval_results,
            #"cv_iterations": per_iter_results,
            "cv_rmse_summary": cv_summary,
            "alpha_path": {
                "path_alphas": [float(a) for a in path_alphas],
                "per_alpha_metrics": per_alpha_metrics,
                "unique_realized_alphas_mapped": unique_realized_alphas_mapped,
            },
            # NEW: frequency table for TeX
            "cv_alpha_frequency_table": cv_alpha_freq_table,
            # Histogram with labelled groups for plotting
            "cv_hist_by_mapped_alpha": cv_hist,
        },
    }

    # ----- Write to a single file -----
    os.makedirs(".data", exist_ok=True)
    out_path = ".data/experiment_dump.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

from matplotlib.ticker import MaxNLocator, PercentFormatter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import GridSearchCV  # <-- NEW

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from models import CCPPTree
from data import load_dataset


# ------------------------------ Config ------------------------------


@dataclass
class TreeParams:
    max_depth: int = 8
    min_samples_split: int = 20
    random_state: int = 0


@dataclass
class CVParams:
    k_folds: int = 5
    n_iterations: int = 500  # how many CV runs (different fold shuffles)
    scoring: str = "rmse"  # "rmse" or "mse"  <-- NEW


# ------------------------------ Core logic ------------------------------


def train_regression_tree(
    X,
    y,
    max_depth=8,
    min_samples_split=20,
    ccp_alpha=0.0,
    random_state=0,
):
    """
    Train a regression tree with given hyperparameters.
    Returns a fitted sklearn DecisionTreeRegressor.
    """
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    tree.fit(X, y)
    return tree


def pick_alpha_via_kfold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_folds: int,
    tree_params: TreeParams,
    seed: int,
) -> float:
    """
    Choose ccp_alpha via K-fold CV on the training set.
    For each fold, search along that fold's cost-complexity path
    and pick the alpha with the smallest validation MSE.
    Return the average of the fold-wise best alphas.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    best_alphas: List[float] = []

    for tr_idx, va_idx in kf.split(X_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        # Base tree to get the pruning path for this fold
        base = train_regression_tree(
            X_tr,
            y_tr,
            max_depth=tree_params.max_depth,
            min_samples_split=tree_params.min_samples_split,
            random_state=tree_params.random_state,
        )
        ccp_path = base.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas

        # Grid search over the path
        best_alpha, best_mse = 0.0, float("inf")
        for alpha in ccp_path:
            t = train_regression_tree(
                X_tr,
                y_tr,
                max_depth=tree_params.max_depth,
                min_samples_split=tree_params.min_samples_split,
                ccp_alpha=float(alpha),
                random_state=tree_params.random_state,
            )
            mse = mean_squared_error(y_va, t.predict(X_va))
            if mse < best_mse:
                best_mse, best_alpha = mse, float(alpha)

        best_alphas.append(best_alpha)

    return float(np.mean(best_alphas))


# -------------------- NEW: GridSearchCV-based alpha picker --------------------


def pick_alpha_via_gridsearchcv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_folds: int,
    tree_params: TreeParams,
    seed: int = 0,
    scoring: str = "rmse",  # "rmse" or "mse"
):
    """
    Choose ccp_alpha via GridSearchCV over the cost-complexity path.
    Returns (best_alpha, gridsearch_object).
    """
    # Build alpha grid from the full training data (fast + standard for DT pruning)
    base = train_regression_tree(
        X_train,
        y_train,
        max_depth=tree_params.max_depth,
        min_samples_split=tree_params.min_samples_split,
        ccp_alpha=0.0,
        random_state=tree_params.random_state,
    )
    ccp_path = base.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    alphas = np.unique(ccp_path)

    # Drop the largest alpha which often collapses to root-only stump
    if alphas.size > 1:
        alphas = alphas[:-1]
    if alphas.size == 0:
        # Very rare backstop
        alphas = np.logspace(-6, 0, 30)

    # Scorer (built-in strings; avoids custom scorer plumbing)
    if scoring.lower() == "rmse":
        scorer = "neg_root_mean_squared_error"
    elif scoring.lower() == "mse":
        scorer = "neg_mean_squared_error"
    else:
        raise ValueError("scoring must be 'rmse' or 'mse'")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Estimator with structural params fixed; tune only ccp_alpha
    est = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=tree_params.max_depth,
        min_samples_split=tree_params.min_samples_split,
        random_state=tree_params.random_state,
    )

    gs = GridSearchCV(
        estimator=est,
        param_grid={"ccp_alpha": alphas},
        scoring=scorer,
        cv=kf,
        n_jobs=-1,
        refit=True,  # keep best estimator trained on full data
        return_train_score=False,
    )
    gs.fit(X_train, y_train)

    best_alpha = float(gs.best_params_["ccp_alpha"])
    return best_alpha, gs


def _one_iteration(
    i: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_params,
    tree_params,
) -> Tuple[int, float, DecisionTreeRegressor]:
    # Choose alpha via GridSearchCV (RMSE or MSE based on cv_params.scoring)
    alpha, _ = pick_alpha_via_gridsearchcv(
        X_train,
        y_train,
        cv_params.k_folds,
        tree_params,
        seed=i,
        scoring=cv_params.scoring,  # <-- uses "rmse" or "mse"
    )

    # Train final tree (optionally vary random_state per iteration to increase diversity)
    final_tree = train_regression_tree(
        X_train,
        y_train,
        max_depth=tree_params.max_depth,
        min_samples_split=tree_params.min_samples_split,
        ccp_alpha=alpha,
        random_state=(
            tree_params.random_state if tree_params.random_state is not None else i
        ),
    )
    return i, alpha, final_tree


def cv_based_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_params,
    tree_params,
) -> List[DecisionTreeRegressor]:
    # Use all cores; tweak n_jobs if you want fewer
    results = Parallel(n_jobs=-1, prefer="processes", verbose=0)(
        delayed(_one_iteration)(i, X_train, y_train, cv_params, tree_params)
        for i in range(cv_params.n_iterations)
    )

    # Preserve original order by iteration index
    results.sort(key=lambda t: t[0])
    trees = [t[2] for t in results]

    # Optional: compact progress summary
    for i, alpha, _ in results:
        print(
            f"\rIteration {i + 1}/{cv_params.n_iterations}: alpha={alpha:.5f}", end=""
        )

    return trees


def collect_leaf_counts(trees: List[DecisionTreeRegressor]) -> List[int]:
    """
    Given a list of fitted sklearn DecisionTreeRegressor,
    """
    return [int(t.tree_.n_leaves) for t in trees]


def pval_based_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tree_params: CVParams,
    deltas: List[float] = [0.01, 0.05, 0.1],
) -> List[CCPPTree]:
    trees = []
    for i, delta in enumerate(deltas):
        final_tree = CCPPTree(
            significance_level=delta,
            **tree_params.__dict__,
        )
        final_tree.fit(X_train, y_train)
        trees.append(final_tree)
        # print(
        #    f"\rIteration {i + 1}/{len(deltas)}",
        #    end="",
        # )

    return trees


def collect_leaf_counts_pval(trees: List[CCPPTree]) -> List[int]:
    """
    Repeat the CV alpha selection many times (different shuffles) and
    record the number of leaves of the final tree fit on all training data.
    """
    return [int(t.n_leaves_) for t in trees if t.n_leaves_ is not None]


def make_frequency_table(leaves: List[int]) -> List[Tuple[int, int, float]]:
    """
    Turn a list of leaf counts into a sorted table:
      (n_leaves, count, share)
    """
    ctr = Counter(leaves)
    total = sum(ctr.values())
    rows = [(k, v, v / total) for k, v in ctr.items()]
    rows.sort(key=lambda r: (r[0]))  # ascending by n_leaves
    return rows


def plot_leaf_histogram(table, output_path="leaf_frequencies.pdf"):
    # table: (n_leaves, count, share). Rebuild a dense series.
    if not table:
        print("No data to plot.")
        return

    xs_present = [row[0] for row in table]
    counts_present = {row[0]: row[1] for row in table}

    xmin, xmax = min(xs_present), max(xs_present)
    xs = list(range(xmin, xmax + 1))
    counts = [counts_present.get(x, 0) for x in xs]
    total = sum(counts) if sum(counts) > 0 else 1
    shares = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(xs, shares, width=0.8, edgecolor="black", linewidth=0.6)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ymax = max(shares) if shares else 1.0
    ax.set_ylim(0, ymax * 1.15)

    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_xlabel("Number of leaves")
    # ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram to {output_path}")


def trees_rmse(trees: List[DecisionTreeRegressor | CCPPTree], X, y) -> List[float]:
    """
    Given a list of fitted trees (sklearn or CCPPTree),
    compute their RMSE on (X, y).
    """
    return [float(np.sqrt(mean_squared_error(y, t.predict(X)))) for t in trees]


# ------------------------------ Run ------------------------------


def main():
    seed = 1 
    np.random.seed(seed)
    n_iterations = 500

    # Generate simple linear data (use your existing helper)
    X, y, beta = load_dataset("linear_model", n=500,
                              sigma=1, p=6, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    print(beta)

    tree_params = TreeParams(max_depth=8, min_samples_split=20, random_state=0)
    cv_params = CVParams(
        k_folds=5, n_iterations=n_iterations, scoring="rmse"
    )  # <-- pick "rmse" or "mse"

    ### CV BASED METHOD (now uses GridSearchCV inside) ###
    cv_trees = cv_based_trees(X_train, y_train, cv_params, tree_params)
    leaves = collect_leaf_counts(cv_trees)
    rmses = trees_rmse(cv_trees, X_test, y_test)
    table = make_frequency_table(leaves)

    print("CV-based method:")
    print("n_leaves\tcount\tshare")
    for n_leaves, count, share in table:
        print(f"{n_leaves}\t{count}\t{share:.3f}")
    # Plot and save histogram
    plot_leaf_histogram(table)

    argmin_rmse = np.argmin(rmses)
    print(
        f"Best RMSE (CV-based): {rmses[argmin_rmse]:.4f} "
        f"with {leaves[argmin_rmse]} leaves"
    )

    print("Pval-based method:")
    print("n_leaves\tcount\tshare")
    ### PVAL BASED METHOD ###
    pval_trees = pval_based_trees(X_train, y_train, tree_params)
    leaves = collect_leaf_counts_pval(pval_trees)
    rmses = trees_rmse(pval_trees, X_test, y_test)

    for i, tree in enumerate(pval_trees):
        print(
            f"Delta={tree.significance_level}: RMSE={rmses[i]:.4f} "
            f"with {leaves[i]} leaves"
        )


if __name__ == "__main__":
    main()

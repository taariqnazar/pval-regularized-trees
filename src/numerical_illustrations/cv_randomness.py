from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

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


def _one_iteration(
    i: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_params,
    tree_params,
) -> Tuple[int, float, DecisionTreeRegressor]:
    # Pick alpha (consider making this function accept n_jobs too; see section 2)
    alpha = pick_alpha_via_kfold(
        X_train, y_train, cv_params.k_folds, tree_params, seed=i
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
        print(
            f"\rIteration {i + 1}/{len(deltas)}",
            end="",
        )

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
    """
    Plot histogram of leaf frequencies from the frequency table.

    Parameters
    ----------
    table : list of tuples
        Each element is (n_leaves, count, share)
    output_path : str
        File path to save the resulting PDF.
    """
    # unpack
    n_leaves = [row[0] for row in table]
    shares = [row[2] * 100 for row in table]  # convert to percentages

    # create figure
    plt.figure(figsize=(9, 5))
    plt.bar(
        n_leaves,
        shares,
        width=0.8,
        color="#4472C4",
        edgecolor="black",
        linewidth=0.5,
    )

    # labels and ticks
    # plt.xlabel("Number of leaves", fontsize=12)
    # plt.ylabel("Frequency (%)", fontsize=12)
    # plt.xticks(n_leaves)  # show every second tick
    plt.yticks([0, 5, 10, 15], [f"{t}%" for t in [0, 5, 10, 15]])
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # make layout tight and save as PDF
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to {output_path}")


def trees_rmse(trees: List[DecisionTreeRegressor | CCPPTree], X, y) -> List[float]:
    """
    Given a list of fitted trees (sklearn or CCPPTree),
    compute their RMSE on (X, y).
    """
    return [float(np.sqrt(mean_squared_error(y, t.predict(X)))) for t in trees]


# ------------------------------ Run ------------------------------


def main():
    np.random.seed(42)

    # Generate simple linear data (use your existing helper)
    X, y, beta = load_dataset("linear_model", n=500,
                              sigma=1, p=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    tree_params = TreeParams(max_depth=8, min_samples_split=20, random_state=0)
    cv_params = CVParams(k_folds=5, n_iterations=500)

    ### CV BASED METHOD ###
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

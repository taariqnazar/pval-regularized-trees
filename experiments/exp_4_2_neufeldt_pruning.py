"""Section 4.2 — Simulated examples from Neufeldt et al.

Reproduces:
- Fig 3: regression tree corresponding to Eq. (19), schematic for a=b=1.
- Fig 4: MSEP vs MSE + cumulative p-values along CCP path, b=1.
- Fig 5: same with b=0.5.
- Fig 6: regularised output trees for b=1 and b=0.5 with red-shaded nodes
  that violate the cumulative p-value condition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from trees.data import load_dataset
from trees.models import PSumTree
from trees.plotting import plot_tree_with_pvalues
from trees.stats import get_cum_p_val, get_p_values, mse


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / Path(__file__).stem


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--quick", action="store_true")
    return p


def make_figure3(out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")
    ax.text(0.5, 0.95, "X1 <= 0\n(b)", ha="center",
            bbox=dict(boxstyle="round"))
    ax.text(0.25, 0.7, "X2 <= 0\n(2b)", ha="center",
            bbox=dict(boxstyle="round"))
    ax.text(0.75, 0.7, "0", ha="center", bbox=dict(boxstyle="round"))
    ax.text(0.1, 0.45, "X3 <= 0\n(1.5b)", ha="center",
            bbox=dict(boxstyle="round"))
    ax.text(0.4, 0.45, "X3 <= 0\n(2.5b)", ha="center",
            bbox=dict(boxstyle="round"))
    for x, lab in [(0.05, "2b"), (0.15, "b"), (0.35, "2b"), (0.45, "3b")]:
        ax.text(x, 0.2, lab, ha="center", bbox=dict(boxstyle="round"))
    fig.suptitle("Fig 3 — Neufeldt tree schematic (Eq. 19, a=1)")
    fig.savefig(out_dir / "figure3.pdf", bbox_inches="tight")
    plt.close(fig)


def run_one_b(b: float, N: int, seed: int):
    X, Y = load_dataset("neufeldt", a=1.0, b=b, n=N, p=10, sigma=1.0,
                        random_state=seed)
    X_test, Y_test = load_dataset("neufeldt", a=1.0, b=b, n=N, p=10,
                                  sigma=1.0, random_state=seed + 10_000)
    K, min_leaf = 4, 20
    cart = DecisionTreeRegressor(max_depth=K, min_samples_leaf=min_leaf,
                                 random_state=seed).fit(X, Y)
    ccp_alphas = cart.cost_complexity_pruning_path(X, Y)["ccp_alphas"]
    flipped = np.flip(ccp_alphas)
    mse_train, mse_test, cum_p, n_leaves = [], [], [], []
    for alpha in flipped:
        sub = DecisionTreeRegressor(
            max_depth=K, min_samples_leaf=min_leaf,
            random_state=seed, ccp_alpha=alpha,
        ).fit(X, Y)
        mse_train.append(mse(Y, sub.predict(X)))
        mse_test.append(mse(Y_test, sub.predict(X_test)))
        cum_p.append(get_cum_p_val(sub, X.shape[1]))
        n_leaves.append(sub.get_n_leaves())
    return (np.asarray(n_leaves), np.asarray(mse_train),
            np.asarray(mse_test), np.asarray(cum_p))


def _plot_subtree_panels(b, n_leaves, mse_train, mse_test, cum_p, out_path,
                         delta=0.05):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.plot(n_leaves, mse_test, label="MSEP", color="C0")
    ax_l.plot(n_leaves, mse_train, label="MSE", color="C1")
    eligible = np.where(cum_p <= delta)[0]
    if len(eligible) > 0:
        k_star = int(eligible.max())
        ax_l.axvline(n_leaves[k_star], color="C0", linestyle="--")
        ax_r.axvline(n_leaves[k_star], color="C0", linestyle="--")
    ax_l.set_xlabel("number of leaves")
    ax_l.legend()
    ax_r.plot(n_leaves, cum_p, color="C0")
    ax_r.set_xlabel("number of leaves")
    ax_r.set_ylabel("cumulative p-value")
    fig.suptitle(f"b = {b}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_figure6(out_dir: Path, seed: int):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    for ax, b in zip(axes, (1.0, 0.5)):
        X, Y = load_dataset("neufeldt", a=1.0, b=b, n=500, p=10, sigma=1.0,
                            random_state=seed)
        m = PSumTree(significance_level=0.05, max_depth=4,
                     min_samples_leaf=20).fit(X, Y, random_state=seed)
        d = X.shape[1]
        node_p = get_p_values(m.model, d)
        cum_p_scalar = get_cum_p_val(m.model, d)
        cum_p = np.full(m.model.tree_.node_count, cum_p_scalar)
        plot_tree_with_pvalues(m.model, node_pvalues=node_p,
                               cumulative_pvalues=cum_p, delta=0.05, ax=ax)
        ax.set_title(f"b = {b}")
    fig.tight_layout()
    fig.savefig(out_dir / "figure6.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    N = 200 if args.quick else 500
    make_figure3(args.out_dir)
    for b, path in [(1.0, args.out_dir / "figure4.pdf"),
                    (0.5, args.out_dir / "figure5.pdf")]:
        n_leaves, mse_tr, mse_te, cum_p = run_one_b(b=b, N=N, seed=args.seed)
        _plot_subtree_panels(b, n_leaves, mse_tr, mse_te, cum_p, path)
    make_figure6(args.out_dir, seed=args.seed)
    print(f"Wrote figure3-6 to {args.out_dir}")


if __name__ == "__main__":
    main()

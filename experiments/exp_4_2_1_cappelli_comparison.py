"""Section 4.2.1 — comparison to Cappelli et al.

Reproduces:
- Table 2: per-internal-node F^ts, U_max, p^F, P_max on a single CART grown
  on a 350 / 150 (grow / prune) split of n=500 Neufeldt-a1b1 data.
- Fig 7 (left): leaf-count distributions of Cappelli and PSum over 200 reps.
- Fig 7 (right): MSEP scatter (p-sum vs Cappelli, n=500 train + 500 test).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from trees.data import load_dataset
from trees.models import CappelliSTP, PSumTree
from trees.models.cappelli import _f_pvalues, _test_impurities
from trees.stats import get_YD_statistics, get_p_values


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / Path(__file__).stem


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--quick", action="store_true")
    return p


def _f_statistic_table(cart, impurity_test, n_test):
    n = cart.tree_.node_count
    out = np.full(n, -1.0)
    children_left = cart.tree_.children_left
    children_right = cart.tree_.children_right
    for k in range(n):
        if children_left[k] == -1:
            continue
        WSS_l = n_test[children_left[k]] * impurity_test[children_left[k]]
        WSS_r = n_test[children_right[k]] * impurity_test[children_right[k]]
        WSS = WSS_l + WSS_r
        BSS = 0.0
        L = 0
        stack = [k]
        while stack:
            inner = stack.pop()
            ll = children_left[inner]
            rr = children_right[inner]
            if ll == -1:
                continue
            L += 1
            BSS += n_test[inner] * impurity_test[inner] - (
                n_test[ll] * impurity_test[ll] + n_test[rr] * impurity_test[rr]
            )
            stack.extend([rr, ll])
        n_node = n_test[k]
        if WSS <= 0 or n_node - L - 1 <= 0 or L == 0:
            out[k] = 0.0
            continue
        out[k] = (BSS / WSS) * ((n_node - L - 1) / L)
    return out


def make_table2(out_dir: Path, seed: int) -> None:
    """Table 2: F^ts, U_max, p^F, P_max on a 350/150 split of n=500 Neufeldt-a1b1."""
    X, Y = load_dataset("neufeldt", a=1.0, b=1.0, n=500, p=10, sigma=1.0,
                        random_state=seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    grow = perm[:350]
    prune = perm[350:]
    cart = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=20, random_state=seed
    ).fit(X[grow], Y[grow])
    impurity_test, n_test = _test_impurities(cart, X[prune], Y[prune])
    p_F = _f_pvalues(cart, impurity_test, n_test)
    F_ts = _f_statistic_table(cart, impurity_test, n_test)

    cart_full = DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=20, random_state=seed
    ).fit(X, Y)
    U_max = get_YD_statistics(cart_full)
    P_max = get_p_values(cart_full, d=X.shape[1])

    rows = []
    n_nodes = cart.tree_.node_count
    for k in range(n_nodes):
        if cart.tree_.children_left[k] == -1:
            continue
        rows.append({
            "node_id": k,
            "F_ts": float(F_ts[k]),
            "U_max": float(U_max[k]) if k < len(U_max) else float("nan"),
            "p_F": float(p_F[k]),
            "P_max": float(P_max[k]) if k < len(P_max) else float("nan"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "table2.csv", index=False)


def make_figure7(out_dir: Path, reps: int, seed: int) -> None:
    leaves_psum, leaves_cap, mse_psum, mse_cap = [], [], [], []
    for rep in range(reps):
        X_tr, Y_tr = load_dataset("neufeldt", a=1.0, b=1.0, n=500, p=10,
                                  sigma=1.0, random_state=rep + seed)
        X_te, Y_te = load_dataset("neufeldt", a=1.0, b=1.0, n=500, p=10,
                                  sigma=1.0, random_state=rep + seed + 10_000)
        ps = PSumTree(significance_level=0.05, max_depth=4,
                      min_samples_leaf=20).fit(X_tr, Y_tr, random_state=rep)
        cap = CappelliSTP(significance_level=0.01, train_frac=0.7,
                          max_depth=4, min_samples_leaf=20
                          ).fit(X_tr, Y_tr, random_state=rep)
        leaves_psum.append(ps.n_leaves_)
        leaves_cap.append(cap.n_leaves_)
        mse_psum.append(float(np.mean((ps.predict(X_te) - Y_te) ** 2)))
        mse_cap.append(float(np.mean((cap.predict(X_te) - Y_te) ** 2)))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    ax_left.scatter(range(reps), leaves_cap, c="C0", label="Cappelli (F)", s=8)
    ax_left.scatter(range(reps), leaves_psum, c="C1", marker="x",
                    label="p-sum", s=12)
    ax_left.set_xlabel("rep")
    ax_left.set_ylabel("number of leaves")
    ax_left.legend()

    lo = min(min(mse_psum), min(mse_cap))
    hi = max(max(mse_psum), max(mse_cap))
    ax_right.scatter(mse_psum, mse_cap, s=10)
    ax_right.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax_right.set_xlabel("MSEP — p-sum")
    ax_right.set_ylabel("MSEP — Cappelli")
    fig.tight_layout()
    fig.savefig(out_dir / "figure7.pdf", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "rep": range(reps),
        "leaves_psum": leaves_psum,
        "leaves_cappelli": leaves_cap,
        "msep_psum": mse_psum,
        "msep_cappelli": mse_cap,
    }).to_csv(out_dir / "figure7_raw.csv", index=False)


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reps = 4 if args.quick else args.reps
    make_table2(args.out_dir, seed=args.seed)
    make_figure7(args.out_dir, reps=reps, seed=args.seed)
    print(f"Wrote table2 and figure7 to {args.out_dir}")


if __name__ == "__main__":
    main()

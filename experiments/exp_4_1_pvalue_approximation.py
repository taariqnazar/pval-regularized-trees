"""Section 4.1 — The p-value approximation for a single split.

Reproduces:
- Fig 1: empirical vs approximated CDFs of U_max for varying n, d, rho.
- Table 1: 0.95-quantiles of U_max under H_0.
- Fig 2: detection rate vs sample size for independent / dependent covariates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from trees.stats import Psi, get_YD_statistics, get_p_values


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / Path(__file__).stem


def _argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--quick", action="store_true")
    return p


def simulate_empirical_quantile(M, N, d, rho, sigma=1.0, alpha=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    mu = np.zeros(d)
    YD = np.empty(M)
    for i in range(M):
        X = rng.multivariate_normal(mu, cov, size=N)
        Y = rng.normal(0.0, sigma, size=N)
        cart = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1).fit(X, Y)
        YD[i] = get_YD_statistics(cart)[0]
    Z = np.sort(YD)
    emp_q = float(np.quantile(Z, alpha))
    x = np.arange(0.1, 20, 0.001)
    F_approx = np.array([1 - Psi(N, u, d) for u in x])
    idx = int(np.searchsorted(F_approx, alpha))
    approx_q = float(x[min(idx, len(x) - 1)])
    F_emp = np.arange(len(Z)) / float(len(Z))
    return emp_q, approx_q, Z, F_emp, x, F_approx


def make_figure1_and_table1(out_dir: Path, M: int, seed: int):
    rng = np.random.default_rng(seed)
    Ns = [50, 1000]
    ds = [1, 2, 10]
    rhos = [0.0, 0.8]

    emp = {}
    approx = {}
    for rho in rhos:
        for d in ds:
            for N in Ns:
                eq, aq, *_ = simulate_empirical_quantile(M, N, d, rho, rng=rng)
                emp[(rho, d, N)] = eq
                approx[(d, N)] = aq

    rows = []
    for d in ds:
        for N in Ns:
            rows.append({
                "d": d, "n": N,
                "emp_rho_0.0": emp[(0.0, d, N)],
                "emp_rho_0.8": emp[(0.8, d, N)],
                "approx": approx[(d, N)],
            })
    pd.DataFrame(rows).to_csv(out_dir / "table1.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, rho in enumerate(rhos):
        for j, N in enumerate(Ns):
            d = 10
            _, _, Z, F_emp, x, F_approx = simulate_empirical_quantile(
                M, N, d, rho, rng=rng
            )
            ax = axes[i, j]
            ax.plot(Z, F_emp, label="Empirical CDF", color="blue")
            ax.plot(x, F_approx, label="Approximation", color="orange")
            ax.axhline(y=0.95, linestyle="--", color="blue")
            ax.set_xlim(10, 18)
            ax.set_ylim(0.5, 1.0)
            ax.set_title(f"rho={rho}, N={N}, d={d}")
            if i == 1:
                ax.set_xlabel("u")
            if j == 0:
                ax.set_ylabel("F(u)")
    axes[0, 0].legend()
    fig.savefig(out_dir / "figure1.pdf", bbox_inches="tight")
    plt.close(fig)


def _approx_quantile(N, delta, d):
    x = np.arange(0.1, 20, 0.01)
    for u in x:
        if Psi(N, u, d) <= delta:
            return float(u)
    return float(x[-1])


def make_figure2(out_dir: Path, M: int, seed: int):
    rng = np.random.default_rng(seed + 1)
    d = 10
    rho = 0.8
    sigma = 1.0
    delta = 0.05
    r = 0.2
    x_axis_bound = 6000
    step = 500 if M <= 50 else 100
    x_axis = np.arange(100, x_axis_bound, step)

    muX = np.zeros(d)
    covX = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)

    frac_indep, frac_dep, kappas = [], [], []
    for N in x_axis:
        kappa = N ** (-r)
        kappas.append(kappa)
        _approx_quantile(N, delta, d)

        pi = []
        for _ in range(M):
            X = rng.normal(0, 1, size=(N, d))
            mus = np.where(X[:, 0] <= 0, 0.0, kappa)
            Y = rng.normal(mus, sigma, size=N)
            cart = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1).fit(X, Y)
            pi.append(get_p_values(cart, d)[0])
        frac_indep.append(float(np.mean(np.asarray(pi) <= delta)))

        pd_ = []
        for _ in range(M):
            X = rng.multivariate_normal(muX, covX, size=N)
            mus = np.where(X[:, 0] <= 0, 0.0, kappa)
            Y = rng.normal(mus, sigma, size=N)
            cart = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1).fit(X, Y)
            pd_.append(get_p_values(cart, d)[0])
        frac_dep.append(float(np.mean(np.asarray(pd_) <= delta)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_axis, frac_dep, label="Dependent covariates", color="C0")
    ax.plot(x_axis, frac_indep, label="Independent covariates", color="C1")
    ax.plot(x_axis, kappas, label=r"$\kappa = N^{-r}$", color="C2")
    ax.axhline(y=0.95, linestyle="--", color="gray")
    ax.set_ylim(0, 1)
    ax.set_xlim(float(x_axis.min()), float(x_axis.max()))
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Fraction of correct detections")
    ax.legend()
    fig.savefig(out_dir / "figure2.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = _argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    M_fig1 = 200 if args.quick else 10_000
    M_fig2 = 20 if args.quick else 1_000
    make_figure1_and_table1(args.out_dir, M=M_fig1, seed=args.seed)
    make_figure2(args.out_dir, M=M_fig2, seed=args.seed)
    print(f"Wrote figure1, figure2, table1 to {args.out_dir}")


if __name__ == "__main__":
    main()

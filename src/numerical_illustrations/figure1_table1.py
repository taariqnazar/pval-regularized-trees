import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from utils.stats.pvalues import get_YD_statistics, Psi  # your custom functions

plt.style.use("default")

# -----------------------
# Parameters
# -----------------------
M = 10_000  # Monte Carlo repetitions
Ns = [50, 1000]  # sample sizes
ds = [1, 2, 10]  # feature dimensions
rhos = [0.0, 0.8]  # independent vs dependent covariates
sigma = 1.0
min_leaves = 1
delta = 0.05


# -----------------------
# Helper
# -----------------------
def simulate_empirical_quantile(N, d, rho, alpha=0.95):
    """Simulate empirical and approximate quantile of YD statistic."""
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    mu = np.zeros(d)

    YD_statistics = []
    for _ in range(M):
        X = np.random.multivariate_normal(mu, cov, size=N)
        Y = np.random.normal(0.0, sigma, size=N)

        cart = DecisionTreeRegressor(
            max_depth=1, min_samples_leaf=min_leaves).fit(X, Y)
        YD_statistics.append(get_YD_statistics(cart)[0])

    # empirical quantile
    Z = np.sort(YD_statistics)
    emp_q = np.quantile(Z, alpha)

    # approximate quantile
    x = np.arange(0.1, 20, 0.001)
    F_approx = [1 - Psi(N, u, d) for u in x]
    approx_q = x[np.searchsorted(F_approx, alpha)]

    return emp_q, approx_q, (Z, np.arange(len(Z)) / float(len(Z)), x, F_approx)


# -----------------------
# Table 1 results
# -----------------------
table_emp = {rho: {} for rho in rhos}
table_approx = {}

for rho in rhos:
    for d in ds:
        for N in Ns:
            emp_q, approx_q, _ = simulate_empirical_quantile(N, d, rho)
            table_emp[rho][(d, N)] = emp_q
            table_approx[(d, N)] = approx_q

# Print in LaTeX table format
print("\n--- Table 1 values ---")
for rho in rhos:
    print(f"\nRho = {rho}")
    for d in ds:
        row = f"d={d} : "
        for N in Ns:
            row += f"{table_emp[rho][(d, N)]:.2f}  "
        print(row)

print("\nApproximation (rho-independent of rho):")
for d in ds:
    row = f"d={d} : "
    for N in Ns:
        row += f"{table_approx[(d, N)]:.2f}  "
    print(row)


# -----------------------
# Figure 1 (4-panel plot)
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for i, rho in enumerate(rhos):  
    for j, N in enumerate(Ns):  
        d = 10  # figure uses d=10 fixed
        emp_q, approx_q, (Z, F_emp, x, F_approx) = simulate_empirical_quantile(N, d, rho)

        ax = axes[i, j]
        ax.plot(Z, F_emp, label="Empirical CDF", color="blue")   # <-- fixed
        ax.plot(x, F_approx, label="Approximation", color="orange")
        ax.axhline(y=0.95, linestyle="--", color="blue")

        ax.set_xlim(10, 18)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"rho={rho}, N={N}, d={d}")
        if i == 1:
            ax.set_xlabel("u (YD statistic)")
        if j == 0:
            ax.set_ylabel("F(u)")

axes[0, 0].legend()
plt.savefig("Figure1.pdf", bbox_inches="tight", transparent=True)
plt.show()

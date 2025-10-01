import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from utils.stats.pvalues import Psi, get_p_values  # your custom functions

# -----------------------
# Parameters
# -----------------------
M = 1000            # Monte Carlo repetitions
d = 10              # feature dimension
rho = 0.8           # correlation for dependent covariates
sigma = 1.0         # noise standard deviation
delta = 0.05        # significance level
r = 0.2             # decay rate for kappa = N^(-r)
min_leaves = 1      # minimal leaf size for CART
x_axis_bound = 6000 # maximum sample size
step = 100          # step size for sample sizes

# Covariance for dependent covariates
muX = np.zeros(d)
covX = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)

# -----------------------
# Helper: approximate quantile from Psi
# -----------------------
def get_appr_quantile(N, delta, d):
    x = np.arange(0.1, 20, 0.01)
    for u in x:
        if Psi(N, u, d) <= delta:
            return u
    return x[-1]


# -----------------------
# Arrays for results
# -----------------------
x_axis = np.arange(100, x_axis_bound, step)
frac_correct_indep = []
frac_correct_dep = []
kappas = []


# -----------------------
# Independent covariates
# -----------------------
for N in x_axis:
    kappa = N ** (-r)
    kappas.append(kappa)
    u = get_appr_quantile(N, delta, d)

    p_vals = []
    for _ in range(M):
        # Independent standard normal features
        X = np.random.normal(0, 1, size=(N, d))
        mus = [0 if x[0] <= 0 else kappa for x in X]
        Y = np.random.normal(mus, sigma, size=N)

        cart = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_leaves).fit(X, Y)
        p_vals.append(get_p_values(cart, d)[0])

    frac_correct_indep.append(np.mean([p <= delta for p in p_vals]))


# -----------------------
# Dependent covariates
# -----------------------
for N in x_axis:
    kappa = N ** (-r)
    u = get_appr_quantile(N, delta, d)

    p_vals = []
    for _ in range(M):
        # Correlated Gaussian features
        X = np.random.multivariate_normal(muX, covX, size=N)
        mus = [0 if x[0] <= 0 else kappa for x in X]
        Y = np.random.normal(mus, sigma, size=N)

        cart = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_leaves).fit(X, Y)
        p_vals.append(get_p_values(cart, d)[0])

    frac_correct_dep.append(np.mean([p <= delta for p in p_vals]))


# -----------------------
# Plot
# -----------------------
plt.style.use("default")
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(x_axis, frac_correct_dep, label="Dependent covariates", color="C0")
ax.plot(x_axis, frac_correct_indep, label="Independent covariates", color="C1")
ax.plot(x_axis, kappas, label=r"$\kappa = N^{-r}$", color="C2")
ax.axhline(y=0.95, linestyle="--", color="gray")

ax.set_ylim(0, 1)
ax.set_xlim(min(x_axis), max(x_axis))
ax.set_xlabel("Sample size $N$")
ax.set_ylabel("Fraction of correct detections")
ax.legend()

plt.title("Figure 2: Detection rate vs sample size", fontsize=12)
plt.savefig("Figure2.pdf", bbox_inches="tight", transparent=True)
plt.show()

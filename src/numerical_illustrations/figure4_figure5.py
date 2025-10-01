import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from utils.stats.pvalues import get_cum_p_val
from utils.metrics import mse


def run_and_plot(
    a=1,
    b=1,
    N=500,
    d=10,
    sigma=1.0,
    delta=0.05,
    K=4,
    min_dp_leaf=20,
    seed_train=20,
    seed_test=99,
    title="Figure",
):
    """
    Run the Neufeld regression tree experiment for given (a, b)
    and plot two side-by-side panels:
      Left: Test MSEP (blue) vs Train MSE (orange)
      Right: Cumulative p-values across subtrees
    """
    np.random.seed(seed_train)

    # Gaussian design
    covX = np.eye(d)  # independent features
    muX = np.zeros(d)
    X = np.random.multivariate_normal(muX, covX, size=N)

    # Neufeld regression function
    def mu(x):
        return (
            b
            * (1 if x[0] <= 0 else 0)
            * (1 + a * (1 if x[1] > 0 else 0) + (1 if x[1] * x[2] > 0 else 0))
        )

    Y = np.random.normal([mu(x) for x in X], sigma, N)

    # Fit CART
    cart = DecisionTreeRegressor(
        max_depth=K, min_samples_leaf=min_dp_leaf, random_state=seed_train
    ).fit(X, Y)

    # Validation data
    np.random.seed(seed_test)
    X_test = np.random.normal(0, 1, size=(N, d))
    Y_test = np.random.normal([mu(x) for x in X_test], sigma, N)

    # Subtree pruning
    ccp_alphas = cart.cost_complexity_pruning_path(X, Y)["ccp_alphas"]
    flipped_ccps = np.flip(ccp_alphas)

    mse_train_list, mse_test_list, cum_pvals = [], [], []
    n_leaves = [0]
    max_subtree_cum_pvals05 = 0

    for alpha in flipped_ccps:
        subtree = DecisionTreeRegressor(
            max_depth=K,
            min_samples_leaf=min_dp_leaf,
            random_state=seed_train,
            ccp_alpha=alpha,
        ).fit(X, Y)

        pred_subtree_train = subtree.predict(X)
        pred_subtree_test = subtree.predict(X_test)

        mse_sub = mse(Y, pred_subtree_train)
        msep_sub = mse(Y_test, pred_subtree_test)
        cum_pval = get_cum_p_val(subtree, d)

        n_leaves.append(subtree.get_n_leaves())
        n_diff = n_leaves[-1] - n_leaves[-2]

        mse_train_list.extend([mse_sub] * n_diff)
        mse_test_list.extend([msep_sub] * n_diff)
        cum_pvals.extend([cum_pval] * n_diff)

        if cum_pval <= delta:
            max_subtree_cum_pvals05 = len(mse_test_list) + 1 - n_diff

    # --- Plot ---
    x_axis = range(1, len(mse_test_list) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.style.use("default")

    # Left: errors
    axes[0].plot(x_axis, mse_test_list, label="MSEP (test)", color="blue")
    axes[0].plot(x_axis, mse_train_list, label="MSE (train)", color="orange")
    axes[0].axvline(x=max_subtree_cum_pvals05, linestyle="--", color="blue")
    axes[0].set_xlabel("Number of leaves")
    axes[0].set_ylabel("Error")
    axes[0].legend()

    # Right: cumulative p-values
    axes[1].plot(x_axis, cum_pvals, color="blue")
    axes[1].axvline(x=max_subtree_cum_pvals05, linestyle="--", color="blue")
    axes[1].set_xlabel("Number of leaves")
    axes[1].set_ylabel("Cumulative p-value")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# -----------------------
#   Run Figure 4 (b=1)
# -----------------------
run_and_plot(a=1, b=1, title="Figure 4 (a=1, b=1)")

# -----------------------
#   Run Figure 5 (b=0.5)
# -----------------------
run_and_plot(a=1, b=0.5, title="Figure 5 (a=1, b=0.5)")

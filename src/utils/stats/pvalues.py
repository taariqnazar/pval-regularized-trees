import numpy as np
from math import log, sqrt
from scipy.stats import norm

from sklearn.tree import _tree, DecisionTreeRegressor


def Psi(n: int, u: float, d: int) -> float:
    """
    Your p-value mapping (with gentle guards):

        K = sqrt(u) - (2 log log n)^(-1/2) * (log log log n + log 2)
        p = d * (1 - Phi(K)^(2 log(n/2)))

    Guards:
    - If n <= 16 or u <= 0, return 1.0 (conservative large p).
    - Clamp Phi(K) to (0,1) to avoid 0^exp / 1^exp issues.
    - Clip p to [0,1].
    """
    if n <= 16 or u <= 0:
        return 1.0

    ll = log(log(n))  # log log n
    lll = log(ll)  # log log log n
    denom = sqrt(2.0 * ll)  # (2 log log n)^(1/2)
    K = sqrt(u) - (lll + log(2.0)) / denom

    pow_exp = 2.0 * log(n / 2.0)
    phiK = min(max(norm.cdf(K), 1e-12), 1 - 1e-12)

    p = d * (1.0 - (phiK**pow_exp))
    return float(min(max(p, 0.0), 1.0))


# -------------------- YD statistic & p-values --------------------


def get_YD_statistics(cart: DecisionTreeRegressor) -> np.ndarray:
    T = cart.tree_
    node_count = T.node_count
    YD_statistics = np.full(node_count, -1.0, dtype=float)

    children_left = T.children_left
    children_right = T.children_right
    impurities = T.impurity
    sample_size = T.n_node_samples

    for k in range(node_count):
        if children_left[k] == _tree.TREE_LEAF:
            continue
        n = sample_size[k]
        nL = sample_size[children_left[k]]
        nR = sample_size[children_right[k]]

        S = n * impurities[k]
        SL = nL * impurities[children_left[k]]
        SR = nR * impurities[children_right[k]]

        YD_statistics[k] = 0.0 if S <= 0.0 else (S - SL - SR) / (S / n)

    return YD_statistics


def get_p_values(cart: DecisionTreeRegressor, d: int) -> np.ndarray:
    YD = get_YD_statistics(cart)
    n_node = cart.tree_.n_node_samples
    pvals = np.full_like(YD, -1.0, dtype=float)
    for k, yd in enumerate(YD):
        if yd != -1.0:
            pvals[k] = Psi(int(n_node[k]), float(yd), int(d))
    return pvals


def get_cum_p_val(tree: DecisionTreeRegressor, d: int) -> float:
    """
    Return the cumulative p-value of a tree = sum of all node p-values.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        A fitted regression tree.
    d : int
        Number of features (passed to get_p_values).

    Returns
    -------
    float
        Sum of all valid p-values (ignores sentinel -1 entries).
    """
    p_vals = get_p_values(tree, d)
    mask = p_vals != -1
    return float(np.sum(p_vals[mask]))

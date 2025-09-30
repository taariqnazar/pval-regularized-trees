import numpy as np
from math import log, sqrt
from typing import Optional
from scipy.stats import norm
from sklearn.base import clone
from sklearn.tree import _tree, DecisionTreeRegressor

# -------------------- p-value mapping --------------------


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


# -------------------- helpers --------------------


def _cond_scalar(p: float, thr: float, op: str) -> bool:
    if p < 0:  # leaves carry -1 sentinel
        return False
    if op == ">=":
        return p >= thr
    if op == ">":
        return p > thr
    if op == "<=":
        return p <= thr
    if op == "<":
        return p < thr
    raise ValueError("prune_if must be one of '>=', '>', '<=', '<'")


def _cond_vec(p: np.ndarray, thr: float, op: str) -> np.ndarray:
    if op == ">=":
        return p >= thr
    if op == ">":
        return p > thr
    if op == "<=":
        return p <= thr
    if op == "<":
        return p < thr
    raise ValueError("prune_if must be one of '>=', '>', '<=', '<'")


def _reachable_mask(T) -> np.ndarray:
    """Mark nodes reachable from the root after pruning."""
    reach = np.zeros(T.node_count, dtype=bool)
    stack = [0]
    reach[0] = True
    while stack:
        nid = stack.pop()
        L, R = T.children_left[nid], T.children_right[nid]
        if L == _tree.TREE_LEAF:
            continue
        reach[L] = True
        reach[R] = True
        stack.extend([L, R])
    return reach


# -------------------- train + prune (regression) --------------------


def train_model(
    tree: DecisionTreeRegressor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    d: int,
    prune_if: str = ">=",
    copy: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> DecisionTreeRegressor:
    """
    Fit a regression tree on (X, y), compute YD-based p-values at each split,
    prune top-down by the (p, threshold, prune_if) rule, return the pruned tree.

    Notes:
    - We DO NOT recompute node predictions when pruning; we keep the stored
      node mean so it's consistent with possible sample_weight.
    - Validation only checks REACHABLE internal nodes, using the SAME p-values
      and comparator that were used to prune.
    """
    est = clone(tree) if copy else tree
    est.fit(X, y, sample_weight=sample_weight)
    T = est.tree_

    # 1) Compute p-values ONCE on the fitted tree (these drive pruning)
    pvals_used = get_p_values(est, d=int(d))

    # 2) Stash for optional validation/debug
    est._pvals_used = pvals_used
    est._prune_threshold = float(threshold)
    est._prune_if = prune_if

    # 3) Prune top-down (depth-first)
    def _set_leaf(node_id: int) -> None:
        # keep existing T.value[node_id, 0, 0] (already the node mean, weighted if applicable)
        T.children_left[node_id] = _tree.TREE_LEAF
        T.children_right[node_id] = _tree.TREE_LEAF

    def _prune_subtree(root_id: int) -> None:
        """Mark root_id and all its descendants as leaves."""
        stack = [root_id]
        while stack:
            nid = stack.pop()
            L, R = T.children_left[nid], T.children_right[nid]

            # If this node has children, push them to wipe them too
            if L != _tree.TREE_LEAF:
                stack.append(L)
                stack.append(R)

            # Mark this node as a leaf
            T.children_left[nid] = _tree.TREE_LEAF
            T.children_right[nid] = _tree.TREE_LEAF

            # (Optional but tidy) also reset split metadata to leaf semantics
            # T.feature[nid]   = _tree.TREE_UNDEFINED  # usually -2
            # T.threshold[nid] = -2.0

    def _dfs(node_id: int) -> None:
        L, R = T.children_left[node_id], T.children_right[node_id]
        if L == _tree.TREE_LEAF:
            return

        p = float(pvals_used[node_id])
        if _cond_scalar(p, threshold, prune_if):
            # wipes all children; node_id becomes a leaf
            _prune_subtree(node_id)
            return

        _dfs(L)
        _dfs(R)

    _dfs(0)

    return est

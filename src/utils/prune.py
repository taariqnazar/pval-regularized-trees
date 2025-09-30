from __future__ import annotations

from typing import Iterable
import numpy as np
from sklearn.tree import _tree, DecisionTreeRegressor

__all__ = [
    "cond_scalar",
    "cond_vec",
    "prune_subtree_inplace",
    "prune_topdown_inplace",
    "reachable_mask",
]


# -------------------- comparators --------------------


def cond_scalar(p: float, thr: float, op: str) -> bool:
    """
    Compare a single p-value against threshold.

    Leaves often carry a sentinel (e.g. -1). We treat p < 0 as 'not applicable'.

    op ∈ {'>=', '>', '<=', '<'}
    """
    if p < 0:
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


def cond_vec(p: np.ndarray, thr: float, op: str) -> np.ndarray:
    """
    Vectorized comparator for p-values; shape-preserving boolean mask.
    """
    if op == ">=":
        return p >= thr
    if op == ">":
        return p > thr
    if op == "<=":
        return p <= thr
    if op == "<":
        return p < thr
    raise ValueError("prune_if must be one of '>=', '>', '<=', '<'")


# -------------------- pruning primitives --------------------


def prune_subtree_inplace(tree: DecisionTreeRegressor, root_id: int) -> None:
    """
    Mark `root_id` and all its descendants as leaves in-place.

    Notes
    -----
    - Does NOT recompute node predictions/values; keeps the stored means.
    - For regression trees in sklearn, setting both children to TREE_LEAF makes a node a leaf.
    """
    T = tree.tree_
    stack = [root_id]
    while stack:
        nid = stack.pop()
        L, R = T.children_left[nid], T.children_right[nid]
        if L != _tree.TREE_LEAF:
            stack.extend([L, R])
        T.children_left[nid] = _tree.TREE_LEAF
        T.children_right[nid] = _tree.TREE_LEAF
        # (optional clean-up)
        # T.feature[nid] = _tree.TREE_UNDEFINED
        # T.threshold[nid] = -2.0


def prune_topdown_inplace(
    tree: DecisionTreeRegressor,
    pvals: np.ndarray,
    *,
    threshold: float,
    prune_if: str = ">=",
) -> None:
    """
    Depth-first top-down pruning using node p-values.

    At node k:
      if cond_scalar(pvals[k], threshold, prune_if) is True -> prune entire subtree at k.

    Parameters
    ----------
    tree : fitted DecisionTreeRegressor
    pvals : ndarray, per-node p-values; leaves can be marked with a negative sentinel (e.g. -1)
    threshold : float
    prune_if : str, one of {'>=', '>', '<=', '<'}
    """
    T = tree.tree_

    def _dfs(node_id: int) -> None:
        L, R = T.children_left[node_id], T.children_right[node_id]
        if L == _tree.TREE_LEAF:
            return
        if cond_scalar(float(pvals[node_id]), threshold, prune_if):
            prune_subtree_inplace(tree, node_id)
            return
        _dfs(L)
        _dfs(R)

    _dfs(0)


# -------------------- diagnostics --------------------


def reachable_mask(tree: DecisionTreeRegressor) -> np.ndarray:
    """
    Boolean mask over nodes indicating reachability from the root
    after any in-place pruning (i.e., following child pointers).
    """
    T = tree.tree_
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

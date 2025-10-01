from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree

from utils.stats.pvalues import get_p_values
from utils.prune import cond_scalar


class RecursivePTree:
    """
    A regression tree (CART) pruned using recursive hypothesis testing.

    Each internal split is assigned a p-value via the YD statistic + Psi mapping.
    The tree is then pruned **top-down**, removing subtrees if their p-value
    meets the (threshold, prune_if) condition.

    Parameters (constructor)
    ------------------------
    threshold : float
        The p-value threshold used for pruning.
    prune_if : str, default=">="
        Comparator applied to p-values against threshold.
        Must be one of {">=", ">", "<=", "<"}.
        Example: if prune_if=">=" and threshold=0.05, then
        nodes with p >= 0.05 are pruned.
    **tree_params : Any
        Additional keyword arguments passed directly to
        sklearn.tree.DecisionTreeRegressor
        (e.g., max_depth, min_samples_leaf, random_state, etc.).

    Attributes (after fitting)
    --------------------------
    model : DecisionTreeRegressor
        The fitted and pruned sklearn decision tree.
    d : int
        Number of features in the training data (X.shape[1]).
    pvals_ : ndarray of shape (n_nodes,)
        Vector of per-node p-values (=-1.0 for leaves).
    n_leaves_ : int
        Number of leaves in the final pruned tree.
    """

    def __init__(self, *, threshold: float, prune_if: str = ">=", **tree_params: Any):
        if prune_if not in {">=", ">", "<=", "<"}:
            raise ValueError("prune_if must be one of '>=', '>', '<=', '<'")

        self.threshold = float(threshold)
        self.d: Optional[int] = None
        self.prune_if = prune_if
        self.tree_params: Dict[str, Any] = dict(tree_params)

        # Learned after fit
        self.model: Optional[DecisionTreeRegressor] = None
        self.pvals_: Optional[np.ndarray] = None
        self.n_leaves_: Optional[int] = None

    def fit(self, X, y, *, random_state: int = 0):
        """
        Fit and prune the recursive p-value regression tree.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        random_state : int, default=0
            Ensures determinism if user didn’t already pass a random_state
            in tree_params. Forwarded to sklearn’s DecisionTreeRegressor.

        Returns
        -------
        self : RecursivePTree
            The fitted/pruned model.
        """
        # Ensure determinism if user didn’t provide random_state
        self.tree_params.setdefault("random_state", random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        self.d = int(X.shape[1])

        est = DecisionTreeRegressor(**self.tree_params)
        est.fit(X, y)

        # Compute p-values once on fitted tree
        pvals = get_p_values(est, d=self.d)
        self._prune_topdown_inplace(est, pvals, self.threshold, self.prune_if)

        self.model = est
        self.updated_pvals_ = get_p_values(est, d=self.d)
        self.n_leaves_ = count_leaves(est)
        return self

    def predict(self, X):
        """
        Predict regression targets for new samples.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values from the fitted/pruned tree.
        """
        if self.model is None:
            raise RuntimeError(
                "RecursivePTree is not fitted. Call .fit() first.")
        return self.model.predict(X)

    # --------- internal pruning ---------

    @staticmethod
    def _prune_topdown_inplace(
        est: DecisionTreeRegressor, pvals: np.ndarray, threshold: float, prune_if: str
    ) -> None:
        """
        Modify a fitted DecisionTreeRegressor in place,
        pruning subtrees whose root node meets the condition
        (p, threshold, prune_if).
        """
        T = est.tree_

        def _prune_subtree(root_id: int) -> None:
            """Mark root_id and all its descendants as leaves."""
            stack = [root_id]
            while stack:
                nid = stack.pop()
                L, R = T.children_left[nid], T.children_right[nid]
                if L != _tree.TREE_LEAF:
                    stack.extend([L, R])
                T.children_left[nid] = _tree.TREE_LEAF
                T.children_right[nid] = _tree.TREE_LEAF

        def _dfs(node_id: int) -> None:
            L, R = T.children_left[node_id], T.children_right[node_id]
            if L == _tree.TREE_LEAF:
                return
            p = float(pvals[node_id])
            if cond_scalar(p, threshold, prune_if):
                _prune_subtree(node_id)
                return
            _dfs(L)
            _dfs(R)

        _dfs(0)

    # --------- sklearn-style helpers ---------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return init parameters (threshold, d, prune_if + tree_params)."""
        return {
            "threshold": self.threshold,
            "d": self.d,
            "prune_if": self.prune_if,
            **self.tree_params,
        }

    def set_params(self, **params):
        """Set init parameters (sklearn-style)."""
        if "threshold" in params:
            self.threshold = float(params.pop("threshold"))
        if "d" in params:
            self.d = int(params.pop("d"))
        if "prune_if" in params:
            val = params.pop("prune_if")
            if val not in {">=", ">", "<=", "<"}:
                raise ValueError(
                    "prune_if must be one of '>=', '>', '<=', '<'")
            self.prune_if = val
        self.tree_params.update(params)
        return self

    def __repr__(self) -> str:
        inner = ", ".join(
            [
                f"threshold={self.threshold}",
                f"d={self.d}",
                f"prune_if='{self.prune_if}'",
            ]
            + [f"{k}={v!r}" for k, v in self.tree_params.items()]
        )
        return f"RecursivePTree({inner})"


def count_leaves(est):
    """Count leaves in a (possibly pruned) DecisionTreeRegressor."""
    T = est.tree_
    n_leaves = 0
    stack = [0]
    while stack:
        nid = stack.pop()
        L, R = T.children_left[nid], T.children_right[nid]
        if L == _tree.TREE_LEAF:
            n_leaves += 1
        else:
            stack.extend([L, R])
    return n_leaves

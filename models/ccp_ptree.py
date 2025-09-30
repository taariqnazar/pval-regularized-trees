from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import get_cum_p_val


class CCPPTree:
    """
    Cost-complexity-pruned regression tree where pruning is selected by a
    cumulative p-value rule along the CCP path.

    The tree starts from the most-pruned solution (largest alpha),
    walks toward less-pruned (smaller alpha), and stops as soon as
    the cumulative p-value becomes non-significant.

    Parameters (constructor)
    ------------------------
    significance_level : float, default=0.05
        Threshold for the cumulative p-value pruning rule.
        Smaller values → stricter pruning.
    **tree_params : Any
        Additional keyword arguments passed directly to
        sklearn.tree.DecisionTreeRegressor
        (e.g., max_depth, min_samples_leaf, random_state, etc.).

    Attributes (after fitting)
    --------------------------
    model : DecisionTreeRegressor
        The fitted and pruned sklearn decision tree.
    ccp_alpha_ : float
        The alpha (complexity parameter) chosen along the CCP path.
    p_value_ : float
        The cumulative p-value corresponding to the chosen tree.
    n_leaves_ : int
        Number of leaves in the chosen pruned tree.
    _path_ : List[Tuple[float, float]]
        Sequence of (ccp_alpha, p_value) pairs checked during pruning.
    """

    def __init__(self, *, significance_level: float = 0.05, **tree_params: Any):
        if not (0 < significance_level < 1):
            raise ValueError("significance_level must be in (0, 1).")
        self.significance_level = float(significance_level)

        # Params forwarded to sklearn's DecisionTreeRegressor
        self.tree_params: Dict[str, Any] = dict(tree_params)

        # Learned after fit
        self.model: Optional[DecisionTreeRegressor] = None
        self.ccp_alpha_: Optional[float] = None
        self.p_value_: Optional[float] = None
        self.n_leaves_: Optional[int] = None

        # Diagnostics: list of (alpha, p_value) checked
        self._path_: List[Tuple[float, float]] = []

    # ----- internal helper -----

    def _fit_with_alpha(self, X, y, alpha: float) -> DecisionTreeRegressor:
        """Fit a decision tree with a given CCP alpha."""
        est = DecisionTreeRegressor(ccp_alpha=float(alpha), **self.tree_params)
        est.fit(X, y)
        return est

    # ----- main training -----

    def fit(self, X, y, *, random_state: int = 0):
        """
        Fit/prune a regression tree by cost-complexity + cumulative p-value rule.

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
        self : CCPPTree
            The fitted/pruned model.
        """
        # Ensure determinism if user didn’t provide random_state
        self.tree_params.setdefault("random_state", random_state)

        X = np.asarray(X)
        y = np.asarray(y)
        d = X.shape[1]

        # Probe a tree to get the pruning path
        probe = DecisionTreeRegressor(**self.tree_params)
        path = probe.cost_complexity_pruning_path(X, y)
        alphas = np.asarray(path.ccp_alphas, dtype=float)

        # If path is empty → fallback: just fit a plain tree
        if alphas.size == 0:
            self.model = DecisionTreeRegressor(**self.tree_params).fit(X, y)
            self.ccp_alpha_ = float(getattr(self.model, "ccp_alpha", 0.0))
            self.p_value_ = get_cum_p_val(self.model, d)
            self.n_leaves_ = int(self.model.tree_.n_leaves)
            self._path_ = []
            return self

        # Traverse alphas from most-pruned (largest) to least-pruned (smallest)
        alphas = np.unique(alphas)[::-1]

        prev = self._fit_with_alpha(X, y, alphas[0])
        prev_p = get_cum_p_val(prev, d)
        self._path_ = [(float(alphas[0]), float(prev_p))]

        best = prev
        best_alpha = float(alphas[0])
        best_p = float(prev_p)

        for a in alphas[1:]:
            curr = self._fit_with_alpha(X, y, a)
            p = get_cum_p_val(curr, d)
            self._path_.append((float(a), float(p)))

            # stop: return previous tree once p-value becomes non-significant
            if p > self.significance_level:
                break
            best = curr
            best_alpha = float(a)
            best_p = float(p)

        # store chosen model
        self.model = best
        self.ccp_alpha_ = best_alpha
        self.p_value_ = best_p
        self.n_leaves_ = int(best.tree_.n_leaves)
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
            raise RuntimeError("CCPPTree is not fitted. Call .fit() first.")
        return self.model.predict(X)

    # ----- convenience -----

    def get_staged_complexity(self) -> List[int]:
        """
        Return leaf counts for staged learners.

        Since CCPPTree is a single tree (not boosting), this just returns
        a list with one entry: [n_leaves_].
        """
        if self.model is None:
            raise RuntimeError("CCPPTree is not fitted. Call .fit() first.")
        return [int(self.model.tree_.n_leaves)]

    def path_diagnostics(self) -> List[Tuple[float, float]]:
        """
        Return diagnostic information from pruning.

        Returns
        -------
        path : list of (ccp_alpha, p_value)
            Pairs of CCP complexity parameter and p-value checked during pruning.
        """
        return list(self._path_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return init parameters (significance_level + tree_params)."""
        return {"significance_level": self.significance_level, **self.tree_params}

    def set_params(self, **params):
        """Set init parameters (sklearn-style)."""
        if "significance_level" in params:
            self.significance_level = float(params.pop("significance_level"))
        self.tree_params.update(params)
        return self

    def __repr__(self) -> str:
        inner = ", ".join(
            [f"significance_level={self.significance_level}"]
            + [f"{k}={v!r}" for k, v in self.tree_params.items()]
        )
        return f"CCPPTree({inner})"

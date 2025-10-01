from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np

from models.ccp_ptree import CCPPTree


class CCPPTreeBoosting:
    """
    Gradient boosting (squared loss) using CCPPTree as base learners.

    The boosting procedure:
    - Start with F_0(x) = 0 (no prediction).
    - At each stage t:
        1. Compute residuals r = y - F_{t-1}(x).
        2. Fit a CCPPTree to r.
        3. Update F_t(x) = F_{t-1}(x) + learning_rate * tree_t(x).
    - Stop early if a tree degenerates to a single leaf.

    Parameters (constructor)
    ------------------------
    learning_rate : float, default=0.1
        Shrinkage applied to each stage’s predictions.
    max_estimators : int, default=500
        Maximum number of boosting stages (trees).
    **base_params : Any
        Additional keyword arguments forwarded to CCPPTree,
        e.g. significance_level, max_depth, min_samples_leaf, etc.

    Attributes (after fitting)
    --------------------------
    estimators_ : List[CCPPTree]
        The sequence of fitted base learners.
    n_estimators_ : int
        Actual number of stages fitted (<= max_estimators).
    train_mse_ : List[float]
        Training mean squared error after each stage.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_estimators: int = 500,
        **base_params: Any,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if max_estimators < 1:
            raise ValueError("max_estimators must be >= 1")
        self.learning_rate = float(learning_rate)
        self.max_estimators = int(max_estimators)
        self.base_params: Dict[str, Any] = dict(base_params)

        # Learned after fitting
        self.estimators_: List[CCPPTree] = []
        self.n_estimators_: int = 0
        self.train_mse_: List[float] = []

    def fit(self, X, y, *, random_state: int = 0):
        """
        Fit a boosting ensemble of CCPPTree learners.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        random_state : int, default=0
            Random seed forwarded to each CCPPTree for reproducibility.

        Returns
        -------
        self : CCPPTreeBoosting
            The fitted boosting model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.estimators_.clear()
        self.train_mse_.clear()

        # initial prediction F_0(x) = 0
        F = np.zeros(X.shape[0], dtype=float)

        for _ in range(self.max_estimators):
            # residuals (squared loss)
            r = y - F

            # base learner
            base = CCPPTree(**self.base_params)
            base.fit(X, r, random_state=random_state)

            self.estimators_.append(base)

            # update ensemble prediction
            F += self.learning_rate * base.predict(X)
            self.train_mse_.append(float(np.mean((y - F) ** 2)))

            # early-stop if base learner collapses to a single leaf
            if base.n_leaves_ == 1:
                break

        self.n_estimators_ = len(self.estimators_)
        return self

    def predict(self, X, *, sub_tree: Optional[int] = None):
        """
        Predict with the boosting ensemble.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)
        sub_tree : int or None, default=None
            If None, use all fitted trees.
            If int, use only the first `sub_tree` trees.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Ensemble predictions.
        """
        if not self.estimators_:
            raise RuntimeError(
                "CCPPTreeBoosting is not fitted. Call .fit() first.")
        X = np.asarray(X)
        T = (
            self.n_estimators_
            if sub_tree is None
            else max(0, min(sub_tree, self.n_estimators_))
        )
        F = np.zeros(X.shape[0], dtype=float)
        for base in self.estimators_[:T]:
            F += self.learning_rate * base.predict(X)
        return F

    def staged_predict(self, X):
        """
        Yield predictions after each stage (1..n_estimators_).

        Useful for plotting learning curves.
        """
        if not self.estimators_:
            raise RuntimeError(
                "CCPPTreeBoosting is not fitted. Call .fit() first.")
        X = np.asarray(X)
        F = np.zeros(X.shape[0], dtype=float)
        for base in self.estimators_:
            F += self.learning_rate * base.predict(X)
            yield F.copy()

    def get_staged_complexity(self) -> List[int]:
        """
        Return number of leaves per fitted base learner.
        """
        if not self.estimators_:
            raise RuntimeError(
                "CCPPTreeBoosting is not fitted. Call .fit() first.")
        return [int(est.model.tree_.n_leaves) for est in self.estimators_]

    # ----- sklearn-style helpers -----

    def get_params(self, deep: bool = True):
        """Return init parameters (learning_rate, max_estimators + base_params)."""
        return {
            "learning_rate": self.learning_rate,
            "max_estimators": self.max_estimators,
            **self.base_params,
        }

    def set_params(self, **params):
        """Set init parameters (sklearn-style)."""
        if "learning_rate" in params:
            self.learning_rate = float(params.pop("learning_rate"))
        if "max_estimators" in params:
            self.max_estimators = int(params.pop("max_estimators"))
        self.base_params.update(params)
        return self

    def __repr__(self) -> str:
        inner = ", ".join(
            [
                f"learning_rate={self.learning_rate}",
                f"max_estimators={self.max_estimators}",
            ]
            + [f"{k}={v!r}" for k, v in self.base_params.items()]
        )
        return f"CCPPTreeBoosting({inner})"

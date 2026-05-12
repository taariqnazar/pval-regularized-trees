from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np

from .recursive_psum import RecursivePSumTree


class RecursivePSumBoosting:
    """
    Gradient boosting (squared loss) using RecursivePSumTree as base learners.

    Parameters (constructor)
    ------------------------
    learning_rate : float, default=0.1
        Shrinkage factor applied to each stage’s predictions.
    max_estimators : int, default=500
        Maximum number of boosting stages.
    **base_params : Any
        Additional keyword arguments passed to RecursivePSumTree,
        e.g. threshold, prune_if, max_depth, min_samples_leaf, etc.

    Attributes (after fit)
    ----------------------
    estimators_ : List[RecursivePSumTree]
        Sequence of fitted base learners.
    n_estimators_ : int
        Number of stages actually fitted (<= max_estimators).
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
        self.estimators_: List[RecursivePSumTree] = []
        self.n_estimators_: int = 0
        self.train_mse_: List[float] = []

    def fit(self, X, y, *, random_state: int = 0):
        """
        Fit a boosting ensemble of RecursivePSumTree learners.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        random_state : int, default=0
            Base seed for reproducibility.
            Each stage uses random_state + t (stage index) so splits differ.

        Returns
        -------
        self : RecursivePSumBoosting
            The fitted boosting model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.estimators_.clear()
        self.train_mse_.clear()

        # initial prediction F_0(x) = 0
        F = np.zeros(X.shape[0], dtype=float)

        for t in range(self.max_estimators):
            # residuals (gradient for squared loss)
            r = y - F

            # fit a RecursivePSumTree to residuals
            base = RecursivePSumTree(**self.base_params)
            base.fit(X, r, random_state=random_state + t)

            self.estimators_.append(base)

            # update ensemble prediction
            F += self.learning_rate * base.predict(X)
            self.train_mse_.append(float(np.mean((y - F) ** 2)))

            # early-stop if tree degenerates to a stump
            if base.n_leaves_ == 1:
                break

        self.n_estimators_ = len(self.estimators_)
        return self

    def predict(self, X, *, sub_tree: Optional[int] = None):
        """
        Predict with the fitted boosting ensemble.

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
                "RecursivePSumBoosting is not fitted. Call .fit() first."
            )
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

        Useful for diagnostics and learning curves.
        """
        if not self.estimators_:
            raise RuntimeError(
                "RecursivePSumBoosting is not fitted. Call .fit() first."
            )
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
                "RecursivePSumBoosting is not fitted. Call .fit() first."
            )
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
        return f"RecursivePSumBoosting({inner})"

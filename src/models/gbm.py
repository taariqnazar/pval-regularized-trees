from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class GBM:
    """
    Gradient Boosting wrapper with optional CV-based early stopping.

    If k_folds >= 2, the number of boosting stages used is chosen by cross-validation:
      - Train a GradientBoostingRegressor up to `max_estimators`.
      - Track validation error using staged predictions.
      - Pick the number of stages that minimizes MSE on each fold.
      - Use the mean (rounded) across folds as the final number of stages.

    If k_folds = 1 (default), the model simply trains for `max_estimators` stages.

    Parameters (constructor)
    ------------------------
    k_folds : int, default=1
        Number of cross-validation folds to select the number of stages.
        If 1, disables CV and uses `max_estimators` directly.
    max_estimators : int, default=500
        Upper bound on the number of boosting stages to train.
    **parameters : Any
        Additional keyword arguments passed directly to
        sklearn.ensemble.GradientBoostingRegressor
        (e.g., learning_rate, max_depth, subsample, etc.).

    Attributes (after fitting)
    --------------------------
    model : GradientBoostingRegressor
        The fitted sklearn GradientBoostingRegressor.
    n_estimators_ : int
        The number of stages actually chosen (via CV or equal to `max_estimators` when k_folds=1).
    """

    def __init__(
        self,
        *,
        k_folds: int = 1,
        max_estimators: int = 500,
        **parameters,
    ):
        if k_folds < 1:
            raise ValueError("k_folds must be a positive integer")
        self.k_folds = int(k_folds)
        self.max_estimators = int(max_estimators)
        self.parameters = dict(parameters)

        self.model: Optional[GradientBoostingRegressor] = None
        self.n_estimators_: Optional[int] = None  # chosen number of stages after fit

    def _best_iters_cv(
        self, X: np.ndarray, y: np.ndarray, random_state: int
    ) -> Sequence[int]:
        """
        Perform k-fold CV to find the best number of boosting stages (1-based).

        Returns
        -------
        best_iters : list of int
            The optimal number of stages found on each fold (1..max_estimators).
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=random_state)
        best_iters = []
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            model = GradientBoostingRegressor(
                n_estimators=self.max_estimators,  # sklearn's arg
                random_state=random_state,
                **self.parameters,
            )
            model.fit(X_tr, y_tr)

            best_mse = float("inf")
            best_iter = 1
            for t, y_pred in enumerate(model.staged_predict(X_te), start=1):
                mse = mean_squared_error(y_te, y_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_iter = t
            best_iters.append(best_iter)
        return best_iters

    def fit(self, X, y, random_state: int = 0):
        """
        Fit the Gradient Boosting model, optionally selecting the number of stages by CV.

        Inputs
        ------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        random_state : int, default=0
            Seed for reproducibility, passed to sklearn’s GBR.

        Returns
        -------
        self : GBM
            The fitted wrapper object.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.k_folds >= 2:
            best_iters = self._best_iters_cv(X, y, random_state)
            # Use rounded mean; ensure at least 1
            n_best = max(1, int(round(sum(best_iters) / len(best_iters))))
        else:
            n_best = self.max_estimators

        self.n_estimators_ = n_best
        self.model = GradientBoostingRegressor(
            n_estimators=n_best,  # sklearn's arg
            random_state=random_state,
            **self.parameters,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict regression targets for new samples."""
        if self.model is None:
            raise RuntimeError("GBM is not fitted. Call .fit() first.")
        return self.model.predict(X)

    def staged_predict(self, X):
        """
        Yield predictions after each stage (1..n_estimators_).
        """
        if self.model is None:
            raise RuntimeError("GBM is not fitted. Call .fit() first.")
        return self.model.staged_predict(X)

    def get_staged_complexity(self):
        """
        Return number of leaves per weak learner.

        For single-output GBR, estimators_ has shape (n_estimators_, 1).

        Returns
        -------
        list of int
            Leaf counts for each boosting stage.
        """
        if self.model is None:
            raise RuntimeError("GBM is not fitted. Call .fit() first.")
        return [est[0].tree_.n_leaves for est in self.model.estimators_]

    # ----- sklearn-style helpers -----

    def get_params(self, deep: bool = True):
        """Return init parameters (wrapper + GBR)."""
        return {
            "k_folds": self.k_folds,
            "max_estimators": self.max_estimators,
            **self.parameters,
        }

    def set_params(self, **params):
        """Set init parameters (wrapper + GBR)."""
        if "k_folds" in params:
            self.k_folds = int(params.pop("k_folds"))
        if "max_estimators" in params:
            self.max_estimators = int(params.pop("max_estimators"))
        self.parameters.update(params)
        return self

    def __repr__(self) -> str:
        inner = ", ".join(
            [f"k_folds={self.k_folds}", f"max_estimators={self.max_estimators}"]
            + [f"{k}={v!r}" for k, v in self.parameters.items()]
        )
        return f"GBM({inner})"

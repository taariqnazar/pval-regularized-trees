from __future__ import annotations
import numpy as np

from .gbm import GBM
from .psum import PSumTree


class DoubleRegression:
    """
    Simple two-stage regressor.

    Stage 1: fit GBM on (X, y)
    Stage 2: fit PSumTree on pseudo-covariate = GBM.predict(X).reshape(-1, 1)

    Parameters
    ----------
    ordering_regressor_params : dict
        Keyword args for GBM(...)
    prediction_regressor_params : dict
        Keyword args for PSumTree(...)
    """

    def __init__(
        self,
        ordering_regressor_params: dict,
        prediction_regressor_params: dict,
    ):
        self.ordering_regressor_params = dict(ordering_regressor_params or {})
        self.prediction_regressor_params = dict(
            prediction_regressor_params or {})

        # Will be created at init so __repr__ looks nice; re-fit on .fit
        self.ordering_regressor_ = GBM(**self.ordering_regressor_params)
        self.prediction_regressor_ = PSumTree(
            **self.prediction_regressor_params)

        self.n_leaves_ = None

    def fit(self, X, y, random_state: int = 0):
        X = np.asarray(X)
        y = np.asarray(y)

        # (re)initialize in case params changed
        self.ordering_regressor_ = GBM(**self.ordering_regressor_params)
        self.prediction_regressor_ = PSumTree(
            **self.prediction_regressor_params)

        # Stage 1
        self.ordering_regressor_.fit(X, y, random_state=random_state)

        # Pseudo-covariate from stage 1 (in-sample, as requested)
        pseudo_covariates = self.ordering_regressor_.predict(X).reshape(-1, 1)

        # Stage 2
        self.prediction_regressor_.fit(
            pseudo_covariates, y, random_state=random_state)

        self.n_leaves_ = int(self.prediction_regressor_.n_leaves_)
        return self

    def predict(self, X):
        X = np.asarray(X)
        pseudo_covariates = self.ordering_regressor_.predict(X).reshape(-1, 1)
        return self.prediction_regressor_.predict(pseudo_covariates)

    # Optional: sklearn-style helpers so config tweaks are easy
    def get_params(self, deep: bool = True):
        return {
            "ordering_regressor_params": dict(self.ordering_regressor_params),
            "prediction_regressor_params": dict(self.prediction_regressor_params),
        }

    def set_params(self, **params):
        if "ordering_regressor_params" in params:
            self.ordering_regressor_params = dict(
                params["ordering_regressor_params"])
        if "prediction_regressor_params" in params:
            self.prediction_regressor_params = dict(
                params["prediction_regressor_params"]
            )
        # Rebuild inner estimators so __repr__ is accurate after set_params
        self.ordering_regressor_ = GBM(**self.ordering_regressor_params)
        self.prediction_regressor_ = PSumTree(
            **self.prediction_regressor_params)
        return self

    def __repr__(self) -> str:
        return (
            "DoubleRegression("
            f"ordering_regressor={self.ordering_regressor_}, "
            f"prediction_regressor={self.prediction_regressor_}"
            ")"
        )

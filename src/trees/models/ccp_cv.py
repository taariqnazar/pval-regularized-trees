"""sklearn cost-complexity-pruned regression tree with 1-SE CV rule.

Reference: paper §4.2.2 / §4.3 — used as the non-statistical CV baseline.
The 1-SE rule picks the smallest tree (largest ccp_alpha) whose mean CV-MSE
is within one standard error of the minimum.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


class CCPCV:
    """sklearn DecisionTreeRegressor + CCP path + 1-SE CV rule baseline."""

    def __init__(self, n_folds: int = 5, min_samples_leaf: int = 20,
                 rule: str = "1se", **tree_params):
        self.n_folds = n_folds
        self.min_samples_leaf = min_samples_leaf
        self.rule = rule
        self.tree_params = tree_params

    def fit(self, X, y, random_state: int = 0):
        X = np.asarray(X)
        y = np.asarray(y)
        base = DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state,
            **self.tree_params,
        ).fit(X, y)
        ccp_alphas = np.unique(
            base.cost_complexity_pruning_path(X, y)["ccp_alphas"]
        )
        if len(ccp_alphas) == 0:
            self.model_ = base
            self.ccp_alpha_ = 0.0
            self.n_leaves_ = int((base.tree_.children_left == -1).sum())
            return self

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)
        fold_mse = np.zeros((len(ccp_alphas), self.n_folds))
        for f_idx, (tr, va) in enumerate(kf.split(X)):
            for k, alpha in enumerate(ccp_alphas):
                t = DecisionTreeRegressor(
                    min_samples_leaf=self.min_samples_leaf,
                    ccp_alpha=alpha,
                    random_state=random_state,
                    **self.tree_params,
                ).fit(X[tr], y[tr])
                fold_mse[k, f_idx] = float(np.mean((t.predict(X[va]) - y[va]) ** 2))

        mean = fold_mse.mean(axis=1)
        if self.rule == "min":
            k_pick = int(np.argmin(mean))
        elif self.rule == "1se":
            sem = fold_mse.std(axis=1, ddof=1) / np.sqrt(self.n_folds)
            k_min = int(np.argmin(mean))
            threshold = mean[k_min] + sem[k_min]
            eligible = np.where(mean <= threshold)[0]
            k_pick = int(eligible.max())
        else:
            raise ValueError(f"rule must be '1se' or 'min', got {self.rule!r}")

        chosen = float(ccp_alphas[k_pick])
        final = DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            ccp_alpha=chosen,
            random_state=random_state,
            **self.tree_params,
        ).fit(X, y)
        self.model_ = final
        self.ccp_alpha_ = chosen
        self.n_leaves_ = int((final.tree_.children_left == -1).sum())
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

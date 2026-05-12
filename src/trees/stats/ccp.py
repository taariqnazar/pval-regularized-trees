"""CCP chain helpers shared across PSumTree, GSellStop, and exp_4_3."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def build_ccp_chain(X, y, min_samples_leaf: int, random_state: int = 0
                    ) -> List[DecisionTreeRegressor]:
    """Return the nested CCP subtree chain in root-first order."""
    base = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf, random_state=random_state
    ).fit(X, y)
    ccp_alphas = base.cost_complexity_pruning_path(X, y)["ccp_alphas"]
    if len(ccp_alphas) == 0:
        return [base]
    return [
        DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            ccp_alpha=float(a),
        ).fit(X, y)
        for a in np.flip(ccp_alphas)
    ]

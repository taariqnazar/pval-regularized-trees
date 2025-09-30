from sklearn.base import clone
from typing import Optional
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from utils import get_cum_p_val


def train_regression_tree(X, y, **kwargs):
    tree = DecisionTreeRegressor(**kwargs)
    tree.fit(X, y)
    return tree


def ccp_pval_regularised_tree(X, y, significance_level, **kwargs):
    rt = DecisionTreeRegressor(**kwargs)
    d = X.shape[1]

    path = rt.cost_complexity_pruning_path(X, y)
    try:
        ccps = path.ccp_alphas[1:]
        prev_tree = train_regression_tree(X, y, ccp_alpha=ccps[-1], **kwargs)
        for ccp in ccps[::-1][1:]:
            tree = train_regression_tree(X, y, ccp_alpha=ccp, **kwargs)
            p_value = get_cum_p_val(tree, d)
            if p_value > significance_level:
                return prev_tree
            prev_tree = tree

        return tree
    except:
        ccp = path.ccp_alphas[-1]
        tree = train_regression_tree(X, y, ccp_alpha=ccp, **kwargs)
        return tree


def train_model(
    tree: DecisionTreeRegressor,
    X,
    y,
    *,
    significance_level: float,
    sample_weight: Optional[np.ndarray] = None,
) -> DecisionTreeRegressor:
    """
    Fit/prune a regression tree by cost-complexity + cumulative p-value rule.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        An *unfitted* instance with your desired hyperparameters already set
        (e.g., max_depth, min_samples_leaf, random_state, etc.).
        This function will clone it internally for each alpha tried.
    X, y : training data
    significance_level : float
        Threshold to compare `get_cum_p_val(fitted_tree, d)` against.
    sample_weight : optional, passed to .fit

    Returns
    -------
    Fitted (and possibly pruned) DecisionTreeRegressor.
    """
    # Use a probe clone (no prior fit needed) to get the CCP path
    probe = clone(tree)
    path = probe.cost_complexity_pruning_path(
        X, y, sample_weight=sample_weight)
    ccps = np.asarray(path.ccp_alphas, dtype=float)

    # Fallbacks for degenerate paths
    if ccps.size == 0:
        est = clone(tree)
        est.fit(X, y, sample_weight=sample_weight)
        return est
    ccps = np.unique(ccps)
    d = X.shape[1]

    def _fit_with_alpha(alpha: float) -> DecisionTreeRegressor:
        est = clone(tree)
        est.set_params(ccp_alpha=float(alpha))
        est.fit(X, y, sample_weight=sample_weight)
        return est

    # Traverse from most-pruned (largest alpha) toward less-pruned (smaller alpha)
    alphas_desc = ccps[::-1]

    prev_tree = _fit_with_alpha(alphas_desc[0])

    for alpha in alphas_desc[1:]:
        curr_tree = _fit_with_alpha(alpha)
        p_value = get_cum_p_val(curr_tree, d)
        # Stop *as soon as* model becomes non-significant at your threshold
        if p_value > significance_level:
            return prev_tree
        prev_tree = curr_tree

    # If never crossed the threshold, return the least-pruned we tried
    return prev_tree

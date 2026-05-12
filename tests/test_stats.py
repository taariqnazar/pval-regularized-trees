import numpy as np
from sklearn.tree import DecisionTreeRegressor

from trees.stats import Psi, get_YD_statistics, get_p_values, get_cum_p_val
from trees.stats import mse, rmse


def test_psi_returns_probability():
    p = Psi(n=100, u=10.0, d=5)
    assert 0.0 <= p <= 1.0


def test_psi_small_n_is_conservative():
    assert Psi(n=10, u=20.0, d=5) == 1.0


def test_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert mse(y_true, y_pred) == 1.0 / 3.0
    assert abs(rmse(y_true, y_pred) - (1.0 / 3.0) ** 0.5) < 1e-12


def test_yd_and_pvalues_on_fitted_tree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + 0.3 * rng.normal(size=200)
    cart = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
    yd = get_YD_statistics(cart)
    pv = get_p_values(cart, d=4)
    assert len(yd) == cart.tree_.node_count
    assert len(pv) == cart.tree_.node_count
    cum = get_cum_p_val(cart, d=4)
    assert cum >= 0.0

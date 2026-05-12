import numpy as np

from trees.data import load_dataset
from trees.models import PSumTree, RecursivePSumTree, GBM


def _data():
    X, y, _ = load_dataset("linear_model", n=200, p=5, sigma=1.0, random_state=0)
    return X, y


def test_psum_tree_fits_and_predicts():
    X, y = _data()
    m = PSumTree(significance_level=0.05, max_depth=4, min_samples_leaf=10)
    m.fit(X, y, random_state=0)
    pred = m.predict(X)
    assert pred.shape == y.shape


def test_cappelli_stp_fits_and_predicts():
    from trees.models import CappelliSTP
    X, y = _data()
    m = CappelliSTP(significance_level=0.05, train_frac=0.7,
                    min_samples_leaf=10, max_depth=4)
    m.fit(X, y, random_state=0)
    pred = m.predict(X)
    assert pred.shape == y.shape
    assert m.n_leaves_ >= 1


def test_recursive_psum_tree_fits_and_predicts():
    X, y = _data()
    m = RecursivePSumTree(threshold=0.05, prune_if=">=", max_depth=4, min_samples_leaf=10)
    m.fit(X, y, random_state=0)
    pred = m.predict(X)
    assert pred.shape == y.shape


def test_gbm_fits_and_predicts():
    X, y = _data()
    m = GBM(k_folds=1, max_estimators=10, max_depth=2, learning_rate=0.1)
    m.fit(X, y, random_state=0)
    pred = m.predict(X)
    assert pred.shape == y.shape

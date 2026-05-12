"""Cappelli–Mola–Siciliano (2001) F-test-based pruning of CART trees.

Reference: paper §4.2.1, ref [14]. The training data is split 70/30 into a
'grow' slice and an independent 'prune' slice; an F-test is evaluated at
every internal node of T_max using statistics computed on the prune slice;
nodes whose F p-value exceeds delta are pruned.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy.stats import f as f_distribution
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def _mse(values: Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean((arr - arr.mean()) ** 2))


def _test_impurities(cart: DecisionTreeRegressor, X_test, y_test):
    """Push the prune slice down T_max; return per-node (impurity, n)."""
    n_nodes = cart.tree_.node_count
    node_indicator = cart.decision_path(X_test)
    n_node_samples_test = np.asarray(node_indicator.sum(axis=0)).ravel()
    node_to_labels: dict[int, list[float]] = defaultdict(list)
    for i in range(X_test.shape[0]):
        nodes = node_indicator.indices[
            node_indicator.indptr[i] : node_indicator.indptr[i + 1]
        ]
        for node in nodes:
            node_to_labels[node].append(y_test[i])
    impurity_test = np.zeros(n_nodes)
    for node in range(n_nodes):
        impurity_test[node] = _mse(node_to_labels[node])
    return impurity_test, n_node_samples_test


def _iter_internal(cart, start):
    stack = [start]
    while stack:
        nid = stack.pop()
        left, right = cart.tree_.children_left[nid], cart.tree_.children_right[nid]
        if left != -1:
            yield nid
            stack.append(right)
            stack.append(left)


def _f_pvalues(cart, impurity_test, n_node_samples_test):
    """One-shot F-test at every internal node of the original T_max."""
    n_nodes = cart.tree_.node_count
    children_left = cart.tree_.children_left
    children_right = cart.tree_.children_right
    p_vals = np.full(n_nodes, -1.0, dtype=float)
    for k in range(n_nodes):
        if children_left[k] == -1:
            continue
        n = n_node_samples_test[k]
        S_left = n_node_samples_test[children_left[k]] * impurity_test[children_left[k]]
        S_right = n_node_samples_test[children_right[k]] * impurity_test[children_right[k]]
        WSS = S_left + S_right
        BSS = 0.0
        L = 0
        for inner in _iter_internal(cart, k):
            l_child = children_left[inner]
            r_child = children_right[inner]
            if l_child == -1:
                continue
            L += 1
            n_l = n_node_samples_test[l_child]
            n_r = n_node_samples_test[r_child]
            BSS += n_node_samples_test[inner] * impurity_test[inner] - (
                n_l * impurity_test[l_child] + n_r * impurity_test[r_child]
            )
        T_size = L + 1
        if T_size <= 1 or WSS <= 0 or n - T_size <= 0:
            p_vals[k] = 1.0
            continue
        F = (BSS / WSS) * ((n - T_size) / L)
        p_vals[k] = float(1.0 - f_distribution.cdf(F, L, n - T_size))
    return p_vals


def _prune_in_place(cart: DecisionTreeRegressor, nodes: Sequence[int]) -> None:
    """Mutate cart.tree_ to make `nodes` leaves (drop their children)."""
    stack = list(nodes)
    while stack:
        nid = stack.pop()
        left = cart.tree_.children_left[nid]
        right = cart.tree_.children_right[nid]
        if left != -1:
            stack.append(left)
            cart.tree_.children_left[nid] = -1
        if right != -1:
            stack.append(right)
            cart.tree_.children_right[nid] = -1


class CappelliSTP:
    """Cappelli STP (single-tree pruning) via F-test on a held-out slice."""

    def __init__(self, significance_level: float = 0.05, train_frac: float = 0.7,
                 **tree_params):
        self.significance_level = significance_level
        self.train_frac = train_frac
        self.tree_params = tree_params

    def fit(self, X, y, random_state: int = 0):
        X = np.asarray(X)
        y = np.asarray(y)
        Xg, Xp, yg, yp = train_test_split(
            X, y, train_size=self.train_frac, random_state=random_state
        )
        cart = DecisionTreeRegressor(random_state=random_state, **self.tree_params).fit(Xg, yg)
        impurity_test, n_test = _test_impurities(cart, Xp, yp)
        p_vals = _f_pvalues(cart, impurity_test, n_test)
        to_prune = [k for k, p in enumerate(p_vals) if p > self.significance_level]
        _prune_in_place(cart, to_prune)
        self.model_ = cart
        self.p_values_ = p_vals
        self.n_leaves_ = int((cart.tree_.children_left == -1).sum())
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

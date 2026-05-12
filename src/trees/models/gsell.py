"""G'Sell ForwardStop / StrongStop sequential FDR stopping along the CCP path.

Reference: G'Sell, Wager, Chouldechova, Tibshirani, JRSS-B 2016 (ref [13]
in the paper). Two p-value sequences are supported:

* strategy="sum": one hypothesis per CCP step, p_k = cumulative Psi-sum
  over internal nodes of T_k.
* strategy="wl": one hypothesis per internal node of T_max, ordered by
  CCP birth (weakest-link order).

These methods are exploratory and are not reported in the revised paper;
they are kept available for users who want to compare.
"""
from __future__ import annotations

import copy
from typing import List, Literal

import numpy as np

from trees.stats.ccp import build_ccp_chain
from trees.stats.pvalues import get_cum_p_val, get_p_values
from trees.stats.prune import prune_subtree_inplace


def _internal_nodes_set(subtree) -> set[int]:
    n = subtree.tree_.node_count
    left = subtree.tree_.children_left
    return {k for k in range(n) if left[k] != -1}


def _sanitize(p_seq: List[float]) -> np.ndarray:
    p = np.asarray(p_seq, dtype=float)
    return np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)


def _forward_stop_k(p_seq, alpha: float, eps: float = 1e-12) -> int:
    if len(p_seq) == 0:
        return 0
    p = np.clip(_sanitize(p_seq), 0.0, 1.0 - eps)
    y = -np.log(1.0 - p)
    running_mean = np.cumsum(y) / np.arange(1, len(y) + 1)
    elig = np.where(running_mean <= alpha)[0]
    return 0 if len(elig) == 0 else int(elig.max()) + 1


def _strong_stop_k(p_seq, alpha: float, eps: float = 1e-12) -> int:
    if len(p_seq) == 0:
        return 0
    p = np.clip(_sanitize(p_seq), eps, 1.0)
    m = len(p)
    idx = np.arange(1, m + 1)
    weights = np.log(p) / idx
    tail = np.cumsum(weights[::-1])[::-1]
    q = np.exp(tail)
    thresh = alpha * idx / m
    elig = np.where(q <= thresh)[0]
    return 0 if len(elig) == 0 else int(elig.max()) + 1


def _strategy_sum(chain, d: int, alpha: float, rule: str):
    if not chain:
        return None
    subtrees, p_seq = [], []
    for t in chain:
        if _internal_nodes_set(t):
            subtrees.append(t)
            p_seq.append(get_cum_p_val(t, d))
    if not subtrees:
        return chain[-1]
    k_hat = (_forward_stop_k if rule == "forward" else _strong_stop_k)(p_seq, alpha)
    return chain[0] if k_hat == 0 else subtrees[k_hat - 1]


def _strategy_wl(chain, d: int, alpha: float, rule: str):
    T_max = chain[-1]
    all_internals = _internal_nodes_set(T_max)
    if not all_internals:
        return copy.deepcopy(T_max)
    ordered, seen = [], set()
    for sub in chain:
        new_nodes = _internal_nodes_set(sub) - seen
        for node in sorted(new_nodes):
            ordered.append(node)
        seen.update(new_nodes)
    p_vals_all = get_p_values(T_max, d)
    p_seq = [p_vals_all[n] for n in ordered]
    k_hat = (_forward_stop_k if rule == "forward" else _strong_stop_k)(p_seq, alpha)
    keep = set(ordered[:k_hat])
    to_prune = list(all_internals - keep)
    result = copy.deepcopy(T_max)
    for node_id in to_prune:
        prune_subtree_inplace(result, node_id)
    return result


class GSellStop:
    """G'Sell ForwardStop / StrongStop along the CCP chain."""

    def __init__(
        self,
        significance_level: float = 0.05,
        rule: Literal["forward", "strong"] = "forward",
        strategy: Literal["sum", "wl"] = "sum",
        min_samples_leaf: int = 20,
    ):
        self.significance_level = significance_level
        self.rule = rule
        self.strategy = strategy
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y, random_state: int = 0):
        X = np.asarray(X)
        y = np.asarray(y)
        chain = build_ccp_chain(X, y, self.min_samples_leaf, random_state=random_state)
        d = X.shape[1]
        if self.strategy == "sum":
            tree = _strategy_sum(chain, d, self.significance_level, self.rule)
        elif self.strategy == "wl":
            tree = _strategy_wl(chain, d, self.significance_level, self.rule)
        else:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        self.model_ = tree
        self.n_leaves_ = int((tree.tree_.children_left == -1).sum())
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

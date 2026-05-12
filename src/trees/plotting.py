"""Tree-rendering helpers for the paper's figures.

Section 4.2 (Fig 6) shows regularised output trees with each node annotated by
the split rule, the mean prediction, the per-node p-value, and the cumulative
p-value of the smallest subtree this node appears in as a non-leaf. Nodes
whose cumulative p-value exceeds the chosen tolerance delta are highlighted.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.tree import DecisionTreeRegressor, plot_tree


def plot_tree_with_pvalues(
    cart: DecisionTreeRegressor,
    feature_names: Sequence[str] | None = None,
    node_pvalues: Sequence[float] | None = None,
    cumulative_pvalues: Sequence[float] | None = None,
    delta: float = 0.05,
    ax: Axes | None = None,
) -> Axes:
    """Render a CART regression tree with optional p-value annotations.

    Nodes whose `cumulative_pvalues[k] > delta` (i.e. those that the p-sum
    rule would prune) are shaded in red.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    annotations = plot_tree(
        cart,
        feature_names=list(feature_names) if feature_names is not None else None,
        impurity=False,
        proportion=False,
        filled=False,
        rounded=True,
        ax=ax,
    )
    if node_pvalues is None and cumulative_pvalues is None:
        return ax

    for node_id, ann in enumerate(annotations):
        text = ann.get_text()
        rows = [text]
        if node_pvalues is not None and node_pvalues[node_id] >= 0:
            rows.append(f"p = {node_pvalues[node_id]:.2g}")
        if cumulative_pvalues is not None and cumulative_pvalues[node_id] >= 0:
            rows.append(f"P_sum = {cumulative_pvalues[node_id]:.2g}")
        ann.set_text("\n".join(rows))
        if (
            cumulative_pvalues is not None
            and cumulative_pvalues[node_id] > delta
        ):
            ann.set_bbox({"boxstyle": "round", "facecolor": "#ffcccc", "edgecolor": "black"})
    return ax

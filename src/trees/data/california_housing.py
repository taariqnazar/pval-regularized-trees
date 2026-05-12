# data/california.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.datasets import fetch_california_housing
from .core import register_dataset


@register_dataset("california_housing")
def load_california(
    *, data_home: str = "data/raw", return_X_y: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Delegates to sklearn; always returns (X, y) for our API
    X, y = fetch_california_housing(data_home=data_home, return_X_y=True)
    return X, y

from __future__ import annotations
from typing import Callable, Dict, Tuple, Any, Optional, List
import numpy as np

DatasetFn = Callable[..., Tuple[np.ndarray, np.ndarray]]

_DATASETS: Dict[str, DatasetFn] = {}


def register_dataset(name: str):
    """Decorator to register a dataset loader under a string name."""

    def _wrap(fn: DatasetFn) -> DatasetFn:
        if name in _DATASETS:
            raise ValueError(f"Dataset '{name}' already registered.")
        _DATASETS[name] = fn
        return fn

    return _wrap


def available_datasets() -> List[str]:
    return sorted(_DATASETS.keys())


def load_dataset(name: str, /, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified entry point: returns (X, y) as NumPy arrays.
    Example:
        X, y = load_dataset("california_housing", data_home="data/raw")
        X, y = load_dataset("bemtpl16", csv_path="data/raw/BEMTPL16.csv")
    """
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {
                       available_datasets()}")
    return _DATASETS[name](**kwargs)


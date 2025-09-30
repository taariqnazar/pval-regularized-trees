"""
Dataset loaders and generators.

This module provides a unified interface to load real and synthetic datasets
through `load_dataset(name, **kwargs)`.

Available datasets can be listed with `available_datasets()`.
"""

from .core import load_dataset, available_datasets, register_dataset

# Import modules so their @register_dataset functions run
from . import california  # noqa: F401
from . import bemtpl16  # noqa: F401
from . import synthetic  # noqa: F401

__all__ = [
    "load_dataset",
    "available_datasets",
    "register_dataset",
]

"""
Dataset loaders and generators.

This module provides a unified interface to load real and synthetic datasets
through `load_dataset(name, **kwargs)`.

Available datasets can be listed with `available_datasets()`.
"""

from .core import load_dataset, available_datasets, register_dataset

# Import modules so their @register_dataset functions run
from .california_housing import load_california
from .bemtpl16 import load_bemtpl16
from .synthetic import generate_linear_data, generate_neufeldt_data
from .swemotorcycle import load_swemotorscycle
# from .freMTPL2freq import load_freMTPL2freq

__all__ = [
    "load_dataset",
    "available_datasets",
    "register_dataset",
]

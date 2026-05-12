"""Dataset loaders.

Unified API:
    >>> from trees.data import load_dataset, available_datasets
    >>> available_datasets()
    >>> X, y = load_dataset("boston_housing")
"""
from .core import load_dataset, available_datasets, register_dataset

# Importing modules registers their @register_dataset functions.
from . import california_housing  # noqa: F401
from . import bemtpl16            # noqa: F401
from . import synthetic           # noqa: F401  (registers linear_model, neufeldt)
from . import swemotorcycle       # noqa: F401
from . import boston_housing      # noqa: F401
from . import box_lunch           # noqa: F401

__all__ = ["load_dataset", "available_datasets", "register_dataset"]

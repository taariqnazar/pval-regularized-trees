"""
Synthetic dataset generators for experiments.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np

from .core import register_dataset


@register_dataset("linear_model")
def generate_linear_data(
    n: int = 200, p: int = 10, sigma: float = 5.0, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate linear regression data with Gaussian noise.

    Parameters
    ----------
    n : int, default=200
        Number of samples.
    p : int, default=10
        Number of features.
    sigma : float, default=5.0
        Standard deviation of noise.
    random_state : int, default=0
        Random seed.

    Returns
    -------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Target vector.
    beta : ndarray of shape (p,)
        True regression coefficients.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, (n, p))
    beta = rng.normal(0, 1, p)
    y = X @ beta + rng.normal(0, sigma, n)
    return X, y, beta


@register_dataset("neufeldt")
def generate_neufeldt_data(
    a: float,
    b: float,
    n: int = 200,
    p: int = 10,
    sigma: float = 5.0,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Neufeldt-style data with interaction/nonlinear effects.

    Parameters
    ----------
    a : float
        Effect size for condition (X[:, 1] > 0).
    b : float
        Overall scaling factor.
    n : int, default=200
        Number of samples.
    p : int, default=10
        Number of features.
    sigma : float, default=5.0
        Standard deviation of noise.
    random_state : int, default=0
        Random seed.

    Returns
    -------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Target vector.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, (n, p))

    mu = 1 + a * (X[:, 1] > 0) + ((X[:, 2] * X[:, 1]) > 0)
    mu = (X[:, 0] <= 0) * mu
    mu = b * mu

    y = mu + rng.normal(0, sigma, n)
    return X, y

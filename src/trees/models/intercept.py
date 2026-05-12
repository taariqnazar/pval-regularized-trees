from __future__ import annotations
from typing import Optional

import numpy as np


class Intercept:
    def __init__(
        self,
    ):
        self.intercept_: Optional[float] = None  # chosen intercept after fit

    def fit(self, X, y, random_state: int = 0):
        self.intercept_ = np.mean(y)

    def predict(self, X):
        return self.intercept_ * np.ones(X.shape[0])

    def __repr__(self) -> str:
        return f"Intercept()"

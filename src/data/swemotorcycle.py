# data/bemtpl16.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from .core import register_dataset


@register_dataset("swemotorcycle")
def load_swemotorscycle(
    *, csv_path: str = "src/data/raw/swemotorcycle.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    y = df["ClaimAmount"].to_numpy(dtype=np.float64)
    X = df[
        ["OwnerAge", "Gender", "Area", "RiskClass", "VehAge", "BonusClass"]
    ].to_numpy(dtype=np.float64)

    idx = y > 0
    return X[idx], y[idx]

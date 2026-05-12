"""Boston Housing dataset loader (StatLib snapshot, ref [21] in the paper).

Source: http://lib.stat.cmu.edu/datasets/boston (accessed 2026-04-15).
Target: MEDV (median home value, $1000s).
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .core import register_dataset


_RAW = Path(__file__).parent / "raw" / "boston_housing.csv"


@register_dataset("boston_housing")
def load_boston_housing(csv_path: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path) if csv_path is not None else _RAW
    df = pd.read_csv(path)
    y = df["MEDV"].astype(float).to_numpy()
    X = df.drop(columns=["MEDV"]).select_dtypes(include="number").to_numpy(dtype=float)
    return X, y

"""Box Lunch Study (blsdata) loader (R visTree::blsdata, refs [22], [23]).

Target: kcal24h0 (24-hour kcal intake).
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .core import register_dataset


_RAW = Path(__file__).parent / "raw" / "blsdata.csv"


@register_dataset("box_lunch")
def load_box_lunch(csv_path: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path) if csv_path is not None else _RAW
    df = pd.read_csv(path)
    y = df["kcal24h0"].astype(float).to_numpy()
    X = df.drop(columns=["kcal24h0"]).select_dtypes(include="number").to_numpy(dtype=float)
    return X, y

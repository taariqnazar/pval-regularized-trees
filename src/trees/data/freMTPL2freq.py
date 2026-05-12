# data/bemtpl16.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from .core import register_dataset


@register_dataset("freMTPL2freq")
def load_freMTPL2freq(
    *, csv_path: str = "src/data/raw/freMTPL2freq.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    y = df["ClaimNb"].to_numpy(dtype=np.float64)
    X = df[
        [
            "insured_birth_year",
            "vehicle_age",
            "policy_holder_age",
            "driver_license_age",
            "mileage",
            "vehicle_power",
        ]
    ].to_numpy(dtype=np.float64)
    return X, y

import numpy as np


def mse(y_true, y_pred) -> float:
    """Mean Squared Error (MSE) between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (RMSE) between true and predicted values."""
    return float(np.sqrt(mse(y_true, y_pred)))

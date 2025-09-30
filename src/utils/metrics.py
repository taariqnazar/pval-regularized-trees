import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    """Compute Root Mean Squared Error between true and predicted values."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

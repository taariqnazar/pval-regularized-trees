import os 
import yaml

import numpy as np 
from sklearn.metrics import mean_squared_error 
from src.models import GBM, CCPBoostingTree 

def _load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _train_models(models, X, y):
    return {
        model_name: _train_model(
            model_name=model_name, parameters=model_config, X=X, y=y
        )
        for model_name, model_config in models.items()
    }


def _train_model(model_name, parameters, X, y):
    if model_name == "gbm":
        model = GBM(parameters)
    elif model_name[:9] == "pboosting":
        model = CCPBoostingTree(parameters)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    model.fit(X, y)
    return model


def _evaluate_models(models, X, y):
    return {
        model_name: _evaluate_model(model, X, y) for model_name, model in models.items()
    }


def _evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return {
        "rmse": rmse,
    }


def _evaluate_staged_models(models, X, y):
    return {
        model_name: _evaluate_staged_model(model, X, y)
        for model_name, model in models.items()
    }


def _evaluate_staged_model(model, X, y):
    staged_prediction = model.staged_predict(X)
    return {
        "rmse": [
            np.sqrt(mean_squared_error(y, y_pred)) for y_pred in staged_prediction
        ],
    }


# Solution: Convert NumPy types to Python native types
def _convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # Convert np.int64 → int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert np.float32/np.float64 → float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array → Python list
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

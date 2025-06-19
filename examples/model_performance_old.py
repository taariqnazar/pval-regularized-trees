from collections import defaultdict
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.models import GBM, CCPBoostingTree
from src.california_housing_tree_data import (
    sample_california_housing_tree_data,
    sample_california_housing_gbm_data,
    sample_california_housing_data,
    sample_california_housing_gbm_normal_features,
    sample_california_housing_tree_normal_features,
)

# Apply global scientific settings
plt.rcParams.update(
    {
        "font.size": 12,  # Increase font size for readability
        "lines.linewidth": 2,  # Thicker lines
        "figure.dpi": 300,  # High resolution
        "savefig.dpi": 300,  # High-quality figure saving
        #"alpha": 0.7,  # Partial transparency
    }
)
plt.style.use('default')

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

if __name__ == "__main__":
    config = _load_config("examples/model_performance.yml")
    seed = config["random_seed"]
    # Set consistent random seed for all random number generators
    np.random.seed(seed)  # seed for numpy

    datasets = {
       # "gbm_tree": (sample_california_housing_gbm_data, .5), #0,24 true
       # "gbm_tree_normal": (sample_california_housing_gbm_normal_features,
       #                     .5), #0.24 true
       # "tree": (sample_california_housing_tree_data, .5),
       # "tree_normal": (sample_california_housing_tree_normal_features, .5),
        "raw": (sample_california_housing_data, np.nan),
    }

    output_data = defaultdict(dict)

    for dataset_name, data in datasets.items():
        dataset, rmse = data
        if rmse is not np.nan:
            X, y = dataset(sigma=rmse)
            output_data[dataset_name]["rmse"] = rmse
        else:
            X, y = dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        models = _train_models(config["models"], X_train, y_train)
        evaluation = _evaluate_models(models, X_test, y_test)
        staged_evaluation = _evaluate_staged_models(models, X_test, y_test)

        for model_name, model_evaluation in staged_evaluation.items():
            print(f"Evaluation for {model_name}:")
            print(f"Number of estimators: {models[model_name].n_estimators}")
            (line,) = plt.plot(model_evaluation["rmse"], label=model_name)
            model_complexity = models[model_name].get_staged_complexity()
            output_data[dataset_name][model_name] = {
                "rmse": model_evaluation["rmse"],
                "n_estimators": models[model_name].n_estimators,
                "staged_evaluation": staged_evaluation[model_name],
                "evaluation": evaluation[model_name],
                "staged_complexity": model_complexity,
            }
#            plt.axvline(
#                models[model_name].n_estimators - 1,
#                linestyle="--",
#                color=line.get_color(),
#            )
#        plt.ylim(0.45, 1)
#        plt.savefig(f".data/{dataset_name}_validation.png")
#        plt.clf()


    with open(".data/gbm_pboost_comparison_abt.json", "w") as f:
        json.dump(output_data, f, indent=4, default=_convert_numpy)

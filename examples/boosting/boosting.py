import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from examples.boosting_utils import (
    _evaluate_models,
    _evaluate_staged_models,
    _load_config,
    _train_models,
)


def sample_california_housing_data():
    return fetch_california_housing(data_home=".data/", return_X_y=True)


def sample_bemtpl16_data():
    df = pd.read_csv(".data/BEMTPL16.csv")
    y = df["number_of_liability_claims"]
    X = df[
        [
            "insured_birth_year",
            "vehicle_age",
            "policy_holder_age",
            "driver_license_age",
            "mileage",
            "vehicle_power",
        ]
    ]

    return X.values, np.array(y.values, dtype=np.float64)


def run_experiment(X, y):
    config = _load_config("examples/boosting_config.yml")
    seed = config["random_seed"]
    # Set consistent random seed for all random number generators
    np.random.seed(seed)  # seed for numpy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = _train_models(config["models"], X_train, y_train)
    evaluation = _evaluate_models(models, X_test, y_test)
    staged_evaluation = _evaluate_staged_models(models, X_test, y_test)

    return models, evaluation, staged_evaluation


def main():
    experiments_results = {}
    experiments = {
        "gbm_vs_pboost_bemtpl": sample_bemtpl16_data,
        "gbm_vs_pboost_california_housing": sample_california_housing_data,
    }

    for experiment_name, data_func in experiments.items():
        X, y = data_func()
        models, evaluation, staged_evaluation = run_experiment(X, y)
        experiments_results[experiment_name] = {
            "staged_evaluation": staged_evaluation,
            "evaluation": evaluation,
            "models": models,
        }
        print("Done with experiment:", experiment_name)

    with open(f".data/boosting.pkl", "wb") as f:
        pickle.dump(
            experiments_results,
            f,
        )

    # for model_name, model_evaluation in staged_evaluation.items():
    #    n_estimators = models[model_name].n_estimators
    #    print(model_name, ":", n_estimators)
    #    rmse = model_evaluation["rmse"]
    #    min_rmse = min(rmse)
    #    if model_name == "gbm":
    #        (line,) = plt.plot(rmse, label=model_name, linewidth=2, linestyle="--")
    #    else:
    #        (line,) = plt.plot(rmse, label=model_name, linewidth=2)
    #    plt.axvline(
    #        n_estimators - 1,
    #        linestyle="--",
    #        color=line.get_color(),
    #    )  # Mark minimum with a red dot
    #    # plt.plot(n_estimators, rmse[n_estimators], 'o',
    #    #         color=line.get_color(), markersize=10)

    #    plt.axhline(
    #        min_rmse,
    #        linestyle="--",
    #        color=line.get_color(),
    #    )
    # plt.ylim(0.4, 0.8)
    # plt.xlim(left=0, right=250)
    # plt.tight_layout()
    # plt.savefig(f".data/{experiment_name}_validation.png")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()

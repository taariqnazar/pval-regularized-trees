import pickle
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from src.california_housing_tree_data import sample_california_housing_data

parameters = {
    "learning_rate": 0.1,
    "max_depth": 3,
}


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

    return X.values, y.values


def main():
    n_iterations = 500
    #X, y = sample_california_housing_data()
    X, y = sample_bemtpl16_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rmses = []
    for i in range(n_iterations):
        seed = i
        rmse = run_iteration(seed, X_train, y_train, X_test, y_test)
        print(f"Iteration {i}: RMSE: {rmse}")
        rmses.append(rmse)
    
    with open(".data/rmse_gbm_histogram_bemtmp16.pkl", "wb") as f:
        pickle.dump(rmses, f)


def run_iteration(seed, X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    best_iters = []

    for train_indices, validation_indices in kf.split(X_train):
        X_train_k, X_validation = X_train[train_indices], X_train[validation_indices]
        y_train_k, y_validation = y_train[train_indices], y_train[validation_indices]

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=3,
            criterion="squared_error",
        )
        model.fit(X_train_k, y_train_k)

        best_iter = min(
            enumerate(model.staged_predict(X_validation)),
            key=lambda x: mean_squared_error(y_validation, x[1]),
        )[0]
        best_iters.append(best_iter)

    final_iter = sum(best_iters) // len(best_iters)
    print(f"Final Iteration: {final_iter}")
    final_model = GradientBoostingRegressor(
        n_estimators=final_iter,
        learning_rate=0.1,
        min_samples_leaf=20,
        max_depth=3,
        criterion="squared_error",
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()

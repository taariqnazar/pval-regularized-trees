import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np

from single_tree import train_regression_tree


def generate_gbm_model():
    pass

def generate_single_tree_():
    max_depth = 3 
    min_samples_split = 5

    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    tree = train_regression_tree(
        X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split, random_state=0
    )

    tree.cost_complexity_pruning_path(X_train, y_train)
    ccps_alpha = tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas

    optimal_tree = None
    min_mse = float("inf")
    for ccp in ccps_alpha:
        tree = train_regression_tree(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=0,
            ccp_alpha=ccp,
        )
        y_pred = tree.predict(X_test)
        msep = mean_squared_error(y_test, y_pred)
        if msep < min_mse:
            min_mse = msep
            optimal_tree = tree

    with open(".data/california_housing_tree.pkl", "wb") as f:
        pickle.dump(optimal_tree, f)


def sample_california_housing_tree_data(sigma=0.01):
    try:
        with open(".data/california_housing_tree.pkl", "rb") as f:
            tree = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "You need to run generate_data() first to generate the tree model."
        )

    X, _ = fetch_california_housing(data_home=".data/", return_X_y=True)

    noise = np.random.normal(0, 1, X.shape[0])
    y = tree.predict(X) + sigma * noise

    return X, y


def main():
    sample_california_housing_tree_data()

if __name__ == "__main__":
    main()

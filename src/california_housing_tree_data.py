import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

import numpy as np

from single_tree import train_regression_tree


weak_learner_params = {
    "max_depth": 3,
    "min_samples_leaf": 20,
    "random_state": 0,
    "learning_rate": 0.1,
}


def generate_gbm_model():
    n_estimators = 10000
    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators,
        **weak_learner_params,
    )
    gbm.fit(X_train, y_train)

    gbm_preds = gbm.staged_predict(X_test)
    best_iter = None
    min_mse = float("inf")
    for i in range(n_estimators):
        gbm_pred = next(gbm_preds)
        gbm_msep = mean_squared_error(y_test, gbm_pred)
        if gbm_msep < min_mse:
            min_mse = gbm_msep
            best_iter = i

    gbm = GradientBoostingRegressor(n_estimators=best_iter,
                                    **weak_learner_params)
    gbm.fit(X, y)

    with open(".data/california_housing_gbm.pkl", "wb") as f:
        #gbm.estimators_ = gbm.estimators_[: best_iter + 1]
        pickle.dump(gbm, f)
        print(f"Best MSEP: {gbm_msep} @ iteration {best_iter}")


def generate_single_tree_model():
    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    del weak_learner_params["learning_rate"] 

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    alphas = []
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train)):
        X_train_, X_test_ = (
            X_train[train_indices],
            X_train[test_indices],
        )
        y_train_, y_test_ = (
            y_train[train_indices],
            y_train[test_indices],
        )
        # Train single tree
        tree = train_regression_tree(
            X_train_,
            y_train_,
            **weak_learner_params,
        )
        tree.cost_complexity_pruning_path(X_train_, y_train_)
        ccps_alpha = tree.cost_complexity_pruning_path(X_train_, y_train_).ccp_alphas

        optimal_tree = None
        min_mse = float("inf")
        for ccp in ccps_alpha:
            tree = train_regression_tree(
                X_train_,
                y_train_,
                ccp_alpha=ccp,
                **weak_learner_params,
            )
            y_pred = tree.predict(X_test_)
            msep = mean_squared_error(y_test_, y_pred)
            if msep < min_mse:
                best_alpha = ccp
        alphas.append(best_alpha)
    alpha = sum(alphas) / len(alphas)
    optimal_tree = train_regression_tree(
        X_train, y_train, ccp_alpha=alpha, **weak_learner_params
    )
    with open(".data/california_housing_tree.pkl", "wb") as f:
        pickle.dump(optimal_tree, f)
    print(f"Tree msep: {min_mse}")


def sample_california_housing_data():
    return fetch_california_housing(data_home=".data/", return_X_y=True)


def sample_california_housing_gbm_data(sigma=0.01):
    try:
        with open(".data/california_housing_gbm.pkl", "rb") as f:
            tree = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "You need to run generate_data() first to generate the tree model."
        )

    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    y_pred = tree.predict(X)

    noise = np.random.normal(0, 1, X.shape[0])
    y = y_pred + sigma * noise

    return X, y

def sample_california_housing_gbm_normal_features(sigma=0.01):
    try:
        with open(".data/california_housing_gbm.pkl", "rb") as f:
            tree = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "You need to run generate_data() first to generate the tree model."
        )

    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    X = np.random.multivariate_normal(mean, cov, X.shape[0])

    y_pred = tree.predict(X)

    noise = np.random.normal(0, 1, X.shape[0])
    y = y_pred + sigma * noise

    return X, y

def sample_california_housing_tree_data(sigma=0.01):
    try:
        with open(".data/california_housing_tree.pkl", "rb") as f:
            tree = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "You need to run generate_data() first to generate the tree model."
        )

    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    y_pred = tree.predict(X)

    noise = np.random.normal(0, 1, X.shape[0])

    y = y_pred + sigma * noise

    return X, y

def sample_california_housing_tree_normal_features(sigma=0.01):
    try:
        with open(".data/california_housing_tree.pkl", "rb") as f:
            tree = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "You need to run generate_data() first to generate the tree model."
        )

    X, y = fetch_california_housing(data_home=".data/", return_X_y=True)
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    X = np.random.multivariate_normal(mean, cov, X.shape[0])
    y_pred = tree.predict(X)

    noise = np.random.normal(0, 1, X.shape[0])

    y = y_pred + sigma * noise

    return X, y


def main():
    # generate_gbm_model()
    generate_single_tree_model()

if __name__ == "__main__":
    main()

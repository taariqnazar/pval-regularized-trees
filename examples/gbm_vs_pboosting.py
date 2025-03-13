import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from src.boosting import CCPBoostingTree
from src.california_housing_tree_data import sample_california_housing_tree_data


def main():
    weak_learner_depth = 2
    weak_learner_min_samples_split = 10

    X, y = sample_california_housing_tree_data(sigma=0.1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    significance_level = 0.05
    ccp_tree = CCPBoostingTree()
    ccp_tree.fit(
        X_train,
        y_train,
        significance_level=significance_level,
        max_depth=weak_learner_depth,
        min_samples_split=weak_learner_min_samples_split,
    )


    n_estimators = 2*len(ccp_tree.trees)
    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.5,
        criterion="friedman_mse",
        max_depth=weak_learner_depth,
        min_samples_split=weak_learner_min_samples_split,
    )
    gbm.fit(X_train, y_train)

    gbm_preds = gbm.staged_predict(X_test)

    ccp_loss= []
    for i in range(len(ccp_tree.trees)):
        ccp_pred = ccp_tree.predict(X_test, sub_tree=i + 1)
        ccp_msep = mean_squared_error(y_test, ccp_pred)
        ccp_loss.append((i+1, ccp_msep))
        print(f"iteration {i+1}")

    gbm_loss = []
    for i in range(n_estimators):
        gbm_pred = next(gbm_preds)
        gbm_msep = mean_squared_error(y_test, gbm_pred)
        gbm_loss.append( (i+1, gbm_msep))
        print(f"iteration {i+1}")

    plt.plot(*zip(*ccp_loss), label="CCP Boosting")
    plt.plot(*zip(*gbm_loss), label="Gradient Boosting Machine")
    plt.legend(("CCP Boosting", "Gradient Boosting Machine"))
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("CCP Boosting vs Gradient Boosting Machine")
    plt.show()


if __name__ == "__main__":
    main()

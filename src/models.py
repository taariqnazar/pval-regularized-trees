import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from .single_tree import ccp_pval_regularised_tree

max_estimators = 500

class GBM:
    def __init__(self, parameters):
        self.parameters = parameters.copy()

        self.model = None
        self.n_estimators = 0

        if "k_folds" in self.parameters.keys():
            self.k_folds = self.parameters["k_folds"]
            del self.parameters["k_folds"]
        else:
            self.k_folds = 1

    def fit(self, X, y, random_state=0):
        global max_estimators

        if self.k_folds > 1:
            best_iters = []
            for i, (train_indices, test_indices) in enumerate(
                KFold(n_splits=self.k_folds, random_state=random_state,
                      shuffle=True).split(X)
            ):
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]

                model = GradientBoostingRegressor(
                    n_estimators=max_estimators, **self.parameters
                )
                model.fit(X_train, y_train)
                # Determine best number of iterations
                staged_predictions = model.staged_predict(X_test)
                best_msep = float("inf")
                best_iter = None
                for i, y_pred in enumerate(staged_predictions):
                    msep = mean_squared_error(y_test, y_pred)
                    if msep < best_msep:
                        best_msep = msep
                        best_iter = i
                best_iters.append(best_iter)

            # Average best iteration
            best_iter = sum(best_iters) // len(best_iters)
            self.n_estimators = best_iter
            self.model = GradientBoostingRegressor(
                n_estimators=max_estimators, **self.parameters
            )
            self.model.fit(X, y)

        elif self.k_folds == 1:
            self.model = GradientBoostingRegressor(
                n_estimators=max_estimators, **self.parameters
            )
            self.model.fit(X, y)
            self.n_estimators = max_estimators
        else:
            raise ValueError("k_folds must be a positive integer")


    def predict(self, X):
        return self.model.predict(X)

    def staged_predict(self, X):
        return self.model.staged_predict(X)

    def get_staged_complexity(self):
        complexity = []
        for estimator in self.model.estimators_:
            complexity.append(estimator[0].tree_.n_leaves)
        return complexity

class CCPBoostingTree:
    def __init__(self, parameters):
        self.trees = []
        self.parameters = parameters
        self.n_estimators = 0

        if "learning_rate" in self.parameters.keys():
            self.learning_rate = self.parameters["learning_rate"]
            del self.parameters["learning_rate"]
        else:
            self.learning_rate = 0.1

    def fit(self, X, y):
        global max_estimators

        if self.trees:
            self.trees = []
            print("Model already contains trees")
            print("Cleaning prev model")

        working_y = y.copy()
        for i in range(max_estimators):
            tree = ccp_pval_regularised_tree(X, working_y, **self.parameters)
            self.trees.append(tree)
            if tree.tree_.n_leaves == 1:
                break
            y_pred = self.trees[-1].predict(X)
            working_y = working_y - self.learning_rate * y_pred
            self.n_estimators = len(self.trees) - 1
        return self

    def predict(self, X, sub_tree=None):
        if not self.trees:
            raise ValueError("The model has not been fit.")

        prediction = False
        if not sub_tree:
            sub_tree = len(self.trees)

        for tree in self.trees[:sub_tree]:
            prediction += self.learning_rate*tree.predict(X)

        return prediction

    def staged_predict(self, X):
        global max_estimators

        if not self.trees:
            raise ValueError("The model has not been fit.")

        #n = len(self.trees)
        n = max_estimators
        for i in range(1, n):
            yield self.predict(X, i)

    def get_staged_complexity(self):
        return [tree.tree_.n_leaves for tree in self.trees]

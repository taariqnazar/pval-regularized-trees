from sklearn.tree import DecisionTreeRegressor

from stats import get_tree_split_pvalue
from single_tree import train_regression_tree, ccp_pval_regularised_tree

class BoostingTree:
    def __init__(self, trees=[]):
        self.trees = trees

    def add_tree(self, tree: DecisionTreeRegressor):
        self.trees.append(tree)

    def predict(self, X, sub_tree=None):
        if not self.trees:
            raise ValueError("No trees in the model")

        prediction = False
        if not sub_tree:
            sub_tree = len(self.trees)

        for tree in self.trees[:sub_tree]:
            prediction += tree.predict(X)

        return prediction

    #def fit(self, X, y, n_estimators=1000, **kwargs):
    #    #if self.trees:
    #    #    raise ValueError("Model already contains trees")

    #    working_y = y.copy()
    #    for i in range(n_estimators):
    #        self.add_tree(train_regression_tree(X, working_y, **kwargs))
    #        working_y -= self.trees[-1].predict(X)

    #    return self

class CCPBoostingTree(BoostingTree):
    def __init__(self, trees=[]):
        super().__init__(trees)

    def fit(self, X, y, significance_level, max_estimators=1000, **kwargs):
        if self.trees:
            self.trees = []
            print("Model already contains trees")
            print("Cleaning prev model")

        working_y = y.copy()
        for i in range(max_estimators):
            tree = ccp_pval_regularised_tree(X, working_y, significance_level, **kwargs)
            self.add_tree(tree)
            if tree.tree_.n_leaves == 1:
                break
            working_y -= self.trees[-1].predict(X)

        return self


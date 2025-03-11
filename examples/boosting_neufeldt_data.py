import numpy as np

from src.boosting import CCPBoostingTree
from src.stats import get_tree_split_pvalue
from src.testing import computed_mse

def sample_neufeld(N, p=10, a=0.5, b=5, sig=5):
    # Data generation
    sig = 5
    X = np.random.normal(0, 1, (N, p))
    eps = np.random.normal(0, sig, N)

    mu = b * (X[:, 0] <= 0) * \
        (1 + a * (X[:, 1] > 0) + ((X[:, 1] * X[:, 2]) > 0))

    y = mu + eps

    return X, y


def main():
    n_samples = 10000
    significanced_level = 0.05

    X_train, y_train = sample_neufeld(n_samples)
    X_test, y_test = sample_neufeld(n_samples)


    trees = CCPBoostingTree()
    trees.fit(X_train, y_train, significanced_level=significanced_level, max_depth=3, min_samples_split=10)

    print(len(trees.trees))
    print(get_tree_split_pvalue(trees.trees[-1]))
    return

    for i in range(len(trees.trees)):
        pval = get_tree_split_pvalue(trees.trees[i])
        print(pval)
        if pval > significance_level:
            print(f"Tree {i} is not significant")
            pred_train = trees.predict(X_train, sub_tree=i-1)
            pred_test = trees.predict(X_test, sub_tree=i-1)
            in_sampled_loss = computed_mse(y_train, pred_train)
            out_sampled_loss = computed_mse(y_test, pred_test)
            print(f"Tree {i}: In-sample loss: {in_sampled_loss}, Out-sample loss: {out_sampled_loss}")
            break

if __name__ == "__main__":
    main()

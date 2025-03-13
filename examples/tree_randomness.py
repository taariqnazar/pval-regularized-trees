import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from src.single_tree import train_regression_tree
from sklearn.metrics import mean_squared_error
from src.utils import sample_neufeld_data


def main():
    N = 500
    X, y = sample_neufeld_data(N)

    trees = []
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=i)
        tree = train_regression_tree(X_train, y_train, max_depth=2,
                                     min_samples_split=5,
                                     random_state=0)
        tree.cost_complexity_pruning_path(X_train, y_train)
        ccps_alpha = tree.cost_complexity_pruning_path(X_train,
                                                       y_train).ccp_alphas


        optimal_tree = None
        min_mse = float('inf')
        for ccp in ccps_alpha:
            tree = train_regression_tree(X_train, y_train, max_depth=2,\
                                         min_samples_split=5,\
                                         random_state=0, ccp_alpha=ccp)
            y_pred = tree.predict(X_test)
            msep = mean_squared_error(y_test, y_pred)
            if msep < min_mse:
                min_mse = msep
                optimal_tree = tree 

        trees.append(optimal_tree)


    # Plot the trees
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    for i, tree in enumerate(trees):
        plot_tree(tree, ax=axes[i], filled=True, rounded=True, proportion=True,
                  fontsize=10, impurity=False)

        axes[i].set_title(f'Tree {i+1}')

    plt.tight_layout()
    plt.savefig('figures/tree_randomness.png')
    plt.show()



if __name__ == "__main__":
    main()

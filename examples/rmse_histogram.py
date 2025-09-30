import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import plot_tree

from single_tree import ccp_pval_regularised_tree, train_regression_tree
from utils import generate_neufeldt_data, generate_linear_data


def main():
    n_iterations = 200
    n_samples = 1000
    # X, y = generate_neufeldt_data(a=1, b=1, n=n_samples, sigma=1)
    X, y = generate_linear_data(n=n_samples, sigma=1, p=5, random_state=42)

    fold_test = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for delta in [0.1, 0.05, 0.01]:
        our_tree = ccp_pval_regularised_tree(
            X_train,
            y_train,
            significance_level=delta,
            max_depth=3,
            min_samples_split=20,
            random_state=0,
        )
        y_pred = our_tree.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Delta: {delta} RMSE: {rmse} ccp_alpha: {our_tree.ccp_alpha}")

    unique_alphas = set()
    for k_folds in [5]:
        rmses = []
        for i in range(n_iterations):
            kf = KFold(n_splits=k_folds, random_state=i, shuffle=True)
            alphas_all = []
            for j, (train_indices, validation_indices) in enumerate(kf.split(X_train)):
                X_train_k, X_validation = (
                    X_train[train_indices],
                    X_train[validation_indices],
                )
                y_train_k, y_validation = (
                    y_train[train_indices],
                    y_train[validation_indices],
                )
                # Train single tree
                tree = train_regression_tree(
                    X_train_k,
                    y_train_k,
                    max_depth=3,
                    min_samples_split=20,
                    random_state=0,
                )

                ccps_alpha = tree.cost_complexity_pruning_path(
                    X_train_k, y_train_k
                ).ccp_alphas

                optimal_tree = None
                best_alpha = 0
                min_mse = float("inf")
                for ccp in ccps_alpha:
                    tree = train_regression_tree(
                        X_train_k,
                        y_train_k,
                        max_depth=3,
                        min_samples_split=20,
                        ccp_alpha=ccp,
                        random_state=0,
                    )
                    y_pred = tree.predict(X_validation)
                    msep = mean_squared_error(y_validation, y_pred)
                    if msep < min_mse:
                        min_mse = msep
                        best_alpha = ccp

                alphas_all.append(best_alpha)

            alpha = sum(alphas_all) / len(alphas_all)
            optimal_tree = train_regression_tree(
                X_train,
                y_train,
                max_depth=3,
                min_samples_split=20,
                ccp_alpha=alpha,
                random_state=0,
            )
            unique_alphas.add(optimal_tree.ccp_alpha)
            # compute rmse
            y_pred = optimal_tree.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmses.append(rmse)

        fold_test[k_folds] = rmses

        plt.hist(rmses, bins=20, label="5 fold")
        plt.savefig(
            ".data/neufeldt_rmse_histogram.pdf", bbox_inches="tight", transparent=True
        )
        plt.show()
        tree = train_regression_tree(
            X_train,
            y_train,
            max_depth=3,
            min_samples_split=20,
            random_state=0,
        )

        ccp_alpha = tree.cost_complexity_pruning_path(
            X_train, y_train).ccp_alphas

        print("list of ccps:", ccp_alpha)

        realised_alphas = [
            get_realiased_ccp(ccp_alpha, alpha) for alpha in unique_alphas
        ]
        print("list of realised ccps:", np.unique(realised_alphas))

        for alpha in ccp_alpha:
            tree = train_regression_tree(
                X_train,
                y_train,
                max_depth=3,
                min_samples_split=20,
                ccp_alpha=alpha,
                random_state=0,
            )
            y_pred = tree.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(alpha, rmse, tree.get_n_leaves())

        for alpha in np.unique(realised_alphas):
            tree = train_regression_tree(
                X_train,
                y_train,
                max_depth=3,
                min_samples_split=20,
                ccp_alpha=alpha,
                random_state=0,
            )
            print("alpha:", alpha)
            plot_tree(tree)
            plt.show()
        plot_tree(our_tree)
        plt.show()


def get_realiased_ccp(ccp_alphas, alpha):
    for a in ccp_alphas[::-1]:
        if a <= alpha:
            return a


if __name__ == "__main__":
    np.random.seed(1)
    main()

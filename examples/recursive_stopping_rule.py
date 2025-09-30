from sklearn.tree import _tree, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.recursive_early_stopping import train_model as recursive_split_train_model
from src.single_tree import train_model as ccp_pval_regularised_train_model


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


# --- helpers ---------------------------------------------------------------


def n_leaves(model: DecisionTreeRegressor) -> int:
    """Count leaves in a fitted sklearn tree."""
    T = model.tree_
    return int(np.sum(T.children_left == _tree.TREE_LEAF))


def evaluate_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    models: {"Model name": fitted_estimator, ...}
    Returns a DataFrame with complexity (#leaves) and MSEs.
    """
    rows = []
    for name, m in models.items():
        yhat_tr = m.predict(X_train)
        yhat_te = m.predict(X_test)
        rows.append(
            {
                "model": name,
                "n_leaves": n_leaves(m),
                "mse_train": mean_squared_error(y_train, yhat_tr),
                "mse_test": mean_squared_error(y_test, yhat_te),
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["n_leaves", "mse_test"]).reset_index(drop=True)
    return df


# --- your main() with the table -------------------------------------------


def main():
    reg = DecisionTreeRegressor(random_state=0)

    # Example data
    X, y = sample_bemtpl16_data()
    d = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    threshold = 0.05

    # 1) CCP + cumulative p-value regularization (your alpha-path search)
    ccp_pval_regularised_tree = ccp_pval_regularised_train_model(
        reg,
        X_train,
        y_train,
        significance_level=threshold,
    )

    # 2) recursive split (top-down p-value pruning using your yd/psi rule)
    recursive_split_model = recursive_split_train_model(
        reg,
        X_train,
        y_train,
        threshold=threshold,
        d=d,
    )

    print(n_leaves(recursive_split_train_model))

    # Build the comparison table
    results = evaluate_models(
        {
            f"CCP+Pval (α={threshold})": ccp_pval_regularised_tree,
            f"Recursive split (p{'>='}{threshold})": recursive_split_model,
        },
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

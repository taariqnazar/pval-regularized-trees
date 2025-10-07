import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Union
from data import load_dataset
from sklearn.model_selection import train_test_split

# assumes: from your_code import load_dataset


def plot_all_from_pickle(
    pickle_path: Union[str, Path],
    outdir: Union[str, Path] = "rank_plots",
    step_outputs: bool = True,
    order_descending: bool = False,
    seed: int = 42,
) -> None:
    """
    Load models from a pickle with structure {ds_name: list_or_dict_of_models},
    and for each dataset:
      - load X, y via load_dataset(ds_name)
      - rank samples by the *first* model's score (x-axis is rank)
      - plot y (blue dots) and outputs of the remaining models (red)
      - save figure to outdir/{ds_name}_rank_plot.png

    Raises:
        FileNotFoundError, ValueError (for missing/invalid data).
    """
    p = Path(pickle_path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle not found: {p}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(p, "rb") as f:
        model_dict: Dict[str, Any] = pickle.load(f)

    if not isinstance(model_dict, dict) or not model_dict:
        raise ValueError(
            "Expected a non-empty dict {ds_name: models} in the pickle.")

    for ds_name, models_obj in model_dict.items():
        # normalize to ordered dict-like {name/index: model}
        ordering_model = models_obj.get("DoubleRegression/gbm_ccp05", None)
        if ordering_model is None:
            raise ValueError(
                f"Expected an ordering model for dataset '{ds_name}'.")

        ordering_model = ordering_model.ordering_regressor_

        # ---- load data ----
        X, y = load_dataset(ds_name)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        if X is None or y is None:
            raise ValueError(f"load_dataset('{ds_name}') returned no data.")

        X = Xte
        y = yte
        # ---- ordering by the Ordering model ----
        pseudo_X = ordering_model.predict(X).reshape(-1)
        order = np.argsort(pseudo_X, axis=0)

        x_ord = np.asarray(pseudo_X).reshape(-1)[order]
        y_sorted = np.asarray(y).reshape(-1)[order]

        plt.figure(figsize=(9, 5.5))
        plt.scatter(
            [i + 1 for i in range(len(y_sorted))],
            y_sorted,
            s=24,
            alpha=0.9,
            label="observations y",
        )

        # plot ALL models in models_obj
        for name, mdl in models_obj.items():
            if name == "DoubleRegression/gbm_ccp05":
                yhat_sorted = np.asarray(mdl.predict(X)).reshape(-1)[order]
                plt.plot(
                    yhat_sorted,
                    linewidth=2.0,
                    alpha=0.9,
                    label=str(name),
                    color="red",
                )

            if name == "GBM/gbm":
                yhat_sorted = np.asarray(mdl.predict(X)).reshape(-1)[order]
                plt.plot(
                    yhat_sorted,
                    linewidth=2.0,
                    alpha=0.9,
                    label=str(name),
                    color="orange",
                )

        # plt.title(f"{ds_name} — ordered by Ordering model")
        # plt.xlabel("ordered output of ordering model")
        # plt.ylabel("responses")
        # plt.legend(loc="best", ncol=2)  # tweak as you like
        plt.tight_layout()

        out_png = outdir / f"{ds_name}_rank_plot.pdf"
        plt.savefig(out_png, dpi=160)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    # adjust as needed
    pickle_path = "src/numerical_illustrations/outputs/run_variants_models.pkl"
    plot_all_from_pickle(pickle_path)

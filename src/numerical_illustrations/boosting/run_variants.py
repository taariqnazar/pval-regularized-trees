from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, List

import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data import load_dataset
from models import GBM, CCPPTreeBoosting, RecursivePTreeBoosting
from utils.metrics import rmse

MODEL_REGISTRY = {
    "GBM": GBM,
    "CCPPTreeBoosting": CCPPTreeBoosting,
    "RecursivePTreeBoosting": RecursivePTreeBoosting,
}


def _load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_models(models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand {ModelType: {variants: {name: params}}} into concrete instances.
    Keys use 'ModelType/variant' for clarity in results.
    """
    built: Dict[str, Any] = {}
    for model_type, block in models_cfg.items():
        if model_type not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model type '{model_type}'.")
        Model = MODEL_REGISTRY[model_type]
        variants = block.get("variants", {})
        if not variants:
            raise ValueError(f"No variants provided for '{model_type}'.")
        for var_name, params in variants.items():
            # copy to avoid mutating cfg
            p = dict(params)
            key = f"{model_type}/{var_name}"
            built[key] = Model(**p)
    return built


def _evaluate(model, Xtr, ytr, Xte, yte):
    y_tr = model.predict(Xtr)
    y_te = model.predict(Xte)

    out = {
        "rmse_train": float(np.sqrt(rmse(ytr, y_tr))),
        "rmse_test": float(np.sqrt(rmse(yte, y_te))),
        "n_estimators_": int(getattr(model, "n_estimators_", 0))
        if getattr(model, "n_estimators_", None) is not None
        else None,
    }

    if hasattr(model, "get_staged_complexity"):
        try:
            sc = model.get_staged_complexity()
            out["staged_complexity"] = [
                int(v) for v in sc] if sc is not None else None
        except Exception:
            out["staged_complexity"] = None

    if hasattr(model, "staged_predict"):
        staged = []
        for yp in model.staged_predict(Xte):
            staged.append(float(np.sqrt(mean_squared_error(yte, yp))))
        out["staged_rmse"] = staged
    else:
        out["staged_rmse"] = None

    return out


def run(cfg_path: str, outdir: str | Path = "src/numerical_illustrations/outputs"):
    cfg = _load_cfg(cfg_path)
    seed = int(cfg.get("random_seed", 0))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {"config": cfg, "datasets": {}}

    for ds_name in cfg["datasets"]:
        print(f"=== Dataset: {ds_name} ===")
        X, y = load_dataset(ds_name)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed)

        models = _build_models(cfg["models"])
        ds_results: Dict[str, Any] = {}
        for key, model in models.items():
            print(f"  -> Training {key}")
            model.fit(Xtr, ytr, random_state=seed)
            ds_results[key] = _evaluate(model, Xtr, ytr, Xte, yte)

        all_results["datasets"][ds_name] = ds_results

    outpath = outdir / "run_variants.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results → {outpath}")


if __name__ == "__main__":
    run("src/numerical_illustrations/boosting/config.yaml")

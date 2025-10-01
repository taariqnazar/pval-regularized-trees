# Regularisation of CART Trees by Summation of *p*-values — Code

Reproducible code for the experiments in the paper:

> Engler, Lindholm, Lindskog, Nazar — *Regularisation of CART trees by summation of p-values*. arXiv:2505.18769  
> https://arxiv.org/abs/2505.18769

This repository contains:
- Dataset loaders/generators (`src/data`)
- Tree models and boosting wrappers (`src/models`)
- Numerical illustrations and figure scripts (`src/numerical_illustrations`)
- Utility code for pruning, metrics, and *p*-values (`src/utils`)

---

## Quick start

```bash
# clone and enter the code directory
git clone <your-repo-url>
cd <repo>/code

# (recommended) create and activate a virtual env
python3 -m venv .venv
source ./.venv/bin/activate  # Windows: .\.venv\Scripts\activate

# install dependencies
pip install -U pip
# Option A: add src/ to Python path (simple & robust)
export PYTHONPATH=$PWD/src     # Windows PowerShell: $env:PYTHONPATH="$PWD/src"
# Option B (if your setuptools packaging is wired up): editable install
pip install -e .               # if this fails to expose packages, use Option A
```

> **Packaging note:** The project uses a `src/` layout but no explicit package discovery config in `pyproject.toml`. If `pip install -e .` doesn’t expose imports like `from data import ...`, set `PYTHONPATH=$PWD/src` before running scripts.

---

## Requirements

- Python ≥ 3.8
- Python packages (see `pyproject.toml`):  
  `numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, jupyterlab, dtreeviz, graphviz`

---

## Repository layout

```
code/
├── pyproject.toml
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── bemtpl16.py
    │   ├── california_housing.py
    │   ├── core.py
    │   └── synthetic.py
    ├── models/
    │   ├── __init__.py
    │   ├── ccp_ptree.py
    │   ├── ccp_ptree_boosting.py
    │   ├── gbm.py
    │   ├── recursive_ptree.py
    │   └── recursive_ptree_boosting.py
    ├── numerical_illustrations/
    │   ├── figure1_table1.py
    │   ├── figure2.py
    │   ├── figure4_figure5.py
    │   └── boosting/
    │       ├── generate_results_table.py
    │       └── run_variants.py
    └── utils/
        ├── __init__.py
        ├── metrics.py
        ├── protocols.py
        ├── prune.py
        └── stats/
            ├── __init__.py
            └── pvalues.py
```

---

## Datasets

### Built-in loaders / generators

Unified API:

```python
from data import load_dataset, available_datasets

print(available_datasets())
# ['bemtpl16', 'california_housing', 'linear_model', 'neufeldt']

X, y = load_dataset("california_housing")                   # sklearn fetch
X, y = load_dataset("bemtpl16", csv_path="src/data/raw/BEMTPL16.csv")  # CSV
X, y, beta = load_dataset("linear_model", n=200, p=10)      # synthetic
X, y = load_dataset("neufeldt", a=1.0, b=0.5)               # synthetic
```

**BEMTPL16 CSV:** download the dataset and place it at `src/data/raw/BEMTPL16.csv`, or pass a custom path via `csv_path=...`.

---

## Models

Import aliases:

```python
from models import CCPPTree, RecursivePTree
from models import CCPPTreeBoosting, RecursivePTreeBoosting, GBM
```

### Standalone trees

```python
from data import load_dataset
from models import CCPPTree, RecursivePTree

X, y = load_dataset("linear_model", n=500, p=10)

# Cost-complexity path + cumulative p-value rule
tree = CCPPTree(significance_level=0.05, max_depth=6, min_samples_leaf=10)
tree.fit(X, y, random_state=0)
yhat = tree.predict(X)

# Recursive top-down pruning by node p-values
rtree = RecursivePTree(threshold=0.05, prune_if=">=", max_depth=6, min_samples_leaf=10)
rtree.fit(X, y, random_state=0)
yhat2 = rtree.predict(X)
```

### Boosting wrappers

```python
from models import CCPPTreeBoosting, RecursivePTreeBoosting

X, y = load_dataset("linear_model", n=1000, p=10)

ensemble = CCPPTreeBoosting(learning_rate=0.1, max_estimators=200,
                            significance_level=0.05, max_depth=2, min_samples_leaf=20)
ensemble.fit(X, y, random_state=0)
pred = ensemble.predict(X)
```

`GBM` is a thin wrapper around `sklearn.ensemble.GradientBoostingRegressor` with optional CV-based early stopping (`k_folds>=2`).

---

## Reproducing figures / experiments

> Make sure `PYTHONPATH=$PWD/src` (or that an editable install works). Run commands from `code/`.

### Figure 1 + Table 1

Monte Carlo for empirical vs approximated quantiles of the YD statistic.

```bash
python src/numerical_illustrations/figure1_table1.py
# Outputs:
# - Figure1.pdf
# - Table values printed to stdout
```

### Figure 2

Detection rate vs sample size for independent/dependent covariates.

```bash
python src/numerical_illustrations/figure2.py
# Outputs:
# - Figure2.pdf
```

### Figures 4 & 5

Neufeld regression tree experiment with cumulative *p*-value pruning along the CCP path.

```bash
python src/numerical_illustrations/figure4_figure5.py
# Displays two figures sequentially
```

### Boosting experiments

Config-driven runs across datasets and model variants; results saved to JSON.

1) Create a config file at `src/numerical_illustrations/boosting/config.yaml` (example below).  
2) Run experiments and then print result tables.

```bash
python src/numerical_illustrations/boosting/run_variants.py
python src/numerical_illustrations/boosting/generate_results_table.py
# Outputs:
# - src/numerical_illustrations/outputs/run_variants.json
# - Tables printed to stdout
```

**Example `config.yaml`:**

```yaml
random_seed: 0

datasets:
  - california_housing
  - linear_model

models:
  GBM:
    variants:
      default:
        k_folds: 1
        max_estimators: 300
        learning_rate: 0.1
        max_depth: 2

  CCPPTreeBoosting:
    variants:
      sl05_depth2:
        learning_rate: 0.1
        max_estimators: 300
        significance_level: 0.05
        max_depth: 2
        min_samples_leaf: 20

  RecursivePTreeBoosting:
    variants:
      thr05_depth2:
        learning_rate: 0.1
        max_estimators: 300
        threshold: 0.05
        prune_if: ">="
        max_depth: 2
        min_samples_leaf: 20
```

---

## Tips & troubleshooting

- **Long Monte Carlo runs**: Scripts like `figure1_table1.py`/`figure2.py` use large `M` by default. Reduce it when testing.
- **Graphviz issues**: Ensure the system Graphviz is installed and on `PATH` if `dtreeviz` is used.
- **Import errors**: Prefer `export PYTHONPATH=$PWD/src` when in doubt.

---

## Reference

For details on the methodology and experimental design, see the paper:  
https://arxiv.org/abs/2505.18769

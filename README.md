# pval-regularized-trees

Reproducibility code for:

> Engler, Lindholm, Lindskog, Nazar. *Regularisation of regression trees by summation of p-values.* arXiv:2505.18769.

Each script under `experiments/` reproduces exactly one (sub)section of the paper. Outputs land in `results/<same-stem>/`.

## Quick start

```bash
git clone https://github.com/taariqnazar/pval-regularized-trees
cd pval-regularized-trees
python -m venv .venv && source .venv/bin/activate
make install
make all          # regenerates every figure and table from Section 4
make test         # smoke test: runs each experiment with --quick and checks outputs exist
```

## What reproduces what

| Paper section | Script | Outputs |
|---|---|---|
| 4.1 — *p*-value approximation for a single split | `experiments/exp_4_1_pvalue_approximation.py` | `figure1.pdf`, `figure2.pdf`, `table1.csv` |
| 4.2 — Simulated examples from Neufeldt et al. | `experiments/exp_4_2_neufeldt_pruning.py` | `figure3.pdf`, `figure4.pdf`, `figure5.pdf`, `figure6.pdf` |
| 4.2.1 — Comparison to Cappelli et al. | `experiments/exp_4_2_1_cappelli_comparison.py` | `table2.csv`, `figure7.pdf` |
| 4.2.2 — Randomness via cross-validation | `experiments/exp_4_2_2_cv_randomness.py` | `figure8.pdf`, `table3.csv` |
| 4.3 — Real-data comparison | `experiments/exp_4_3_real_data.py` | `table4.csv`, `table5.csv`, `raw.csv` |
| 4.4 — Auto-calibrated predictor | *(out of scope; not in this repo)* | — |

Each script accepts `--seed`, `--out-dir`, and `--quick` (small Monte Carlo for smoke testing). Run any script with `--help` to see its options.

## Library API

```python
from trees.data import load_dataset, available_datasets
from trees.models import PSumTree, CappelliSTP, CCPCV, GSellStop
from trees.stats import Psi, get_p_values, build_ccp_chain

X, y = load_dataset("boston_housing")
m = PSumTree(significance_level=0.05, max_depth=6, min_samples_leaf=20)
m.fit(X, y, random_state=0)
y_hat = m.predict(X)
```

Available models:

| Class | Section | Notes |
|---|---|---|
| `PSumTree` | 4.* | Cumulative Ψ-sum stopping along the CCP path. *Paper's main method.* |
| `CCPCV` | 4.2.2, 4.3 | sklearn CCP path + 1-SE CV-rule baseline. |
| `CappelliSTP` | 4.2.1, 4.3 | F-test pruning on a 70/30 grow/prune split. |
| `GSellStop` | — | ForwardStop / StrongStop, two sequence strategies. *Exploratory; not in revised paper.* |
| `RecursivePSumTree` | — | v1 recursive top-down node-wise pruning. *Kept for v1 reproducibility.* |
| `PSumBoosting`, `RecursivePSumBoosting`, `GBM`, `Intercept`, `DoubleRegression` | — | Boosting wrappers / baselines. |

## Datasets

- `boston_housing` — 506 × 13, MEDV. CSV ships with the repo (StatLib snapshot, ref [21]).
- `box_lunch` — 226 × 16, kcal24h0. CSV ships with the repo (R `visTree::blsdata`).
- `california_housing` — 20640 × 8, MedHouseVal. Fetched via sklearn; cached CSV included.
- `linear_model` (synthetic) — Gaussian linear, used by Section 4.2.2.
- `neufeldt` (synthetic) — Eq. (19), used by Section 4.2 / 4.2.1.
- `bemtpl16` — v1 dataset, manual download required (not used by any current experiment).

## Legacy v1 experiments (not in revised paper)

`experiments/legacy/` holds v1 boosting-variant scripts that don't map to a section of the revised manuscript. Run via `make legacy`.

## Reproducibility notes

- All randomness is seeded; default seeds match the paper's Monte Carlo settings.
- Pinned floors in `pyproject.toml`: `numpy>=1.24`, `scipy>=1.11`, `scikit-learn>=1.3`.
- No R, no rpy2: the repo is pure Python.
- Reproducibility target: *qualitative* (figures and tables look the same, conclusions hold). Bit-exact match across NumPy/SciPy versions is not guaranteed.

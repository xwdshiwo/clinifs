# clinifs — Clinical Feature Selection Toolkit

**clinifs** is a scikit-learn-compatible Python package for benchmarking and applying feature selection methods to clinical cancer gene-expression panels. It accompanies the paper:

> *Multi-dimensional benchmarking of feature selection methods for clinical small-panel cancer gene-expression classification*, Briefings in Bioinformatics, 2026.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/badge/PyPI-clinifs-orange)](https://pypi.org/project/clinifs)

---

## Installation

```bash
pip install clinifs
```

Or from source:

```bash
git clone https://github.com/<repo>/clinifs.git
cd clinifs
pip install -e ".[dev]"
```

---

## Quick start (< 5 seconds)

```python
from clinifs import FeatureSelector

# Load your data (rows = samples, cols = genes)
import pandas as pd
X = pd.read_csv("examples/example_microarray.csv", index_col=0).values
y = pd.read_csv("examples/example_labels.csv", index_col=0).values.ravel()

# Adaptive method selection (paper's two-tier recommendation)
fs = FeatureSelector(method="auto", k=20)
fs.fit(X, y)

print(f"Tier detected : {fs.tier_}")          # 'easy_medium' or 'hard'
print(f"Selected genes: {fs.selected_indices_}")
X_panel = fs.transform(X)                      # shape (n_samples, 20)
```

Run `python examples/quickstart.py` or `clinifs demo` for a full walk-through.

---

## Available methods

| Family | Method | Key | Speed |
|---|---|---|---|
| **Filter** | Variance Threshold | `variance` | ⚡ |
| Filter | ANOVA F-score | `anova` | ⚡ |
| Filter | Mutual Information | `mi` | ⚡ |
| Filter | mRMR | `mrmr` | 🔵 |
| Filter | ReliefF | `relieff` | 🔵 |
| **Embedded** | L1-Logistic | `l1` | ⚡ |
| Embedded | ElasticNet | `elasticnet` | 🔵 |
| Embedded | LinearSVC-L1 | `linsvc` | ⚡ |
| **Wrapper** | BorutaPy | `boruta` | 🔴 |
| Wrapper | ExtraTrees | `extratrees` | ⚡ |
| Wrapper | RFECV | `rfecv` | 🔵 |
| Wrapper | Genetic Algorithm | `ga` | 🔴 |
| Wrapper | Binary PSO | `bpso` | 🔴 |
| **RRA (aggregate)** | Adaptive RRA | `auto` / `rra` | ⚡ |

⚡ < 1 s/fit · 🔵 1–10 s/fit · 🔴 30–120 s/fit (on microarray after ANOVA pre-filter)

---

## Adaptive RRA wrapper

The paper's recommended default: **RRA(ANOVA + MI)** for Easy/Medium datasets, **RRA(ANOVA + MI + MEL)** for Hard datasets. The package auto-detects the tier from a random-gene baseline AUC probe.

```python
from clinifs import RankAggregateFilter

# Two-method RRA (Easy / Medium)
rra = RankAggregateFilter(methods=["anova", "mi"], k=20)
rra.fit(X_train, y_train)
X_sel = rra.transform(X_test)

# Three-method RRA with external MEL scores (Hard tier)
rra3 = RankAggregateFilter(methods=["anova", "mi", "mel"], k=20)
rra3.fit(X_train, y_train, extra_scores={"mel": mel_scores})
```

---

## Stability metrics

```python
from clinifs.evaluation import stability_report

# feature_sets: list of arrays of selected feature indices across folds
report = stability_report(feature_sets, n_features=X.shape[1])
print(report["nogueira_phi"])     # Nogueira Φ ∈ (-∞, 1]
print(report["kuncheva_jaccard"]) # mean pairwise Jaccard
```

---

## CLI

```bash
clinifs --help
clinifs methods                            # list all methods
clinifs version
clinifs demo                               # run quickstart on built-in example data
clinifs run --x X.csv --y y.csv --method auto --k 20 --out selected.csv
```

---

## Reproducing paper results

See [`docs/reproduce_paper.md`](docs/reproduce_paper.md) for step-by-step instructions to reproduce the 1 155 benchmark evaluation units, 6-pair external validation, and RRA Hard-tier experiments.

---

## Citing

```bibtex
@article{clinifs2026,
  title   = {Multi-dimensional benchmarking of feature selection methods for
             clinical small-panel cancer gene-expression classification},
  author  = {Author(s) TBD},
  journal = {Briefings in Bioinformatics},
  year    = {2026},
  doi     = {TBD}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

# Reproducing Paper Results

This document describes how to reproduce the main benchmark results reported in:

> *Multi-dimensional benchmarking of feature selection methods for clinical small-panel cancer gene-expression classification*, accompanying manuscript (submitted).

---

## Requirements

```bash
pip install "git+https://github.com/xwdshiwo/clinifs.git@v0.1.0" "skrebate>=0.62" "mrmr-selection>=0.2.6" "boruta>=0.3"
pip install pandas numpy scipy scikit-learn
```

Python 3.10+, tested on Python 3.12.13.

---

## Data

All datasets are publicly available from GEO (accession numbers in paper Table 1). The dataset manifest and reconstruction entry points are maintained in the lightweight benchmark repository:

https://github.com/xwdshiwo/clinifs-benchmark

Directory structure expected:

```
data/
  Bladder_GSE31189.csv
  Prostate_GSE6919.csv
  Renal_GSE53757.csv
  Colorectal_GSE44861.csv
  Breast_GSE70947.csv
  Pancreatic_GSE16515.csv
  Liver_GSE14520.csv
  Liver_GSE76427.csv
  Lung_GSE19804.csv
  Colorectal_GSE44076.csv
  Leukemia_GSE63270.csv
```

---

## Main benchmark (1 155 evaluation units)

The full 5 × 5 repeated outer-CV benchmark is implemented in the public benchmark repository:

https://github.com/xwdshiwo/clinifs-benchmark

Using clinifs selectors directly mirrors the paper's implementation:

```python
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from clinifs.selectors import ANOVASelector  # substitute any selector

df = pd.read_csv("data/Bladder_GSE31189.csv", index_col=0)
X = df.drop("label", axis=1).values.astype(float)
y = df["label"].values

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr, te in outer_cv.split(X, y):
    sel = ANOVASelector(k=20)
    sel.fit(X[tr], y[tr])
    X_tr_sel = sel.transform(X[tr])
    X_te_sel = sel.transform(X[te])
    clf = LogisticRegression(C=1, solver="liblinear", max_iter=300, random_state=42)
    clf.fit(X_tr_sel, y[tr])
    prob = clf.predict_proba(X_te_sel)[:, 1]
    aucs.append(roc_auc_score(y[te], prob))

print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
```

For all 15 methods × 7 k × 11 datasets, use the benchmark runner in the paper repository (`run_v2_dispatcher.py` or the lightweight `run_all.py` smoke-test entry point).

Expected result for ANOVA on Bladder_GSE31189, k=20: AUC ≈ 0.62 ± 0.08 (Hard tier).

---

## Adaptive RRA (§3.5, §3.6)

```python
from clinifs import FeatureSelector

fs = FeatureSelector(method="auto", k=20)
fs.fit(X[tr], y[tr])
print(fs.tier_)   # 'hard' for Bladder/Prostate; 'easy_medium' for others
```

Expected: Bladder and Prostate datasets trigger `tier_='hard'`; all remaining 9 datasets trigger `'easy_medium'`.

---

## Stability metrics

```python
from sklearn.model_selection import StratifiedKFold
from clinifs.selectors import ANOVASelector
from clinifs.evaluation import stability_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_sets = []
for tr, _ in skf.split(X, y):
    sel = ANOVASelector(k=20)
    sel.fit(X[tr], y[tr])
    fold_sets.append(sel.selected_indices_)

report = stability_report(fold_sets, n_features=X.shape[1])
# Expected for ANOVA on Easy-tier dataset: Nogueira Φ ≈ 0.75–0.80
```

---

## Pre-computed results

Full fold-level outputs (AUC per fold, stability per fold, biological annotations) are not stored directly in GitHub because of size constraints, but can be reconstructed from the public accessions, dataset manifest, and benchmark runners. The manuscript reports summary-level outputs organized under:

```
benchmark_outputs/
  E2_main_benchmark/      # 1155 evaluation units
  E3_external_validation/ # 6-pair cohort validation
  E5_random_baseline/     # 96 250 random-gene baseline runs
  E6c_rra_hard/           # RRA Hard-tier comparison (Table 3, SI §S10)
```

---

## Compute environment

Main benchmark: Windows workstation, Intel i7, 16 GB RAM, 5 parallel processes.  
Total runtime: ~8 h (main) + 25 min (external val) + 57 min (diagnostics + baseline).  
Dependency versions: `scikit-learn==1.3.x`, `skrebate==0.62`, `mrmr-selection==0.2.8`.

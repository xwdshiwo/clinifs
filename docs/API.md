# clinifs API Reference

## Top-level imports

```python
from clinifs import FeatureSelector, RankAggregateFilter
from clinifs import __version__
```

---

## `FeatureSelector`

High-level adaptive wrapper following the paper's two-tier recommendation.

### Constructor

```python
FeatureSelector(method='auto', k=20, extra_scores=None, selector_kwargs=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `'auto'` | Method key. `'auto'` estimates difficulty and picks RRA variant. See method table below. |
| `k` | int | `20` | Panel size (number of features to select). |
| `extra_scores` | dict \| None | `None` | External scorer arrays, e.g. `{'mel': <array>}` for three-method RRA. |
| `selector_kwargs` | dict \| None | `None` | Forwarded to underlying selector `__init__`. |

### Method keys

| Key | Underlying class | Notes |
|---|---|---|
| `'auto'` | adaptive | Estimates tier; Easy/Medium → RRA(anova,mi); Hard → RRA(anova,mi,[mel]) |
| `'rra'` | `RankAggregateFilter(['anova','mi'])` | Best all-rounder on Easy/Medium |
| `'anova'` | `ANOVASelector` | Single ANOVA filter |
| `'mi'` | `MISelector` | Mutual information |
| `'mrmr'` | `mRMRSelector` | min-Redundancy max-Relevance |
| `'relieff'` | `ReliefFSelector` | ReliefF |
| `'variance'` | `VarianceSelector` | Unsupervised; control baseline |
| `'l1'` | `L1LogisticSelector` | L1-Logistic (Embedded) |
| `'elasticnet'` | `ElasticNetSelector` | ElasticNet (Embedded) |
| `'linsvc'` | `LinSVCL1Selector` | LinearSVC-L1 (Embedded) |
| `'boruta'` | `BorutaSelector` | Boruta (slow) |
| `'extratrees'` | `ExtraTreesSelector` | ExtraTrees importance |
| `'rfecv'` | `RFECVSelector` | Recursive Feature Elimination + CV |
| `'ga'` | `GASelector` | Genetic Algorithm (slow) |
| `'bpso'` | `BPSOSelector` | Binary PSO (slow) |

### Attributes after `fit`

| Attribute | Type | Description |
|---|---|---|
| `selected_indices_` | `ndarray[int]` | Indices of selected features (original space) |
| `selector_` | fitted selector | Underlying fitted selector |
| `tier_` | str \| None | `'easy_medium'` or `'hard'` (set when `method='auto'`) |
| `random_baseline_auc_` | float \| None | Median random-gene baseline AUC used for tier detection |
| `n_features_in_` | int | Number of input features |

### Methods

```python
fs.fit(X, y)            # fit; returns self
fs.transform(X)         # returns X[:, selected_indices_]
fs.fit_transform(X, y)  # shortcut
```

---

## `RankAggregateFilter`

Aggregate 2–N ranked feature lists via Robust Rank Aggregation (Kolde et al. 2012).

### Constructor

```python
RankAggregateFilter(methods=('anova', 'mi'), k=20, prefilter=2000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `methods` | list[str] | `['anova','mi']` | Built-in: `'anova'`, `'mi'`. External (e.g. MEL) passed via `extra_scores`. |
| `k` | int | `20` | Panel size. |
| `prefilter` | int | `2000` | Reduce to top-n ANOVA features before aggregation (for speed on high-dim data). |

### fit signature

```python
rra.fit(X, y, extra_scores=None)
```

`extra_scores`: `dict[str, array-like]` — additional score arrays (length n_features or n_sub).

### Attributes after `fit`

| Attribute | Type | Description |
|---|---|---|
| `selected_indices_` | `ndarray[int]` | Selected feature indices (original space) |
| `rho_scores_` | `ndarray[float]` | RRA ρ-score per feature (lower = better); Bonferroni-corrected |
| `n_features_in_` | int | — |

---

## Individual selectors

All selectors share the same API:

```python
sel = <SelectorClass>(k=20)
sel.fit(X, y)
sel.transform(X)          # -> X[:, sel.selected_indices_]
sel.fit_transform(X, y)
```

Attributes: `selected_indices_`, `scores_` (or `importances_`), `n_features_in_`.

### Filter family

```python
from clinifs.selectors import (
    VarianceSelector, ANOVASelector, MISelector, mRMRSelector, ReliefFSelector
)
```

### Embedded family

```python
from clinifs.selectors import (
    L1LogisticSelector, ElasticNetSelector, LinSVCL1Selector
)
```

### Wrapper family

```python
from clinifs.selectors import (
    BorutaSelector, ExtraTreesSelector, RFECVSelector,  # classical
    GASelector, BPSOSelector,                           # evolutionary
)
```

---

## Stability metrics

```python
from clinifs.evaluation import nogueira_phi, kuncheva_jaccard, stability_report
```

### `nogueira_phi(feature_sets, n_features)`

Nogueira Φ stability index (Nogueira et al. 2018). Returns float ∈ (−∞, 1]; 0 = random baseline.

### `kuncheva_jaccard(feature_sets)`

Mean pairwise Jaccard similarity across all fold pairs. Returns float ∈ [0, 1].

### `stability_report(feature_sets, n_features)`

Returns `dict` with keys `nogueira_phi`, `kuncheva_jaccard`, `mean_panel_size`, `n_folds`.

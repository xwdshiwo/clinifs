"""
clinifs quickstart — run with: python quickstart.py  OR  clinifs demo

Demonstrates:
  1. Adaptive feature selection (auto tier)
  2. Explicit RRA(ANOVA + MI)
  3. Single-method selectors
  4. Stability metric
"""
import pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
# Allow running from any working directory
sys.path.insert(0, str(HERE.parent))

# ── Generate example data if not present ──────────────────────────────────────
X_path = HERE / "example_microarray.csv"
y_path = HERE / "example_labels.csv"
if not X_path.exists():
    print("Building example data …")
    from build_example_data import build
    build()

import pandas as pd
X = pd.read_csv(X_path, index_col=0).values.astype(float)
y = pd.read_csv(y_path, index_col=0).values.ravel()
print(f"\nLoaded:  X {X.shape}  |  classes {np.unique(y)}")

# ── 1. Adaptive RRA (paper recommendation) ────────────────────────────────────
from clinifs import FeatureSelector

fs = FeatureSelector(method="auto", k=20)
fs.fit(X, y)
X_panel = fs.transform(X)
print(f"\n[1] Auto tier  : {fs.tier_}")
print(f"    Selected   : {fs.selected_indices_[:5]} … ({len(fs.selected_indices_)} genes)")
print(f"    X_panel    : {X_panel.shape}")

# ── 2. Explicit RRA(ANOVA + MI) ───────────────────────────────────────────────
from clinifs import RankAggregateFilter

rra = RankAggregateFilter(methods=["anova", "mi"], k=20)
rra.fit(X, y)
print(f"\n[2] RRA(ANOVA+MI) top-5 ρ-scores: {rra.rho_scores_[rra.selected_indices_[:5]].round(4)}")

# ── 3. Single-method selector ─────────────────────────────────────────────────
from clinifs.selectors import ANOVASelector, mRMRSelector

anova = ANOVASelector(k=10)
anova.fit(X, y)
print(f"\n[3] ANOVA top-10: {anova.selected_indices_}")

# ── 4. Stability report ───────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from clinifs.evaluation import stability_report

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_sets = []
for tr, _ in skf.split(X, y):
    sel = ANOVASelector(k=20)
    sel.fit(X[tr], y[tr])
    fold_sets.append(sel.selected_indices_)

report = stability_report(fold_sets, n_features=X.shape[1])
print(f"\n[4] Stability (5-fold ANOVA, k=20)")
print(f"    Nogueira Φ     : {report['nogueira_phi']:.4f}")
print(f"    Kuncheva Jaccard: {report['kuncheva_jaccard']:.4f}")

print("\n✓ Quickstart complete.\n")

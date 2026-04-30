"""
Wrapper (classical) feature selectors (3 methods).

These use sequential or tree-based search strategies with an internal
classifier.  They are slower than filter/embedded methods on microarray
data; use with `prefilter=2000` (default) for reasonable runtime.

**RFECV warning**: On small datasets (< 80 samples) RFECV is prone to
overfitting the discovery cohort; check external validation AUC before
reporting RFECV panels in clinical settings.
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from clinifs._prefilter import anova_prefilter


# ──────────────────────────────────────────────────────────────────────────────
# 1. BorutaSelector
# ──────────────────────────────────────────────────────────────────────────────

class BorutaSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper feature selection via BorutaPy (Kursa & Rudnicki 2010).

    Requires the ``boruta`` package::

        pip install boruta

    Confirmed features are ranked by RF importance; tentative features fill
    remaining slots if confirmed set has fewer than k members.
    """

    def __init__(self, k: int = 20, prefilter: int = 2000,
                 n_estimators: int = 50, max_depth: int = 5,
                 max_iter: int = 50, random_state: int = 42):
        self.k = k
        self.prefilter = prefilter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from boruta import BorutaPy
        except ImportError as e:
            raise ImportError(
                "boruta is required for BorutaSelector. "
                "Install with: pip install boruta"
            ) from e

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            n_jobs=2, random_state=self.random_state
        )
        selector = BorutaPy(
            rf, n_estimators="auto", max_iter=self.max_iter,
            verbose=0, random_state=self.random_state
        )
        selector.fit(X_sub, y)

        # Get RF importances for ranking
        rf.fit(X_sub, y)
        importances = rf.feature_importances_

        confirmed = np.where(selector.support_)[0]
        tentative = np.where(selector.support_weak_)[0]
        others = np.setdiff1d(
            np.arange(len(importances)), np.union1d(confirmed, tentative)
        )
        pool = (
            list(confirmed[np.argsort(-importances[confirmed])])
            + list(tentative[np.argsort(-importances[tentative])])
            + list(others[np.argsort(-importances[others])])
        )
        sub_idx = np.array(pool[: self.k])
        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = importances

        self.scores_ = scores_full
        self.selected_indices_ = pre_idx[sub_idx]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 2. ExtraTreesSelector
# ──────────────────────────────────────────────────────────────────────────────

class ExtraTreesSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper feature selection via Extra-Trees feature importance.
    Faster than BorutaPy with comparable performance on microarray data.
    """

    def __init__(self, k: int = 20, prefilter: int = 2000,
                 n_estimators: int = 100, random_state: int = 42):
        self.k = k
        self.prefilter = prefilter
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators, n_jobs=2,
            random_state=self.random_state
        )
        et.fit(X_sub, y)
        imp = et.feature_importances_

        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = imp
        self.scores_ = scores_full

        sub_idx = np.argsort(imp)[::-1][: min(self.k, len(imp))]
        self.selected_indices_ = pre_idx[sub_idx]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 3. RFECVSelector
# ──────────────────────────────────────────────────────────────────────────────

class RFECVSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper feature selection via Recursive Feature Elimination with CV.

    .. warning::
        On small discovery cohorts (< 80 samples) RFECV is prone to deep
        overfitting.  Benchmark results show val_AUC = 0.556 on the Prostate
        cohort-shift pair.  Use with caution in clinical settings.
    """

    def __init__(self, k: int = 20, prefilter: int = 2000,
                 C: float = 1.0, step: float = 0.1, cv: int = 3,
                 random_state: int = 42):
        self.k = k
        self.prefilter = prefilter
        self.C = C
        self.step = step
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.feature_selection import RFECV

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        if len(y) < 80:
            warnings.warn(
                "RFECVSelector: sample size < 80. "
                "Deep RFECV overfitting risk is elevated. "
                "Validate on an independent cohort before clinical use.",
                UserWarning,
                stacklevel=2,
            )

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        estimator = LogisticRegression(
            C=self.C, solver="liblinear",
            max_iter=500, random_state=self.random_state
        )
        selector = RFECV(
            estimator=estimator, step=self.step, cv=self.cv,
            scoring="roc_auc", n_jobs=-1,
            min_features_to_select=self.k
        )
        selector.fit(X_sub, y)
        ranking = selector.ranking_
        sub_idx = np.argsort(ranking)[: self.k]

        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = 1.0 / ranking
        self.scores_ = scores_full
        self.selected_indices_ = pre_idx[sub_idx]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]

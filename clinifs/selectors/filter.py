"""
Filter-family feature selectors (5 methods).

All classes follow the scikit-learn Transformer API:
    fit(X, y) -> self
    transform(X) -> X[:, selected_indices_]
    fit_transform(X, y) -> shortcut via TransformerMixin

Attributes set after fit
------------------------
selected_indices_ : np.ndarray[int]   indices into the *original* feature axis
scores_           : np.ndarray[float] | None   per-feature relevance score
n_features_in_    : int
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from clinifs._prefilter import anova_prefilter


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]


# ──────────────────────────────────────────────────────────────────────────────
# 1. VarianceSelector
# ──────────────────────────────────────────────────────────────────────────────

class VarianceSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by sample variance (unsupervised)."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.scores_ = np.var(X, axis=0)
        self.selected_indices_ = _topk(self.scores_, self.k)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 2. ANOVASelector
# ──────────────────────────────────────────────────────────────────────────────

class ANOVASelector(BaseEstimator, TransformerMixin):
    """Select top-k features by ANOVA F-score."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        scores, _ = f_classif(X, y)
        self.scores_ = np.nan_to_num(scores, nan=0.0)
        self.selected_indices_ = _topk(self.scores_, self.k)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 3. MISelector  (Mutual Information)
# ──────────────────────────────────────────────────────────────────────────────

class MISelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features by Mutual Information.

    Uses ANOVA pre-filter (top 2 000) before computing MI to keep runtime
    reasonable on microarray-scale data (50 K+ features).
    """

    def __init__(self, k: int = 20, prefilter: int = 2000,
                 random_state: int = 42):
        self.k = k
        self.prefilter = prefilter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        mi_sub = mutual_info_classif(X_sub, y, random_state=self.random_state)

        # Map back to full feature space
        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = mi_sub

        self.scores_ = scores_full
        sub_top = _topk(mi_sub, self.k)
        self.selected_indices_ = pre_idx[sub_top]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 4. mRMRSelector  (Minimum Redundancy Maximum Relevance)
# ──────────────────────────────────────────────────────────────────────────────

class mRMRSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features by mRMR (Ding & Peng 2005).

    Requires the ``mrmr-selection`` package::

        pip install mrmr-selection

    Uses ANOVA pre-filter (top 2 000) for speed on high-dimensional inputs.
    """

    def __init__(self, k: int = 20, prefilter: int = 2000):
        self.k = k
        self.prefilter = prefilter

    def fit(self, X, y):
        try:
            from mrmr import mrmr_classif
        except ImportError as e:
            raise ImportError(
                "mrmr-selection is required for mRMRSelector. "
                "Install with: pip install mrmr-selection"
            ) from e
        import contextlib
        import io
        import pandas as pd

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        k_eff = min(self.k, X_sub.shape[1])

        feat_names = [f"f{i}" for i in range(X_sub.shape[1])]
        df = pd.DataFrame(X_sub, columns=feat_names)
        with contextlib.redirect_stderr(io.StringIO()):
            selected_names = mrmr_classif(X=df, y=pd.Series(y), K=k_eff)

        sub_idx = np.array([int(n[1:]) for n in selected_names])
        self.selected_indices_ = pre_idx[sub_idx]
        self.scores_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 5. ReliefFSelector
# ──────────────────────────────────────────────────────────────────────────────

class ReliefFSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features by ReliefF (Robnik-Šikonja & Kononenko 2003).

    Requires the ``skrebate`` package::

        pip install skrebate

    Uses ANOVA pre-filter (top 2 000) before ReliefF for speed.
    """

    def __init__(self, k: int = 20, prefilter: int = 2000,
                 n_neighbors: int = 10):
        self.k = k
        self.prefilter = prefilter
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        try:
            from skrebate import ReliefF
        except ImportError as e:
            raise ImportError(
                "skrebate is required for ReliefFSelector. "
                "Install with: pip install skrebate"
            ) from e

        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        k_eff = min(self.k, X_sub.shape[1])
        n_nb = min(self.n_neighbors, len(y) - 1)

        rf = ReliefF(n_features_to_select=k_eff, n_neighbors=n_nb, n_jobs=1)
        rf.fit(X_sub, y)

        sub_idx = rf.top_features_[:k_eff]
        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = rf.feature_importances_

        self.scores_ = scores_full
        self.selected_indices_ = pre_idx[sub_idx]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]

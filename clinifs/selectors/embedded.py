"""
Embedded-family feature selectors (3 methods).

All three regularised linear models assign a coefficient magnitude to each
feature; we select the top-k by |coef|.  A global ANOVA pre-filter is applied
for saga/libsvm speed on high-dimensional data.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from clinifs._prefilter import anova_prefilter


def _coef_topk(coef: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(coef))[::-1][:min(k, len(coef))]


# ──────────────────────────────────────────────────────────────────────────────
# 1. L1LogisticSelector
# ──────────────────────────────────────────────────────────────────────────────

class L1LogisticSelector(BaseEstimator, TransformerMixin):
    """
    Embedded feature selection via L1-penalised Logistic Regression.
    Top-k features are chosen by |coefficient| magnitude.
    """

    def __init__(self, k: int = 20, C: float = 0.1,
                 max_iter: int = 1000, random_state: int = 42):
        self.k = k
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        clf = LogisticRegression(
            penalty="l1", C=self.C, solver="liblinear",
            max_iter=self.max_iter, random_state=self.random_state
        )
        clf.fit(X, y)
        coef = np.abs(clf.coef_).ravel()
        self.scores_ = coef
        self.selected_indices_ = _coef_topk(coef, self.k)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 2. ElasticNetSelector
# ──────────────────────────────────────────────────────────────────────────────

class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """
    Embedded feature selection via Elastic-Net-penalised Logistic Regression.
    Uses ANOVA pre-filter (top 2 000) before fitting the saga solver.
    """

    def __init__(self, k: int = 20, C: float = 0.1, l1_ratio: float = 0.5,
                 prefilter: int = 2000, max_iter: int = 1000,
                 random_state: int = 42):
        self.k = k
        self.C = C
        self.l1_ratio = l1_ratio
        self.prefilter = prefilter
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=self.l1_ratio, C=self.C,
            max_iter=self.max_iter, random_state=self.random_state, n_jobs=1
        )
        clf.fit(X_sub, y)
        coef = np.abs(clf.coef_).ravel()

        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = coef
        self.scores_ = scores_full
        self.selected_indices_ = pre_idx[_coef_topk(coef, self.k)]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


# ──────────────────────────────────────────────────────────────────────────────
# 3. LinSVCL1Selector
# ──────────────────────────────────────────────────────────────────────────────

class LinSVCL1Selector(BaseEstimator, TransformerMixin):
    """
    Embedded feature selection via L1-penalised Linear SVC.
    Uses ANOVA pre-filter (top 2 000) before fitting.
    """

    def __init__(self, k: int = 20, C: float = 0.1,
                 prefilter: int = 2000, max_iter: int = 2000,
                 random_state: int = 42):
        self.k = k
        self.C = C
        self.prefilter = prefilter
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        clf = LinearSVC(
            penalty="l1", dual=False, C=self.C,
            max_iter=self.max_iter, random_state=self.random_state
        )
        clf.fit(X_sub, y)
        coef = np.abs(clf.coef_).ravel()

        scores_full = np.zeros(X.shape[1])
        scores_full[pre_idx] = coef
        self.scores_ = scores_full
        self.selected_indices_ = pre_idx[_coef_topk(coef, self.k)]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]

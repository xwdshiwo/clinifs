"""
Robust Rank Aggregation (RRA) wrapper and adaptive FeatureSelector.

Reference
---------
Kolde R, Laur S, Adler P, Vilo J (2012).
  Robust rank aggregation for gene list integration and meta-analysis.
  *Bioinformatics*, 28(4):573–580.  https://doi.org/10.1093/bioinformatics/btr709

Classes
-------
RankAggregateFilter
    Aggregate 2–N ranked feature lists via β-order statistics + Bonferroni.
    Built-in scorers: 'anova', 'mi'.
    External scorers (e.g. MEL) can be passed via ``extra_scores``.

FeatureSelector
    High-level adaptive wrapper.  Set ``method='auto'`` to follow the
    paper's two-tier recommendation:
      • Easy/Medium (median_auc ≥ 0.80) → RRA(['anova', 'mi'])
      • Hard (median_auc < 0.80)        → RRA(['anova', 'mi'])  + user can
                                          supply MEL via extra_scores

    Other method shortcuts: 'anova', 'mi', 'mrmr', 'relieff',
    'l1', 'elasticnet', 'linsvc', 'boruta', 'extratrees', 'rfecv',
    'rra'  (alias for RRA(['anova', 'mi'])),
    'ga', 'bpso'  (slow; built-in, no extra install).
"""
import warnings
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.utils.validation import check_is_fitted

from clinifs._prefilter import anova_prefilter, PREFILTER_N


# ─────────────────────────────────────────────────────────────────────────────
# RankAggregateFilter
# ─────────────────────────────────────────────────────────────────────────────

class RankAggregateFilter(BaseEstimator, TransformerMixin):
    """
    Aggregate ranked feature lists from multiple scoring methods using
    Robust Rank Aggregation (Kolde et al. 2012).

    Parameters
    ----------
    methods : list[str]
        Scoring methods to aggregate.  Built-in: 'anova', 'mi'.
        For external scores (e.g. 'mel') pass them via ``extra_scores``
        argument of ``fit``.
    k : int
        Number of features to select.
    prefilter : int
        Reduce to top-n ANOVA features before aggregation.

    Attributes
    ----------
    selected_indices_ : np.ndarray[int]
    rho_scores_ : np.ndarray[float]
        RRA ρ-score (Bonferroni-corrected); lower = more consistently ranked.
    n_features_in_ : int

    Examples
    --------
    >>> rra = RankAggregateFilter(methods=['anova', 'mi'], k=20)
    >>> rra.fit(X_train, y_train)
    >>> X_sel = rra.transform(X_test)

    Three-method RRA with external MEL scores:

    >>> rra3 = RankAggregateFilter(methods=['anova', 'mi', 'mel'], k=20)
    >>> rra3.fit(X_train, y_train, extra_scores={'mel': mel_scores_array})
    """

    _BUILTIN = ('anova', 'mi')

    def __init__(self, methods=('anova', 'mi'), k: int = 20,
                 prefilter: int = PREFILTER_N):
        self.methods = list(methods)
        self.k = k
        self.prefilter = prefilter

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, extra_scores: dict | None = None):
        """
        Fit the RRA aggregator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        extra_scores : dict[str, array-like], optional
            Additional score arrays keyed by method name (e.g.
            ``{'mel': mel_scores}``).  Arrays must have length n_features
            (original space) or n_sub (after pre-filter).
        """
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # ANOVA pre-filter
        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        n_sub = X_sub.shape[1]

        # Collect score arrays (in pre-filtered space)
        all_scores = []
        for method in self.methods:
            s = self._score(method, X_sub, y, extra_scores, n_features, pre_idx)
            all_scores.append(np.nan_to_num(s, nan=0.0))

        # Normalised ranks: (L, n_sub), values in (0, 1]
        L = len(all_scores)
        ranks = np.zeros((L, n_sub))
        for i, s in enumerate(all_scores):
            order = np.argsort(-s)          # highest score → rank 0
            rank_pos = np.empty_like(order)
            rank_pos[order] = np.arange(n_sub)
            ranks[i] = (rank_pos + 1.0) / n_sub

        # RRA ρ-scores
        rho_raw = self._compute_rho(ranks, n_sub, L)
        rho_corrected = np.minimum(rho_raw * n_sub, 1.0)  # Bonferroni

        # Select top-k
        k_eff = min(self.k, n_sub)
        sub_idx = np.argsort(rho_corrected)[:k_eff]

        self._pre_idx = pre_idx
        self.rho_scores_ = rho_corrected
        self.selected_indices_ = pre_idx[sub_idx]
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _score(self, method, X_sub, y, extra_scores, n_orig, pre_idx):
        if method == 'anova':
            s, _ = f_classif(X_sub, y)
            return s
        if method == 'mi':
            return mutual_info_classif(X_sub, y, random_state=42)
        # External scorer
        if extra_scores is not None and method in extra_scores:
            s = np.asarray(extra_scores[method], dtype=float)
            if len(s) == n_orig:
                return s[pre_idx]          # map from original to sub-space
            if len(s) == len(pre_idx):
                return s                   # already in sub-space
            raise ValueError(
                f"extra_scores['{method}'] has length {len(s)}; "
                f"expected {n_orig} (original) or {len(pre_idx)} (pre-filtered)."
            )
        raise ValueError(
            f"Unknown method '{method}'. Built-in methods: {self._BUILTIN}. "
            "For custom scores pass them via extra_scores."
        )

    @staticmethod
    def _compute_rho(ranks: np.ndarray, n_features: int, L: int) -> np.ndarray:
        """
        Vectorised RRA ρ computation.

        For each feature j, sort its ranks across methods, then compute
        the Beta CDF for each order statistic and take the minimum
        (= most significant rank position).

        ranks : (L, n_features), values in (0, 1]
        """
        sorted_r = np.sort(ranks, axis=0)   # shape (L, n_features)
        rho = np.ones(n_features)
        for ord_k in range(L):
            # k-th order statistic ~ Beta(k+1, L-k)
            p_k = beta_dist.cdf(sorted_r[ord_k], ord_k + 1, L - ord_k)
            rho = np.minimum(rho, p_k)
        return rho


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSelector  (high-level adaptive API)
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_MAP = {
    'variance'  : ('clinifs.selectors.filter',    'VarianceSelector'),
    'anova'     : ('clinifs.selectors.filter',    'ANOVASelector'),
    'mi'        : ('clinifs.selectors.filter',    'MISelector'),
    'mrmr'      : ('clinifs.selectors.filter',    'mRMRSelector'),
    'relieff'   : ('clinifs.selectors.filter',    'ReliefFSelector'),
    'l1'        : ('clinifs.selectors.embedded',  'L1LogisticSelector'),
    'elasticnet': ('clinifs.selectors.embedded',  'ElasticNetSelector'),
    'linsvc'    : ('clinifs.selectors.embedded',  'LinSVCL1Selector'),
    'boruta'    : ('clinifs.selectors.classical', 'BorutaSelector'),
    'extratrees': ('clinifs.selectors.classical', 'ExtraTreesSelector'),
    'rfecv'     : ('clinifs.selectors.classical', 'RFECVSelector'),
    'ga'        : ('clinifs.selectors.evolutionary', 'GASelector'),
    'bpso'      : ('clinifs.selectors.evolutionary', 'BPSOSelector'),
    'rra'       : None,   # handled inline → RankAggregateFilter(['anova','mi'])
    'auto'      : None,   # handled inline → adaptive tier detection
}

_HARD_AUC_THRESHOLD = 0.80
_RANDOM_BASELINE_K  = 10


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    High-level adaptive feature selector following the benchmark paper's
    two-tier recommendation.

    Parameters
    ----------
    method : str
        'auto'       – estimate dataset difficulty, then choose RRA variant.
        'rra'        – RRA(ANOVA + MI), best all-rounder on Easy/Medium.
        'anova'      – single ANOVA filter.
        'mi', 'mrmr', 'relieff', 'variance' – other filter methods.
        'l1', 'elasticnet', 'linsvc' – embedded methods.
        'boruta', 'extratrees', 'rfecv' – classical wrapper methods.
        'ga', 'bpso' – evolutionary wrappers (slow; built-in, no extra install).
    k : int
        Panel size (number of features).
    extra_scores : dict, optional
        External scorer arrays, e.g. ``{'mel': <array>}`` for three-method RRA.
    **selector_kwargs
        Forwarded to the underlying selector's ``__init__``.

    Attributes
    ----------
    selector_ : fitted selector object
    tier_ : str | None   'easy_medium' or 'hard' (set only when method='auto')

    Examples
    --------
    >>> fs = FeatureSelector(method='auto', k=20)
    >>> fs.fit(X_train, y_train)
    >>> X_sel = fs.transform(X_test)
    >>> print(fs.tier_)         # 'easy_medium' or 'hard'
    >>> print(fs.selected_indices_)
    """

    def __init__(self, method: str = 'auto', k: int = 20,
                 extra_scores: dict | None = None,
                 selector_kwargs: dict | None = None):
        self.method = method
        self.k = k
        self.extra_scores = extra_scores
        self.selector_kwargs = selector_kwargs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        method = self.method.lower().strip()

        kw = self.selector_kwargs or {}
        if method == 'auto':
            self._fit_auto(X, y)
        elif method == 'rra':
            self.tier_ = None
            self.selector_ = RankAggregateFilter(
                methods=['anova', 'mi'], k=self.k, **kw
            )
            self.selector_.fit(X, y, extra_scores=self.extra_scores)
        elif method in _METHOD_MAP and _METHOD_MAP[method] is not None:
            mod_path, cls_name = _METHOD_MAP[method]
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            self.tier_ = None
            self.selector_ = cls(k=self.k, **kw)
            self.selector_.fit(X, y)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid options: {sorted(_METHOD_MAP.keys())}."
            )

        self.selected_indices_ = self.selector_.selected_indices_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)

    # ------------------------------------------------------------------
    # adaptive tier detection
    # ------------------------------------------------------------------

    def _fit_auto(self, X, y):
        """
        Estimate dataset difficulty via random-gene baseline median AUC,
        then select the appropriate RRA configuration.

          median_auc < 0.80  → Hard  → RRA(['anova', 'mi'])
                                        (+ MEL if extra_scores provided)
          median_auc ≥ 0.80  → Easy/Medium → RRA(['anova', 'mi'])
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression as _LR

        k_probe = min(_RANDOM_BASELINE_K, X.shape[1])
        rng = np.random.RandomState(0)
        n_features = X.shape[1]
        n_reps = 20
        aucs = []

        y_arr = np.asarray(y)
        min_class_count = int(np.bincount(y_arr.astype(int)).min())
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            # Too few samples per class – skip probe, assume medium difficulty
            self.random_baseline_auc_ = _HARD_AUC_THRESHOLD
            self.tier_ = 'easy_medium'
            kw = self.selector_kwargs or {}
            self.selector_ = RankAggregateFilter(
                methods=['anova', 'mi'], k=self.k, **kw
            )
            self.selector_.fit(X, y, extra_scores=None)
            return

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for _ in range(n_reps):
            feat_idx = rng.choice(n_features, size=k_probe, replace=False)
            X_probe = X[:, feat_idx]
            fold_aucs = []
            for tr, te in skf.split(X_probe, y):
                clf = _LR(C=1, solver="liblinear", max_iter=300, random_state=42)
                try:
                    clf.fit(X_probe[tr], y[tr])
                    from sklearn.metrics import roc_auc_score
                    prob = clf.predict_proba(X_probe[te])[:, 1]
                    fold_aucs.append(roc_auc_score(y[te], prob))
                except Exception:
                    fold_aucs.append(0.5)
            aucs.append(np.mean(fold_aucs))

        median_auc = float(np.median(aucs))
        self.random_baseline_auc_ = median_auc

        if median_auc < _HARD_AUC_THRESHOLD:
            self.tier_ = 'hard'
            # Check if MEL scores provided → three-method RRA
            extra = self.extra_scores or {}
            methods = ['anova', 'mi', 'mel'] if 'mel' in extra else ['anova', 'mi']
            if 'mel' not in extra:
                warnings.warn(
                    f"Hard tier detected (random baseline AUC={median_auc:.3f}). "
                    "For best performance supply MEL scores via "
                    "FeatureSelector(extra_scores={'mel': <array>}).",
                    UserWarning, stacklevel=3,
                )
        else:
            self.tier_ = 'easy_medium'
            methods = ['anova', 'mi']
            extra = self.extra_scores or {}

        kw2 = self.selector_kwargs or {}
        self.selector_ = RankAggregateFilter(
            methods=methods, k=self.k, **kw2
        )
        self.selector_.fit(X, y, extra_scores=extra if extra else None)

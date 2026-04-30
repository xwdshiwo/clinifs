"""
Shared ANOVA pre-filter utility.

For high-dimensional inputs (e.g. microarray with 50 K features), running
mRMR / ReliefF / MI directly is prohibitively slow.  We first reduce to
the top-`n` features by ANOVA F-score and carry both the reduced matrix
and the index mapping so callers can correctly map back to original space.
"""
import numpy as np
from sklearn.feature_selection import f_classif


PREFILTER_N = 2000


def anova_prefilter(X: np.ndarray, y: np.ndarray, n: int = PREFILTER_N):
    """
    Reduce feature space to top-n by ANOVA F-score.

    Returns
    -------
    X_sub : np.ndarray, shape (n_samples, min(n, p))
    pre_idx : np.ndarray, shape (min(n, p),)
        Indices into the *original* feature axis.
    anova_scores : np.ndarray | None
        Full ANOVA score array (length p) if pre-filtering was applied,
        else None.
    """
    n_features = X.shape[1]
    if n_features <= n:
        return X, np.arange(n_features), None

    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    pre_idx = np.argsort(scores)[::-1][:n]
    return X[:, pre_idx], pre_idx, scores

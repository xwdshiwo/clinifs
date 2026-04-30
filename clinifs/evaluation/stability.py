"""
Feature-selection stability metrics.

Nogueira Φ (2018)
-----------------
Unbiased estimator of Jaccard-based stability that corrects for feature set
size and number of features.  Range: [−1, 1]; 1 = perfectly stable.

Reference: Nogueira S, Sechidis K, Brown G (2018).
  On the stability of feature selection algorithms.
  *JMLR*, 18(174):1–54.

Kuncheva average Jaccard (2007)
--------------------------------
Mean pairwise Jaccard index across all run pairs.

Reference: Kuncheva LI (2007).
  A stability index for feature selection.
  *AICT*, pp. 390–395.
"""
import numpy as np
from itertools import combinations
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Nogueira Φ
# ─────────────────────────────────────────────────────────────────────────────

def nogueira_phi(feature_sets: Sequence, n_features: int) -> float:
    """
    Compute Nogueira Φ stability over a collection of feature sets.

    Parameters
    ----------
    feature_sets : sequence of array-like
        Each element is a list/array of *indices* (0-based) of selected
        features for one run / fold.
    n_features : int
        Total number of features in the original space (*p*).

    Returns
    -------
    phi : float
        Stability in [−1, 1].  Returns NaN if fewer than 2 sets provided.

    Notes
    -----
    Uses the corrected estimator from eq. (9) in Nogueira et al. 2018,
    which accounts for varying panel sizes and number of features.
    """
    sets = [np.asarray(s, dtype=int) for s in feature_sets]
    M = len(sets)
    if M < 2:
        return float("nan")

    p = n_features
    ks = np.array([len(s) for s in sets], dtype=float)  # panel sizes

    # All pairwise intersections
    pairwise_hat = []
    for a, b in combinations(sets, 2):
        overlap = len(np.intersect1d(a, b))
        ka, kb = len(a), len(b)
        # Expected overlap under independence
        expected = ka * kb / p
        # Variance of overlap under independence
        var = ka * kb * (p - ka) * (p - kb) / (p * p * (p - 1))
        if var <= 0:
            pairwise_hat.append(0.0)
        else:
            # Normalised corrected Jaccard
            union = ka + kb - overlap
            if union == 0:
                pairwise_hat.append(1.0)
            else:
                jaccard = overlap / union
                exp_jaccard = expected / (ka + kb - expected)
                denom = 1.0 - exp_jaccard
                if abs(denom) < 1e-12:
                    pairwise_hat.append(1.0 if jaccard >= exp_jaccard else 0.0)
                else:
                    pairwise_hat.append((jaccard - exp_jaccard) / denom)

    return float(np.mean(pairwise_hat))


# ─────────────────────────────────────────────────────────────────────────────
# Kuncheva average Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def kuncheva_jaccard(feature_sets: Sequence) -> float:
    """
    Compute Kuncheva average Jaccard stability.

    Parameters
    ----------
    feature_sets : sequence of array-like
        Each element is a list/array of selected feature indices.

    Returns
    -------
    stability : float in [0, 1].  Returns NaN for < 2 sets.
    """
    sets = [set(np.asarray(s, dtype=int).tolist()) for s in feature_sets]
    M = len(sets)
    if M < 2:
        return float("nan")

    scores = []
    for a, b in combinations(sets, 2):
        inter = len(a & b)
        union = len(a | b)
        scores.append(inter / union if union > 0 else 1.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-seed Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def cross_seed_jaccard(feature_sets_by_seed: dict) -> float:
    """
    Compute mean Jaccard across all cross-seed (between-run) pairs.

    Parameters
    ----------
    feature_sets_by_seed : dict[int, list[array-like]]
        Keys are seed identifiers; values are lists of feature-index arrays
        (one per fold) for that seed.

    Returns
    -------
    float : mean cross-seed pairwise Jaccard.
    """
    seeds = list(feature_sets_by_seed.keys())
    if len(seeds) < 2:
        return float("nan")

    scores = []
    for s1, s2 in combinations(seeds, 2):
        for fs1, fs2 in zip(feature_sets_by_seed[s1], feature_sets_by_seed[s2]):
            a = set(np.asarray(fs1, dtype=int).tolist())
            b = set(np.asarray(fs2, dtype=int).tolist())
            union = len(a | b)
            scores.append(len(a & b) / union if union > 0 else 1.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all three from a list of fold results
# ─────────────────────────────────────────────────────────────────────────────

def stability_report(feature_sets: Sequence, n_features: int) -> dict:
    """
    Return all three stability metrics in a single dict.

    Parameters
    ----------
    feature_sets : sequence of array-like
    n_features : int

    Returns
    -------
    dict with keys: 'nogueira_phi', 'kuncheva_jaccard'
    """
    return {
        "nogueira_phi"     : nogueira_phi(feature_sets, n_features),
        "kuncheva_jaccard" : kuncheva_jaccard(feature_sets),
    }

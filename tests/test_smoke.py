"""
Smoke tests for clinifs package.

Run with:
    pytest tests/test_smoke.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="module")
def toy_data():
    X, y = make_classification(
        n_samples=120, n_features=200, n_informative=20,
        n_redundant=10, random_state=42
    )
    return X.astype(np.float32), y


# ─────────────────────────────────────────────────────────────────────────────
# Filter selectors
# ─────────────────────────────────────────────────────────────────────────────

def test_variance_selector(toy_data):
    from clinifs.selectors import VarianceSelector
    X, y = toy_data
    sel = VarianceSelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)
    assert len(sel.selected_indices_) == 10


def test_anova_selector(toy_data):
    from clinifs.selectors import ANOVASelector
    X, y = toy_data
    sel = ANOVASelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


def test_mi_selector(toy_data):
    from clinifs.selectors import MISelector
    X, y = toy_data
    sel = MISelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Embedded selectors
# ─────────────────────────────────────────────────────────────────────────────

def test_l1_selector(toy_data):
    from clinifs.selectors import L1LogisticSelector
    X, y = toy_data
    sel = L1LogisticSelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


def test_elasticnet_selector(toy_data):
    from clinifs.selectors import ElasticNetSelector
    X, y = toy_data
    sel = ElasticNetSelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


def test_linsvc_selector(toy_data):
    from clinifs.selectors import LinSVCL1Selector
    X, y = toy_data
    sel = LinSVCL1Selector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Classical wrapper selectors
# ─────────────────────────────────────────────────────────────────────────────

def test_extratrees_selector(toy_data):
    from clinifs.selectors import ExtraTreesSelector
    X, y = toy_data
    sel = ExtraTreesSelector(k=10)
    Xt = sel.fit_transform(X, y)
    assert Xt.shape == (120, 10)


def test_rfecv_selector():
    from clinifs.selectors import RFECVSelector
    X_small, y_small = make_classification(
        n_samples=60, n_features=100, n_informative=10, random_state=0
    )
    with pytest.warns(UserWarning, match="sample size"):
        sel = RFECVSelector(k=5)
        Xt = sel.fit_transform(X_small.astype(np.float32), y_small)
    assert Xt.shape == (60, 5)


# ─────────────────────────────────────────────────────────────────────────────
# RRA aggregator
# ─────────────────────────────────────────────────────────────────────────────

def test_rra_two_method(toy_data):
    from clinifs import RankAggregateFilter
    X, y = toy_data
    rra = RankAggregateFilter(methods=['anova', 'mi'], k=10)
    Xt = rra.fit_transform(X, y)
    assert Xt.shape == (120, 10)
    assert rra.rho_scores_.shape[0] == X.shape[1]


def test_rra_three_method_external(toy_data):
    """Three-method RRA with simulated external MEL scores."""
    from clinifs import RankAggregateFilter
    X, y = toy_data
    # Simulate MEL scores (e.g. from a pre-run MEL experiment)
    fake_mel = np.random.RandomState(7).random(X.shape[1])
    rra = RankAggregateFilter(methods=['anova', 'mi', 'mel'], k=10)
    rra.fit(X, y, extra_scores={'mel': fake_mel})
    Xt = rra.transform(X)
    assert Xt.shape == (120, 10)


def test_rra_explicit_method_string(toy_data):
    from clinifs import FeatureSelector
    X, y = toy_data
    fs = FeatureSelector(method='rra', k=10)
    Xt = fs.fit_transform(X, y)
    assert Xt.shape == (120, 10)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSelector – auto mode (no true Hard tier expected on toy data)
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_selector_auto(toy_data):
    from clinifs import FeatureSelector
    X, y = toy_data
    fs = FeatureSelector(method='auto', k=10)
    Xt = fs.fit_transform(X, y)
    assert Xt.shape == (120, 10)
    assert fs.tier_ in ('easy_medium', 'hard')


# ─────────────────────────────────────────────────────────────────────────────
# Stability metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_nogueira_phi():
    from clinifs.evaluation import nogueira_phi
    sets = [np.array([0, 1, 2]), np.array([0, 1, 3]), np.array([0, 1, 4])]
    phi = nogueira_phi(sets, n_features=20)
    assert isinstance(phi, float)
    assert -1.0 <= phi <= 1.0


def test_kuncheva_jaccard():
    from clinifs.evaluation import kuncheva_jaccard
    sets = [np.array([0, 1, 2]), np.array([0, 1, 3])]
    jac = kuncheva_jaccard(sets)
    assert abs(jac - 0.5) < 1e-9   # intersection=2, union=4


def test_stability_report():
    from clinifs.evaluation import stability_report
    sets = [np.arange(5), np.arange(3, 8)]
    report = stability_report(sets, n_features=20)
    assert "nogueira_phi" in report
    assert "kuncheva_jaccard" in report

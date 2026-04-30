"""
clinifs – Clinical Feature Selection toolkit

Quick start
-----------
>>> from clinifs import FeatureSelector, RankAggregateFilter
>>> fs = FeatureSelector(method='auto', k=20)
>>> fs.fit(X_train, y_train)
>>> X_sel = fs.transform(X_test)

All individual selectors are also importable:

>>> from clinifs.selectors import ANOVASelector, MISelector, mRMRSelector
>>> from clinifs.selectors import ReliefFSelector, VarianceSelector
>>> from clinifs.selectors import L1LogisticSelector, ElasticNetSelector, LinSVCL1Selector
>>> from clinifs.selectors import BorutaSelector, ExtraTreesSelector, RFECVSelector
>>> # EA selectors (GA/BPSO; no extra install needed)
>>> from clinifs.selectors import GASelector, BPSOSelector

Stability metrics:

>>> from clinifs.evaluation import nogueira_phi, kuncheva_jaccard, stability_report
"""

from clinifs.aggregators.rra import RankAggregateFilter, FeatureSelector

__version__ = "0.1.0"
__all__ = ["RankAggregateFilter", "FeatureSelector"]

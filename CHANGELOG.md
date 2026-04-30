# Changelog

## v0.1.0 (2026-04-30)

Initial release accompanying the paper submission to *Briefings in Bioinformatics*.

### Added
- 5 Filter selectors: `VarianceSelector`, `ANOVASelector`, `MISelector`, `mRMRSelector`, `ReliefFSelector`
- 3 Embedded selectors: `L1LogisticSelector`, `ElasticNetSelector`, `LinSVCL1Selector`
- 5 Wrapper selectors: `BorutaSelector`, `ExtraTreesSelector`, `RFECVSelector`, `GASelector`, `BPSOSelector`
- `RankAggregateFilter`: two- or three-method Robust Rank Aggregation (Kolde et al. 2012)
- `FeatureSelector`: high-level adaptive wrapper with automatic difficulty-tier detection
- Stability metrics: `nogueira_phi`, `kuncheva_jaccard`, `stability_report`
- Smoke test suite (`tests/test_smoke.py`)
- Example data and quickstart script (`examples/`)
- CLI entry-point: `clinifs --help`

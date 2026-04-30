from clinifs.selectors.filter import (
    VarianceSelector,
    ANOVASelector,
    MISelector,
    mRMRSelector,
    ReliefFSelector,
)
from clinifs.selectors.embedded import (
    L1LogisticSelector,
    ElasticNetSelector,
    LinSVCL1Selector,
)
from clinifs.selectors.classical import (
    BorutaSelector,
    ExtraTreesSelector,
    RFECVSelector,
)

__all__ = [
    "VarianceSelector", "ANOVASelector", "MISelector",
    "mRMRSelector", "ReliefFSelector",
    "L1LogisticSelector", "ElasticNetSelector", "LinSVCL1Selector",
    "BorutaSelector", "ExtraTreesSelector", "RFECVSelector",
]

# EA selectors imported lazily to avoid hard dependency
def __getattr__(name):
    if name in ("GASelector", "BPSOSelector"):
        from clinifs.selectors.evolutionary import GASelector, BPSOSelector
        return {"GASelector": GASelector, "BPSOSelector": BPSOSelector}[name]
    raise AttributeError(f"module 'clinifs.selectors' has no attribute '{name}'")

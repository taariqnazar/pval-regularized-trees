"""Tree models for the regularised regression-tree paper.

Naming convention (post-revision):
    PSumTree, PSumBoosting              -> cumulative Psi-sum stopping (paper's main method)
    RecursivePSumTree, RecursivePSumBoosting -> v1 recursive top-down pruning (kept for v1 reproducibility)
    CappelliSTP                          -> F-test-based pruning, Cappelli et al. 2001 (Section 4.2.1)
    GSellStop                            -> ForwardStop / StrongStop, G'Sell et al. 2016 (exploratory; not in paper)
    CCPCV                                -> sklearn CCP path + 1-SE CV rule baseline (Section 4.2.2 / 4.3)
    GBM, Intercept, DoubleRegression    -> baselines / wrappers
"""

from .psum import PSumTree
from .psum_boosting import PSumBoosting
from .recursive_psum import RecursivePSumTree
from .recursive_psum_boosting import RecursivePSumBoosting
from .gbm import GBM
from .intercept import Intercept
from .double_regression import DoubleRegression
# Added in later tasks:
from .cappelli import CappelliSTP
from .gsell import GSellStop
from .ccp_cv import CCPCV

__all__ = [
    "PSumTree",
    "PSumBoosting",
    "RecursivePSumTree",
    "RecursivePSumBoosting",
    "GBM",
    "Intercept",
    "DoubleRegression",
    "CappelliSTP",
    "GSellStop",
    "CCPCV",
]

MODEL_REGISTRY = {
    "PSumTree": PSumTree,
    "PSumBoosting": PSumBoosting,
    "RecursivePSumTree": RecursivePSumTree,
    "RecursivePSumBoosting": RecursivePSumBoosting,
    "GBM": GBM,
    "Intercept": Intercept,
    "DoubleRegression": DoubleRegression,
    "CappelliSTP": CappelliSTP,
    "GSellStop": GSellStop,
    "CCPCV": CCPCV,
}

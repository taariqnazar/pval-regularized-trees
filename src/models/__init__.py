from .intercept import Intercept
from .ccp_ptree import CCPPTree
from .recursive_ptree import RecursivePTree
from .ccp_ptree_boosting import CCPPTreeBoosting
from .recursive_ptree_boosting import RecursivePTreeBoosting
from .gbm import GBM
from .double_regression import DoubleRegression

__all__ = [
    "CCPPTree",
    "RecursivePTree",
    "CCPPTreeBoosting",
    "RecursivePTreeBoosting",
    "GBM",
    "Intercept",
    "DubleRegression",
]

# Optional: registry for config-driven experiments
MODEL_REGISTRY = {
    "CCPPTree": CCPPTree,
    "RecursivePTree": RecursivePTree,
    "CCPPTreeBoosting": CCPPTreeBoosting,
    "RecursivePTreeBoosting": RecursivePTreeBoosting,
    "Intercept": Intercept,
    "GBM": GBM,
    "DoubleRegression": DoubleRegression,
}

from .ccp_ptree import CCPPTree
from .recursive_ptree import RecursivePTree
from .ccp_ptree_boosting import CCPPTreeBoosting
from .recursive_ptree_boosting import RecursivePTreeBoosting
from .gbm import GBM

__all__ = [
    "CCPPTree",
    "RecursivePTree",
    "CCPPTreeBoosting",
    "RecursivePTreeBoosting",
    "GBM",
]

# Optional: registry for config-driven experiments
MODEL_REGISTRY = {
    "CCPPTree": CCPPTree,
    "RecursivePTree": RecursivePTree,
    "CCPPTreeBoosting": CCPPTreeBoosting,
    "RecursivePTreeBoosting": RecursivePTreeBoosting,
    "GBM": GBM,
}

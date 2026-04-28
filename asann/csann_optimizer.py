"""Backward-compat shim: csann.csann_optimizer -> asann.asann_optimizer.

Re-exports everything from asann_optimizer so that legacy imports of
the form `from csann.csann_optimizer import CSANNOptimizerConfig` work.
"""
from .asann_optimizer import *  # noqa: F401, F403
from .asann_optimizer import (
    ASANNOptimizer,
    create_asann_parameter_groups,
)

# Class-level aliases
CSANNOptimizer = ASANNOptimizer

# Config alias re-imported from config module
from .config import ASANNOptimizerConfig
CSANNOptimizerConfig = ASANNOptimizerConfig

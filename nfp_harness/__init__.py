"""Estimation harness for nested-fixed-point GMM demand models (docs/estimation_harness/DESIGN.md).

The exact pure characteristics model is the first model (``nfp_harness.models.pcm``). The optimizer,
moments, weighting, diagnostics and checkpointing are model-agnostic.
"""

from .types import (
    ConvergenceClass,
    EvalStatus,
    Evaluation,
    HistoryRecord,
    InnerModel,
    InnerSolution,
    KernelRequests,
    LinearFit,
    MarketIndex,
    MicroPartRequest,
    Moment,
    MomentBlock,
    OptimizeResult,
    ParameterSpec,
    Problem,
    Report,
    StepLimit,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ConvergenceClass",
    "EvalStatus",
    "Evaluation",
    "HistoryRecord",
    "InnerModel",
    "InnerSolution",
    "KernelRequests",
    "LinearFit",
    "MarketIndex",
    "MicroPartRequest",
    "Moment",
    "MomentBlock",
    "OptimizeResult",
    "ParameterSpec",
    "Problem",
    "Report",
    "StepLimit",
    "__version__",
]

"""Integration stability and multi-start summary (DESIGN 5.7, 5.8). Owned by WP6."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..types import DiagnosticBlock, OptimizeResult

__all__ = ["integration_stability", "multistart_summary"]


def integration_stability(result: OptimizeResult, reevaluate: Callable[[np.ndarray], OptimizeResult],
                          units: np.ndarray, se: np.ndarray | None = None) -> DiagnosticBlock:
    """integration_stability(result, reevaluate, units, se=None) -> DiagnosticBlock.

    ``reevaluate`` re-runs at most 20 iterations from the estimate with draws doubled. STABLE if every
    parameter moves less than 0.5 SE (or 0.1 natural units without SEs), else UNSTABLE with the list.
    """
    raise NotImplementedError("integration_stability(result, reevaluate, units, se=None): WP6, DESIGN 5.7")


def multistart_summary(results: Sequence[OptimizeResult], units: np.ndarray, tol: float = 0.1) -> dict[str, Any]:
    """multistart_summary(results, units, tol=0.1) -> endpoints, q, class, number of distinct endpoints
    (closer than ``tol`` natural units in every parameter count as one) and the q spread."""
    raise NotImplementedError("multistart_summary(results, units, tol=0.1): WP6, DESIGN 5.8")

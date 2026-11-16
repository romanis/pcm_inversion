"""Convergence class (DESIGN 4.6, 5.1). Owned by WP5."""

from __future__ import annotations

from typing import Any, Sequence

from ..types import ConvergenceClass, DiagnosticBlock, HistoryRecord, OptimizeResult

__all__ = ["classify", "convergence_block"]


def classify(history: Sequence[HistoryRecord], projected_gradient: float, gn_decrease: float, q: float,
             options: Any, at_cap: bool = False, limit: str | None = None) -> ConvergenceClass:
    """classify(history, projected_gradient, gn_decrease, q, options, at_cap=False, limit=None) -> class.

    CONVERGED: status OK, ``max|pg| <= pg_tol (1 + q)^{1/2}`` and ``dq_gn <= gn_tol (1 + q)``;
    CONVERGED_AT_CAP: as CONVERGED with a cap parameter at its cap; otherwise STALLED, FLAT_PROGRESS,
    ITERATION_LIMIT, TIME_LIMIT or INVERSION_FAILED by the rules of DESIGN 4.6. Never CONVERGED on a
    flat stop with a large gradient.
    """
    raise NotImplementedError("classify(history, projected_gradient, gn_decrease, q, options, ...): WP5, DESIGN 4.6")


def convergence_block(result: OptimizeResult) -> DiagnosticBlock:
    """convergence_block(result) -> DiagnosticBlock with the message template of DESIGN 5.1:
    "Convergence: {class}. Projected gradient {pg:.2e} (tolerance {tol:.2e}); predicted Gauss-Newton
    decrease {dq:.2e} (tolerance {tolq:.2e}); {n_acc} accepted steps, {n_eval} evaluations, {hours:.1f} h."
    """
    raise NotImplementedError("convergence_block(result) -> DiagnosticBlock: WP5, DESIGN 5.1")

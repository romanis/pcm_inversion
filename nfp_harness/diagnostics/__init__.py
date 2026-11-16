"""End-of-stage diagnostics (DESIGN 5). ``run_diagnostics`` is owned by WP6.

Each block returns a ``types.DiagnosticBlock`` with status in {"ok", "flag", "fail", "not_run"},
the numbers, the thresholds (from ``spec.diagnostics`` with the documented defaults) and its
message lines.
"""

from __future__ import annotations

from typing import Any

from ..types import OptimizeResult, Problem, Report

__all__ = ["run_diagnostics"]


def run_diagnostics(problem: Problem, result: OptimizeResult, spec: Any, options: Any = None) -> Report:
    """run_diagnostics(problem, result, spec, options) -> Report.

    Runs, in order: convergence class (5.1), flat directions and units versus identification (5.2),
    collinearity and weak instruments (5.3), named degeneracies (5.4), overidentification (5.5),
    economic sanity (5.6), integration stability (5.7), multi-start (5.8), standard errors (5.9).
    Blocks that need a stationary point print "not interpreted: objective did not converge" unless
    the class is CONVERGED or CONVERGED_AT_CAP.
    """
    raise NotImplementedError("run_diagnostics(problem, result, spec, options) -> Report: WP6, DESIGN 5")

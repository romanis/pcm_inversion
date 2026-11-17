"""Convergence class (DESIGN 4.6, 5.1). Owned by WP5.

``classify`` is the single place where the stopping rules of DESIGN 4.6 live. The optimizer calls it
at every iteration; the report calls ``convergence_block`` on the finished ``OptimizeResult``. A stop
is ``CONVERGED`` only when both the projected-gradient and the Gauss-Newton-decrease tests pass, so a
flat stop with a large gradient (the CUE run of LEARNINGS 13: q = 0.80, |grad| = 0.38) is never
reported as converged.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..types import ConvergenceClass, DiagnosticBlock, EvalStatus, HistoryRecord, OptimizeResult

__all__ = ["classify", "convergence_block", "converged_test", "noise_stop", "thresholds"]

_LIMITS = {
    "iterations": ConvergenceClass.ITERATION_LIMIT,
    "time": ConvergenceClass.TIME_LIMIT,
    "inversion_failed": ConvergenceClass.INVERSION_FAILED,
}


def _opt(options: Any, name: str, default: Any) -> Any:
    return getattr(options, name, default) if options is not None else default


def thresholds(q: float, options: Any = None) -> tuple[float, float]:
    """thresholds(q, options) -> (pg threshold, gn threshold) = (pg_tol (1+q)^{1/2}, gn_tol (1+q))."""
    q = max(float(q), 0.0) if math.isfinite(q) else 0.0
    return _opt(options, "pg_tol", 1e-3) * math.sqrt(1.0 + q), _opt(options, "gn_tol", 1e-6) * (1.0 + q)


def converged_test(projected_gradient: float, gn_decrease: float, q: float, options: Any = None,
                   status: EvalStatus = EvalStatus.OK) -> bool:
    """converged_test(pg, dq_gn, q, options, status) -> True iff the CONVERGED rule of DESIGN 4.6 holds."""
    if status is not EvalStatus.OK or not (math.isfinite(projected_gradient) and math.isfinite(gn_decrease)):
        return False
    pg_thr, gn_thr = thresholds(q, options)
    return projected_gradient <= pg_thr and gn_decrease <= gn_thr


def _trailing_rejections(history: Sequence[HistoryRecord]) -> int:
    count = 0
    for rec in reversed(history):
        if rec.accepted:
            break
        count += 1
    return count


def _flat(history: Sequence[HistoryRecord], q: float, options: Any) -> bool:
    window = int(_opt(options, "flat_window", 25))
    qs = [rec.q for rec in history if rec.accepted]
    if len(qs) < window + 1:
        return False
    return (qs[-window - 1] - qs[-1]) < _opt(options, "flat_tol", 1e-3) * (1.0 + max(q, 0.0))


def noise_stop(history: Sequence[HistoryRecord], q: float, noise_floor: float | None, options: Any) -> bool:
    """noise_stop(history, q, noise_floor, options) -> True iff the CONVERGED_NOISE rule holds.

    The last ``noise_rejections`` (3) records of ``history`` are all rejected trials that evaluated (status OK,
    finite q), each tried with a trust radius below ``noise_Delta`` (1e-4 natural units), and each changed q by
    less than ``noise_floor`` against the accepted ``q``: ``|q_trial - q| < noise_floor``. The floor is the
    largest block change of q measured at the stage start over probes of +-``noise_probe`` units per free
    parameter (``optimizer``), so a step this small cannot be told from the objective's own jumps (the supply
    block's frontier jumps, LEARNINGS 25). False without a positive finite floor.
    """
    if noise_floor is None or not (math.isfinite(noise_floor) and noise_floor > 0):
        return False
    m = int(_opt(options, "noise_rejections", 3))
    D = float(_opt(options, "noise_Delta", 1e-4))
    if m <= 0 or len(history) < m:
        return False
    tail = list(history)[-m:]
    return all((not r.accepted) and r.status is EvalStatus.OK and math.isfinite(r.q) and math.isfinite(r.Delta)
               and r.Delta < D and abs(r.q - q) < noise_floor for r in tail)


def classify(history: Sequence[HistoryRecord], projected_gradient: float, gn_decrease: float, q: float,
             options: Any, at_cap: bool = False, limit: str | None = None,
             Delta: float | None = None, noise_floor: float | None = None) -> ConvergenceClass | None:
    """classify(history, projected_gradient, gn_decrease, q, options, at_cap=False, limit=None, Delta=None,
    noise_floor=None) -> class, or None when no stopping rule fires and the optimizer should continue.

    Rules, in this order (DESIGN 4.6):

    - ``limit == "inversion_failed"``: INVERSION_FAILED (start point or accepted point cannot be evaluated).
    - CONVERGED: status OK and ``max|pg| <= pg_tol (1 + q)^{1/2}`` and ``dq_gn <= gn_tol (1 + q)``;
      CONVERGED_AT_CAP: as CONVERGED with a cap parameter at its cap (``at_cap``).
    - CONVERGED_NOISE: ``noise_stop`` with the stage's measured ``noise_floor`` (round-3 analysis 4 (ii)).
    - STALLED: ``Delta < Delta_min`` or ``max_rejections`` consecutive rejections at the end of ``history``.
    - FLAT_PROGRESS: the last ``flat_window`` accepted steps lowered q by less than ``flat_tol (1 + q)``.
    - ITERATION_LIMIT / TIME_LIMIT: ``limit == "iterations"`` / ``"time"``.

    ``history`` is the current stage's records (accepted and rejected) in order; the last accepted
    record is the current point. Every non-converged class is returned only when the CONVERGED test
    failed, so a flat stop with a large gradient is never CONVERGED.
    """
    if limit == "inversion_failed":
        return ConvergenceClass.INVERSION_FAILED
    accepted = [rec for rec in history if rec.accepted]
    status = accepted[-1].status if accepted else EvalStatus.OK
    if converged_test(projected_gradient, gn_decrease, q, options, status):
        return ConvergenceClass.CONVERGED_AT_CAP if at_cap else ConvergenceClass.CONVERGED
    if noise_stop(history, q, noise_floor, options):
        return ConvergenceClass.CONVERGED_NOISE
    if Delta is not None and Delta < _opt(options, "Delta_min", 1e-8):
        return ConvergenceClass.STALLED
    if _trailing_rejections(history) >= int(_opt(options, "max_rejections", 10)):
        return ConvergenceClass.STALLED
    if _flat(history, q, options):
        return ConvergenceClass.FLAT_PROGRESS
    if limit is not None:
        if limit not in _LIMITS:
            raise ValueError(f"unknown limit {limit!r}; expected one of {sorted(_LIMITS)}")
        return _LIMITS[limit]
    return None


def convergence_block(result: OptimizeResult) -> DiagnosticBlock:
    """convergence_block(result) -> DiagnosticBlock with the message template of DESIGN 5.1:
    "Convergence: {class}. Projected gradient {pg:.2e} (tolerance {tol:.2e}); predicted Gauss-Newton
    decrease {dq:.2e} (tolerance {tolq:.2e}); {n_acc} accepted steps, {n_eval} evaluations, {hours:.1f} h."

    ``result.pg_tol`` and ``result.gn_tol`` hold the threshold values at the final q. Status is "ok" for
    CONVERGED, CONVERGED_AT_CAP and CONVERGED_NOISE, "fail" otherwise. ``result.message`` (bound and cap notes, stop
    reason) is added as a second line.
    """
    cls = result.convergence
    line = (f"Convergence: {cls.value}. Projected gradient {result.projected_gradient:.2e} "
            f"(tolerance {result.pg_tol:.2e}); predicted Gauss-Newton decrease {result.gn_decrease:.2e} "
            f"(tolerance {result.gn_tol:.2e}); {result.accepted_steps} accepted steps, "
            f"{result.evaluations} evaluations, {result.wall_seconds / 3600.0:.1f} h.")
    messages = [line]
    if result.message:
        messages.append(result.message)
    return DiagnosticBlock(
        name="convergence",
        status="ok" if cls.converged else "fail",
        values={
            "class": cls.value,
            "projected_gradient": float(result.projected_gradient),
            "gn_decrease": float(result.gn_decrease),
            "q": float(result.q),
            "accepted_steps": int(result.accepted_steps),
            "evaluations": int(result.evaluations),
            "wall_seconds": float(result.wall_seconds),
            "kink_limited_steps": int(result.kink_limited_steps),
            "noise_floor": float(result.noise_floor),
            "Delta": float(result.Delta),
            "q_mass": float(result.q_mass),
            "stage": result.stage,
            "weighting": result.weighting,
        },
        thresholds={"pg_tol": float(result.pg_tol), "gn_tol": float(result.gn_tol)},
        messages=messages,
    )

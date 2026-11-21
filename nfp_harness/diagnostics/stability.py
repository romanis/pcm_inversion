"""Integration stability and multi-start summary (DESIGN 5.7, 5.8). Owned by WP6."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..types import ConvergenceClass, DiagnosticBlock, OptimizeResult

__all__ = ["STABILITY_DEFAULTS", "integration_stability", "multistart_summary"]

STABILITY_DEFAULTS: dict[str, float] = {
    "se_change": 0.5,  # parameter change in SE units below this: stable
    "unit_change": 0.1,  # parameter change in natural units below this (no SEs): stable
    "headline_change": 0.1,  # relative change of a headline statistic below this: stable
}


def integration_stability(result: OptimizeResult, reevaluate: Callable[[np.ndarray], OptimizeResult],
                          units: np.ndarray, se: np.ndarray | None = None, names: Sequence[str] | None = None,
                          headlines: Mapping[str, tuple[float, float]] | None = None,
                          thresholds: Mapping[str, Any] | None = None) -> DiagnosticBlock:
    """integration_stability(result, reevaluate, units, se=None) -> DiagnosticBlock.

    ``reevaluate`` re-runs at most 20 iterations from the estimate with draws doubled. STABLE if every
    parameter moves less than 0.5 SE (or 0.1 natural units without SEs), else UNSTABLE with the list.
    INCONCLUSIVE (block status ``flag``) when the re-run did not evaluate the estimate: it ended INVERSION_FAILED
    (e.g. its start point failed, so it took no step) or its q is not finite. Its parameters then cannot have moved, so a zero change says
    nothing; the message gives the re-run's own reason (which markets failed).

    The change in q at the estimate is read from the first history record of the re-run when that
    record is at the estimate, else the endpoint q is compared. ``headlines`` maps a statistic name
    (median own elasticity, counterfactual headline) to (value at estimate, value after the re-run);
    a relative change above 0.1 also makes the block UNSTABLE.
    """
    th = dict(STABILITY_DEFAULTS)
    if thresholds:
        th.update({k: v for k, v in thresholds.items() if k in STABILITY_DEFAULTS})
    block = DiagnosticBlock(name="integration", thresholds=dict(th))
    theta = np.asarray(result.theta, dtype=float)
    names = list(names) if names is not None else [f"theta_{i}" for i in range(theta.size)]
    try:
        re = reevaluate(theta)
    except Exception as exc:  # noqa: BLE001 - any failure of the re-run is the finding
        block.status = "fail"
        block.values.update(status="FAILED", dq=None, changes={})
        block.messages.append(f"FAIL: Integration check failed: re-run with doubled draws raised "
                              f"{type(exc).__name__}: {exc}.")
        return block
    failed = _rerun_failure(re)
    if failed:
        block.status = "flag"
        block.values.update(status="INCONCLUSIVE", dq=None, dq_endpoint=None, changes={}, headlines={},
                            rerun_class=re.convergence.value, rerun_iterations=re.accepted_steps, unstable=[],
                            reason=failed)
        block.messages.append(f"FLAG: Integration INCONCLUSIVE with doubled draws: {failed}. No parameter or q "
                              f"comparison is possible.")
        return block
    first = re.history[0] if re.history else None
    if first is not None and np.allclose(first.theta, theta, rtol=0, atol=1e-12) and np.isfinite(first.q):
        dq_at = float(first.q - result.q)
    else:
        dq_at = float(re.q - result.q)
    new = np.asarray(re.theta, dtype=float)
    diff = new - theta
    use_se = se is not None and np.all(np.isfinite(np.asarray(se, dtype=float))) and np.all(np.asarray(se) > 0)
    scale = np.asarray(se, dtype=float) if use_se else np.asarray(units, dtype=float)
    tol = th["se_change"] if use_se else th["unit_change"]
    rel = diff / scale
    changes = {nm: {"from": float(a), "to": float(b), "change": float(r), "in": "se" if use_se else "natural_units"}
               for nm, a, b, r in zip(names, theta, new, rel)}
    moved = [nm for nm, r in zip(names, rel) if abs(r) >= tol]
    head: dict[str, Any] = {}
    for nm, (a, b) in (headlines or {}).items():
        r = abs(b - a) / abs(a) if a else (0.0 if b == a else float("inf"))
        head[nm] = {"from": float(a), "to": float(b), "relative_change": float(r)}
        if r >= th["headline_change"]:
            moved.append(nm)
    verdict = "UNSTABLE" if moved else "STABLE"
    block.values.update(status=verdict, dq=dq_at, dq_endpoint=float(re.q - result.q), changes=changes,
                        headlines=head, rerun_class=re.convergence.value, rerun_iterations=re.accepted_steps,
                        unstable=moved)
    unit_txt = f"{tol} SE" if use_se else f"{tol} natural units"
    if moved:
        block.status = "flag"
        block.messages.append(f"FLAG: Integration UNSTABLE with doubled draws: {', '.join(moved)} moved by at least "
                              f"{unit_txt}; q at the estimate changed by {dq_at:+.4g}.")
    else:
        block.status = "ok"
        block.messages.append(f"Integration STABLE with doubled draws: every parameter moved less than {unit_txt}; "
                              f"q at the estimate changed by {dq_at:+.4g}.")
    return block


def _rerun_failure(re: OptimizeResult) -> str:
    """Why the re-run did not evaluate the estimate, or '' when it did."""
    msg = f" ({re.message})" if getattr(re, "message", "") else ""
    if re.convergence is ConvergenceClass.INVERSION_FAILED:
        return f"the re-run ended INVERSION_FAILED after {re.accepted_steps} accepted steps{msg}"
    if not np.isfinite(re.q):
        return f"the re-run ended {re.convergence.value} with q = {re.q}{msg}"
    return ""


def multistart_summary(results: Sequence[OptimizeResult], units: np.ndarray, tol: float = 0.1) -> dict[str, Any]:
    """multistart_summary(results, units, tol=0.1) -> endpoints, q, class, number of distinct endpoints
    (closer than ``tol`` natural units in every parameter count as one) and the q spread."""
    units = np.asarray(units, dtype=float)
    reps: list[np.ndarray] = []
    rows = []
    for i, r in enumerate(results):
        t = np.asarray(r.theta, dtype=float)
        cid = None
        for j, c in enumerate(reps):
            if np.all(np.abs(t - c) / units < tol):
                cid = j
                break
        if cid is None:
            reps.append(t)
            cid = len(reps) - 1
        rows.append({"start": i, "theta": t.tolist(), "q": float(r.q), "class": r.convergence.value, "endpoint": cid})
    qs = np.array([r["q"] for r in rows], dtype=float)
    conv = [r["q"] for r, res in zip(rows, results) if res.convergence.converged]
    out = {"runs": rows, "n_runs": len(rows), "distinct_endpoints": len(reps),
           "q_min": float(qs.min()) if qs.size else None, "q_max": float(qs.max()) if qs.size else None,
           "q_spread": float(qs.max() - qs.min()) if qs.size else None,
           "converged_runs": len(conv), "tol_natural_units": tol}
    out["message"] = (f"{'FLAG: ' if len(reps) > 1 else ''}Multi-start: {len(rows)} runs, {len(reps)} distinct "
                      f"endpoint(s), {len(conv)} converged, q spread {out['q_spread']:.4g}." if rows
                      else "Multi-start: no runs.")
    return out

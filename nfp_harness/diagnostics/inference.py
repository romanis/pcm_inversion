"""Standard-error gating and sandwich standard errors (DESIGN 5.9, 2.4). Owned by WP6."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..types import ConvergenceClass

__all__ = ["se_gate", "sandwich_covariance", "standard_errors"]

_BLOCKING = ("RUN_TO_BOUND", "RUN_TO_INFINITY")


def se_gate(convergence: ConvergenceClass, rc_jacobi: float | None, degeneracies: list[dict],
            rc_min: float = 1e-6) -> tuple[bool, str | None]:
    """se_gate(convergence, rc_jacobi, degeneracies) -> (computed, reason).

    SEs only if the class is CONVERGED or CONVERGED_AT_CAP, ``rc(A_jac) >= 1e-6`` and no RUN_TO_BOUND or
    RUN_TO_INFINITY; otherwise "Standard errors omitted: {reason}."
    """
    reasons = []
    if not convergence.converged:
        reasons.append(f"objective did not converge (class {convergence.value})")
    if rc_jacobi is None or not np.isfinite(rc_jacobi):
        reasons.append("the Jacobian condition is unknown")
    elif rc_jacobi < rc_min:
        reasons.append(f"the Jacobian is ill conditioned (relative condition {rc_jacobi:.1e} < {rc_min:.0e} "
                       f"in Jacobi units)")
    hit = [d["name"] for d in degeneracies if d.get("detected") and d.get("name") in _BLOCKING]
    if hit:
        reasons.append(f"{' and '.join(hit)} detected")
    if reasons:
        return False, f"Standard errors omitted: {'; '.join(reasons)}."
    return True, None


def sandwich_covariance(G_full: np.ndarray, W: np.ndarray, S: np.ndarray, n: int) -> np.ndarray:
    """sandwich_covariance(G_full, W, S, n) -> V = (G'WG)^{-1} G'W S W G (G'WG)^{-1} / n over (beta, theta).

    Cap parameters are treated as fixed and get no SE: the caller drops their columns.
    Raises ``numpy.linalg.LinAlgError`` if G'WG is singular.
    """
    G = np.asarray(G_full, dtype=float)
    W = np.asarray(W, dtype=float)
    S = np.asarray(S, dtype=float)
    B = G.T @ W @ G
    Binv = np.linalg.inv(0.5 * (B + B.T))
    M = G.T @ W @ S @ W @ G
    V = Binv @ M @ Binv / n
    return 0.5 * (V + V.T)


def standard_errors(G_full: np.ndarray, W: np.ndarray, S: np.ndarray, n: int, names: Sequence[str],
                    fixed: Sequence[bool] | None = None) -> dict[str, Any]:
    """Sandwich SEs by name. Columns flagged ``fixed`` (cap parameters at their cap, parameters
    held fixed) are dropped and get ``None``."""
    names = list(names)
    keep = np.ones(len(names), dtype=bool) if fixed is None else ~np.asarray(fixed, dtype=bool)
    V = sandwich_covariance(np.asarray(G_full)[:, keep], W, S, n)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    out: dict[str, float | None] = {nm: None for nm in names}
    for nm, v in zip([nm for nm, k in zip(names, keep) if k], se):
        out[nm] = float(v)
    return {"se": out, "covariance": V, "names": [nm for nm, k in zip(names, keep) if k]}

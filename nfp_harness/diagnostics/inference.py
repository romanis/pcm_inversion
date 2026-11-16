"""Standard-error gating and sandwich standard errors (DESIGN 5.9, 2.4). Owned by WP6."""

from __future__ import annotations

import numpy as np

from ..types import ConvergenceClass

__all__ = ["se_gate", "sandwich_covariance"]


def se_gate(convergence: ConvergenceClass, rc_jacobi: float, degeneracies: list[dict]) -> tuple[bool, str | None]:
    """se_gate(convergence, rc_jacobi, degeneracies) -> (computed, reason).

    SEs only if the class is CONVERGED or CONVERGED_AT_CAP, ``rc(A_jac) >= 1e-6`` and no RUN_TO_BOUND or
    RUN_TO_INFINITY; otherwise "Standard errors omitted: {reason}."
    """
    raise NotImplementedError("se_gate(convergence, rc_jacobi, degeneracies) -> (bool, reason): WP6, DESIGN 5.9")


def sandwich_covariance(G_full: np.ndarray, W: np.ndarray, S: np.ndarray, n: int) -> np.ndarray:
    """sandwich_covariance(G_full, W, S, n) -> V = (G'WG)^{-1} G'W S W G (G'WG)^{-1} / n over (beta, theta).

    Cap parameters are treated as fixed and get no SE.
    """
    raise NotImplementedError("sandwich_covariance(G_full, W, S, n) -> np.ndarray: WP6, DESIGN 5.9")

"""Concentrated linear parameters: IV and absorbed fixed effects (DESIGN 2.4). Owned by WP4."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import LinearFit

__all__ = ["LinearIV", "Absorb"]


class LinearIV:
    """Concentrates beta out with the aggregate block of W.

    ``beta = (X1' Z W_aa Z' X1)^{-1} X1' Z W_aa Z' delta``, ``M = I - X1 (...)``, ``xi = M delta``.
    Exact when W is block diagonal between aggregate and micro moments and micro moments do not
    depend on beta. Z must contain the exogenous X1 columns: ``Z = [X1, excluded]`` (F1, LEARNINGS 16).
    For inference the aggregate rows of the full Jacobian over (beta, theta) are
    ``[-Z' X1 / n, Z' D / n]``.
    """

    def __init__(self, X1: np.ndarray, Z: np.ndarray, W_aa: np.ndarray | None = None,
                 names: Sequence[str] | None = None) -> None:
        """LinearIV(X1, Z, W_aa=None, names=None). W_aa defaults to (Z'Z/n)^{-1}."""
        raise NotImplementedError("LinearIV(X1, Z, W_aa=None, names=None): WP4, DESIGN 2.4")

    def set_weighting(self, W_aa: np.ndarray) -> None:
        """set_weighting(W_aa): recompute beta's projection and M for a new aggregate block of W."""
        raise NotImplementedError("LinearIV.set_weighting(W_aa): WP4, DESIGN 2.4")

    def fit(self, delta: np.ndarray) -> LinearFit:
        """fit(delta) -> LinearFit(names, beta, xi, M)."""
        raise NotImplementedError("LinearIV.fit(delta) -> LinearFit: WP4, DESIGN 2.4")


class Absorb:
    """Replaces X1 with within-group demeaning for fixed effects (Nevo brand dummies)."""

    def __init__(self, groups: np.ndarray) -> None:
        """Absorb(groups): groups is the (n,) fixed-effect id of each product."""
        raise NotImplementedError("Absorb(groups): WP4, DESIGN 2.4")

    def set_weighting(self, W_aa: np.ndarray) -> None:
        """set_weighting(W_aa): no-op for pure absorption; kept for the LinearIV interface."""
        raise NotImplementedError("Absorb.set_weighting(W_aa): WP4, DESIGN 2.4")

    def fit(self, delta: np.ndarray) -> LinearFit:
        """fit(delta) -> LinearFit with xi = within-group demeaned delta and empty beta."""
        raise NotImplementedError("Absorb.fit(delta) -> LinearFit: WP4, DESIGN 2.4")

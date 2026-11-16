"""Box trust-region Gauss-Newton for q(theta) = n g' W g (DESIGN 4.1-4.6). Owned by WP5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import OptimizeResult, Problem

__all__ = ["OptimizerOptions", "minimize", "projected_gradient"]


@dataclass(frozen=True)
class OptimizerOptions:
    """Tuning constants of DESIGN 4.2-4.6 with their documented defaults."""

    Delta_0: float = 0.3  # initial box radius, natural units
    Delta_max: float = 5.0
    Delta_min: float = 1e-8
    rho_accept: float = 1e-4
    q_mass_0: float = 0.2  # initial affected-mass threshold of the max_step cap
    q_mass_max: float = 0.8
    ridge: float = 1e-10  # mu = ridge * max(d)^2
    pg_tol: float = 1e-3  # max |pg| <= pg_tol (1 + q)^{1/2}
    gn_tol: float = 1e-6  # dq_gn <= gn_tol (1 + q)
    max_rejections: int = 10
    flat_window: int = 25
    flat_tol: float = 1e-3
    max_iterations: int | None = None
    max_seconds: float | None = None


def minimize(problem: Problem, weighting: Any, options: OptimizerOptions, checkpoint: Any = None) -> OptimizeResult:
    """minimize(problem, weighting, options, checkpoint) -> OptimizeResult.

    Minimises ``q(theta) = n g' W g`` over the free parameters within ``[lower, upper]``. Steps solve
    ``min ||r + A d||^2 + mu ||diag(d_k) d||^2`` in a box of radius Delta (natural units) with
    ``scipy.optimize.lsq_linear``; ``max_step`` truncates steps by affected mass; acceptance on the
    gain ratio rho; non-OK trials are rejected and ``restore`` is called; a checkpoint is written
    after every accepted step. Stopping classes are those of ``types.ConvergenceClass``.
    """
    raise NotImplementedError("minimize(problem, weighting, options, checkpoint) -> OptimizeResult: WP5, DESIGN 4")


def projected_gradient(G: np.ndarray, W: np.ndarray, g: np.ndarray, n: int, units: np.ndarray,
                       theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """projected_gradient(G, W, g, n, units, theta, lower, upper) -> pg.

    ``grad_k = 2 n (G' W g)_k unit_k``, set to zero for parameters at a bound whose gradient points outward.
    """
    raise NotImplementedError("projected_gradient(G, W, g, n, units, theta, lower, upper): WP5, DESIGN 4.6")

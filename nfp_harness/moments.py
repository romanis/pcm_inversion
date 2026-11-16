"""Moment objects: aggregate IV moments and pooled micro moments (DESIGN 2.3). Owned by WP4.

A moment returns its value and partial derivatives in delta and theta (``MomentBlock``); the
framework applies the implicit-function chain rule ``G_b = d_theta_b + sum_t d_delta_b[:, rows_t] @ D_t``.
Moments never see D.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import InnerSolution, KernelRequests, LinearFit, MicroPartRequest, Moment, MomentBlock

__all__ = ["Moment", "IVMoments", "MicroDataset", "MicroPart", "RatioMicroMoment"]

#: A micro part, a pair (agent weight h, product weight v with outside value v0); see ``types.MicroPartRequest``.
MicroPart = MicroPartRequest


class IVMoments:
    """Aggregate IV moments ``g = Z' xi / n`` with ``xi = M delta``.

    ``d_delta = Z' M / n``, ``d_theta = 0``. Covariance block: clustered
    ``S = (1/n) sum_c (sum_{j in c} z_j xi_j)(sum_{j in c} z_j xi_j)'``.
    """

    name: str

    def __init__(self, Z: np.ndarray, clustering_ids: np.ndarray | None = None, name: str = "aggregate") -> None:
        """IVMoments(Z, clustering_ids=None, name="aggregate"). No clustering means one cluster per product."""
        raise NotImplementedError("IVMoments(Z, clustering_ids=None, name='aggregate'): WP4, DESIGN 2.3")

    def requests(self) -> KernelRequests:
        """requests() -> KernelRequests(): IV moments need no extra kernel outputs."""
        raise NotImplementedError("IVMoments.requests() -> KernelRequests: WP4, DESIGN 2.3")

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value, d_delta=Z'M/n, d_theta=0)."""
        raise NotImplementedError("IVMoments.evaluate(sol, linear, need_jacobian) -> MomentBlock: WP4, DESIGN 2.3")

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> clustered S_aggregate, (k_b, k_b)."""
        raise NotImplementedError("IVMoments.covariance_block(sol, linear) -> np.ndarray: WP4, DESIGN 2.3")


class MicroDataset:
    """A micro dataset pooled across markets (pyblp 1.3.0 ``MicroDataset``).

    ``Den = sum_t sum_i w_it`` over the dataset's markets; part covariances are divided by
    ``observations`` (29,125 for Petrin's CEX).
    """

    def __init__(self, name: str, observations: int, market_ids: Sequence | None = None) -> None:
        """MicroDataset(name, observations, market_ids=None). None means all markets."""
        raise NotImplementedError("MicroDataset(name, observations, market_ids=None): WP4, DESIGN 2.3")


class RatioMicroMoment:
    """Ratio micro moments ``g_r = part(num_r) / part(den_r) - value_r``, one row per ratio.

    ``part = (1 / Den) sum_t [ v0 H_t + sum_i w_it h(i) sum_j (v_j - v0) s_ijt ]``. Partials are
    ``grad_f . d parts / d delta`` and ``grad_f . d parts / d theta``. The covariance block is
    ``grad_f Sigma_parts grad_f'`` with part covariances ``E[v1 v2] - E[v1] E[v2]`` divided by
    ``dataset.observations``; product parts are requested only when ``covariance_block`` is called.
    """

    name: str

    def __init__(self, name: str, dataset: MicroDataset, numerators: Sequence[MicroPart],
                 denominators: Sequence[MicroPart], values: np.ndarray,
                 labels: Sequence[str] | None = None) -> None:
        """RatioMicroMoment(name, dataset, numerators, denominators, values, labels=None)."""
        raise NotImplementedError(
            "RatioMicroMoment(name, dataset, numerators, denominators, values, labels=None): WP4, DESIGN 2.3"
        )

    def requests(self) -> KernelRequests:
        """requests() -> KernelRequests with every numerator and denominator part."""
        raise NotImplementedError("RatioMicroMoment.requests() -> KernelRequests: WP4, DESIGN 2.3")

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value, d_delta, d_theta, data=values)."""
        raise NotImplementedError("RatioMicroMoment.evaluate(sol, linear, need_jacobian) -> MomentBlock: WP4, DESIGN 2.3")

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> V_micro, (k_b, k_b); the stacked S uses n * V_micro."""
        raise NotImplementedError("RatioMicroMoment.covariance_block(sol, linear) -> np.ndarray: WP4, DESIGN 2.3")

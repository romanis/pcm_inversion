"""PCM adapter: markets, draws, kernel calls, threads, warm starts (DESIGN 3.5, 2.2). Owned by WP7.

Implements ``types.InnerModel`` on ``nfp_harness._native.pcm_kernel``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import InnerSolution, KernelRequests, MarketIndex, StepLimit

__all__ = ["PCMModel", "allocate_threads"]


def allocate_threads(weights: np.ndarray, total: int) -> np.ndarray:
    """allocate_threads(weights, total) -> (T,) threads per market.

    ``n_t = max(1, floor(total w_t / sum w))``, leftover threads to the markets with the largest weight.
    First evaluation weights ``J_t^2 R_t``; afterwards an exponential moving average (factor 0.5) of
    ``seconds_t * n_threads_t``.
    """
    raise NotImplementedError("allocate_threads(weights, total) -> np.ndarray: WP7, DESIGN 3.5")


class PCMModel:
    """Exact PCM inner model.

    Markets run in a ``ThreadPoolExecutor`` with one Python thread per market; each ``solve_market``
    call gets its own OpenMP team size. Keeps an accepted state (per market delta and the theta it was
    solved at) and a trial state: ``solve`` warm-starts from the accepted state and writes the trial
    state; ``commit`` promotes it; ``restore`` discards it (F3). ``max_step`` uses
    ``pcm_kernel.max_step``; trial points are checked with ``frontier_signature`` (DESIGN 4.4).
    """

    markets: list[MarketIndex]
    n_products: int

    def __init__(self, spec: Any, data: Any, layout: Any, nu_per_agent: int, seed: int,
                 total_threads: int | None = None) -> None:
        """PCMModel(spec, data, layout, nu_per_agent, seed, total_threads=None). None uses os.cpu_count()."""
        raise NotImplementedError("PCMModel(spec, data, layout, nu_per_agent, seed, total_threads=None): WP7")

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        """solve(theta, requests, need_jacobian) -> InnerSolution."""
        raise NotImplementedError("PCMModel.solve(theta, requests, need_jacobian) -> InnerSolution: WP7, DESIGN 2.2")

    def max_step(self, sol: InnerSolution, dtheta: np.ndarray) -> StepLimit:
        """max_step(sol, dtheta) -> StepLimit from the per-market kernel events (DESIGN 3.3, 4.2)."""
        raise NotImplementedError("PCMModel.max_step(sol, dtheta) -> StepLimit: WP7, DESIGN 3.3")

    def realised_mass(self, theta: np.ndarray) -> float:
        """realised_mass(theta) -> max_j A_j between the trial and the accepted frontier signatures (DESIGN 4.4)."""
        raise NotImplementedError("PCMModel.realised_mass(theta) -> float: WP7, DESIGN 4.4")

    def commit(self, theta: np.ndarray) -> None:
        """commit(theta): promote the trial state to the accepted state."""
        raise NotImplementedError("PCMModel.commit(theta): WP7, DESIGN 2.1")

    def restore(self, theta: np.ndarray) -> None:
        """restore(theta): discard the trial state."""
        raise NotImplementedError("PCMModel.restore(theta): WP7, DESIGN 2.1")

    def state_dict(self) -> dict[str, Any]:
        """state_dict() -> per-market accepted delta and theta, thread allocation weights."""
        raise NotImplementedError("PCMModel.state_dict() -> dict: WP7, DESIGN 4.7")

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """load_state_dict(d): restore what ``state_dict`` wrote."""
        raise NotImplementedError("PCMModel.load_state_dict(d): WP7, DESIGN 4.7")

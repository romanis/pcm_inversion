"""PCM postestimation: elasticities, diversion, Nash costs, consumer surplus, counterfactuals
(DESIGN 5.6, 6.5). Owned by WP8."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["elasticities", "diversion", "nash_markups", "consumer_surplus", "drop_products_counterfactual", "compare_blp"]


def elasticities(s: np.ndarray, ds_dp: np.ndarray, p: np.ndarray) -> np.ndarray:
    """elasticities(s, ds_dp, p) -> (J, J) matrix ``e_jk = (ds_j/dp_k) p_k / s_j`` for one market."""
    raise NotImplementedError("elasticities(s, ds_dp, p) -> np.ndarray: WP8, DESIGN 5.6")


def diversion(ds_dp: np.ndarray) -> np.ndarray:
    """diversion(ds_dp) -> (J, J) ``D_jk = -(ds_k/dp_j) / (ds_j/dp_j)``; outside diversion is ``1 - sum_k D_jk``."""
    raise NotImplementedError("diversion(ds_dp) -> np.ndarray: WP8, DESIGN 5.6")


def nash_markups(s: np.ndarray, ds_dp: np.ndarray, firm_ids: np.ndarray, rcond_min: float = 1e-12) -> dict[str, Any]:
    """nash_markups(s, ds_dp, firm_ids, rcond_min=1e-12) -> markups ``-(Omega * (ds/dp)')^{-1} s``.

    Reports NASH_JACOBIAN_SINGULAR when the ownership-weighted ds/dp has rcond below ``rcond_min``.
    """
    raise NotImplementedError("nash_markups(s, ds_dp, firm_ids, rcond_min=1e-12) -> dict: WP8, DESIGN 6.5")


def consumer_surplus(model: Any, theta: np.ndarray, delta: np.ndarray, market: Any) -> dict[str, Any]:
    """consumer_surplus(model, theta, delta, market) -> total and per-draw CS via ``pcm_kernel.consumer_surplus``
    ($1000 per household), with tabulations by alpha quintile and family size."""
    raise NotImplementedError("consumer_surplus(model, theta, delta, market) -> dict: WP8, DESIGN 3.4, 6.5")


def drop_products_counterfactual(model: Any, theta: np.ndarray, market: Any, drop: np.ndarray,
                                 costs: np.ndarray, damping: float = 0.5, tol: float = 1e-7,
                                 max_iterations: int = 500) -> dict[str, Any]:
    """drop_products_counterfactual(model, theta, market, drop, costs, damping=0.5, tol=1e-7,
    max_iterations=500) -> new prices, shares, CS and convergence.

    Delta and xi of the remaining products unchanged; prices from ``p <- p + damping (c + markup(p) - p)``
    with prices re-sorted and ties spread each iteration. Non-convergence is reported, not hidden.
    """
    raise NotImplementedError("drop_products_counterfactual(model, theta, market, drop, costs, ...): WP8, DESIGN 6.5")


def compare_blp(pcm_summary: Mapping[str, Any], blp_summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """compare_blp(pcm_summary, blp_summaries) -> comparison table against the BLP summary JSON files."""
    raise NotImplementedError("compare_blp(pcm_summary, blp_summaries) -> str: WP8, DESIGN 6.5")

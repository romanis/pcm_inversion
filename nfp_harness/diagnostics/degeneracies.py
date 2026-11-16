"""Named degeneracy detectors (DESIGN 5.4). Owned by WP6.

Each detector reads the final point and the accepted-step history and returns
``{"name", "detected", "evidence", "message"}``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..types import HistoryRecord

__all__ = [
    "DEGENERACIES",
    "detect_all",
    "scale_artifact",
    "sd_xi_blowup",
    "run_to_bound",
    "normalisation_cap_active",
    "run_to_infinity",
    "segmented_corner",
    "vertical_collapse",
]

DEGENERACIES = (
    "SCALE_ARTIFACT",
    "SD_XI_BLOWUP",
    "RUN_TO_BOUND",
    "NORMALISATION_CAP_ACTIVE",
    "RUN_TO_INFINITY",
    "SEGMENTED_CORNER",
    "VERTICAL_COLLAPSE",
)


def detect_all(history: Sequence[HistoryRecord], final: Mapping[str, Any], spec: Any,
               thresholds: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """detect_all(history, final, spec, thresholds=None) -> one entry per name in ``DEGENERACIES``."""
    raise NotImplementedError("detect_all(history, final, spec, thresholds=None): WP6, DESIGN 5.4")


def scale_artifact(history: Sequence[HistoryRecord], thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SCALE_ARTIFACT (LEARNINGS 1): q_fixed falls by more than 10x while q_cue changes by less than 2x
    and sd(xi) falls by more than 2x."""
    raise NotImplementedError("scale_artifact(history, thresholds=None): WP6, DESIGN 5.4")


def sd_xi_blowup(history: Sequence[HistoryRecord], xi: np.ndarray,
                 thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SD_XI_BLOWUP (LEARNINGS 13): q falls by more than 50% while sd(xi) rises by more than 2x; or
    kurtosis(xi) > 10; or the top 1% of products carry more than 25% of sum xi^2."""
    raise NotImplementedError("sd_xi_blowup(history, xi, thresholds=None): WP6, DESIGN 5.4")


def run_to_bound(theta: np.ndarray, gradient: np.ndarray, layout: Any,
                 thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """RUN_TO_BOUND: a non-cap parameter within 1e-3 units of a bound with outward gradient."""
    raise NotImplementedError("run_to_bound(theta, gradient, layout, thresholds=None): WP6, DESIGN 5.4")


def normalisation_cap_active(theta: np.ndarray, layout: Any) -> dict[str, Any]:
    """NORMALISATION_CAP_ACTIVE: a cap parameter at its cap (a stated normalisation, not a failure)."""
    raise NotImplementedError("normalisation_cap_active(theta, layout): WP6, DESIGN 5.4")


def run_to_infinity(history: Sequence[HistoryRecord], layout: Any,
                    thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """RUN_TO_INFINITY: over the last 10 accepted steps |theta_k| rose monotonically to above 5 times
    its start in natural units while q fell by less than 10%."""
    raise NotImplementedError("run_to_infinity(history, layout, thresholds=None): WP6, DESIGN 5.4")


def segmented_corner(theta: np.ndarray, layout: Any, price_range: float, cross_group_diversion: float | None,
                     thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SEGMENTED_CORNER (LEARNINGS 13): for a taste spread on a binary column,
    ``2 (1 - Phi(range / sigma)) > 0.2``, or cross-group diversion below 1%."""
    raise NotImplementedError("segmented_corner(theta, layout, price_range, ...): WP6, DESIGN 5.4")


def vertical_collapse(theta: np.ndarray, layout: Any, economics: Mapping[str, Any] | None,
                      thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """VERTICAL_COLLAPSE: all non-cap taste spreads below 0.05 natural units, or median nonzero
    cross-elasticities per product <= 2, or median own elasticity below -50."""
    raise NotImplementedError("vertical_collapse(theta, layout, economics, thresholds=None): WP6, DESIGN 5.4")

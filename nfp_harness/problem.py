"""Generic nested-fixed-point GMM problem (DESIGN 2.1-2.2). Owned by WP4."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .params import ParameterLayout
from .types import Evaluation, InnerModel, Moment, StepLimit

__all__ = ["NFPGMMProblem"]


class NFPGMMProblem:
    """Implements ``types.Problem`` for any model with an inner inversion.

    ``__init__`` collects ``moment.requests()`` from every moment once. ``evaluate`` solves the inner
    model, fits the linear parameters, evaluates every moment and applies the chain rule
    ``D_t = -(ds_t/ddelta_t)^{-1} (ds_t/dtheta)``, ``G_b = d_theta_b + sum_t d_delta_b[:, rows_t] @ D_t``.
    It never raises for model failures; it returns a non-OK ``EvalStatus``.
    """

    layout: ParameterLayout

    def __init__(self, model: InnerModel, moments: Sequence[Moment], linear: Any, layout: ParameterLayout) -> None:
        """NFPGMMProblem(model, moments, linear, layout)."""
        raise NotImplementedError("NFPGMMProblem(model, moments, linear, layout): WP4, DESIGN 2.2")

    def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation:
        """evaluate(theta, need_jacobian=True) -> Evaluation."""
        raise NotImplementedError("NFPGMMProblem.evaluate(theta, need_jacobian=True) -> Evaluation: WP4, DESIGN 2.1")

    def covariance(self, ev: Evaluation) -> np.ndarray:
        """covariance(ev) -> S(theta) = blockdiag(S_aggregate, n V_micro), (k, k)."""
        raise NotImplementedError("NFPGMMProblem.covariance(ev) -> np.ndarray: WP4, DESIGN 2.1, 2.3")

    def set_weighting(self, W: np.ndarray) -> None:
        """set_weighting(W): store W and re-concentrate the linear parameters with its aggregate block."""
        raise NotImplementedError("NFPGMMProblem.set_weighting(W): WP4, DESIGN 2.1, 2.4")

    def max_step(self, ev: Evaluation, dtheta: np.ndarray) -> StepLimit:
        """max_step(ev, dtheta) -> StepLimit, delegated to the model when it has ``max_step``."""
        raise NotImplementedError("NFPGMMProblem.max_step(ev, dtheta) -> StepLimit: WP4, DESIGN 2.1")

    def commit(self, theta: np.ndarray) -> None:
        """commit(theta): theta accepted, promote the model's trial state."""
        raise NotImplementedError("NFPGMMProblem.commit(theta): WP4, DESIGN 2.1")

    def restore(self, theta: np.ndarray) -> None:
        """restore(theta): trial rejected, back to the model's accepted state."""
        raise NotImplementedError("NFPGMMProblem.restore(theta): WP4, DESIGN 2.1")

    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]:
        """cue_terms(ev) -> (V: C x k, dV: C x k x P), the per-cluster terms of the CUE gradient."""
        raise NotImplementedError("NFPGMMProblem.cue_terms(ev) -> (V, dV): WP4, DESIGN 2.1, 4.5")

    def state_dict(self) -> dict[str, Any]:
        """state_dict() -> warm starts etc. for checkpoints."""
        raise NotImplementedError("NFPGMMProblem.state_dict() -> dict: WP4, DESIGN 2.1")

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """load_state_dict(d): restore what ``state_dict`` wrote."""
        raise NotImplementedError("NFPGMMProblem.load_state_dict(d): WP4, DESIGN 2.1")

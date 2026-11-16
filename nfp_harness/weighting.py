"""Weighting modes and covariance algebra: fixed, two-step, CUE (DESIGN 4.5, 2.3). Owned by WP4."""

from __future__ import annotations

import numpy as np

__all__ = ["Fixed", "TwoStep", "CUE", "initial_weighting", "clustered_covariance", "stacked_covariance"]


class Fixed:
    """Fixed(W): W given by the spec."""

    def __init__(self, W: np.ndarray) -> None:
        """Fixed(W)."""
        raise NotImplementedError("Fixed(W): WP4, DESIGN 4.5")


class TwoStep:
    """TwoStep(W1): run with W1, then W2 = S(theta_hat_1)^{-1} and restart at theta_hat_1.

    Step 2 starts only if the step-1 report has no SCALE_ARTIFACT or SD_XI_BLOWUP.
    """

    def __init__(self, W1: np.ndarray) -> None:
        """TwoStep(W1)."""
        raise NotImplementedError("TwoStep(W1): WP4, DESIGN 4.5")


class CUE:
    """CUE: q = n g' S(theta)^{-1} g with the micro block of S held at its value at the CUE start point."""

    def __init__(self, hold_micro_block: bool = True) -> None:
        """CUE(hold_micro_block=True)."""
        raise NotImplementedError("CUE(hold_micro_block=True): WP4, DESIGN 4.5")


def initial_weighting(Z: np.ndarray, V_micro: np.ndarray | None) -> np.ndarray:
    """initial_weighting(Z, V_micro) -> W1 = blockdiag((Z'Z/n)^{-1}, (n V_micro(theta_0))^{-1}).

    pyblp's initial weighting with micro moments; with V_micro None only the aggregate block.
    """
    raise NotImplementedError("initial_weighting(Z, V_micro) -> W1: WP4, DESIGN 4.5")


def clustered_covariance(Z: np.ndarray, xi: np.ndarray, clustering_ids: np.ndarray | None) -> np.ndarray:
    """clustered_covariance(Z, xi, clustering_ids) -> S = (1/n) sum_c (sum_{j in c} z_j xi_j)(...)'."""
    raise NotImplementedError("clustered_covariance(Z, xi, clustering_ids) -> np.ndarray: WP4, DESIGN 2.3")


def stacked_covariance(S_aggregate: np.ndarray, V_micro: np.ndarray | None, n: int) -> np.ndarray:
    """stacked_covariance(S_aggregate, V_micro, n) -> blockdiag(S_aggregate, n V_micro)."""
    raise NotImplementedError("stacked_covariance(S_aggregate, V_micro, n) -> np.ndarray: WP4, DESIGN 2.3")

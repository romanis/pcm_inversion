"""Flat directions, units versus identification, weak identification (DESIGN 5.2, 5.3). Owned by WP6."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..types import DiagnosticBlock

__all__ = ["identification_block", "weak_identification_block"]


def identification_block(G: np.ndarray, W: np.ndarray, units: np.ndarray, names: Sequence[str], n: int,
                         named_directions: Mapping[str, np.ndarray] | None = None,
                         thresholds: Mapping[str, Any] | None = None) -> DiagnosticBlock:
    """identification_block(G, W, units, names, n, named_directions=None, thresholds=None) -> DiagnosticBlock.

    SVD of ``A_nat = W^{1/2} G diag(unit)`` and of the column-normalised ``A_jac``. For each direction
    with ``s_l/s_1 < 1e-4`` in either: ``dq_per_unit = n s_l^2``, loadings above 0.2 in named
    parameters, verdict UNITS / IDENTIFICATION / WEAK, and the declared named direction with
    |cosine| above 0.8.
    """
    raise NotImplementedError("identification_block(G, W, units, names, n, ...): WP6, DESIGN 5.2")


def weak_identification_block(G: np.ndarray, S: np.ndarray, n: int,
                              thresholds: Mapping[str, Any] | None = None) -> DiagnosticBlock:
    """weak_identification_block(G, S, n, thresholds=None) -> DiagnosticBlock.

    ``CD = n lambda_min(Gs' S^{-1} Gs)`` with ``Gs = G diag(1 / sd_k)`` against the chi-square(k - P + 1)
    95% critical value; below it WEAK_IDENTIFICATION. Ignores the sampling variance of G (stated).
    """
    raise NotImplementedError("weak_identification_block(G, S, n, thresholds=None): WP6, DESIGN 5.3")

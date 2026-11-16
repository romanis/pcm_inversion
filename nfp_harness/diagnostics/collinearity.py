"""Collinearity of X1, Z, Z'X1 and the columns of G (DESIGN 5.3). Owned by WP6."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..types import DiagnosticBlock

__all__ = ["collinearity_block"]


def collinearity_block(X1: np.ndarray | None, Z: np.ndarray, A_jac: np.ndarray | None, names: Sequence[str],
                       thresholds: Mapping[str, Any] | None = None) -> DiagnosticBlock:
    """collinearity_block(X1, Z, A_jac, names, thresholds=None) -> DiagnosticBlock.

    Rank and condition of standardised X1 and Z (rank tolerance 1e-10 of s_1, flag cond > 1e8);
    rank of Z'X1 must equal the number of linear parameters, with the smallest canonical correlation
    (fail on deficiency: the F1 check); pairs of ``A_jac`` columns with |correlation| above 0.99.
    """
    raise NotImplementedError("collinearity_block(X1, Z, A_jac, names, thresholds=None): WP6, DESIGN 5.3")


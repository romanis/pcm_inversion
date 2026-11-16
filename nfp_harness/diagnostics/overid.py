"""Hansen J overidentification test (DESIGN 5.5). Owned by WP6."""

from __future__ import annotations

import numpy as np

from ..types import DiagnosticBlock

__all__ = ["hansen_j"]


def hansen_j(g: np.ndarray, S: np.ndarray, n: int, dof: int, weighting: str) -> DiagnosticBlock:
    """hansen_j(g, S, n, dof, weighting) -> DiagnosticBlock with J = n g' S^{-1} g and its p-value.

    ``dof = k - (P_free + K_linear)`` minus parameters at a cap. Computed only for two-step and CUE;
    for a fixed-W step the block says why it is not a J statistic.
    """
    raise NotImplementedError("hansen_j(g, S, n, dof, weighting) -> DiagnosticBlock: WP6, DESIGN 5.5")

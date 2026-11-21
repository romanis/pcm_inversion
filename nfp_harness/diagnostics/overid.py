"""Hansen J overidentification test (DESIGN 5.5). Owned by WP6."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..types import DiagnosticBlock

__all__ = ["hansen_j"]

_EFFICIENT = ("two_step", "cue")


def hansen_j(g: np.ndarray, S: np.ndarray, n: int, dof: int, weighting: str,
             converged: bool = True, level: float = 0.05) -> DiagnosticBlock:
    """hansen_j(g, S, n, dof, weighting, converged=True, level=0.05) -> DiagnosticBlock with
    J = n g' S^{-1} g and its p-value.

    ``dof = k - (P_free + K_linear)`` minus parameters at a cap. Computed only for two-step and CUE;
    for a fixed-W step the block says why it is not a J statistic. When the objective did not
    converge the numbers are reported and marked "not interpreted".
    """
    block = DiagnosticBlock(name="hansen_j", thresholds={"level": level})
    block.values.update(J=None, dof=int(dof), p_value=None, weighting=weighting)
    if weighting not in _EFFICIENT:
        block.status = "not_run"
        block.messages.append(f"Hansen J not computed: weighting is {weighting!r}, so W is not S^-1 at the estimate "
                              f"and n g'Wg is not chi-square distributed.")
        return block
    if dof < 0:
        block.status = "fail"
        block.messages.append(f"FAIL: Hansen J undefined: {dof} degrees of freedom (fewer moments than unknowns).")
        return block
    if dof == 0:
        block.status = "not_run"
        block.messages.append("Hansen J not computed: the model is exactly identified (0 degrees of freedom).")
        return block
    g = np.asarray(g, dtype=float)
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    a = np.linalg.lstsq(S, g, rcond=None)[0]
    J = float(n * g @ a)
    p = float(stats.chi2.sf(J, dof))
    block.values.update(J=J, p_value=p)
    text = f"Hansen J = {J:.4g} with {dof} degrees of freedom, p = {p:.3g}."
    if not converged:
        block.status = "not_run"
        block.messages.append(f"{text} Not interpreted: objective did not converge.")
    elif p < level:
        block.status = "flag"
        block.messages.append(f"FLAG: {text} The overidentifying restrictions are rejected at {100 * level:.0f}%.")
    else:
        block.status = "ok"
        block.messages.append(f"{text} Not rejected at {100 * level:.0f}%.")
    return block

"""Weighting modes and covariance algebra: fixed, two-step, CUE (DESIGN 4.5, 2.3). Owned by WP4.

Conventions. The stacked moment vector g has k rows; the objective is ``q = n g' W g`` with n the
number of products. ``S`` is the asymptotic covariance of ``sqrt(n) g``:
``S = blockdiag(S_aggregate, n V_micro)``, where ``S_aggregate`` is the clustered covariance of
``z_j xi_j`` and ``V_micro`` is the covariance of the micro moments themselves (part covariances
divided by the dataset's observations; pyblp ``results/problem_results.py`` line 756).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg

__all__ = [
    "Fixed",
    "TwoStep",
    "CUE",
    "initial_weighting",
    "clustered_covariance",
    "cluster_sums",
    "stacked_covariance",
    "weight_from_covariance",
    "cue_objective",
    "bind_calibration",
    "upweight_rows",
    "efficient_on_excluded",
    "STEP2_RULES",
]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def weight_from_covariance(S: np.ndarray, rcond_min: float = 1e-14) -> np.ndarray:
    """W = S^{-1} for a symmetric positive definite S.

    Raises ``np.linalg.LinAlgError`` naming the reciprocal condition number when S is not positive
    definite or ``lambda_min / lambda_max < rcond_min``; a weight matrix from a singular S would
    silently drop moments.
    """
    S = _sym(np.asarray(S, dtype=float))
    if S.size == 0:
        return S.copy()
    evals, evecs = np.linalg.eigh(S)
    top = float(evals[-1])
    rcond = float(evals[0]) / top if top > 0 else -np.inf
    if not (top > 0 and rcond >= rcond_min):
        raise np.linalg.LinAlgError(
            f"covariance matrix is not safely positive definite: lambda_min/lambda_max = {rcond:.3e} "
            f"(threshold {rcond_min:.1e}, lambda_max = {top:.3e})")
    return _sym((evecs / evals) @ evecs.T)


def cluster_sums(a: np.ndarray, clustering_ids: np.ndarray | None) -> np.ndarray:
    """Sum the rows of a (n, ...) array within clusters: (C, ...). None means one cluster per row.

    Clusters are ordered by ``np.unique(clustering_ids)``.
    """
    a = np.asarray(a, dtype=float)
    if clustering_ids is None:
        return a.copy()
    ids = np.asarray(clustering_ids).ravel()
    if ids.shape[0] != a.shape[0]:
        raise ValueError(f"{ids.shape[0]} clustering ids for {a.shape[0]} rows")
    _, codes = np.unique(ids, return_inverse=True)
    out = np.zeros((int(codes.max()) + 1,) + a.shape[1:])
    np.add.at(out, codes, a)
    return out


def clustered_covariance(Z: np.ndarray, xi: np.ndarray, clustering_ids: np.ndarray | None) -> np.ndarray:
    """clustered_covariance(Z, xi, clustering_ids) -> S = (1/n) sum_c (sum_{j in c} z_j xi_j)(...)'."""
    Z = np.asarray(Z, dtype=float)
    xi = np.asarray(xi, dtype=float).ravel()
    n = Z.shape[0]
    V = cluster_sums(Z * xi[:, None], clustering_ids)
    return _sym(V.T @ V / n)


def stacked_covariance(S_aggregate: np.ndarray, V_micro: np.ndarray | None, n: int) -> np.ndarray:
    """stacked_covariance(S_aggregate, V_micro, n) -> blockdiag(S_aggregate, n V_micro)."""
    S_aggregate = np.asarray(S_aggregate, dtype=float)
    if V_micro is None or np.size(V_micro) == 0:
        return _sym(S_aggregate)
    return _sym(scipy.linalg.block_diag(S_aggregate, n * np.asarray(V_micro, dtype=float)))


def initial_weighting(Z: np.ndarray, V_micro: np.ndarray | None) -> np.ndarray:
    """initial_weighting(Z, V_micro) -> W1 = blockdiag((Z'Z/n)^{-1}, (n V_micro(theta_0))^{-1}).

    pyblp's initial weighting with micro moments; with V_micro None only the aggregate block.
    n is the number of rows of Z (products).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    W_aa = weight_from_covariance(Z.T @ Z / n)
    if V_micro is None or np.size(V_micro) == 0:
        return W_aa
    return _sym(scipy.linalg.block_diag(W_aa, weight_from_covariance(n * np.asarray(V_micro, dtype=float))))


def cue_objective(g: np.ndarray, G: np.ndarray | None, V: np.ndarray, dV: np.ndarray | None, n: int,
                  S_fixed: np.ndarray | None = None) -> tuple[float, np.ndarray | None, np.ndarray]:
    """CUE objective and exact gradient (LEARNINGS 1): ``q = n g' S(theta)^{-1} g``.

    ``S(theta) = V'V / n + S_fixed`` with V the (C, k) per-cluster moment contributions
    (``sum_{j in c} z_j xi_j`` in the aggregate columns, zero elsewhere) and ``S_fixed`` the part
    of S held constant (``blockdiag(0, n V_micro)`` at the CUE start point, DESIGN 4.5).
    With ``dV = dV/dtheta`` of shape (C, k, P):

    ``dq/dtheta_p = 2 n g' S^{-1} G_p - n (S^{-1} g)' dS_p (S^{-1} g)``,
    ``dS_p = (dV_p' V + V' dV_p) / n``.

    Returns ``(q, grad or None, S)``.
    """
    g = np.asarray(g, dtype=float)
    V = np.asarray(V, dtype=float)
    S = V.T @ V / n
    if S_fixed is not None:
        S = S + np.asarray(S_fixed, dtype=float)
    S = _sym(S)
    c, low = scipy.linalg.cho_factor(S)
    a = scipy.linalg.cho_solve((c, low), g)  # S^{-1} g
    q = float(n * g @ a)
    if G is None or dV is None:
        return q, None, S
    Va = V @ a  # (C,)
    # a' dS_p a = 2 (V a)' (dV_p a) / n
    dVa = np.einsum("ckp,k->cp", dV, a)
    grad = 2.0 * n * (np.asarray(G).T @ a) - 2.0 * (Va @ dVa)
    return q, grad, S


class Fixed:
    """Fixed(W): W given by the spec."""

    mode = "fixed"

    def __init__(self, W: np.ndarray) -> None:
        """Fixed(W)."""
        self.W = _sym(np.asarray(W, dtype=float))

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "k": int(self.W.shape[0])}


class TwoStep:
    """TwoStep(W1): run with W1, then W2 = S(theta_hat_1)^{-1} and restart at theta_hat_1.

    Step 2 starts only if the step-1 report has no SCALE_ARTIFACT or SD_XI_BLOWUP; a blocking degeneracy the caller
    names in ``gate(..., warn_only=...)`` is a warning instead (SD_XI_BLOWUP under a scale-free aggregate block).
    """

    mode = "two_step"
    blocking_degeneracies = ("SCALE_ARTIFACT", "SD_XI_BLOWUP")

    def __init__(self, W1: np.ndarray) -> None:
        """TwoStep(W1)."""
        self.W1 = _sym(np.asarray(W1, dtype=float))
        self.W2: np.ndarray | None = None

    def second_step_weight(self, problem: Any, ev: Any) -> np.ndarray:
        """W2 = S(theta_hat_1)^{-1} with S from ``problem.covariance(ev)``. Stores and returns it.

        The caller then calls ``problem.set_weighting(W2)`` (re-concentrates beta) and re-evaluates.
        """
        self.W2 = weight_from_covariance(problem.covariance(ev))
        return self.W2

    def gate(self, detected: Any, warn_only: Any = ()) -> tuple[bool, str]:
        """gate(detected_degeneracy_names, warn_only=()) -> (may_start_step2, reason).

        ``detected`` is an iterable of names of degeneracies detected in the step-1 report. A blocking degeneracy
        also in ``warn_only`` does not stop step 2: the result is (True, warning text). Otherwise the reason is ""
        when step 2 may start.
        """
        found = set(detected) & set(self.blocking_degeneracies)
        warned = sorted(found & set(warn_only))
        blocking = sorted(found - set(warn_only))
        if blocking:
            return False, (f"Step 2 not started: the step-1 point shows {', '.join(blocking)}, and W2 would "
                           f"inherit the degenerate point (DESIGN 4.5).")
        if warned:
            return True, (f"WARNING: the step-1 point shows {', '.join(warned)}; step 2 runs because the aggregate "
                          f"block is scale-free (Sargan form), which bounds the concentrated-xi direction (LEARNINGS "
                          f"19), so W2 does not inherit the fixed-W scale artefact (DESIGN 4.5).")
        return True, ""

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "k": int(self.W1.shape[0]), "W2_set": self.W2 is not None}


class CUE:
    """CUE: q = n g' S(theta)^{-1} g with the micro block of S held at its value at the CUE start point."""

    mode = "cue"

    def __init__(self, hold_micro_block: bool = True) -> None:
        """CUE(hold_micro_block=True)."""
        if not hold_micro_block:
            raise NotImplementedError("CUE with a moving micro covariance block is not implemented (DESIGN 8.9)")
        self.hold_micro_block = hold_micro_block
        self.S_fixed: np.ndarray | None = None

    def start(self, problem: Any, ev: Any) -> None:
        """Hold the micro block of S at its value at the start evaluation ``ev``."""
        self.S_fixed = problem.cue_fixed_covariance(ev)

    def objective(self, problem: Any, ev: Any) -> tuple[float, np.ndarray | None, np.ndarray]:
        """(q, grad, S) at ``ev`` from ``problem.cue_terms(ev)``; see ``cue_objective``."""
        if self.S_fixed is None:
            self.start(problem, ev)
        V, dV = problem.cue_terms(ev)
        return cue_objective(ev.g, ev.G, V, dV if ev.G is not None else None, problem.n, self.S_fixed)

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "hold_micro_block": self.hold_micro_block,
                "note": "micro block of S held at its value at the CUE start point"}


# ------------------------------------------------------------------------------------------------
# W adjustments of the pipeline's stages (declared by a spec, DESIGN 6.0)
# ------------------------------------------------------------------------------------------------


def bind_calibration(problem: Any, W: np.ndarray) -> np.ndarray:
    """W with every calibration row of ``problem`` (e.g. a margin row) set to its binding entry and decoupled from the
    rest; a copy of W when the problem has none."""
    W = np.array(W, dtype=float)
    for m in getattr(problem, "calibration", ()):
        rows = problem.blocks[m.name]
        idx = np.arange(rows.start, rows.stop)
        W[idx, :] = 0.0
        W[:, idx] = 0.0
        W[idx, idx] = m.binding_block()
    return W


def upweight_rows(problem: Any, W: np.ndarray, rows: Any, factor: float,
                  zero_calibration: bool = False) -> np.ndarray:
    """W with the rows ``rows`` = {moment block: (lo, hi)} (positions within the block, Python slice bounds) multiplied
    by ``factor`` in their diagonal block, so those moments are solved with the rest of q only breaking ties (a block
    the problem lacks is skipped); with ``zero_calibration`` the calibration rows are set to zero, so the result does
    not depend on a calibration target."""
    W = np.array(W, dtype=float)
    for block, (lo, hi) in dict(rows).items():
        sl = problem.blocks.get(block)
        if sl is None:
            continue
        idx = np.arange(sl.start, sl.stop)[lo:hi]
        W[np.ix_(idx, idx)] *= factor
    if zero_calibration:
        for m in getattr(problem, "calibration", ()):
            r = problem.blocks[m.name]
            W[r, :] = 0.0
            W[:, r] = 0.0
    return W


def efficient_on_excluded(problem: Any, ev: Any) -> np.ndarray:
    """W2 at an evaluation: every block other than the aggregate keeps its fixed (n V)^-1, and the aggregate block is
    efficient on the excluded instruments only.

    When beta is concentrated by OLS the X1 rows of Z'xi are identically zero and the clustered S of the full aggregate
    block is singular (a cluster with one product carries a row that is exactly zero). In the basis Z~ = Z A =
    [X1, M_X1 Z_ex], A = [[I, -G], [0, I]] with G = (X1'X1)^-1 X1'Z_ex, W_aa = A blockdiag((X1'X1/n)^-1, S_h^-1) A' with
    S_h the clustered covariance of the residualised excluded rows (A' S A restricted to them). That W keeps beta at
    OLS (the excluded rows of Z~'(delta - X1 b) do not depend on b), so the aggregate block is n h' S_h^-1 h with
    h = Z~_ex' xi / n: the aggregate J on the excluded instruments' degrees of freedom. Needs Z = [X1, Z_ex]."""
    S = problem.covariance(ev)
    W = np.zeros_like(S)
    agg = getattr(problem, "agg", None)
    for name, sl in ev.blocks.items():
        if agg is not None and name == agg.name:
            continue
        W[sl, sl] = weight_from_covariance(S[sl, sl])
    if agg is not None:
        a = problem.agg_rows
        Z = np.asarray(agg.Z, dtype=float)
        X1 = np.asarray(problem.linear.X1, dtype=float)
        K = X1.shape[1]
        if not np.array_equal(Z[:, :K], X1):
            raise ValueError("efficient_on_excluded: the aggregate Z does not start with X1")
        n = Z.shape[0]
        G = np.linalg.lstsq(X1, Z[:, K:], rcond=None)[0]
        A = np.eye(Z.shape[1])
        A[:K, K:] = -G
        St = A.T @ S[a, a] @ A
        B = np.zeros_like(St)
        B[:K, :K] = np.linalg.inv(X1.T @ X1 / n)
        B[K:, K:] = weight_from_covariance(St[K:, K:])
        W[a, a] = A @ B @ A.T
    return 0.5 * (W + W.T)


#: named W2 rules a spec may declare as ``step2_weighting`` (the pipeline's default is S^-1 block diagonal)
STEP2_RULES = {"efficient_on_excluded": efficient_on_excluded}

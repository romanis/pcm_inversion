"""Collinearity of X1, Z, Z'X1 and the columns of G (DESIGN 5.3). Owned by WP6."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..types import DiagnosticBlock

__all__ = ["COLLINEARITY_DEFAULTS", "collinearity_block", "matrix_rank_condition"]

COLLINEARITY_DEFAULTS: dict[str, float] = {
    "rank_tol": 1e-10,  # singular values below rank_tol * s_1 count as zero
    "cond_max": 1e8,  # condition number above this: flag
    "pair_corr": 0.99,  # |correlation| of A_jac columns above this: listed
    "span_tol": 1e-8,  # 1 - canonical correlation below this: the X1 column is in span(Z)
}


def _merge(thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(COLLINEARITY_DEFAULTS)
    if thresholds:
        out.update({k: v for k, v in thresholds.items() if k in COLLINEARITY_DEFAULTS})
    return out


def _equilibrate(X: np.ndarray) -> np.ndarray:
    """Scale every column to unit Euclidean norm. Columns are not centred, so a constant column stays."""
    norms = np.linalg.norm(X, axis=0)
    return X / np.where(norms > 0, norms, 1.0)[None, :]


def matrix_rank_condition(X: np.ndarray, rank_tol: float = 1e-10) -> dict[str, Any]:
    """Rank and condition number of X after scaling every column to unit norm."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return {"shape": list(np.shape(X)), "rank": 0, "condition": None, "full_rank": True}
    s = np.linalg.svd(_equilibrate(X), compute_uv=False)
    rank = int(np.sum(s > rank_tol * s[0])) if s.size and s[0] > 0 else 0
    full = rank == X.shape[1]
    cond = float(s[0] / s[-1]) if full and s[-1] > 0 else float("inf")
    return {"shape": [int(X.shape[0]), int(X.shape[1])], "rank": rank, "condition": cond, "full_rank": full}


def _orth(X: np.ndarray, rank_tol: float) -> np.ndarray:
    U, s, _ = np.linalg.svd(_equilibrate(X), full_matrices=False)
    if not s.size or s[0] == 0:
        return U[:, :0]
    return U[:, s > rank_tol * s[0]]


def collinearity_block(X1: np.ndarray | None, Z: np.ndarray | None, A_jac: np.ndarray | None, names: Sequence[str],
                       thresholds: Mapping[str, Any] | None = None,
                       x1_names: Sequence[str] | None = None) -> DiagnosticBlock:
    """collinearity_block(X1, Z, A_jac, names, thresholds=None, x1_names=None) -> DiagnosticBlock.

    Rank and condition of standardised X1 and Z (rank tolerance 1e-10 of s_1, flag cond > 1e8);
    rank of Z'X1 must equal the number of linear parameters, with the smallest canonical correlation
    (fail on deficiency: the F1 check); pairs of ``A_jac`` columns with |correlation| above 0.99.

    Columns are standardised by scaling to unit norm without centring, so the constant in X1 is kept.
    Canonical correlations are uncentred (cosines of the principal angles between span(Z) and each
    direction of span(X1)). X1 columns outside span(Z) are listed (LEARNINGS 16: exogenous X1 should
    instrument itself). The correlation of two Jacobian columns is their uncentred cosine.
    """
    th = _merge(thresholds)
    names = list(names)
    block = DiagnosticBlock(name="collinearity", thresholds=dict(th))
    fails: list[str] = []
    flags: list[str] = []
    info: list[str] = []

    have_z = Z is not None and np.ndim(Z) == 2 and np.shape(Z)[1] > 0
    if not have_z:
        block.values.update(Z={}, X1={}, ZX1={})
        info.append("Instrument matrix Z not supplied: X1, Z and Z'X1 checks not run.")
    else:
        Z = np.asarray(Z, dtype=float)
        z_info = matrix_rank_condition(Z, th["rank_tol"])
        block.values["Z"] = z_info
        if not z_info["full_rank"]:
            flags.append(f"FLAG: Z has rank {z_info['rank']} with {Z.shape[1]} columns: some instruments are "
                         f"redundant and S is singular in those directions.")
        elif z_info["condition"] > th["cond_max"]:
            flags.append(f"FLAG: Z is ill conditioned (condition {z_info['condition']:.2e} > {th['cond_max']:.0e}).")

        if X1 is not None and np.asarray(X1).size and np.asarray(X1).shape[1] > 0:
            X1 = np.asarray(X1, dtype=float)
            K = X1.shape[1]
            x1n = list(x1_names) if x1_names is not None else [f"x1_{i}" for i in range(K)]
            x_info = matrix_rank_condition(X1, th["rank_tol"])
            block.values["X1"] = x_info
            if not x_info["full_rank"]:
                fails.append(f"FAIL: X1 has rank {x_info['rank']} with {K} columns: the linear coefficients are not "
                             f"identified (collinear characteristics).")
            elif x_info["condition"] > th["cond_max"]:
                flags.append(f"FLAG: X1 is ill conditioned (condition {x_info['condition']:.2e} > "
                             f"{th['cond_max']:.0e}).")
            zx = _equilibrate(Z).T @ _equilibrate(X1)
            r_zx = matrix_rank_condition(zx, th["rank_tol"])["rank"] if zx.size else 0
            Qz, Qx = _orth(Z, th["rank_tol"]), _orth(X1, th["rank_tol"])
            cc = np.linalg.svd(Qz.T @ Qx, compute_uv=False) if Qz.shape[1] and Qx.shape[1] else np.zeros(0)
            cc_full = np.concatenate([cc, np.zeros(max(0, Qx.shape[1] - cc.size))])
            # per-column distance of X1 from span(Z)
            Xe = _equilibrate(X1)
            resid = np.linalg.norm(Xe - Qz @ (Qz.T @ Xe), axis=0) if Qz.shape[1] else np.ones(K)
            outside = [x1n[i] for i in range(K) if resid[i] > np.sqrt(2 * th["span_tol"])]
            block.values["ZX1"] = {"rank": int(r_zx), "required": K,
                                   "min_canonical_correlation": float(cc_full.min()) if cc_full.size else 0.0,
                                   "canonical_correlations": cc_full.tolist(),
                                   "x1_outside_span_Z": outside, "x1_in_span_Z": not outside}
            if r_zx < K:
                fails.append(f"FAIL: rank(Z'X1) = {r_zx} < {K} linear parameters: the instruments cannot identify "
                             f"the linear coefficients (DESIGN F1).")
            else:
                info.append(f"Z'X1 has full rank {K}; smallest canonical correlation {float(cc_full.min()):.3f}.")
            if outside:
                flags.append(f"FLAG: X1_NOT_IN_Z: {len(outside)} of {K} X1 columns are not in the span of Z "
                             f"({', '.join(outside)}). Exogenous X1 is not instrumenting itself, so concentrating "
                             f"beta uses up {K} of the {Z.shape[1]} moments (LEARNINGS 16).")
        else:
            block.values["X1"] = {}
            block.values["ZX1"] = {}
            info.append("No concentrated X1 (linear part absorbed or empty).")

    pairs: list[dict[str, Any]] = []
    if A_jac is not None and np.asarray(A_jac).size:
        A = _equilibrate(np.asarray(A_jac, dtype=float))
        C = A.T @ A
        P = C.shape[0]
        for i in range(P):
            for j in range(i + 1, P):
                if abs(C[i, j]) > th["pair_corr"]:
                    pairs.append({"a": names[i], "b": names[j], "correlation": float(C[i, j])})
        for p in pairs:
            flags.append(f"FLAG: Jacobian columns {p['a']} and {p['b']} are nearly collinear "
                         f"(correlation {p['correlation']:+.4f}).")
    block.values["G_pairs"] = pairs

    block.messages = fails + flags + info
    block.status = "fail" if fails else ("flag" if flags else "ok")
    if block.status == "ok" and have_z:
        block.messages.insert(0, "Collinearity: X1, Z, Z'X1 and the Jacobian columns are well conditioned.")
    return block

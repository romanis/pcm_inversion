"""Second-choice moments: correlations between the characteristics of first and second choices (Grieco, Murry &
Yurukoglu 2024, Table 6; their getSecondChoiceStructure.m and ComputeSecondChoiceMoments.m). Model-agnostic: a model
supplies the matrix of second-choice probabilities given the first choice, conditional on an inside second choice.

For one market with J products, first-choice shares s (J,), second-choice matrix C (J, J) with C[j, k] = P(second
= k | first = j, second inside) and characteristics x (J, K), the joint law of (first, second) is
pi[j, k] = (s_j / sum s) C[j, k], renormalised to sum to 1 (a first choice with no respondents has a zero row in
survey data). For each characteristic c the moment is corr_pi(x_first,c, x_second,c).
"""

from __future__ import annotations

import numpy as np

__all__ = ["first_second_correlations", "logit_second_choice_matrix", "correlation_adjoint"]


def first_second_correlations(first_shares: np.ndarray, second_choice: np.ndarray, x: np.ndarray,
                              full: bool = False) -> np.ndarray:
    """first_second_correlations(first_shares, second_choice, x, full=False) -> (K,) correlations of characteristic
    c of the first and of the second choice, or the (K, K) matrix corr(x_first,a, x_second,b) with ``full``."""
    s = np.asarray(first_shares, dtype=float).ravel()
    C = np.asarray(second_choice, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    J = s.size
    if C.shape != (J, J) or x.shape[0] != J:
        raise ValueError(f"shapes: shares {s.shape}, second choice {C.shape}, x {x.shape}")
    pi = (s / s.sum())[:, None] * C
    tot = pi.sum()
    if not tot > 0:
        raise ValueError("the joint first/second-choice law has no mass")
    pi = pi / tot
    p1, p2 = pi.sum(axis=1), pi.sum(axis=0)          # marginals of the first and the second choice
    m1, m2 = p1 @ x, p2 @ x
    d1, d2 = x - m1, x - m2
    sd1 = np.sqrt(p1 @ d1 ** 2)
    sd2 = np.sqrt(p2 @ d2 ** 2)
    cov = d1.T @ pi @ d2                              # (K, K): cov(x_first,a, x_second,b)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd1, sd2)
    return corr if full else np.diag(corr).copy()


def logit_second_choice_matrix(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """logit_second_choice_matrix(probabilities, weights) -> (J, J) C[j, k] = P(second = k | first = j, second inside)
    for a mixed logit with (J, H) purchase probabilities P and household weights w (H,), as ComputeSecondChoiceMoments.m:
    the buyers of j are household h with weight w_h P_jh / sum_h w_h P_jh, and household h's inside second choice with
    j removed is k with probability P_kh / (sum_l P_lh - P_jh). The diagonal is zero."""
    P = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    J, H = P.shape
    if w.size != H:
        raise ValueError(f"{w.size} weights for {H} households")
    buyers = w[None, :] * P
    buyers = buyers / buyers.sum(axis=1, keepdims=True)          # (J, H): household mix of j's buyers
    rest = P.sum(axis=0)[None, :] - P                             # (J, H): inside mass with j removed
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.where(rest > 0, buyers / rest, 0.0)                # (J, H)
    C = B @ P.T                                                   # C[j, k] = sum_h B[j, h] P[k, h]
    C[np.arange(J), np.arange(J)] -= np.einsum("jh,jh->j", B, P)  # remove k = j
    return C


def correlation_adjoint(first_shares: np.ndarray, second_choice: np.ndarray, x: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """correlation_adjoint(first_shares, second_choice, x) -> (corr (K,), A (K, J, J)) with A[c, j, k] the derivative of
    corr_c (``first_second_correlations``) in C[j, k], first shares held fixed.

    With r = (s / S) C, pi = r / sum r, d1 = x - p1 x, d2 = x - p2 x (p1, p2 the marginals of pi):
    ``d corr_c / d pi_jk = [d1_j d2_k - cov / (2 var1) d1_j^2 - cov / (2 var2) d2_k^2] / sqrt(var1 var2)`` (the means'
    derivatives drop out), the normalisation adds nothing because ``sum_jk pi_jk d corr / d pi_jk = 0``, so
    ``A[c, j, k] = (s_j / (S tot)) d corr_c / d pi_jk`` (GMY spec WP-G4)."""
    s = np.asarray(first_shares, dtype=float).ravel()
    C = np.asarray(second_choice, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    J = s.size
    if C.shape != (J, J) or x.shape[0] != J:
        raise ValueError(f"shapes: shares {s.shape}, second choice {C.shape}, x {x.shape}")
    a = s / s.sum()
    r = a[:, None] * C
    tot = r.sum()
    if not tot > 0:
        raise ValueError("the joint first/second-choice law has no mass")
    pi = r / tot
    p1, p2 = pi.sum(axis=1), pi.sum(axis=0)
    d1, d2 = x - p1 @ x, x - p2 @ x                       # (J, K)
    var1, var2 = p1 @ d1 ** 2, p2 @ d2 ** 2               # (K,)
    cov = np.einsum("jc,jk,kc->c", d1, pi, d2)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(var1 * var2)
        corr = cov / root
        dpi = (d1.T[:, :, None] * d2.T[:, None, :]
               - (cov / (2 * var1))[:, None, None] * (d1.T ** 2)[:, :, None]
               - (cov / (2 * var2))[:, None, None] * (d2.T ** 2)[:, None, :]) / root[:, None, None]
    A = dpi * (a / tot)[None, :, None]
    return corr, A

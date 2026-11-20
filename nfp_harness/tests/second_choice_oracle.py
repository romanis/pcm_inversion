"""Brute-force second choices of one PCM draw, for tests only (GMY spec WP-G4 item 2).

A consumer with price sensitivity a buys argmax_k (d_k - a p_k) when that is positive, else nothing; with the first
choice removed the second choice is the argmax over the rest (positive, else none). Nothing here uses the frontier
or the hull: a log-spaced grid of a locates every change of the (first, second) pair, bisection refines each change to
machine precision, and each segment's mass is the truncated lognormal cdf difference over it. So the oracle is
independent of the kernel's window algebra.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def truncated_cdf(a: np.ndarray, sigma: float, a_lo: float) -> np.ndarray:
    """cdf of a ~ LogNormal(0, sigma) truncated below at a_lo (a_lo <= 0: not truncated)."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore"):
        F = norm.cdf(np.log(np.maximum(a, 1e-300)) / sigma)
    if a_lo > 0:
        lo = norm.cdf(np.log(a_lo) / sigma)
        F = np.where(a > a_lo, (F - lo) / (1.0 - lo), 0.0)
    return F


def _pair(d: np.ndarray, p: np.ndarray, a: float) -> tuple[int, int]:
    u = d - a * p
    j = int(np.argmax(u))
    if not u[j] > 0:
        return -1, -1
    u2 = u.copy()
    u2[j] = -np.inf
    k = int(np.argmax(u2))
    return j, (k if u2[k] > 0 else -1)


def second_choices(d: np.ndarray, p: np.ndarray, sigma: float, a_lo: float = 0.0, n_grid: int = 200_000,
                   span: tuple[float, float] = (1e-6, 1e6)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(s (J,), q (J, J), m (J, J)) for one draw: first-choice shares, q[j, k] = P(second = k | first = j, second
    inside) (row zero when j's buyers have no inside second choice) and the joint masses m[j, k]."""
    d, p = np.asarray(d, dtype=float), np.asarray(p, dtype=float)
    J = d.size
    grid = np.geomspace(span[0], span[1], n_grid)
    U = d[None, :] - grid[:, None] * p[None, :]
    first = np.argmax(U, axis=1)
    top = U[np.arange(n_grid), first]
    U[np.arange(n_grid), first] = -np.inf
    second = np.argmax(U, axis=1)
    sec_u = U[np.arange(n_grid), second]
    f_code = np.where(top > 0, first, -1)
    s_code = np.where((top > 0) & (sec_u > 0), second, -1)
    pairs = list(zip(f_code.tolist(), s_code.tolist()))
    cuts = [span[0]]
    for i in range(n_grid - 1):
        lo_cell, end = grid[i], pairs[i + 1]
        cur = pairs[i]
        while cur != end:   # every change inside the cell: the pair sequence never repeats as a rises
            lo, hi = lo_cell, grid[i + 1]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if mid in (lo, hi):
                    break
                if _pair(d, p, mid) == cur:
                    lo = mid
                else:
                    hi = mid
            cuts.append(hi)
            lo_cell, cur = hi, _pair(d, p, hi)
    cuts.append(span[1])
    F = truncated_cdf(np.array(cuts), sigma, a_lo)
    s = np.zeros(J)
    m = np.zeros((J, J))
    for i in range(len(cuts) - 1):
        j, k = _pair(d, p, float(np.sqrt(cuts[i] * cuts[i + 1])))
        mass = F[i + 1] - F[i]
        if j >= 0:
            s[j] += mass
            if k >= 0:
                m[j, k] += mass
    M = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(M[:, None] > 0, m / M[:, None], 0.0)
    return s, q, m


def market_second_choices(dc: np.ndarray, p: np.ndarray, w: np.ndarray, sigma: float, a_lo: np.ndarray | float,
                          **kw: object) -> tuple[np.ndarray, np.ndarray]:
    """(s (J,), C (J, J)) of a market of draws with conditional qualities ``dc`` (R, J) and weights w (R,):
    N[j, k] = sum_i w_i s_ij q_ijk, s_j = sum_i w_i s_ij, C = N / s (rows of products nobody buys zero)."""
    R, J = dc.shape
    a_lo = np.broadcast_to(np.asarray(a_lo, dtype=float), (R,))
    s = np.zeros(J)
    N = np.zeros((J, J))
    for i in range(R):
        si, qi, _ = second_choices(dc[i], p, sigma, float(a_lo[i]), **kw)   # type: ignore[arg-type]
        s += w[i] * si
        N += w[i] * si[:, None] * qi
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(s[:, None] > 0, N / s[:, None], 0.0)
    return s, C

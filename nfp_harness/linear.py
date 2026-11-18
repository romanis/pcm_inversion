"""Concentrated linear parameters: IV and absorbed fixed effects (DESIGN 2.4). Owned by WP4."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import LinearFit

__all__ = ["LinearIV", "Absorb", "IdentificationError", "matrix_rank"]


class IdentificationError(ValueError):
    """The moment conditions cannot identify the unknowns (rank or count check failed; LEARNINGS 16)."""


def matrix_rank(A: np.ndarray, rtol: float = 1e-10) -> int:
    """Rank of A after scaling every column to unit norm; singular values below ``rtol * s_1`` count as zero."""
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0
    norms = np.linalg.norm(A, axis=0)
    A = A[:, norms > 0] / norms[norms > 0]
    if A.shape[1] == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


class LinearIV:
    """Concentrates beta out with the aggregate block of W.

    ``beta = (X1' Z W_aa Z' X1)^{-1} X1' Z W_aa Z' delta``, ``M = I - X1 (...)``, ``xi = M delta``.
    Exact when W is block diagonal between aggregate and micro moments and micro moments do not
    depend on beta. Z must contain the exogenous X1 columns: ``Z = [X1, excluded]`` (F1, LEARNINGS 16).
    For inference the aggregate rows of the full Jacobian over (beta, theta) are
    ``[-Z' X1 / n, Z' D / n]``.

    Construction refuses (``IdentificationError``) when ``rank(Z' X1) < K``: beta is then not
    identified by the aggregate moments.
    """

    def __init__(self, X1: np.ndarray, Z: np.ndarray, W_aa: np.ndarray | None = None,
                 names: Sequence[str] | None = None) -> None:
        """LinearIV(X1, Z, W_aa=None, names=None). W_aa defaults to (Z'Z/n)^{-1}."""
        X1 = np.asarray(X1, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if X1.ndim != 2 or Z.ndim != 2 or X1.shape[0] != Z.shape[0]:
            raise ValueError(f"X1 {X1.shape} and Z {Z.shape} must be 2-d with the same number of rows")
        self.X1, self.Z = X1, Z
        self.n, self.K = X1.shape
        self.names = tuple(names) if names is not None else tuple(f"beta{k}" for k in range(self.K))
        if len(self.names) != self.K:
            raise ValueError(f"{len(self.names)} names for {self.K} linear parameters")
        self.n_linear = self.K
        self.rank_Z = matrix_rank(Z)
        if self.rank_Z < Z.shape[1]:
            raise IdentificationError(
                f"instrument matrix Z has rank {self.rank_Z} < {Z.shape[1]} columns: drop collinear instruments")
        self.rank_ZX1 = matrix_rank(Z.T @ X1)
        if self.rank_ZX1 < self.K:
            raise IdentificationError(
                f"rank(Z'X1) = {self.rank_ZX1} < {self.K} linear parameters: the aggregate moments do not "
                f"identify beta. Z must contain the exogenous X1 columns, Z = [X1, excluded] (LEARNINGS 16)")
        self.ZtX1 = Z.T @ X1
        self.W_aa: np.ndarray
        self._P: np.ndarray  # (K, n) beta = P delta
        self._M: np.ndarray | None = None
        self.set_weighting(W_aa if W_aa is not None else np.linalg.inv(Z.T @ Z / self.n))

    def set_weighting(self, W_aa: np.ndarray) -> None:
        """set_weighting(W_aa): recompute beta's projection and M for a new aggregate block of W."""
        W_aa = _sym(np.asarray(W_aa, dtype=float))
        if W_aa.shape != (self.Z.shape[1],) * 2:
            raise ValueError(f"W_aa has shape {W_aa.shape}, expected {(self.Z.shape[1],) * 2}")
        B = self.ZtX1.T @ W_aa  # (K, kz) = X1' Z W
        A = _sym(B @ self.ZtX1)  # X1' Z W Z' X1
        self.W_aa = W_aa
        self._P = np.linalg.solve(A, B @ self.Z.T)
        self._M = None

    @property
    def M(self) -> np.ndarray:
        """(n, n) annihilator, xi = M delta. Formed on first use and cached until the weighting changes."""
        if self._M is None:
            self._M = np.eye(self.n) - self.X1 @ self._P
        return self._M

    def projection(self) -> np.ndarray:
        """(K, n) matrix P with beta = P delta."""
        return self._P

    def fit(self, delta: np.ndarray) -> LinearFit:
        """fit(delta) -> LinearFit(names, beta, xi, M)."""
        delta = np.asarray(delta, dtype=float)
        beta = self._P @ delta
        xi = delta - self.X1 @ beta
        return LinearFit(names=self.names, beta=beta, xi=xi, M=self.M)

    def full_jacobian_linear_block(self) -> np.ndarray:
        """(kz, K) aggregate-row derivative of g = Z'(delta - X1 beta)/n in beta: ``-Z' X1 / n`` (DESIGN 2.4)."""
        return -self.ZtX1 / self.n


class Absorb:
    """Replaces X1 with within-group demeaning for fixed effects (Nevo brand dummies).

    xi = delta minus its group mean, so ``M = I - G (G'G)^{-1} G'`` with G the group indicator
    matrix. The absorbed effects are not counted as concentrated linear parameters
    (``n_linear = 0``): the instruments must be within-transformed by the spec, which removes the
    same directions from the moment conditions.
    """

    def __init__(self, groups: np.ndarray) -> None:
        """Absorb(groups): groups is the (n,) fixed-effect id of each product."""
        groups = np.asarray(groups).ravel()
        self.groups = groups
        self.n = groups.shape[0]
        _, self._codes = np.unique(groups, return_inverse=True)
        self._counts = np.bincount(self._codes).astype(float)
        self.names: tuple[str, ...] = ()
        self.K = 0
        self.n_linear = 0
        self._M: np.ndarray | None = None

    def set_weighting(self, W_aa: np.ndarray) -> None:
        """set_weighting(W_aa): no-op for pure absorption; kept for the LinearIV interface."""

    def demean(self, a: np.ndarray) -> np.ndarray:
        """Within-group demeaning of the rows of a (n,) or (n, m) array."""
        a = np.asarray(a, dtype=float)
        flat = a.reshape(self.n, -1)
        means = np.zeros((self._counts.size, flat.shape[1]))
        np.add.at(means, self._codes, flat)
        means /= self._counts[:, None]
        return (flat - means[self._codes]).reshape(a.shape)

    @property
    def M(self) -> np.ndarray:
        if self._M is None:
            self._M = self.demean(np.eye(self.n))
        return self._M

    def fit(self, delta: np.ndarray) -> LinearFit:
        """fit(delta) -> LinearFit with xi = within-group demeaned delta and empty beta."""
        return LinearFit(names=(), beta=np.zeros(0), xi=self.demean(delta), M=self.M)

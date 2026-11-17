"""Moment objects: aggregate IV moments and pooled micro moments (DESIGN 2.3). Owned by WP4.

A moment returns its value and partial derivatives in delta and theta (``MomentBlock``); the
framework applies the implicit-function chain rule ``G_b = d_theta_b + sum_t d_delta_b[:, rows_t] @ D_t``.
Moments never see D.

Kernel output contract for micro parts (what an ``InnerModel`` puts in ``InnerSolution.outputs``):

``sol.outputs[MICRO_OUTPUTS][part.name]`` is a list with one dict per market, in the order of
``model.markets``, with keys

- ``"sum"``: ``sum_i w_i h(i) sum_j u_j s_ij`` with ``u_j = v_j - v0`` (float);
- ``"d_delta"``: (J_t,) partial derivative of ``"sum"`` in delta_t, theta fixed (None without Jacobian);
- ``"d_theta"``: (P,) partial derivative of ``"sum"`` in theta, delta fixed (None without Jacobian);
- ``"h_sum"``: ``H_t = sum_i w_i h(i)`` (float);
- ``"w_sum"``: ``sum_i w_i`` (float).

``h(i)`` is the product of ``part.agent_columns`` and ``v_j`` the product of ``part.product_columns``
(see ``types.MicroPartRequest``). Markets outside a dataset may be omitted by passing None.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import (
    InnerSolution,
    KernelRequests,
    LinearFit,
    MarketIndex,
    MicroPartRequest,
    Moment,
    MomentBlock,
)
from .weighting import cluster_sums, clustered_covariance

__all__ = ["Moment", "IVMoments", "MicroDataset", "MicroPart", "RatioMicroMoment", "MICRO_OUTPUTS",
           "product_part"]

#: A micro part, a pair (agent weight h, product weight v with outside value v0); see ``types.MicroPartRequest``.
MicroPart = MicroPartRequest

#: Key of ``InnerSolution.outputs`` holding the per-market micro part sums (contract above).
MICRO_OUTPUTS = "micro_parts"


def product_part(p1: MicroPart, p2: MicroPart) -> MicroPart:
    """The part ``E[v1 v2]``: h = h1 h2, v = v1 v2 elementwise, v0 = v1_0 v2_0 (DESIGN 2.3)."""
    return MicroPart(
        name=f"{p1.name}*{p2.name}",
        agent_columns=tuple(p1.agent_columns) + tuple(p2.agent_columns),
        product_columns=tuple(p1.product_columns) + tuple(p2.product_columns),
        outside_value=p1.outside_value * p2.outside_value,
    )


class IVMoments:
    """Aggregate IV moments ``g = Z' xi / n`` with ``xi = M delta``.

    ``d_delta = Z' M / n``, ``d_theta = 0``. Covariance block: clustered
    ``S = (1/n) sum_c (sum_{j in c} z_j xi_j)(sum_{j in c} z_j xi_j)'``.
    """

    name: str
    is_aggregate = True
    is_micro = False

    def __init__(self, Z: np.ndarray, clustering_ids: np.ndarray | None = None, name: str = "aggregate") -> None:
        """IVMoments(Z, clustering_ids=None, name="aggregate"). No clustering means one cluster per product."""
        self.Z = np.asarray(Z, dtype=float)
        if self.Z.ndim != 2:
            raise ValueError(f"Z must be 2-d, got shape {self.Z.shape}")
        self.n, self.k = self.Z.shape
        self.clustering_ids = None if clustering_ids is None else np.asarray(clustering_ids).ravel()
        if self.clustering_ids is not None and self.clustering_ids.shape[0] != self.n:
            raise ValueError(f"{self.clustering_ids.shape[0]} clustering ids for {self.n} products")
        self.name = name
        self._M_ref: np.ndarray | None = None
        self._ZtM: np.ndarray | None = None

    def requests(self) -> KernelRequests:
        """requests() -> KernelRequests(): IV moments need no extra kernel outputs."""
        return KernelRequests()

    def _d_delta(self, linear: LinearFit) -> np.ndarray:
        if linear.M is None:
            raise ValueError("IVMoments needs LinearFit.M to form d_delta = Z'M/n")
        if self._M_ref is not linear.M:
            self._ZtM = self.Z.T @ linear.M / self.n
            self._M_ref = linear.M
        assert self._ZtM is not None
        return self._ZtM

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value, d_delta=Z'M/n, d_theta=0)."""
        value = self.Z.T @ linear.xi / self.n
        d_delta = self._d_delta(linear) if need_jacobian else None
        return MomentBlock(name=self.name, value=value, d_delta=d_delta, d_theta=None, data=None)

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> clustered S_aggregate, (k_b, k_b)."""
        return clustered_covariance(self.Z, linear.xi, self.clustering_ids)

    def cluster_terms(self, xi: np.ndarray, dxi: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-cluster contributions for CUE: ``V_c = sum_{j in c} z_j xi_j`` (C, k_b) and
        ``dV_c = sum_{j in c} z_j dxi_j`` (C, k_b, P) with dxi = d xi / d theta (n, P)."""
        V = cluster_sums(self.Z * xi[:, None], self.clustering_ids)
        if dxi is None:
            return V, None
        dV = cluster_sums(self.Z[:, :, None] * dxi[:, None, :], self.clustering_ids)
        return V, dV


class MicroDataset:
    """A micro dataset pooled across markets (pyblp 1.3.0 ``MicroDataset``).

    ``Den = sum_t sum_i w_it`` over the dataset's markets; part covariances are divided by
    ``observations`` (29,125 for Petrin's CEX). Every product and the outside good have dataset
    weight one, so Den does not depend on delta or theta.
    """

    def __init__(self, name: str, observations: int, market_ids: Sequence | None = None) -> None:
        """MicroDataset(name, observations, market_ids=None). None means all markets."""
        if not observations > 0:
            raise ValueError(f"micro dataset {name!r}: observations must be positive, got {observations}")
        self.name = name
        self.observations = int(observations)
        self.market_ids = None if market_ids is None else tuple(market_ids)

    def includes(self, market_id: object) -> bool:
        return self.market_ids is None or market_id in self.market_ids

    def __repr__(self) -> str:
        return f"MicroDataset({self.name!r}, observations={self.observations}, market_ids={self.market_ids})"


class RatioMicroMoment:
    """Ratio micro moments ``g_r = part(num_r) / part(den_r) - value_r``, one row per ratio.

    ``part = (1 / Den) sum_t [ v0 H_t + sum_i w_it h(i) sum_j (v_j - v0) s_ijt ]``. Partials are
    ``grad_f . d parts / d delta`` and ``grad_f . d parts / d theta``. The covariance block is
    ``grad_f Sigma_parts grad_f'`` with part covariances ``E[v1 v2] - E[v1] E[v2]`` divided by
    ``dataset.observations``; product parts are requested only when ``covariance_block`` is called
    (``covariance_requests``).

    ``covariance`` (k, k), when given, is a fixed V_micro used by ``covariance_block`` at every theta
    instead of the model-implied one (e.g. Petrin's CEX sample covariances, pyblp's
    ``PETRIN_COVARIANCES_LOCATION``), so W1's and S's micro blocks do not depend on theta and no product
    parts are requested.

    ``bind(markets, n_products)`` must be called before ``evaluate``; ``NFPGMMProblem`` does it.
    One ``RatioMicroMoment`` per dataset: covariances across moment objects are taken as zero.
    """

    name: str
    is_aggregate = False
    is_micro = True

    def __init__(self, name: str, dataset: MicroDataset, numerators: Sequence[MicroPart],
                 denominators: Sequence[MicroPart], values: np.ndarray,
                 labels: Sequence[str] | None = None, covariance: np.ndarray | None = None) -> None:
        """RatioMicroMoment(name, dataset, numerators, denominators, values, labels=None, covariance=None)."""
        values = np.asarray(values, dtype=float).ravel()
        if not len(numerators) == len(denominators) == values.size:
            raise ValueError(f"{name}: {len(numerators)} numerators, {len(denominators)} denominators and "
                             f"{values.size} values must agree")
        self.name = name
        self.dataset = dataset
        self.numerators = tuple(numerators)
        self.denominators = tuple(denominators)
        self.values = values
        self.k = values.size
        self.fixed_covariance: np.ndarray | None = None
        if covariance is not None:
            V = np.asarray(covariance, dtype=float)
            if V.shape != (self.k, self.k):
                raise ValueError(f"{name}: covariance has shape {V.shape}, expected {(self.k, self.k)}")
            if not np.all(np.isfinite(V)):
                raise ValueError(f"{name}: covariance has non-finite entries")
            if np.max(np.abs(V - V.T)) > 1e-8 * max(float(np.max(np.abs(V))), 1e-300):
                raise ValueError(f"{name}: covariance is not symmetric")
            self.fixed_covariance = 0.5 * (V + V.T)
        self.labels = tuple(labels) if labels is not None else tuple(
            f"{a.name}/{b.name}" for a, b in zip(self.numerators, self.denominators))
        parts: dict[str, MicroPart] = {}
        for p in self.numerators + self.denominators:
            known = parts.get(p.name)
            if known is not None and known != p:
                raise ValueError(f"{name}: part {p.name!r} has two definitions: {known} and {p}")
            parts[p.name] = p
        self.parts: tuple[MicroPart, ...] = tuple(parts.values())
        self._index = {p.name: i for i, p in enumerate(self.parts)}
        self._num = np.array([self._index[p.name] for p in self.numerators])
        self._den = np.array([self._index[p.name] for p in self.denominators])
        self._markets: list[MarketIndex] | None = None
        self._n: int | None = None
        self._P: int | None = None

    # ------------------------------------------------------------------ setup

    def bind(self, markets: Sequence[MarketIndex], n_products: int, P: int | None = None) -> None:
        """Attach the model's markets (their order is the order of the kernel output lists) and,
        optionally, the number of nonlinear parameters P (fixes the width of d_theta)."""
        self._markets = list(markets)
        self._n = int(n_products)
        self._P = None if P is None else int(P)
        self._in = [t for t, m in enumerate(self._markets) if self.dataset.includes(m.market_id)]
        if not self._in:
            raise ValueError(f"{self.name}: dataset {self.dataset.name!r} covers none of the model's markets")

    def requests(self) -> KernelRequests:
        """requests() -> KernelRequests with every numerator and denominator part."""
        return KernelRequests(micro_parts=self.parts)

    def covariance_parts(self) -> tuple[MicroPart, ...]:
        """The product parts ``E[v_a v_b]`` for every pair a <= b of this moment's parts."""
        return tuple(product_part(a, b) for i, a in enumerate(self.parts) for b in self.parts[i:])

    def covariance_requests(self) -> KernelRequests:
        """requests() plus the product parts needed by ``covariance_block`` (none with a fixed covariance)."""
        if self.fixed_covariance is not None:
            return self.requests()
        return KernelRequests(micro_parts=self.parts + self.covariance_parts())

    # ------------------------------------------------------------------ parts

    def _market_outputs(self, sol: InnerSolution, part: MicroPart) -> list[dict]:
        if self._markets is None:
            raise RuntimeError(f"{self.name}: bind(markets, n_products) was not called")
        table = sol.outputs.get(MICRO_OUTPUTS)
        if not isinstance(table, dict) or part.name not in table:
            raise KeyError(f"{self.name}: kernel outputs lack micro part {part.name!r} "
                           f"(sol.outputs[{MICRO_OUTPUTS!r}][{part.name!r}])")
        per_market = table[part.name]
        if len(per_market) != len(self._markets):
            raise ValueError(f"{self.name}: part {part.name!r} has {len(per_market)} market entries, "
                             f"model has {len(self._markets)} markets")
        return per_market

    def denominator(self, sol: InnerSolution) -> float:
        """Den = sum over the dataset's markets of sum_i w_it."""
        per_market = self._market_outputs(sol, self.parts[0])
        return float(sum(per_market[t]["w_sum"] for t in self._in))

    def part_values(self, sol: InnerSolution, parts: Sequence[MicroPart], need_jacobian: bool,
                    P: int | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Pooled part values (m,), and their partials in delta (m, n) and theta (m, P)."""
        assert self._markets is not None and self._n is not None
        if P is None:
            P = self._P
        m = len(parts)
        den = self.denominator(sol)
        value = np.zeros(m)
        d_delta = np.zeros((m, self._n)) if need_jacobian else None
        d_theta: np.ndarray | None = None
        for a, part in enumerate(parts):
            per_market = self._market_outputs(sol, part)
            for t in self._in:
                out = per_market[t]
                value[a] += part.outside_value * out["h_sum"] + out["sum"]
                if need_jacobian:
                    assert d_delta is not None
                    d_delta[a, self._markets[t].rows] += np.asarray(out["d_delta"], dtype=float)
                    dth = np.asarray(out["d_theta"], dtype=float)
                    if d_theta is None:
                        d_theta = np.zeros((m, dth.size if P is None else P))
                    d_theta[a] += dth
        value /= den
        if need_jacobian:
            assert d_delta is not None
            d_delta /= den
            if d_theta is None:
                d_theta = np.zeros((m, 0 if P is None else P))
            d_theta /= den
        return value, d_delta, d_theta

    def ratio_gradient(self, part_values: np.ndarray) -> np.ndarray:
        """(R, m) gradient of the ratios in this moment's parts: 1/b on the numerator, -a/b^2 on the denominator."""
        a = part_values[self._num]
        b = part_values[self._den]
        grad = np.zeros((self.k, len(self.parts)))
        rows = np.arange(self.k)
        np.add.at(grad, (rows, self._num), 1.0 / b)
        np.add.at(grad, (rows, self._den), -a / b**2)
        return grad

    # ------------------------------------------------------------------ Moment protocol

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value, d_delta, d_theta, data=values)."""
        v, dv_delta, dv_theta = self.part_values(sol, self.parts, need_jacobian)
        model = v[self._num] / v[self._den]
        d_delta = d_theta = None
        if need_jacobian:
            grad = self.ratio_gradient(v)
            d_delta = grad @ dv_delta
            d_theta = grad @ dv_theta
        return MomentBlock(name=self.name, value=model - self.values, d_delta=d_delta, d_theta=d_theta,
                           data=self.values.copy())

    def predicted(self, sol: InnerSolution) -> np.ndarray:
        """Model-implied ratio values f(parts), for display next to ``values``."""
        v, _, _ = self.part_values(sol, self.parts, False)
        return v[self._num] / v[self._den]

    def part_covariance(self, sol: InnerSolution) -> np.ndarray:
        """Sigma_parts (m, m): ``(E[v_a v_b] - E[v_a] E[v_b]) / observations`` (pyblp economies/problem.py 1239-1258)."""
        v, _, _ = self.part_values(sol, self.parts, False)
        products = self.covariance_parts()
        e_ab, _, _ = self.part_values(sol, products, False)
        m = len(self.parts)
        sigma = np.zeros((m, m))
        k = 0
        for i in range(m):
            for j in range(i, m):
                sigma[i, j] = sigma[j, i] = (e_ab[k] - v[i] * v[j]) / self.dataset.observations
                k += 1
        return sigma

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> V_micro, (k_b, k_b); the stacked S uses n * V_micro.
        The fixed covariance when one was given."""
        if self.fixed_covariance is not None:
            return self.fixed_covariance.copy()
        v, _, _ = self.part_values(sol, self.parts, False)
        grad = self.ratio_gradient(v)
        V = grad @ self.part_covariance(sol) @ grad.T
        return 0.5 * (V + V.T)

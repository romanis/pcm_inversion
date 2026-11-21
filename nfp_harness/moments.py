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

from dataclasses import dataclass
from typing import Any, Literal, Sequence

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

__all__ = ["Moment", "IVMoments", "SarganIVMoments", "DifferenceIVMoments", "DifferenceSarganIVMoments", "MicroDataset",
           "MicroPart", "RatioMicroMoment", "MICRO_OUTPUTS",
           "product_part", "SupplyMoments", "SupplyDomainError", "MarginMoment", "CostEvent",
           "EventCostContinuityMoments", "GroupMomentRow", "GroupMicroMoments",
           "GROUP_OUTPUTS", "SecondChoiceMoments", "SECOND_CHOICE_OUTPUTS", "SECOND_CHOICE_ADJOINT"]

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
    scale_free = False   # q of the block changes with the scale of xi (SarganIVMoments: it does not)

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


class SarganIVMoments(IVMoments):
    """Scale-free aggregate IV moments ``g = Z' xi / (n s)`` with ``s = sqrt(xi' xi / n)``.

    With ``W = (Z'Z/n)^{-1}`` the block's contribution ``n g' W g = xi' P_Z xi / (xi' xi / n)`` is n times
    the R-squared of xi on Z (the Sargan form). It is invariant to the scale of xi, which closes the
    fixed-W route of shrinking xi (LEARNINGS 1). Unlike CUE it divides by the global variance, not product
    by product, so one huge xi_j drives the block toward n h_jj rather than zero (LEARNINGS 13, 19).

    ``d_delta = (Z'M/n)/s - g (xi'M/n) / s^2`` with g the scaled value. The covariance block is the
    clustered S of IVMoments divided by s^2. Not for CUE.
    """

    scale_free = True

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock of the scaled moments and their delta Jacobian."""
        xi = linear.xi
        s = float(np.sqrt(xi @ xi / self.n))
        if not s > 0:
            raise ValueError("SarganIVMoments: xi is identically zero")
        value = self.Z.T @ xi / (self.n * s)
        d_delta = None
        if need_jacobian:
            ZtM = self._d_delta(linear)
            assert linear.M is not None
            xtM = xi @ linear.M / self.n
            d_delta = ZtM / s - np.outer(value, xtM) / s ** 2
        return MomentBlock(name=self.name, value=value, d_delta=d_delta, d_theta=None, data=None)

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> clustered S_aggregate / s^2."""
        xi = linear.xi
        s2 = float(xi @ xi / self.n)
        return clustered_covariance(self.Z, xi, self.clustering_ids) / s2

    def cluster_terms(self, xi: np.ndarray, dxi: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError("SarganIVMoments is not used with CUE")


class _DifferenceBlock:
    """Mixin of the first-difference IV blocks: a second IV block on xi beside the aggregate one. It does not
    concentrate beta (beta comes from the aggregate block), so it is not ``is_aggregate``; ``is_difference`` asks
    ``NFPGMMProblem.initial_weighting`` for its 2SLS weight ``(Z_d'Z_d/n)^{-1}``."""

    is_aggregate = False
    is_micro = False
    is_difference = True


class DifferenceIVMoments(_DifferenceBlock, IVMoments):
    """First-difference demand moments ``E[dz dxi] = 0`` over pairs of rows of one id in successive periods, as
    ``g = Z_d' xi / n`` with ``Z_d = instruments.difference_instruments(cur, prev, z)`` (``Z_d' xi = dZ' D xi``). xi is the
    aggregate block's, so time-invariant characteristics and fixed effects drop out of every pair."""

    def __init__(self, Z_d: np.ndarray, clustering_ids: np.ndarray | None = None, name: str = "demand_fd") -> None:
        super().__init__(Z_d, clustering_ids, name)


class DifferenceSarganIVMoments(_DifferenceBlock, SarganIVMoments):
    """``DifferenceIVMoments`` in the scale-free form of ``SarganIVMoments`` (divided by ``sqrt(xi'xi/n)``)."""

    def __init__(self, Z_d: np.ndarray, clustering_ids: np.ndarray | None = None, name: str = "demand_fd") -> None:
        super().__init__(Z_d, clustering_ids, name)


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


class SupplyDomainError(ValueError):
    """The supply block cannot be evaluated at this point: an ownership-weighted ds/dp is singular, or some
    marginal cost is not positive, so log mc does not exist. ``NFPGMMProblem.evaluate`` turns it into status
    NUMERICAL with this message as the note. ``info`` carries the diagnostics computed before the failure."""

    def __init__(self, message: str, info: dict | None = None) -> None:
        super().__init__(message)
        self.info = dict(info or {})


class SupplyMoments:
    """Log-cost supply moments under multi-product Bertrand-Nash pricing (Petrin 2002; pyblp ``costs_type="log"``).

    Per market, markups come from the model's own price Jacobian: ``markup = -(Omega * (ds/dp)')^{-1} s`` with
    Omega the ownership matrix of ``firm_ids`` and s the data shares (``pcm_post.nash_markups``); ``mc = p -
    markup``. Then ``log mc = w gamma + omega`` with ``w`` the (n, K_s) cost basis, and the moments are
    ``g_s = Z_s' omega / n``. The cost coefficients gamma are concentrated out by GMM with the supply block of W
    (``set_weighting``; the default ``(Z_s'Z_s/n)^{-1}`` makes it 2SLS), so ``omega = M_s log mc``. When W
    couples the aggregate and supply rows, ``NFPGMMProblem`` concentrates (beta, gamma) jointly instead and
    uses ``log_costs`` directly.

    The evaluation fails (``SupplyDomainError``, status NUMERICAL) when a market's ownership-weighted ds/dp has
    reciprocal condition number below ``rcond_min`` or any mc <= 0.

    Jacobian. ``y = log mc`` depends on theta through ds/dp at (theta, delta(theta)). Its total derivative is
    taken by central differences at fixed inversions: market t is evaluated at ``(theta + h e_k, delta_t + h
    D_t[:, k])`` and ``(theta - h e_k, delta_t - h D_t[:, k])`` with ``pcm_kernel.evaluate_market`` (no
    re-inversion; the shares there differ from the data by O(h^2)). Step rule ``fd_scale``:

    - ``"theta"`` (default): h starts at ``fd_step * max(1, |theta_k|)`` and is divided by 10, at most
      ``fd_shrink`` times, until the frontier signatures at both ends equal the base point's (the slope of
      the smooth piece the base point sits on);
    - ``"unit"``: h = ``fd_step * unit_k`` in the layout's natural units (``bind(..., units)``), never shrunk,
      so the derivative is the secant over a step of the size the trust region takes (round-3 analysis 4 (i):
      ds/dp jumps where a draw's frontier changes, and a slope over 1e-6 is not the change over 1e-3).

    Either way a column whose ends cross a frontier change is counted in ``info["fd_kink_columns"]``
    (market-column pairs), a diagnostic only under ``"unit"``. ``MomentBlock.d_theta`` holds this total
    derivative and ``d_delta`` is None: the block never needs D from the framework.

    ``bind(model, P)`` attaches the PCM model (``kernel_market``, ``kernel_params``, ``markets``,
    ``products``, ``total_threads``); ``NFPGMMProblem`` does it. The block requests ``ds_dp`` from every
    solve. Not for CUE.
    """

    name: str
    is_aggregate = False
    is_micro = False
    is_supply = True

    def __init__(self, Z: np.ndarray, w: np.ndarray, firm_ids: np.ndarray, clustering_ids: np.ndarray | None = None,
                 name: str = "supply", w_names: Sequence[str] | None = None, rcond_min: float = 1e-12,
                 fd_step: float = 1e-6, fd_shrink: int = 2, mc_floor: float | None = None,
                 fd_scale: str = "theta") -> None:
        """SupplyMoments(Z, w, firm_ids, clustering_ids=None, name="supply", w_names=None, rcond_min=1e-12,
        fd_step=1e-6, fd_shrink=2, mc_floor=None, fd_scale="theta"). Z = [w, excluded supply instruments] (n, k_s), w the cost
        basis (n, K_s). ``mc_floor`` (price units) clips FOC costs from below as pyblp's ``costs_bounds``: a
        product whose Nash markup exceeds its price (|own elasticity| < 1) gets mc = mc_floor and zero cost
        derivative, and the count is reported as ``share_mc_clipped``. None keeps the hard mc > 0 guard."""
        from .linear import IdentificationError, matrix_rank

        self.Z = np.asarray(Z, dtype=float)
        self.w = np.asarray(w, dtype=float)
        if self.Z.ndim != 2 or self.w.ndim != 2 or self.Z.shape[0] != self.w.shape[0]:
            raise ValueError(f"Z {self.Z.shape} and w {self.w.shape} must be 2-d with the same number of rows")
        if not (np.all(np.isfinite(self.Z)) and np.all(np.isfinite(self.w))):
            raise ValueError("supply instruments and cost basis must be finite")
        self.n, self.k = self.Z.shape
        self.K = self.w.shape[1]
        self.names = tuple(w_names) if w_names is not None else tuple(f"gamma{k}" for k in range(self.K))
        if len(self.names) != self.K:
            raise ValueError(f"{len(self.names)} names for {self.K} cost coefficients")
        if matrix_rank(self.Z) < self.k:
            raise IdentificationError(f"supply instruments have rank {matrix_rank(self.Z)} < {self.k} columns")
        # rank(Z_s' w) is a property of the column spaces; scaling Z's columns first keeps instruments in large units
        # (e.g. lagged exchange rates, steel weight) from making a full-rank product look rank-deficient
        zn = np.linalg.norm(self.Z, axis=0)
        if matrix_rank((self.Z / np.where(zn > 0, zn, 1.0)).T @ self.w) < self.K:
            raise IdentificationError("rank(Z_s' w) < K_s: the supply moments do not identify gamma; Z_s must "
                                      "contain the cost basis")
        self.firm_ids = np.asarray(firm_ids).ravel()
        if self.firm_ids.size != self.n:
            raise ValueError(f"{self.firm_ids.size} firm ids for {self.n} products")
        self.clustering_ids = None if clustering_ids is None else np.asarray(clustering_ids).ravel()
        if self.clustering_ids is not None and self.clustering_ids.size != self.n:
            raise ValueError(f"{self.clustering_ids.size} clustering ids for {self.n} products")
        self.name = name
        self.rcond_min = float(rcond_min)
        self.fd_step = float(fd_step)
        self.fd_shrink = int(fd_shrink)
        if fd_scale not in ("theta", "unit"):
            raise ValueError(f"fd_scale must be 'theta' or 'unit', got {fd_scale!r}")
        self.fd_scale = fd_scale
        self.units: np.ndarray | None = None
        self.mc_floor = None if mc_floor is None else float(mc_floor)
        self.ZtW = self.Z.T @ self.w / self.n  # (k, K) = -d g_s / d gamma
        self.model: object | None = None
        self.P: int | None = None
        self.s_data: np.ndarray | None = None
        self._cache: tuple[InnerSolution, bool, np.ndarray, np.ndarray | None, dict] | None = None
        self.set_weighting(np.linalg.inv(self.Z.T @ self.Z / self.n))

    # ------------------------------------------------------------------ setup

    def bind(self, model: object, P: int, units: np.ndarray | None = None) -> None:
        """Attach the inner model (a ``PCMModel``), the number of nonlinear parameters and their natural units
        (needed by ``fd_scale="unit"``)."""
        self.model = model
        self.P = int(P)
        self.units = None if units is None else np.asarray(units, dtype=float).ravel()
        if self.fd_scale == "unit" and (self.units is None or self.units.shape != (self.P,)
                                        or not np.all(self.units > 0)):
            raise ValueError(f"{self.name}: fd_scale='unit' needs {self.P} positive natural units, got {self.units}")
        if int(getattr(model, "n_products")) != self.n:
            raise ValueError(f"supply block has {self.n} rows, model has {getattr(model, 'n_products')} products")
        spec = getattr(model, "spec")
        self.s_data = np.asarray(getattr(model, "products")[spec.share_column], dtype=float).ravel()

    def set_weighting(self, W_ss: np.ndarray) -> None:
        """Supply block of W: gamma = (C'W_ss C)^{-1} C'W_ss Z_s' y / n with C = Z_s'w/n."""
        W_ss = np.asarray(W_ss, dtype=float)
        if W_ss.shape != (self.k, self.k):
            raise ValueError(f"W_ss has shape {W_ss.shape}, expected {(self.k, self.k)}")
        self.W_ss = 0.5 * (W_ss + W_ss.T)
        B = self.ZtW.T @ self.W_ss
        self._Pg = np.linalg.solve(B @ self.ZtW, B @ self.Z.T / self.n)  # (K, n): gamma = Pg y

    def requests(self) -> KernelRequests:
        """requests() -> KernelRequests(price_derivatives=True): markups need ds/dp."""
        return KernelRequests(price_derivatives=True)

    def covariance_requests(self) -> KernelRequests:
        return self.requests()

    # ------------------------------------------------------------------ log marginal costs

    def _market_costs(self, t: int, s: np.ndarray, p: np.ndarray, firms: np.ndarray, ds_dp: np.ndarray
                      ) -> tuple[np.ndarray, dict]:
        from .models.pcm_post import nash_markups

        nash = nash_markups(s, ds_dp, firms, self.rcond_min)
        mc = p - nash["markup"]
        if self.mc_floor is not None:
            nash["clipped"] = ~(mc > self.mc_floor)
            mc = np.where(nash["clipped"], self.mc_floor, mc)
        return mc, nash

    def log_costs(self, sol: InnerSolution, need_jacobian: bool) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """log_costs(sol, need_jacobian) -> (y (n,) = log mc, dy (n, P) total derivative or None, info).

        Raises ``SupplyDomainError`` at a singular ownership Jacobian or mc <= 0. ``info``: per-market rcond
        minimum ``nash_rcond_min``, ``margin_median``, ``share_mc_nonpositive``, ``markup``/``mc`` (n,) and the
        FD bookkeeping. Cached for the last solution.
        """
        if self.model is None or self.P is None or self.s_data is None:
            raise RuntimeError(f"{self.name}: bind(model, P) was not called")
        c = self._cache
        if c is not None and c[0] is sol and (c[1] or not need_jacobian):
            return c[2], c[3], c[4]
        model: Any = self.model
        ds_list = sol.outputs.get("ds_dp")
        if not isinstance(ds_list, list) or len(ds_list) != len(model.markets):
            raise KeyError(f"{self.name}: the solution carries no per-market ds_dp (request price_derivatives)")
        markup = np.full(self.n, np.nan)
        mc_all = np.full(self.n, np.nan)
        rconds = []
        singular = []
        for t, m in enumerate(model.markets):
            rows = np.asarray(m.rows)
            p = np.asarray(model.kernel_market(t).p, dtype=float)
            mc, nash = self._market_costs(t, self.s_data[rows], p, self.firm_ids[rows], np.asarray(ds_list[t]))
            rconds.append(float(nash["rcond"]))
            if nash["status"] != "ok":
                singular.append(f"{m.market_id} (rcond {nash['rcond']:.2e})")
            markup[rows] = nash["markup"]
            mc_all[rows] = mc
        bad = ~(mc_all > 0)
        clipped = np.zeros(self.n, dtype=bool)
        if self.mc_floor is not None:
            for t, m in enumerate(model.markets):
                rows = np.asarray(m.rows)
                p_t = np.asarray(model.kernel_market(t).p, dtype=float)
                clipped[rows] = ~(p_t - markup[rows] > self.mc_floor)
        p_all = np.empty(self.n)
        for t, m in enumerate(model.markets):
            p_all[np.asarray(m.rows)] = np.asarray(model.kernel_market(t).p, dtype=float)
        info: dict = {"nash_rcond_min": float(min(rconds)) if rconds else float("nan"),
                      "share_mc_nonpositive": float(np.mean(bad)), "share_mc_clipped": float(np.mean(clipped)),
                      "markup": markup, "mc": mc_all,
                      "margin_median": float(np.nanmedian(markup / p_all)) if np.isfinite(markup).any() else float("nan")}
        if singular:
            raise SupplyDomainError(f"{self.name}: ownership-weighted ds/dp below rcond {self.rcond_min:.0e} in "
                                    f"markets {', '.join(singular)}; no Nash markups", info)
        if bad.any():
            ids = sorted({str(model.markets[t].market_id) for t, m in enumerate(model.markets)
                          if bad[np.asarray(m.rows)].any()})
            raise SupplyDomainError(f"{self.name}: {int(bad.sum())} of {self.n} products have mc <= 0 (share "
                                    f"{info['share_mc_nonpositive']:.4f}, markets {', '.join(ids)}); log mc does "
                                    f"not exist", info)
        y = np.log(mc_all)
        dy = self._jacobian(sol, info) if need_jacobian else None
        self._cache = (sol, need_jacobian, y, dy, info)
        return y, dy, info

    def _jacobian(self, sol: InnerSolution, info: dict) -> np.ndarray:
        """(n, P) total derivative of log mc, central differences at fixed inversions (class docstring)."""
        from concurrent.futures import ThreadPoolExecutor

        from ._native import pcm_kernel
        from .models.pcm import allocate_threads

        model: Any = self.model
        assert self.P is not None and self.s_data is not None
        if sol.D is None:
            raise ValueError(f"{self.name}: the supply Jacobian needs D (solve with need_jacobian=True)")
        theta = np.asarray(sol.outputs["theta"], dtype=float)
        delta = np.asarray(sol.delta, dtype=float)
        T = len(model.markets)
        wts = np.array([float(np.size(m.rows)) ** 2 * np.size(model.kernel_market(t).weights)
                        for t, m in enumerate(model.markets)])
        thr = allocate_threads(wts, int(getattr(model, "total_threads", 1)))

        def one(t: int) -> tuple[np.ndarray, int, int]:
            m = model.markets[t]
            rows = np.asarray(m.rows)
            km = model.kernel_market(t)
            p = np.asarray(km.p, dtype=float)
            s, firms = self.s_data[rows], self.firm_ids[rows]
            d0 = delta[rows]
            D_t = np.asarray(sol.D[t], dtype=float)
            nt = int(thr[t])
            kw = model.kernel_kwargs(theta) if hasattr(model, "kernel_kwargs") else {}
            sig0 = np.asarray(pcm_kernel.frontier_signature(d0, km, *model.kernel_params(theta), nt, **kw))
            out = np.zeros((rows.size, self.P))
            kinks = shrinks = 0
            unit_rule = self.fd_scale == "unit"
            shrink = 0 if unit_rule else self.fd_shrink
            skip = set(getattr(model, "bound_columns", ()))   # fixed bounds: no secant, zero column
            for k in range(self.P):
                if k in skip:
                    continue
                if unit_rule:
                    assert self.units is not None
                    h = self.fd_step * float(self.units[k])
                else:
                    h = self.fd_step * max(1.0, abs(float(theta[k])))
                for attempt in range(shrink + 1):
                    ys, same = [], True
                    for sgn in (1.0, -1.0):
                        th = theta.copy()
                        th[k] += sgn * h
                        dl = d0 + sgn * h * D_t[:, k]
                        kp = model.kernel_params(th)
                        r = pcm_kernel.evaluate_market(dl, km, *kp, derivatives=False, price_derivatives=True,
                                                       n_threads=nt, **kw)
                        mc, _ = self._market_costs(t, s, p, firms, np.asarray(r["ds_dp"]))
                        with np.errstate(invalid="ignore", divide="ignore"):
                            ys.append(np.log(mc))
                        if attempt < shrink or unit_rule:
                            same &= bool(np.array_equal(np.asarray(pcm_kernel.frontier_signature(dl, km, *kp, nt,
                                                                                                 **kw)), sig0))
                    if same or attempt == shrink:
                        if not same:
                            kinks += 1
                        break
                    h /= 10.0
                    shrinks += 1
                out[:, k] = (ys[0] - ys[1]) / (2.0 * h)
            return out, kinks, shrinks

        dy = np.zeros((self.n, self.P))
        with ThreadPoolExecutor(max_workers=max(1, T), thread_name_prefix="supply-fd") as pool:
            res = list(pool.map(one, range(T)))
        kinks = shrinks = 0
        for t, (out, k_, s_) in enumerate(res):
            dy[np.asarray(model.markets[t].rows)] = out
            kinks += k_
            shrinks += s_
        info["fd_kink_columns"] = kinks
        info["fd_step_shrinks"] = shrinks
        if not np.all(np.isfinite(dy)):
            raise SupplyDomainError(f"{self.name}: the finite-difference Jacobian of log mc is not finite (mc <= 0 "
                                    f"or a singular ownership Jacobian within the step)", info)
        return dy

    # ------------------------------------------------------------------ Moment protocol

    def concentrate(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(gamma, omega) from y = log mc with the supply block of W."""
        gamma = self._Pg @ y
        return gamma, y - self.w @ gamma

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value = Z_s' omega / n, d_theta = total
        derivative, d_delta = None)."""
        y, dy, _ = self.log_costs(sol, need_jacobian)
        _, omega = self.concentrate(y)
        d_theta = None
        if need_jacobian:
            assert dy is not None
            d_theta = self.Z.T @ (dy - self.w @ (self._Pg @ dy)) / self.n
        return MomentBlock(name=self.name, value=self.Z.T @ omega / self.n, d_delta=None, d_theta=d_theta, data=None)

    def covariance_from(self, omega: np.ndarray) -> np.ndarray:
        """Clustered S_supply = (1/n) sum_c (sum_{j in c} z_j omega_j)(...)'."""
        return clustered_covariance(self.Z, omega, self.clustering_ids)

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray:
        """covariance_block(sol, linear) -> clustered S_supply at the omega of ``sol`` (2SLS-type gamma)."""
        y, _, _ = self.log_costs(sol, False)
        return self.covariance_from(self.concentrate(y)[1])


class MarginMoment:
    """A calibration row on the markup level (round-3 analysis 5.2): ``g = m(theta) - target`` with ``m`` the
    trimmed mean over all products of the Nash margin ``(p - mc) / p``, between the ``trim`` percentiles
    (25th to 75th by default).

    The margins come from the same Nash markups and the same finite-difference pass as the supply block
    (``supply.log_costs``, cached per solution), so the row adds no kernel work: with ``mc = exp(y)``,
    ``d margin_j = -(mc_j / p_j) dy_j``, and the trimmed mean's derivative is the mean over the products in the
    trimmed set (the order statistics are continuous in the margins, so the value is too; its derivative jumps
    where the set changes). Clipped costs (``mc_floor``) enter at the floor with zero derivative.

    The covariance block is a declared constant ``n sd^2`` (the moment's sd is taken as ``sd``, one margin
    point by default), so ``W1``'s entry is ``1 / (n sd^2)`` and the row contributes ``((m - target) / sd)^2``
    to q. A spec makes it bind by setting that entry to ``binding_block()`` = ``binding_weight / (n sd^2)``.
    ``is_calibration`` marks the row for ``NFPGMMProblem.initial_weighting`` and for reports.
    """

    is_aggregate = False
    is_micro = False
    is_supply = False
    is_calibration = True

    def __init__(self, supply: SupplyMoments, target: float, trim: tuple[float, float] = (0.25, 0.75),
                 sd: float = 0.01, name: str = "margin", binding_weight: float = 1e4) -> None:
        """MarginMoment(supply, target, trim=(0.25, 0.75), sd=0.01, name="margin", binding_weight=1e4)."""
        if not 0.0 <= trim[0] < trim[1] <= 1.0:
            raise ValueError(f"trim must satisfy 0 <= lo < hi <= 1, got {trim}")
        if not sd > 0 or not binding_weight > 0:
            raise ValueError("sd and binding_weight must be positive")
        self.supply = supply
        self.target = float(target)
        self.trim = (float(trim[0]), float(trim[1]))
        self.sd = float(sd)
        self.binding_weight = float(binding_weight)
        self.name = name
        self.k = 1
        self.labels = (f"trimmed mean margin ({100 * self.trim[0]:g}th to {100 * self.trim[1]:g}th percentile) - "
                       f"{self.target:g}",)
        self._last: tuple[InnerSolution, dict[str, Any]] | None = None

    def requests(self) -> KernelRequests:
        return KernelRequests(price_derivatives=True)

    def covariance_requests(self) -> KernelRequests:
        return self.requests()

    def _prices(self) -> np.ndarray:
        model: Any = self.supply.model
        if model is None:
            raise RuntimeError(f"{self.name}: the supply block is not bound")
        p = np.empty(self.supply.n)
        for t, m in enumerate(model.markets):
            p[np.asarray(m.rows)] = np.asarray(model.kernel_market(t).p, dtype=float)
        return p

    def trimmed_set(self, margin: np.ndarray) -> np.ndarray:
        """Indices of the products whose margin ranks between the trim percentiles (stable order)."""
        n = margin.size
        lo = int(np.floor(self.trim[0] * n))
        hi = max(lo + 1, int(np.ceil(self.trim[1] * n)))
        return np.argsort(margin, kind="stable")[lo:hi]

    def margins(self, sol: InnerSolution, need_jacobian: bool) -> tuple[float, np.ndarray | None, dict[str, Any]]:
        """(m, dm/dtheta (P,) or None, info) with ``margin_trimmed_mean``, ``margin_median`` and the set size."""
        y, dy, _ = self.supply.log_costs(sol, need_jacobian)
        p = self._prices()
        mc = np.exp(y)
        margin = 1.0 - mc / p
        idx = self.trimmed_set(margin)
        m = float(np.mean(margin[idx]))
        dm = None
        if need_jacobian:
            assert dy is not None
            dm = np.mean(-(mc / p)[idx, None] * dy[idx], axis=0)
        info = {"margin_trimmed_mean": m, "margin_target": self.target, "margin_moment": m - self.target,
                "margin_median": float(np.median(margin)), "margin_trimmed_products": int(idx.size)}
        self._last = (sol, info)
        return m, dm, info

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value = [m - target], d_theta (1, P), d_delta None)."""
        m, dm, _ = self.margins(sol, need_jacobian)
        return MomentBlock(name=self.name, value=np.array([m - self.target]), d_delta=None,
                           d_theta=None if dm is None else dm[None, :], data=None)

    def summary(self, sol: InnerSolution) -> dict[str, Any]:
        """The scalars of the last evaluation at ``sol`` (for ``Evaluation.aux[name]``)."""
        if self._last is not None and self._last[0] is sol:
            return dict(self._last[1])
        return dict(self.margins(sol, False)[2])

    def covariance_block(self, sol: InnerSolution | None, linear: LinearFit | None) -> np.ndarray:
        """The declared constant ``[[n sd^2]]`` (class docstring)."""
        return np.array([[self.supply.n * self.sd ** 2]])

    def binding_block(self) -> float:
        """The binding W entry ``binding_weight / (n sd^2)``: a deviation of one ``sd`` costs ``binding_weight``."""
        return self.binding_weight / (self.supply.n * self.sd ** 2)


#: Key of ``InnerSolution.outputs`` holding the per-market household sums (contract in ``GroupMicroMoments``).
GROUP_OUTPUTS = "group_sums"


@dataclass(frozen=True)
class CostEvent:
    """One event of ``EventCostContinuityMoments``: products whose ownership incentives change between two periods
    (``treated``) and products whose do not (``control``), each as row pairs (cur, prev) of the same product in the
    two periods. ``match`` (n_treated, n_control), rows summing to one, gives each treated pair its own control mix
    (e.g. controls of the same type); None is the plain control mean."""

    name: str
    treated: tuple[np.ndarray, np.ndarray]
    control: tuple[np.ndarray, np.ndarray]
    match: np.ndarray | None = None


class EventCostContinuityMoments:
    """Cost continuity across ownership events: a difference in differences of implied log marginal cost.

    For event e with treated pairs (c_j, p_j) and control pairs (c'_l, p'_l), dy = y[cur] - y[prev] with y = log mc
    from the Nash FOCs under each period's own ownership (``costs.log_costs``, a ``SupplyMoments``), and

        g_e = mean_j ( dy_j - sum_l A_jl dy'_l ),

    which is zero when costs of products that continue unchanged do not jump at the ownership change beyond the
    controls' change. Under a wrong price sensitivity the recomputed markup change does not match the observed price
    change and the residual shows up in dy. g = L y with L (k, n) fixed by the pairs, so ``d_theta = L dy/dtheta``
    (the cost object's total derivative) and ``d_delta`` is None.

    Covariance: the products are independent draws, so Var(g_e) = var(dy_treated) / n_t + var(dy_control) sum_l
    c_l^2 with c = mean over j of A_j; ``covariance_block`` returns n diag(Var) (q = n g'Wg with W = its inverse is
    the sum of squared studentised contrasts t_e = g_e / sd(g_e)). ``needs_evaluated_weight`` asks
    ``NFPGMMProblem.initial_weighting`` for that weight at theta0. Not for CUE.
    """

    name: str
    is_aggregate = False
    is_micro = False
    is_supply = False
    needs_evaluated_weight = True

    def __init__(self, costs: SupplyMoments, events: Sequence[CostEvent], name: str = "event_cost") -> None:
        """EventCostContinuityMoments(costs, events, name="event_cost"); ``costs`` computes log mc (its own
        instruments are not used)."""
        if not events:
            raise ValueError("no events")
        self.costs = costs
        self.events = list(events)
        self.name = name
        self.n = int(costs.n)
        self.k = len(self.events)
        self.L = np.zeros((self.k, self.n))
        self._parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for e, ev in enumerate(self.events):
            tc, tp = (np.asarray(a, dtype=int).ravel() for a in ev.treated)
            cc, cp = (np.asarray(a, dtype=int).ravel() for a in ev.control)
            if tc.size == 0 or cc.size == 0 or tc.size != tp.size or cc.size != cp.size:
                raise ValueError(f"event {ev.name!r}: treated and control need equal-length non-empty pairs")
            for a in (tc, tp, cc, cp):
                if a.min() < 0 or a.max() >= self.n:
                    raise ValueError(f"event {ev.name!r}: row index outside 0..{self.n - 1}")
            if np.intersect1d(np.r_[tc, tp], np.r_[cc, cp]).size:
                raise ValueError(f"event {ev.name!r}: a row is both treated and control")
            A = np.full((tc.size, cc.size), 1.0 / cc.size) if ev.match is None else np.asarray(ev.match, float)
            if A.shape != (tc.size, cc.size) or np.any(A < 0) or not np.allclose(A.sum(axis=1), 1.0):
                raise ValueError(f"event {ev.name!r}: match must be ({tc.size}, {cc.size}) with rows summing to one")
            c = A.mean(axis=0)
            np.add.at(self.L[e], tc, 1.0 / tc.size)
            np.add.at(self.L[e], tp, -1.0 / tc.size)
            np.add.at(self.L[e], cc, -c)
            np.add.at(self.L[e], cp, c)
            self._parts.append((tc, tp, cc, cp, c))

    def bind_costs(self, model: object, P: int, units: np.ndarray | None = None) -> None:
        """Attach the inner model to the cost object unless the problem already did (a supply block of its own)."""
        if self.costs.model is not model:
            self.costs.bind(model, P, units=units)

    def requests(self) -> KernelRequests:
        return self.costs.requests()

    def covariance_requests(self) -> KernelRequests:
        return self.costs.requests()

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        """evaluate(sol, linear, need_jacobian) -> MomentBlock(value = L log mc, d_theta = L dlog mc/dtheta)."""
        y, dy, _ = self.costs.log_costs(sol, need_jacobian)
        return MomentBlock(name=self.name, value=self.L @ y, d_delta=None,
                           d_theta=self.L @ dy if need_jacobian and dy is not None else None, data=None)

    def variances(self, y: np.ndarray) -> np.ndarray:
        """(k,) Var(g_e) at log mc ``y`` (class docstring)."""
        out = np.empty(self.k)
        for e, (tc, tp, cc, cp, c) in enumerate(self._parts):
            dt, dc = y[tc] - y[tp], y[cc] - y[cp]
            vt = float(np.var(dt, ddof=1)) if dt.size > 1 else 0.0
            vc = float(np.var(dc, ddof=1)) if dc.size > 1 else 0.0
            out[e] = vt / dt.size + vc * float(c @ c)
        return out

    def summary(self, y: np.ndarray) -> list[dict[str, Any]]:
        """Per event: name, pair counts, g_e, sd(g_e), t_e and the mean change of treated and control log mc."""
        g, v = self.L @ y, self.variances(y)
        rows = []
        for e, (ev, (tc, tp, cc, cp, _)) in enumerate(zip(self.events, self._parts)):
            sd = float(np.sqrt(v[e]))
            rows.append({"event": ev.name, "treated": int(tc.size), "control": int(cc.size), "g": float(g[e]),
                         "sd": sd, "t": float(g[e]) / sd if sd > 0 else float("nan"),
                         "d_log_mc_treated": float(np.mean(y[tc] - y[tp])),
                         "d_log_mc_control": float(np.mean(y[cc] - y[cp]))})
        return rows

    def cluster_terms(self, y: np.ndarray, dy: np.ndarray | None, cluster_matrix: Any, n: int
                      ) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-cluster contributions for a joint clustered covariance (``NFPGMMProblem.set_joint``): (C, k) and
        (C, k, P). A pair p of event e contributes ``n L_e[cur_p] (dy_p - mean of its group's dy)`` to the cluster of
        its cur row (``cluster_matrix`` is the (C, n) 0/1 cluster indicator), so ``V'V/n`` of one event is about ``n
        Var(g_e)``, ``covariance_block``'s value, and its scale follows L."""
        C = cluster_matrix.shape[0]
        V = np.zeros((C, self.k))
        dV = None if dy is None else np.zeros((C, self.k, dy.shape[1]))
        for e, (tc, tp, cc, cp, _) in enumerate(self._parts):
            contrib = np.zeros(self.n)
            dcon = None if dy is None else np.zeros((self.n, dy.shape[1]))
            for cur, prev in ((tc, tp), (cc, cp)):
                d = y[cur] - y[prev]
                np.add.at(contrib, cur, n * self.L[e, cur] * (d - d.mean()))
                if dcon is not None:
                    dd = dy[cur] - dy[prev]
                    np.add.at(dcon, cur, n * self.L[e, cur][:, None] * (dd - dd.mean(axis=0)))
            V[:, e] = cluster_matrix @ contrib
            if dV is not None:
                dV[:, e, :] = cluster_matrix @ dcon
        return V, dV

    def covariance_block(self, sol: InnerSolution | None, linear: LinearFit | None) -> np.ndarray:
        """covariance_block(sol, linear) -> n diag(Var(g_e)) at the log mc of ``sol``."""
        assert sol is not None
        y, _, _ = self.costs.log_costs(sol, False)
        return self.n * np.diag(self.variances(y))


@dataclass(frozen=True)
class GroupMomentRow:
    """One GMY micro moment (ExjCondH.m): ``diff`` = E[x_char | buy, set_a] - E[x_char | buy, set_b] with each side the
    household-weighted mean of the households' own conditional means; ``ratio`` = purchase rate in set_a / in set_b.
    ``set_a`` and ``set_b`` are 0/1 agent columns (the survey years are part of the set); ``char`` is a product column
    (None for a ratio row)."""

    kind: Literal["diff", "ratio"]
    char: str | None
    set_a: str
    set_b: str


class GroupMicroMoments:
    """Micro moments built from per-household sums (GMY spec WP-G3; ExjCondH.m, getMicroMoments.m).

    For household g with draws i (weights w_i, sum w_g) and exact PCM choice probabilities s_ij:
    ``P_g = sum_i w_i sum_j s_ij``, ``S_g[c] = sum_i w_i sum_j x_jc s_ij``, ``m_g[c] = S_g[c] / P_g``. A household set H
    pools its households over the model's markets (each year's agent weights sum to one, so years count equally):
    ``M_H[c] = sum_{g in H+} w_g m_g[c] / D_H`` with H+ the households with P_g > 0 and D_H = sum_{H+} w_g
    (``never_buyers="drop"``, the stated intent of ExjCondH.m, deviation D2) or sum_H w_g (``"keep"``, the literal code of
    ExjCondH.m: a never-buyer keeps its weight and adds 0); ``A_H = sum_H P_g / sum_H w_g``. Row values are
    ``M_a - M_b - value`` (diff) and ``A_a / A_b - value`` (ratio). Partials: ``d m_g = (d S_g - m_g d P_g) / P_g``.

    Kernel contract (``sol.outputs["group_sums"]``): a list with one entry per model market (None where the dataset
    has no households) of a dict with ``columns`` (the product columns of S after P, in order), ``S`` (G_t, C+1)
    (column 0 = P_g, then S_g[c]), ``d_delta`` (G_t, C+1, J_t) partials in delta_t (price order), ``d_theta``
    (G_t, C+1, P) partials in theta (layout order), both None without a Jacobian, and ``w`` (G_t,) agent weights.

    The covariance is fixed, ``diag(variances)``, so W1 = (n V)^{-1} and a row contributes (f / sd)^2 to q (GMY's
    weight diag(1 / variance), deviation D6). ``bind(markets, n, P)`` and ``bind_agents(agents, agent_rows)``
    (``agent_rows[t]`` = the agent-table rows of market t's households in group order) are called by
    ``NFPGMMProblem``. ``last_info`` holds the never-buyer counts of the last evaluation.
    """

    name: str
    is_aggregate = False
    is_micro = True

    def __init__(self, name: str, dataset: MicroDataset, rows: Sequence[GroupMomentRow], values: np.ndarray,
                 variances: np.ndarray, labels: Sequence[str] | None = None, never_buyers: str = "drop") -> None:
        """GroupMicroMoments(name, dataset, rows, values, variances, labels=None, never_buyers="drop")."""
        self.name = name
        self.dataset = dataset
        self.rows = tuple(rows)
        self.values = np.asarray(values, dtype=float).ravel()
        var = np.asarray(variances, dtype=float).ravel()
        if not len(self.rows) == self.values.size == var.size:
            raise ValueError(f"{name}: {len(self.rows)} rows, {self.values.size} values, {var.size} variances")
        if not np.all(np.isfinite(var) & (var > 0)):
            raise ValueError(f"{name}: variances must be positive and finite")
        if never_buyers not in ("drop", "keep"):
            raise ValueError(f"never_buyers must be 'drop' or 'keep', got {never_buyers!r}")
        for r in self.rows:
            if r.kind not in ("diff", "ratio") or (r.kind == "diff") != (r.char is not None):
                raise ValueError(f"{name}: bad row {r}")
        self.never_buyers = never_buyers
        self.k = len(self.rows)
        self.variances = var
        self.fixed_covariance = np.diag(var)
        self.labels = tuple(labels) if labels is not None else tuple(
            f"{r.kind} {r.char} {r.set_a} vs {r.set_b}" for r in self.rows)
        chars: list[str] = []
        for r in self.rows:
            if r.char is not None and r.char not in chars:
                chars.append(r.char)
        self.chars = tuple(chars)
        self.sets = tuple(dict.fromkeys([s for r in self.rows for s in (r.set_a, r.set_b)]))
        self._markets: list[MarketIndex] | None = None
        self._n: int | None = None
        self._P: int | None = None
        self._member: list[dict[str, np.ndarray]] | None = None
        self.last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    def bind(self, markets: Sequence[MarketIndex], n_products: int, P: int | None = None) -> None:
        self._markets = list(markets)
        self._n = int(n_products)
        self._P = None if P is None else int(P)
        self._in = [t for t, m in enumerate(self._markets) if self.dataset.includes(m.market_id)]
        if not self._in:
            raise ValueError(f"{self.name}: dataset {self.dataset.name!r} covers none of the model's markets")

    def bind_agents(self, agents: Any, agent_rows: Sequence[np.ndarray]) -> None:
        """Set membership per market: ``agent_rows[t]`` are the agent-table rows of market t's households in the
        kernel's group order; ``agents`` holds the 0/1 set columns."""
        cols = {}
        for s in self.sets:
            try:
                cols[s] = np.asarray(agents[s], dtype=float).ravel()
            except (KeyError, IndexError):
                raise KeyError(f"{self.name}: the agent table has no set column {s!r}") from None
            if not np.all(np.isin(cols[s], (0.0, 1.0))):
                raise ValueError(f"{self.name}: set column {s!r} is not 0/1")
        self._member = [{s: cols[s][np.asarray(r, dtype=int)] > 0.5 for s in self.sets} for r in agent_rows]

    def requests(self) -> KernelRequests:
        return KernelRequests(group_sums=self.chars)

    def covariance_requests(self) -> KernelRequests:
        return self.requests()

    # ------------------------------------------------------------------ evaluation

    def _outputs(self, sol: InnerSolution) -> list[dict[str, Any]]:
        if self._markets is None or self._member is None:
            raise RuntimeError(f"{self.name}: bind(...) and bind_agents(...) were not called")
        table = sol.outputs.get(GROUP_OUTPUTS)
        if not isinstance(table, list) or len(table) != len(self._markets):
            raise KeyError(f"{self.name}: the solution carries no per-market {GROUP_OUTPUTS!r} list")
        out = []
        for t in self._in:
            e = table[t]
            if e is None:
                raise KeyError(f"{self.name}: no group sums for market {self._markets[t].market_id}")
            out.append(e)
        return out

    def _column(self, entry: dict[str, Any], char: str | None) -> int:
        if char is None:
            return 0
        cols = list(entry["columns"])
        if char not in cols:
            raise KeyError(f"{self.name}: group sums lack product column {char!r} (have {cols})")
        return 1 + cols.index(char)

    def moments(self, sol: InnerSolution, need_jacobian: bool) -> tuple[np.ndarray, np.ndarray | None,
                                                                         np.ndarray | None]:
        """(model values (k,), d_delta (k, n), d_theta (k, P)): the GMY quantities M_a - M_b and A_a / A_b."""
        entries = self._outputs(sol)
        assert self._member is not None and self._markets is not None and self._n is not None
        mem = [self._member[t] for t in self._in]
        # pooled set statistics
        W, Wp, never = {}, {}, {}
        for s in self.sets:
            W[s] = sum(float(e["w"][m[s]].sum()) for e, m in zip(entries, mem))
            Wp[s] = sum(float(e["w"][m[s] & (np.asarray(e["S"])[:, 0] > 0)].sum()) for e, m in zip(entries, mem))
            never[s] = sum(int(np.sum(m[s] & ~(np.asarray(e["S"])[:, 0] > 0))) for e, m in zip(entries, mem))
            if not W[s] > 0:
                raise ValueError(f"{self.name}: set {s!r} has no households in the dataset's markets")
        den = Wp if self.never_buyers == "drop" else W
        n_hh = {s: sum(int(m[s].sum()) for m in mem) for s in self.sets}
        self.last_info = {"micro_never_buyers": never, "micro_households": n_hh,
                          "micro_never_buyer_share_max": max(never[s] / max(n_hh[s], 1) for s in self.sets)}
        stats: dict[tuple[str, str | None], float] = {}
        coefs: list[list[np.ndarray]] = [[] for _ in entries]   # per market: list over rows of (G, C+1)

        def side(s: str, char: str | None) -> float:
            key = (s, char)
            if key not in stats:
                tot = 0.0
                for e, m in zip(entries, mem):
                    S = np.asarray(e["S"])
                    if char is None:
                        tot += float(S[m[s], 0].sum())
                    else:
                        c = self._column(e, char)
                        buy = m[s] & (S[:, 0] > 0)
                        tot += float(np.sum(e["w"][buy] * S[buy, c] / S[buy, 0]))
                stats[key] = tot / (W[s] if char is None else den[s])
            return stats[key]

        value = np.empty(self.k)
        for r_i, r in enumerate(self.rows):
            a, b = side(r.set_a, r.char), side(r.set_b, r.char)
            value[r_i] = a - b if r.kind == "diff" else a / b
            if not need_jacobian:
                continue
            for t_i, (e, m) in enumerate(zip(entries, mem)):
                S = np.asarray(e["S"])
                K = np.zeros_like(S)
                if r.kind == "diff":
                    c = self._column(e, r.char)
                    for s, sign in ((r.set_a, 1.0), (r.set_b, -1.0)):
                        buy = m[s] & (S[:, 0] > 0)
                        coef = sign * e["w"][buy] / (den[s] * S[buy, 0])
                        K[buy, c] += coef
                        K[buy, 0] -= coef * S[buy, c] / S[buy, 0]
                else:
                    K[m[r.set_a], 0] += (1.0 / b) / W[r.set_a]
                    K[m[r.set_b], 0] += -(a / b ** 2) / W[r.set_b]
                coefs[t_i].append(K)
        if not need_jacobian:
            return value, None, None
        d_delta = np.zeros((self.k, self._n))
        d_theta: np.ndarray | None = None
        for t_i, (t, e) in enumerate(zip(self._in, entries)):
            if e.get("d_delta") is None or e.get("d_theta") is None:
                raise ValueError(f"{self.name}: group sums without derivatives (solve with need_jacobian=True)")
            Kt = np.stack([K.ravel() for K in coefs[t_i]])                 # (k, G (C+1))
            dD = np.asarray(e["d_delta"], dtype=float)
            dT = np.asarray(e["d_theta"], dtype=float)
            d_delta[:, self._markets[t].rows] += Kt @ dD.reshape(-1, dD.shape[-1])
            part = Kt @ dT.reshape(-1, dT.shape[-1])
            d_theta = part if d_theta is None else d_theta + part
        assert d_theta is not None
        return value, d_delta, d_theta

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        model, dd, dt = self.moments(sol, need_jacobian)
        return MomentBlock(name=self.name, value=model - self.values, d_delta=dd, d_theta=dt, data=self.values.copy())

    def predicted(self, sol: InnerSolution) -> np.ndarray:
        """Model values of the rows (for display next to ``values``)."""
        return self.moments(sol, False)[0]

    def aux_summary(self, sol: InnerSolution) -> dict[str, Any]:
        """Never-buyer counts per set of the last evaluation (``Evaluation.aux[name]``)."""
        return dict(self.last_info)

    def covariance_block(self, sol: InnerSolution | None, linear: LinearFit | None) -> np.ndarray:
        return self.fixed_covariance.copy()


#: Keys of ``InnerSolution.outputs`` for second choices (contract in ``SecondChoiceMoments``).
SECOND_CHOICE_OUTPUTS = "second_choice"
SECOND_CHOICE_ADJOINT = "second_choice_adjoint"


class SecondChoiceMoments:
    """Second-choice correlation moments (GMY spec WP-G4; ComputeSecondChoiceMoments.m, mom_fnc.m).

    For each year y and characteristic c with a finite data correlation: ``g = corr_c(model) - corr_c(data)`` where the
    model joint law is ``(s_j^data / S^data) C[j, k]`` renormalised (D3) and C the model's matrix of P(second = k | first
    = j, second inside), first choices in rows, in the model market's (price) order. Rows with a NaN data correlation
    (electric before EVs) are dropped. Weight: the fixed covariance ``I / weight^2`` so that a row contributes
    ``(weight (corr_model - corr_data))^2`` to q (GMY's 30 x residual, mom_fnc.m line 109; D6).

    Model contract: ``sol.outputs["second_choice"]`` is a list with one entry per model market (None outside the
    requested years) holding ``C`` (J, J) and ``s`` (J,) model shares, with ``C = N / s`` row by row (rows of products
    with s_j = 0 zero); and ``sol.outputs["second_choice_adjoint"]`` is a callable ``(t, A, B) -> (out_delta (K, J),
    out_theta (K, P) layout order)`` returning ``sum_jk A[c, j, k] dN_jk - sum_j B[c, j] ds_j`` in delta_t and theta.
    With ``dC = (dN - C ds) / s`` the moment passes ``A' = A / s_j`` (row scaling) and ``B[c, j] = sum_k A'[c, j, k]
    C[j, k]``, so the Jacobian costs two kernel passes per market.
    """

    name: str
    is_aggregate = False
    is_micro = True

    def __init__(self, name: str, years: Sequence[Any], chars: Sequence[str], data_corr: dict[Any, np.ndarray],
                 products_x: dict[Any, np.ndarray], s_data: dict[Any, np.ndarray], weight: float = 30.0) -> None:
        """SecondChoiceMoments(name, years, chars, data_corr, products_x, s_data, weight=30.0): per year the (K,) data
        correlations, the (J, K) characteristics and the (J,) data shares in the model's price order."""
        self.name = name
        self.years = tuple(years)
        self.chars = tuple(chars)
        if not self.years:
            raise ValueError(f"{name}: no years")
        if not weight > 0:
            raise ValueError(f"{name}: weight must be positive")
        self.weight = float(weight)
        self.data_corr = {y: np.asarray(data_corr[y], dtype=float).ravel() for y in self.years}
        self.x = {y: np.asarray(products_x[y], dtype=float) for y in self.years}
        self.s_data = {y: np.asarray(s_data[y], dtype=float).ravel() for y in self.years}
        K = len(self.chars)
        for y in self.years:
            if self.data_corr[y].size != K or self.x[y].shape != (self.s_data[y].size, K):
                raise ValueError(f"{name}: year {y}: shapes of data_corr {self.data_corr[y].shape}, x {self.x[y].shape}, "
                                 f"shares {self.s_data[y].shape} do not fit {K} characteristics")
        self.keep = {y: np.flatnonzero(np.isfinite(self.data_corr[y])) for y in self.years}
        self.labels = tuple(f"{y} {self.chars[c]}" for y in self.years for c in self.keep[y])
        self.values = np.concatenate([self.data_corr[y][self.keep[y]] for y in self.years])
        self.k = self.values.size
        self.dataset = MicroDataset(f"{name}_survey", 1, market_ids=self.years)
        self.fixed_covariance = np.eye(self.k) / self.weight ** 2
        self._t: dict[Any, int] | None = None
        self._markets: list[MarketIndex] | None = None
        self._n: int | None = None
        self._P: int | None = None

    def bind(self, markets: Sequence[MarketIndex], n_products: int, P: int | None = None) -> None:
        self._markets = list(markets)
        self._n, self._P = int(n_products), (None if P is None else int(P))
        ids = [m.market_id for m in self._markets]
        missing = [y for y in self.years if y not in ids]
        if missing:
            raise ValueError(f"{self.name}: years {missing} are not markets of the model")
        self._t = {y: ids.index(y) for y in self.years}
        for y in self.years:
            J = np.size(self._markets[self._t[y]].rows)
            if self.s_data[y].size != J:
                raise ValueError(f"{self.name}: year {y} has {self.s_data[y].size} products, the model market {J}")

    def requests(self) -> KernelRequests:
        return KernelRequests(second_choice=True, second_choice_markets=self.years)

    def covariance_requests(self) -> KernelRequests:
        return self.requests()

    def correlations(self, sol: InnerSolution, need_jacobian: bool
                     ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """(model correlations of the kept rows (k,), d_delta (k, n), d_theta (k, P))."""
        from .second_choice import correlation_adjoint

        if self._t is None or self._markets is None or self._n is None:
            raise RuntimeError(f"{self.name}: bind(...) was not called")
        table = sol.outputs.get(SECOND_CHOICE_OUTPUTS)
        if not isinstance(table, list) or len(table) != len(self._markets):
            raise KeyError(f"{self.name}: the solution carries no per-market {SECOND_CHOICE_OUTPUTS!r} list")
        adj = sol.outputs.get(SECOND_CHOICE_ADJOINT) if need_jacobian else None
        if need_jacobian and not callable(adj):
            raise KeyError(f"{self.name}: the solution carries no {SECOND_CHOICE_ADJOINT!r} callable")
        vals, dds, dts = [], [], []
        for y in self.years:
            t = self._t[y]
            e = table[t]
            if e is None:
                raise KeyError(f"{self.name}: no second choices for market {y}")
            C = np.asarray(e["C"], dtype=float)
            keep = self.keep[y]
            corr, A = correlation_adjoint(self.s_data[y], C, self.x[y])
            vals.append(corr[keep])
            if not need_jacobian:
                continue
            s = np.asarray(e["s"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(s > 0, 1.0 / s, 0.0)
            Ap = A[keep] * inv[None, :, None]
            B = np.einsum("cjk,jk->cj", Ap, C)
            assert callable(adj)
            od, ot = adj(t, Ap, B)
            dd = np.zeros((keep.size, self._n))
            dd[:, self._markets[t].rows] = np.asarray(od, dtype=float)
            dds.append(dd)
            dts.append(np.asarray(ot, dtype=float))
        value = np.concatenate(vals)
        if not need_jacobian:
            return value, None, None
        return value, np.vstack(dds), np.vstack(dts)

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock:
        model, dd, dt = self.correlations(sol, need_jacobian)
        return MomentBlock(name=self.name, value=model - self.values, d_delta=dd, d_theta=dt, data=self.values.copy())

    def predicted(self, sol: InnerSolution) -> np.ndarray:
        return self.correlations(sol, False)[0]

    def covariance_block(self, sol: InnerSolution | None, linear: LinearFit | None) -> np.ndarray:
        return self.fixed_covariance.copy()

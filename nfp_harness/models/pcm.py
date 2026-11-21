"""PCM adapter: markets, draws, kernel calls, threads, warm starts (DESIGN 2.2, 3.5, 4.4). Owned by WP7.

Implements ``types.InnerModel`` on ``nfp_harness._native.pcm_kernel`` for the exact pure
characteristics model

    u_ij = delta_j + x_j . t_i - c_i a_i p_j,   u_i0 = 0,
    t_i = sum_m theta_m basis_m(i) e_{column(m)},   log c_i = phi . scale_basis_i,   a_i ~ LogNormal(0, sigma_p),

with a_i truncated below when the layout has a bound: at a_lo for every draw (``PCMSpec.bound_kind = "a"``, the
default, LEARNINGS 32), so alpha_i >= a_lo c_i, or at alpha_lo / c_i (``"alpha"``, LEARNINGS 29), so alpha_i >=
alpha_lo.

Parameters come from the ``ParameterLayout``: role ``"taste"`` (``column`` is a characteristic,
``basis`` an agent column for a demographic interaction, any other name for a standard normal taste
draw, None for 1), role ``"scale"`` (``basis`` is an agent column), exactly one ``"sigma_p"`` and at most
one ``"bound"``: the lower bound a_lo on a_i (or alpha_lo on alpha_i = c_i a_i, by ``PCMSpec.bound_kind``), a
fixed parameter (``free=False``; it is never estimated, its column of ``D`` and of the micro ``d_theta`` is zero). The layout order is free; the
kernel order [taste, scale, sigma_p] is mapped internally, so ``D`` and the micro ``d_theta`` are returned
in layout order.

Draws: each market gets ``R_t = 2^ceil(log2(A_t nu_per_agent))`` scrambled Sobol standard normal
points over the taste draw dimensions (seed ``seed + t`` for the t-th market in sorted id order),
split into consecutive blocks over its A_t agents (``data.cross_agents``), so each agent keeps its
weight (DESIGN 9, decisions 3 and 6). ``PCMSpec.draws_per_market`` overrides R_t. With ``antithetic`` (taste draw
dimensions by name, or True for all) the R_t points are R_t / 2 Sobol points each followed by its antithetic partner
(``data.sobol_normal_draws``); the default (none) leaves the draws unchanged. With ``taste_nodes`` (a quadrature rule
over the taste draw dimensions) every agent is instead crossed with every node, agent-major, and a draw's weight is the
agent's weight times the node's (Nevo's Gauss-Hermite grid). a_i has no draws: the kernel integrates its lognormal in
closed form.

Products are sorted by price within each market (the kernel needs strictly increasing prices;
spread ties first with ``data.break_price_ties``). ``MarketIndex.rows`` holds the product rows in that
price order, and every per-market array (``D_t``, micro ``d_delta``, ``ds_dp``) follows it.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import cross_agents, sobol_normal_draws
from ..moments import GROUP_OUTPUTS, MICRO_OUTPUTS, SECOND_CHOICE_ADJOINT, SECOND_CHOICE_OUTPUTS
from ..types import EvalStatus, InnerSolution, KernelRequests, MarketIndex, MicroPartRequest, StepLimit

__all__ = ["PCMModel", "PCMSpec", "allocate_threads", "kernel_status", "group_sums_oracle"]

log = logging.getLogger("nfp_harness.models.pcm")

#: Kernel status string -> harness status. Bad kernel input at a theta inside the layout bounds is
#: a theta the model cannot evaluate (e.g. a non-finite value), so it is OUT_OF_DOMAIN.
_KERNEL_STATUS = {
    "ok": EvalStatus.OK,
    "not_converged": EvalStatus.INVERSION_FAILED,
    "infeasible_start": EvalStatus.INVERSION_FAILED,
    "singular_ds_ddelta": EvalStatus.SINGULAR_INNER_JACOBIAN,
    "invalid_input": EvalStatus.OUT_OF_DOMAIN,
}
# The status of an evaluation is the first of these found among its markets.
_PRIORITY = (EvalStatus.OUT_OF_DOMAIN, EvalStatus.NUMERICAL, EvalStatus.INVERSION_FAILED,
             EvalStatus.SINGULAR_INNER_JACOBIAN)


def kernel_status(name: str) -> EvalStatus:
    """kernel_status(name) -> EvalStatus for a ``pcm_kernel.solve_market`` status string."""
    try:
        return _KERNEL_STATUS[name]
    except KeyError:
        raise ValueError(f"unknown kernel status {name!r}") from None


def allocate_threads(weights: np.ndarray, total: int) -> np.ndarray:
    """allocate_threads(weights, total) -> (T,) threads per market.

    ``n_t = max(1, floor(total w_t / sum w))``, leftover threads one each to the markets with the
    largest weight; if the floor of 1 makes the sum exceed ``total``, the excess comes off the
    largest-weight markets (never below 1). First evaluation weights ``J_t^2 R_t``; afterwards an exponential moving average
    (factor 0.5) of ``seconds_t * n_threads_t`` (DESIGN 3.5). Non-finite or negative weights count as
    zero; all-zero weights mean equal weights.
    """
    w = np.asarray(weights, dtype=float).ravel()
    total = int(total)
    if total < 1:
        raise ValueError(f"total threads must be at least 1, got {total}")
    if w.size == 0:
        return np.zeros(0, dtype=int)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if not w.sum() > 0:
        w = np.ones_like(w)
    n = np.maximum(1, np.floor(total * w / w.sum()).astype(int))
    order = np.argsort(-w, kind="stable")
    left = total - int(n.sum())
    if left > 0:
        for i in range(left):
            n[order[i % w.size]] += 1
    while left < 0 and np.any(n > 1):  # the floor of 1 overshot: take the excess from the largest
        k = order[np.argmax(n[order] > 1)]
        n[k] -= 1
        left += 1
    return n


def group_sums_oracle(kernel: Any, delta: np.ndarray, km: Any, theta: np.ndarray, phi: np.ndarray, sigma_p: float,
                      group_of_draw: np.ndarray, n_groups: int, U: np.ndarray, derivatives: bool = True,
                      n_threads: int = 1, **bound: float) -> dict[str, Any]:
    """Per-household purchase-weighted sums by the existing kernel (GMY spec WP-G3 item 6; tests only, O(G) calls).

    For each group g, a one-household ``Market`` of its draws (weights w_i) and ``evaluate_market`` with the micro
    request H = 1, U' = [1, U]: ``S`` (G, C+1) with column 0 = P_g, ``d_delta`` (G, C+1, J) and ``d_theta``
    (G, C+1, P_kernel) (kernel order; None without derivatives), ``w`` (G,).
    """
    x, p = np.asarray(km.x), np.asarray(km.p)
    w, tb, sb = np.asarray(km.weights), np.asarray(km.taste_basis), np.asarray(km.scale_basis)
    cols = list(km.taste_col)
    g_of = np.asarray(group_of_draw, dtype=int)
    Up = np.column_stack([np.ones(x.shape[0]), np.asarray(U, dtype=float)])
    C1 = Up.shape[1]
    J, P = x.shape[0], np.size(theta) + np.size(phi) + 1
    S = np.zeros((n_groups, C1))
    dD = np.zeros((n_groups, C1, J)) if derivatives else None
    dT = np.zeros((n_groups, C1, P)) if derivatives else None
    wg = np.zeros(n_groups)
    for g in range(n_groups):
        idx = np.flatnonzero(g_of == g)
        wg[g] = w[idx].sum()
        mk = kernel.Market(x, p, w[idx], tb[idx], cols, sb[idx])
        r = kernel.evaluate_market(delta, mk, theta, phi, sigma_p,
                                   {"H": np.ones((idx.size, 1)), "U": Up, "h_of_part": [0] * C1},
                                   derivatives=derivatives, n_threads=n_threads, **bound)
        S[g] = r["micro"]
        if derivatives:
            assert dD is not None and dT is not None
            dD[g] = r["micro_d_delta"]
            dT[g] = r["micro_d_theta"]
    return {"S": S, "d_delta": dD, "d_theta": dT, "w": wg}


@dataclass(frozen=True)
class PCMSpec:
    """What the PCM adapter needs from a paper's spec besides the parameter layout.

    ``characteristics`` are the product columns of x in order; a column named ``"const"`` that the
    product table lacks is a column of ones. The solve settings are passed to
    ``pcm_kernel.solve_market`` (DESIGN 3.2). ``track_realised_mass`` computes the realised affected
    mass of every trial point against the accepted point for the guard of DESIGN 4.4.
    ``max_step_tau`` is the ``tau_max`` of the events kernel: it is 2 so that a direction with no
    vanishing product before tau = 2 leaves the optimizer's ``0.5 tau_vanish_lb`` cap at a full step.
    ``trial_cold_evals`` is the cold-stage budget of a market that has an accepted warm start (trial points
    after the first evaluation); None uses ``cold_evals``. A trial point whose warm stage and theta path
    both fail is usually one the optimizer rejects anyway, and a 40,000-evaluation cold stage there can
    take longer than a whole accepted step. ``bound_kind`` says what the layout's bound parameter bounds: ``"a"``
    the unobserved factor a_i (kernel keyword ``a_lo``) or ``"alpha"`` alpha_i itself (kernel keyword ``alpha_lo``).
    """

    characteristics: tuple[str, ...]
    market_column: str = "market_ids"
    price_column: str = "prices"
    share_column: str = "shares"
    agent_weight_column: str = "weights"
    draws_per_market: int | None = None
    tol: float = 1e-9
    warm_evals: int = 2500
    cold_evals: int = 40000
    path_evals: int = 1500
    path_seconds: float = 90.0
    rcond_min: float = 1e-13
    track_realised_mass: bool = True
    max_step_tau: float = 2.0
    trial_cold_evals: int | None = None
    bound_kind: str = "a"

    @classmethod
    def coerce(cls, spec: Any) -> PCMSpec:
        """A PCMSpec from a PCMSpec, a mapping of its fields, or any object carrying them as attributes."""
        if isinstance(spec, PCMSpec):
            return spec
        names = [f.name for f in dataclasses.fields(cls)]
        if isinstance(spec, Mapping):
            kw = {k: spec[k] for k in names if k in spec}
        else:
            kw = {k: getattr(spec, k) for k in names if hasattr(spec, k)}
        if "characteristics" not in kw:
            raise ValueError("PCM spec needs 'characteristics' (the product columns of x)")
        kw["characteristics"] = tuple(kw["characteristics"])
        return cls(**kw)


def _has(frame: Any, name: str) -> bool:
    try:
        return name in frame
    except TypeError:
        return False


def _col(frame: Any, name: str, what: str) -> np.ndarray:
    if not _has(frame, name):
        raise KeyError(f"{what} table has no column {name!r}")
    return np.asarray(frame[name], dtype=float).ravel()


def _frames(data: Any) -> tuple[Any, Any]:
    if isinstance(data, Mapping):
        return data["products"], data.get("agents")
    return data.products, getattr(data, "agents", None)


@dataclass(eq=False)
class _State:
    """Per-market warm start: delta (price order) and the stacked kernel theta it was solved at."""

    delta: np.ndarray
    theta: np.ndarray  # stacked [taste, scale, sigma_p]
    signature: np.ndarray | None = None  # frontier signature at (delta, theta), computed on demand
    a_lo: float = 0.0  # the bound the state was solved under (of the model's bound_kind)


@dataclass(eq=False)
class _Market:
    t: int
    market_id: Any
    rows: np.ndarray  # product rows in price order
    s_data: np.ndarray
    kernel: Any  # pcm_kernel.Market
    agent_rows: np.ndarray  # rows of the agent table, or empty without one
    agent_of_draw: np.ndarray  # (R,) agent position within the market
    weights: np.ndarray  # (R,)
    J: int
    R: int
    micro_cache: dict[Any, tuple[dict[str, Any], np.ndarray, float]]


class PCMModel:
    """Exact PCM inner model.

    Markets run in a ``ThreadPoolExecutor`` with one Python thread per market; each ``solve_market``
    call starts with its OpenMP team size from ``allocate_threads`` and holds a ``pcm_kernel.ThreadBudget``: when a
    market finishes, its threads go to the markets still running (``_rebalance``). Keeps an accepted state (per market
    delta and the theta it was solved at) and a trial state: ``solve`` warm-starts from the accepted
    state and writes the trial state; ``commit`` promotes it; ``restore`` discards it (F3).
    ``max_step`` uses ``pcm_kernel.max_step``; trial points are checked with ``frontier_signature`` and
    ``affected_mass`` (DESIGN 4.4) and the result is ``InnerSolution.outputs["realised_mass"]``.

    ``InnerSolution.outputs`` holds ``"micro_parts"`` (the contract at the top of ``moments.py``),
    ``"theta"`` (the layout theta solved at), ``"shares"`` ((n,) model shares in row order),
    ``"ds_dp"`` (per-market (J_t, J_t) in price order, only when requested) and ``"realised_mass"``
    (when an accepted state exists and ``track_realised_mass`` is on).
    """

    markets: list[MarketIndex]
    n_products: int

    def __init__(self, spec: Any, data: Any, layout: Any, nu_per_agent: int, seed: int,
                 total_threads: int | None = None, antithetic: Sequence[str] | bool = (),
                 cold_draws: int = 0, taste_nodes: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """PCMModel(spec, data, layout, nu_per_agent, seed, total_threads=None, antithetic=(), cold_draws=0,
        taste_nodes=None). None uses os.cpu_count().

        ``taste_nodes``: (nodes (R_q, dim), weights (R_q,)), a quadrature rule over the taste draw dimensions in the
        order they first appear in the layout. Each market then has R_t = A_t R_q draws: agent a's draws are rows
        a R_q ... (a + 1) R_q - 1, node k with weight w_a w_k. ``nu_per_agent``, ``seed``, ``antithetic`` and
        ``PCMSpec.draws_per_market`` do not apply. None (the default) keeps the Sobol draws. Not part of ``PCMSpec``,
        so spec hashes do not move.

        ``cold_draws``: the kernel's draw homotopy for a market with no warm start (``SolveOptions::cold_draws``):
        invert first on every k-th draw, at least ``cold_draws`` of them, then warm-start each finer level up to all
        draws. 0 (the default) is the plain cold stage. Not part of ``PCMSpec``, so spec hashes do not move.

        ``antithetic``: taste draw dimensions (the layout's taste ``basis`` names that are not agent columns, e.g.
        "nu0") whose draws come in antithetic pairs, or True for every one; off by default. Not part of ``PCMSpec``,
        so spec hashes do not move.

        ``spec`` is a ``PCMSpec`` (or a mapping / object with its fields). ``data`` is a mapping or object
        with ``products`` (market, price, share and characteristic columns) and optionally ``agents``
        (market, weight and demographic columns; without it each market has one agent of weight 1).
        Tables are pandas DataFrames or dicts of arrays; rows are positional. Resolves the layout's
        automatic natural units from the product characteristics and the pooled agent columns.
        """
        from .._native import pcm_kernel

        self._k = pcm_kernel
        self.spec = PCMSpec.coerce(spec)
        self.layout = layout
        self.nu_per_agent = int(nu_per_agent)
        self.seed = int(seed)
        self.total_threads = int(total_threads or os.cpu_count() or 1)
        if self.nu_per_agent < 1:
            raise ValueError(f"nu_per_agent must be at least 1, got {nu_per_agent}")
        if int(cold_draws) < 0:
            raise ValueError(f"cold_draws must be non-negative, got {cold_draws}")
        #: draw homotopy of the kernel's cold stage (0: off)
        self.cold_draws = int(cold_draws)
        sp = self.spec
        products, agents = _frames(data)
        self.products, self.agents = products, agents

        # ------------------------------------------------------------ parameter mapping
        specs = list(layout.specs)
        self.P = len(specs)
        taste = [k for k, s in enumerate(specs) if s.role == "taste"]
        scale = [k for k, s in enumerate(specs) if s.role == "scale"]
        sigp = [k for k, s in enumerate(specs) if s.role == "sigma_p"]
        bound = [k for k, s in enumerate(specs) if s.role == "bound"]
        other = [s.name for s in specs if s.role not in ("taste", "scale", "sigma_p", "bound")]
        if other:
            raise ValueError(f"PCM parameters must have role taste, scale, sigma_p or bound: {other}")
        if len(sigp) != 1:
            raise ValueError(f"PCM needs exactly one sigma_p parameter, got {len(sigp)}")
        if len(bound) > 1:
            raise ValueError(f"PCM takes at most one bound (a_lo) parameter, got {len(bound)}")
        for k in bound:
            if specs[k].free or specs[k].lower < 0:
                raise ValueError(f"{specs[k].name}: the bound a_lo is a fixed, non-negative parameter "
                                 f"(free=False, lower >= 0)")
        if sp.bound_kind not in ("a", "alpha"):
            raise ValueError(f"PCMSpec.bound_kind must be 'a' or 'alpha', got {sp.bound_kind!r}")
        #: what the bound parameter bounds: "a" (a_i, kernel keyword a_lo) or "alpha" (alpha_i, keyword alpha_lo)
        self.bound_kind = sp.bound_kind
        self._taste, self._scale, self._sigp = taste, scale, sigp[0]
        self._bound: int | None = bound[0] if bound else None
        #: layout columns the kernel does not differentiate (the bound): zero in D and the micro d_theta
        self.bound_columns: tuple[int, ...] = tuple(bound)
        self._order = np.array(taste + scale + sigp, dtype=int)  # kernel column c is layout index _order[c]
        self.P_kernel = int(self._order.size)
        agent_cols = set() if agents is None else {c for c in (agents.columns if hasattr(agents, "columns")
                                                              else agents.keys())}
        chars = list(sp.characteristics)
        self.taste_col: list[int] = []
        self.nu_names: list[str] = []
        for k in taste:
            s = specs[k]
            if s.column not in chars:
                raise ValueError(f"{s.name}: column {s.column!r} is not among the characteristics {chars}")
            self.taste_col.append(chars.index(s.column))
            if s.basis is not None and s.basis not in agent_cols and s.basis not in self.nu_names:
                self.nu_names.append(s.basis)
        for k in scale:
            s = specs[k]
            if s.basis is None or s.basis not in agent_cols:
                raise ValueError(f"{s.name}: scale basis {s.basis!r} must be a column of the agent table")
        anti = list(self.nu_names) if antithetic is True else list(antithetic or ())
        unknown = [a for a in anti if a not in self.nu_names]
        if unknown:
            raise ValueError(f"antithetic: {unknown} are not taste draw dimensions {self.nu_names}")
        #: taste draw dimensions drawn in antithetic pairs (empty: plain scrambled Sobol)
        self.antithetic: tuple[str, ...] = tuple(a for a in self.nu_names if a in anti)
        anti_cols = [self.nu_names.index(a) for a in self.antithetic]
        q_nodes, q_w = np.zeros((0, 0)), np.zeros(0)
        if taste_nodes is not None:
            q_nodes = np.asarray(taste_nodes[0], dtype=float)
            q_nodes = q_nodes[:, None] if q_nodes.ndim == 1 else q_nodes
            q_w = np.asarray(taste_nodes[1], dtype=float).ravel()
            if q_nodes.shape != (q_w.size, len(self.nu_names)) or self.antithetic:
                raise ValueError(f"taste_nodes: nodes {q_nodes.shape} and weights {q_w.shape} must cover the taste "
                                 f"draw dimensions {self.nu_names} (and exclude antithetic draws)")
        #: the quadrature rule of the taste draws (None: scrambled Sobol)
        self.taste_nodes = None if taste_nodes is None else (q_nodes, q_w)

        # ------------------------------------------------------------ products
        mkt = np.asarray(products[sp.market_column]).ravel()
        prices = _col(products, sp.price_column, "product")
        shares = _col(products, sp.share_column, "product")
        self.n_products = mkt.size
        x_all = np.column_stack([
            np.ones(self.n_products) if (c == "const" and not _has(products, c)) else _col(products, c, "product")
            for c in chars]) if chars else np.zeros((self.n_products, 0))
        amkt = None if agents is None else np.asarray(agents[sp.market_column]).ravel()
        aw = None if agents is None else _col(agents, sp.agent_weight_column, "agent")

        self._agent_col_cache: dict[str, np.ndarray] = {}
        self._mk: list[_Market] = []
        self.markets = []
        for t, mid in enumerate(np.unique(mkt)):
            pos = np.flatnonzero(mkt == mid)
            rows = pos[np.argsort(prices[pos], kind="stable")]
            p = prices[rows]
            if np.any(np.diff(p) <= 0):
                raise ValueError(f"market {mid!r}: prices are not strictly increasing after sorting; spread ties "
                                 f"with nfp_harness.data.break_price_ties first")
            if agents is None:
                arows = np.zeros(0, dtype=int)
                w_agent = np.ones(1)
            else:
                arows = np.flatnonzero(amkt == mid)
                if arows.size == 0:
                    raise ValueError(f"market {mid!r} has no agents")
                assert aw is not None
                w_agent = aw[arows]
            A = w_agent.size
            if self.taste_nodes is not None:
                R = A * q_w.size
                idx = np.repeat(np.arange(A), q_w.size)
                w = np.repeat(w_agent, q_w.size) * np.tile(q_w, A)
                nu = np.tile(q_nodes, (A, 1))
            else:
                R = sp.draws_per_market or 2 ** int(math.ceil(math.log2(A * self.nu_per_agent)))
                if R < A:
                    raise ValueError(f"market {mid!r}: {R} draws for {A} agents")
                nu = (sobol_normal_draws(R, len(self.nu_names), self.seed + t, antithetic=anti_cols)
                      if self.nu_names else np.zeros((R, 0)))
                w, idx, nu = cross_agents(w_agent, np.arange(A, dtype=float)[:, None], nu)
                idx = idx[:, 0].astype(int)
            basis = np.empty((R, len(taste)))
            for m, k in enumerate(taste):
                b = specs[k].basis
                if b is None:
                    basis[:, m] = 1.0
                elif b in agent_cols:
                    basis[:, m] = self._agent_col(b)[arows][idx]
                else:
                    basis[:, m] = nu[:, self.nu_names.index(b)]
            sbasis = np.column_stack([self._agent_col(specs[k].basis)[arows][idx] for k in scale]) if scale \
                else np.zeros((R, 0))
            km = pcm_kernel.Market(np.ascontiguousarray(x_all[rows]), p, w, basis, self.taste_col, sbasis)
            self._mk.append(_Market(t=t, market_id=mid, rows=rows, s_data=shares[rows], kernel=km,
                                    agent_rows=arows, agent_of_draw=idx, weights=w, J=rows.size, R=R,
                                    micro_cache={}))
            self.markets.append(MarketIndex(market_id=mid, rows=rows))
        self.T = len(self._mk)

        # ------------------------------------------------------------ units, state, threads
        if any(s.unit is None for s in specs):
            unit_data: dict[str, np.ndarray] = {c: x_all[:, i] for i, c in enumerate(chars)}
            for s in specs:
                if s.basis is not None and s.basis in agent_cols:
                    unit_data[s.basis] = self._agent_col(s.basis)
            layout.units(unit_data)
        self._accepted: list[_State | None] = [None] * self.T
        self._trial: list[_State | None] = [None] * self.T
        self._weights = np.array([float(m.J) ** 2 * m.R for m in self._mk])
        self._measured = False
        self.last_threads = np.zeros(self.T, dtype=int)
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.T), thread_name_prefix="pcm-market")
        self._lock = threading.RLock()
        #: how per-household sums are computed: "kernel" (pcm_kernel.group_sums) or "oracle" (group_sums_oracle,
        #: one evaluate_market call per household; tests only). Not part of PCMSpec, so spec hashes do not move.
        self.group_sums_backend = "kernel" if hasattr(pcm_kernel, "group_sums") else "oracle"
        self._group_U: dict[tuple[str, ...], list[np.ndarray]] = {}

    # ---------------------------------------------------------------- helpers

    def _agent_col(self, name: str) -> np.ndarray:
        col = self._agent_col_cache.get(name)
        if col is None:
            if self.agents is None:
                raise KeyError(f"agent column {name!r} requested but the model has no agent table")
            col = self._agent_col_cache[name] = _col(self.agents, name, "agent")
        return col

    def _product_col(self, name: str) -> np.ndarray:
        if name == "const" and not _has(self.products, name):
            return np.ones(self.n_products)
        return _col(self.products, name, "product")

    def kernel_params(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """kernel_params(theta) -> (taste (M,), phi (Q,), sigma_p) for a layout-order theta."""
        theta = np.asarray(theta, dtype=float).ravel()
        return theta[self._taste].copy(), theta[self._scale].copy(), float(theta[self._sigp])

    def a_lo(self, theta: np.ndarray) -> float:
        """a_lo(theta) -> the bound parameter's value at a layout-order theta (0 without a bound parameter): the
        lower bound on a_i under ``bound_kind`` "a", on alpha_i under "alpha"."""
        if self._bound is None:
            return 0.0
        return float(np.asarray(theta, dtype=float).ravel()[self._bound])

    def bound_kwargs(self, value: float) -> dict[str, float]:
        """The kernel keyword for a bound of ``value``: ``{"a_lo": value}`` or ``{"alpha_lo": value}``."""
        return {"a_lo" if self.bound_kind == "a" else "alpha_lo": float(value)}

    def draw_a_lo(self, theta: np.ndarray, c: np.ndarray) -> np.ndarray:
        """(R,) truncation point of a_i for draws with price scales ``c``: a_lo, or alpha_lo / c_i."""
        v = self.a_lo(theta)
        c = np.asarray(c, dtype=float)
        return np.full(c.shape, v) if self.bound_kind == "a" else v / c

    def kernel_kwargs(self, theta: np.ndarray) -> dict[str, float]:
        """Keyword arguments every kernel call takes besides (theta, phi, sigma_p): ``{"a_lo": ...}`` (or
        ``{"alpha_lo": ...}`` under ``bound_kind`` "alpha")."""
        return self.bound_kwargs(self.a_lo(theta))

    def to_kernel_order(self, v: np.ndarray) -> np.ndarray:
        """Layout-order (P,) vector -> stacked kernel order [taste, scale, sigma_p] (the bound is dropped)."""
        return np.asarray(v, dtype=float)[..., self._order]

    def from_kernel_order(self, a: np.ndarray) -> np.ndarray:
        """Array whose last axis is in kernel order -> layout order (zero in the bound's column)."""
        a = np.asarray(a, dtype=float)
        out = np.zeros(a.shape[:-1] + (self.P,))
        out[..., self._order] = a
        return out

    def kernel_market(self, t: int) -> Any:
        """The ``pcm_kernel.Market`` of market t (products in price order, ``markets[t].rows``)."""
        return self._mk[t].kernel

    def market_draws(self, t: int) -> dict[str, np.ndarray]:
        """Draw-level arrays of market t: weights, agent_of_draw (position in the market), agent_rows."""
        m = self._mk[t]
        return {"weights": m.weights, "agent_of_draw": m.agent_of_draw, "agent_rows": m.agent_rows}

    def _group_columns(self, m: _Market, cols: tuple[str, ...]) -> np.ndarray:
        """(J_t, C) product columns ``cols`` of market m in price order (cached)."""
        U = self._group_U.get(cols)
        if U is None:
            full = np.column_stack([self._product_col(c) for c in cols]) if cols else np.zeros((self.n_products, 0))
            U = self._group_U[cols] = [full[mk.rows] for mk in self._mk]
        return U[m.t]

    def group_sums(self, t: int, delta_t: np.ndarray, theta: np.ndarray, cols: tuple[str, ...], derivatives: bool,
                   n_threads: int = 1) -> dict[str, Any]:
        """Per-household sums of market t at the price-order ``delta_t`` (contract of ``moments.GroupMicroMoments``):
        ``columns``, ``S`` (G_t, C+1), ``d_delta`` (G_t, C+1, J_t), ``d_theta`` (G_t, C+1, P) in layout order, ``w``."""
        m = self._mk[t]
        th, ph, sp = self.kernel_params(theta)
        kw = self.kernel_kwargs(theta)
        U = self._group_columns(m, tuple(cols))
        G = int(m.agent_of_draw.max()) + 1
        delta_t = np.asarray(delta_t, dtype=float)
        if self.group_sums_backend == "oracle":
            r = group_sums_oracle(self._k, delta_t, m.kernel, th, ph, sp, m.agent_of_draw, G, U, derivatives,
                                  n_threads, **kw)
            dD, dT = r["d_delta"], r["d_theta"]
        else:
            r = self._k.group_sums(delta_t, m.kernel, th, ph, sp, m.agent_of_draw, G, U, derivatives=derivatives,
                                   n_threads=int(n_threads), **kw)
            dD = np.stack([np.asarray(a) for a in r["d_delta"]]) if derivatives else None
            dT = np.stack([np.asarray(a) for a in r["d_theta"]]) if derivatives else None
        return {"columns": tuple(cols), "S": np.asarray(r["S"], dtype=float),
                "d_delta": None if dD is None else np.asarray(dD, dtype=float),
                "d_theta": None if dT is None else self.from_kernel_order(np.asarray(dT, dtype=float)),
                "w": np.asarray(r["w"], dtype=float)}

    def _micro_for(self, m: _Market, parts: tuple[MicroPartRequest, ...]) -> tuple[dict[str, Any], np.ndarray, float]:
        cached = m.micro_cache.get(parts)
        if cached is not None:
            return cached
        h_keys: list[tuple[str, ...]] = []
        for p in parts:
            if tuple(p.agent_columns) not in h_keys:
                h_keys.append(tuple(p.agent_columns))
        H = np.ones((m.R, len(h_keys)))
        for col, key in enumerate(h_keys):
            for c in key:
                H[:, col] *= self._agent_col(c)[m.agent_rows][m.agent_of_draw]
        U = np.empty((m.J, len(parts)))
        for a, p in enumerate(parts):
            v = np.ones(m.J)
            for c in p.product_columns:
                v = v * self._product_col(c)[m.rows]
            U[:, a] = v - p.outside_value
        h_of_part = [h_keys.index(tuple(p.agent_columns)) for p in parts]
        h_sum = (m.weights @ H)[h_of_part]
        out = ({"H": H, "U": U, "h_of_part": h_of_part}, h_sum, float(m.weights.sum()))
        m.micro_cache[parts] = out
        return out

    # ---------------------------------------------------------------- solve

    def _solve_one(self, m: _Market, th: np.ndarray, ph: np.ndarray, sp: float, parts: tuple[MicroPartRequest, ...],
                   need_jacobian: bool, price_derivatives: bool, n_threads: int, a_lo: float,
                   requests: KernelRequests | None = None, theta: np.ndarray | None = None,
                   budget: Any = None) -> dict[str, Any]:
        r = self._solve_kernel(m, th, ph, sp, parts, need_jacobian, price_derivatives, n_threads, a_lo, budget)
        if budget is not None:
            n_threads = int(budget.value)   # the threads this market holds now (finished markets hand theirs on)
        if requests is not None and theta is not None and r["status"] == "ok":
            if requests.group_sums:
                r["group"] = self.group_sums(m.t, r["delta"], theta, tuple(requests.group_sums), need_jacobian,
                                             n_threads)
            if requests.second_choice and m.market_id in tuple(requests.second_choice_markets):
                r["second"] = self.second_choice(m.t, r["delta"], theta, n_threads=n_threads)
        return r

    def second_choice(self, t: int, delta_t: np.ndarray, theta: np.ndarray, adjoint: dict[str, Any] | None = None,
                      n_threads: int = 1) -> dict[str, Any]:
        """``pcm_kernel.second_choice`` on market t at the price-order ``delta_t``: C, s (and with ``adjoint`` =
        {"A": (K, J, J), "B": (K, J)} the contraction ``out_delta`` (K, J) and ``out_theta`` (K, P) in layout order)."""
        th, ph, sp = self.kernel_params(theta)
        r = self._k.second_choice(np.asarray(delta_t, dtype=float), self._mk[t].kernel, th, ph, sp,
                                  None if adjoint is None else {"A": list(np.asarray(adjoint["A"], dtype=float)),
                                                                "B": np.asarray(adjoint["B"], dtype=float)},
                                  n_threads=int(n_threads), **self.kernel_kwargs(theta))
        out = {"C": np.asarray(r["C"]), "s": np.asarray(r["s"]),
               "draws_with_no_inside_second": int(r["draws_with_no_inside_second"]), "seconds": float(r["seconds"])}
        if adjoint is not None:
            out["out_delta"] = np.asarray(r["out_delta"])
            out["out_theta"] = self.from_kernel_order(np.asarray(r["out_theta"]))
        return out

    def _solve_kernel(self, m: _Market, th: np.ndarray, ph: np.ndarray, sp: float,
                      parts: tuple[MicroPartRequest, ...], need_jacobian: bool, price_derivatives: bool,
                      n_threads: int, a_lo: float, budget: Any = None) -> dict[str, Any]:
        acc = self._accepted[m.t]
        # A state solved under another bound is a warm START: the theta path moves the bound from its a_lo to this
        # one, and a failed path still leaves the full cold stage (with the draw homotopy when it is on).
        other_bound = acc is not None and acc.a_lo != a_lo
        micro = self._micro_for(m, parts)[0] if parts else None
        s = self.spec
        cold = s.cold_evals if acc is None or other_bound or s.trial_cold_evals is None else s.trial_cold_evals
        extra: dict[str, Any] = {}
        if self.cold_draws > 0:
            extra["cold_draws"] = self.cold_draws
        if budget is not None:
            extra["thread_budget"] = budget
        if other_bound:
            assert acc is not None
            extra["lo_warm"] = float(acc.a_lo)
        return self._k.solve_market(
            m.s_data, m.kernel, th, ph, sp,
            None if acc is None else acc.delta, None if acc is None else acc.theta, micro,
            tol=s.tol, warm_evals=s.warm_evals, cold_evals=int(cold), path_evals=s.path_evals,
            path_seconds=s.path_seconds, n_threads=int(n_threads), derivatives=bool(need_jacobian),
            price_derivatives=bool(price_derivatives), rcond_min=s.rcond_min, **self.bound_kwargs(a_lo), **extra)

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        """solve(theta, requests, need_jacobian) -> InnerSolution. Never raises for model failures."""
        with self._lock:
            return self._solve(theta, requests, need_jacobian)

    def _rebalance(self, futures: list[Any], budgets: list[Any]) -> None:
        """As markets finish, give all ``total_threads`` to the markets still running, by ``allocate_threads`` on
        their weights (largest weight first), so the last market does not run on one thread while the rest idle."""
        index = {f: t for t, f in enumerate(futures)}
        running = set(range(self.T))
        for f in as_completed(futures):
            running.discard(index[f])
            if not running:
                break
            live = sorted(running)
            share = allocate_threads(self._weights[live], max(self.total_threads, len(live)))
            for t, n in zip(live, share):
                budgets[t].value = int(n)

    def _solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        theta = np.array(theta, dtype=float).ravel()
        if theta.shape != (self.P,):
            raise ValueError(f"theta has shape {theta.shape}, expected ({self.P},)")
        th, ph, sp = self.kernel_params(theta)
        al = self.a_lo(theta)
        stacked = self.to_kernel_order(theta)
        parts = tuple(requests.micro_parts)
        threads = allocate_threads(self._weights, self.total_threads)
        self.last_threads = threads
        log.info("PCM threads per market: %s", dict(zip([str(m.market_id) for m in self._mk], threads.tolist())))
        # Each market holds a ThreadBudget the kernel reads before every share evaluation: when a market finishes,
        # its threads go to the markets still running, by their weights (results do not depend on thread counts).
        budgets = ([self._k.ThreadBudget(int(n)) for n in threads] if hasattr(self._k, "ThreadBudget")
                   else [None] * self.T)
        futures = [self._pool.submit(self._solve_one, m, th, ph, sp, parts, need_jacobian,
                                     requests.price_derivatives, int(threads[m.t]), al, requests, theta,
                                     budgets[m.t])
                   for m in self._mk]
        if budgets[0] is not None:
            self._rebalance(futures, budgets)
        results = [f.result() for f in futures]

        delta = np.full(self.n_products, np.nan)
        shares = np.full(self.n_products, np.nan)
        D: list[np.ndarray] | None = [] if need_jacobian else None
        per_market: list[dict[str, Any]] = []
        statuses: list[EvalStatus] = []
        table: dict[str, list[dict[str, Any] | None]] = {p.name: [] for p in parts}
        ds_dp: list[np.ndarray] = []
        cpu = np.empty(self.T)
        for m, r in zip(self._mk, results):
            st = kernel_status(r["status"])
            delta[m.rows] = r["delta"] if np.size(r["delta"]) == m.J else np.nan
            if np.size(r["s"]) == m.J:
                shares[m.rows] = r["s"]
            if need_jacobian:
                dd = np.asarray(r["ddelta_dtheta"])
                D_t = self.from_kernel_order(dd) if dd.shape == (m.J, self.P_kernel) else np.full((m.J, self.P), np.nan)
                if st is EvalStatus.OK and not np.all(np.isfinite(D_t)):
                    st = EvalStatus.NUMERICAL
                assert D is not None
                D.append(D_t)
            if st is EvalStatus.OK and not np.all(np.isfinite(r["delta"])):
                st = EvalStatus.NUMERICAL
            statuses.append(st)
            per_market.append({
                "market_id": m.market_id, "status": st.value, "kernel_status": r["status"],
                "converged": r["status"] in ("ok", "singular_ds_ddelta"), "residual": float(r["max_abs_residual"]),
                "evaluations": int(r["share_evaluations"]), "seconds": float(r["seconds"]),
                "threads": int(threads[m.t]), "stages": r["stages"], "rcond": float(r["rcond"]),
                "products": m.J, "draws": m.R,
            })
            cpu[m.t] = float(r["seconds"]) * int(threads[m.t])
            if parts:
                _, h_sum, w_sum = self._micro_for(m, parts)
                mic = np.asarray(r["micro"])
                md = np.asarray(r["micro_d_delta"])
                mt = np.asarray(r["micro_d_theta"])
                jac = need_jacobian and md.shape == (len(parts), m.J) and mt.shape == (len(parts), self.P_kernel)
                for a, p in enumerate(parts):
                    table[p.name].append({
                        "sum": float(mic[a]) if mic.size == len(parts) else float("nan"),
                        "d_delta": md[a].copy() if jac else None,
                        "d_theta": self.from_kernel_order(mt[a]) if jac else None,
                        "h_sum": float(h_sum[a]), "w_sum": w_sum,
                    })
            if requests.price_derivatives:
                ds_dp.append(np.asarray(r["ds_dp"]))
            self._trial[m.t] = (_State(delta=np.array(r["delta"], dtype=float), theta=stacked.copy(), a_lo=al)
                                if st is EvalStatus.OK else None)

        self._weights = 0.5 * self._weights + 0.5 * cpu if self._measured else cpu
        self._measured = True
        status = EvalStatus.OK
        for cand in _PRIORITY:
            if cand in statuses:
                status = cand
                break
        outputs: dict[str, object] = {"theta": theta.copy(), "shares": shares}
        if parts:
            outputs[MICRO_OUTPUTS] = table
        if requests.price_derivatives:
            outputs["ds_dp"] = ds_dp
        if requests.group_sums:
            outputs[GROUP_OUTPUTS] = [r.get("group") for r in results]
        if requests.second_choice:
            outputs[SECOND_CHOICE_OUTPUTS] = [r.get("second") for r in results]
            solved = [np.asarray(r["delta"], dtype=float) for r in results]
            theta_sc = theta.copy()
            nthr = self.total_threads

            def adjoint(t: int, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                o = self.second_choice(t, solved[t], theta_sc, {"A": A, "B": B}, n_threads=nthr)
                return o["out_delta"], o["out_theta"]

            outputs[SECOND_CHOICE_ADJOINT] = adjoint
        if status is EvalStatus.OK and self.spec.track_realised_mass:
            mass = self._realised_mass()
            if mass is not None:
                outputs["realised_mass"] = mass
        bad = [f"{m.market_id}: {pm['kernel_status']} ({pm['stages']})" for m, pm in zip(self._mk, per_market)
               if pm["status"] != "ok"]
        if bad:
            log.info("PCM solve %s in %d of %d markets: %s", status.value, len(bad), self.T, "; ".join(bad))
        return InnerSolution(delta=delta, D=D, status=status, per_market=per_market, outputs=outputs)

    # ---------------------------------------------------------------- frontier guard (DESIGN 4.4)

    def _signature(self, m: _Market, state: _State, n_threads: int) -> np.ndarray:
        if state.signature is None:
            th, ph, sp = self._split_stacked(state.theta)
            state.signature = np.asarray(self._k.frontier_signature(state.delta, m.kernel, th, ph, sp, n_threads,
                                                                    **self.bound_kwargs(state.a_lo)))
        return state.signature

    def _split_stacked(self, stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        M, Q = len(self._taste), len(self._scale)
        return stacked[:M].copy(), stacked[M:M + Q].copy(), float(stacked[M + Q])

    def _mass_one(self, m: _Market, n_threads: int) -> float:
        acc, tri = self._accepted[m.t], self._trial[m.t]
        assert acc is not None and tri is not None
        changed = self._signature(m, acc, n_threads) != self._signature(m, tri, n_threads)
        if not changed.any():
            return 0.0
        th, ph, sp = self._split_stacked(acc.theta)
        A = np.asarray(self._k.affected_mass(acc.delta, m.kernel, th, ph, sp, changed, n_threads,
                                             **self.bound_kwargs(acc.a_lo)))
        return float(A.max()) if A.size else 0.0

    def _realised_mass(self) -> float | None:
        if any(a is None for a in self._accepted) or any(t is None for t in self._trial):
            return None
        threads = self.last_threads if self.last_threads.size == self.T else allocate_threads(self._weights,
                                                                                               self.total_threads)
        futures = [self._pool.submit(self._mass_one, m, int(threads[m.t])) for m in self._mk]
        return max(f.result() for f in futures)

    def realised_mass(self, theta: np.ndarray) -> float:
        """realised_mass(theta) -> max_j A_j between the trial and the accepted frontier signatures (DESIGN 4.4).

        NaN when there is no accepted state or no trial state at theta.
        """
        with self._lock:
            stacked = self.to_kernel_order(np.asarray(theta, dtype=float).ravel())
            al = self.a_lo(theta)
            if any(t is None or not np.array_equal(t.theta, stacked) or t.a_lo != al for t in self._trial):
                return float("nan")
            mass = self._realised_mass()
            return float("nan") if mass is None else mass

    def frontier_signatures(self, theta: np.ndarray, delta: np.ndarray) -> list[np.ndarray]:
        """frontier_signatures(theta, delta) -> per-market uint64 hash of each draw's frontier at (delta, theta).

        ``delta`` is the (n,) row-order vector. Two points share a draw's hash exactly when that draw
        buys the same products (used to pick kink-free finite-difference steps and by the 4.4 guard).
        """
        th, ph, sp = self.kernel_params(theta)
        kw = self.kernel_kwargs(theta)
        delta = np.asarray(delta, dtype=float)
        threads = allocate_threads(self._weights, self.total_threads)
        futures = [self._pool.submit(self._k.frontier_signature, delta[m.rows], m.kernel, th, ph, sp,
                                     int(threads[m.t]), **kw) for m in self._mk]
        return [np.asarray(f.result()) for f in futures]

    # ---------------------------------------------------------------- max_step (DESIGN 3.3, 4.2)

    def max_step(self, sol: InnerSolution, dtheta: np.ndarray) -> StepLimit:
        """max_step(sol, dtheta) -> StepLimit from the per-market kernel events (DESIGN 3.3, 4.2).

        Linearises delta along the step with ``delta_dot_t = D_t dtheta`` and calls
        ``pcm_kernel.max_step`` per market with ``tau_max = spec.max_step_tau``. ``tau_first`` and
        ``tau_vanish_lb`` are minima over markets; the mass profile is the maximum over markets on the
        common tau grid. ``tau_cap`` is 1: the cap comes from the optimizer's mass rule.
        """
        if sol.D is None:
            raise ValueError("max_step needs a solution with D (solve with need_jacobian=True)")
        dtheta = np.asarray(dtheta, dtype=float).ravel()
        if dtheta.shape != (self.P,):
            raise ValueError(f"dtheta has shape {dtheta.shape}, expected ({self.P},)")
        theta = np.asarray(sol.outputs["theta"], dtype=float)
        th, ph, sp = self.kernel_params(theta)
        kw = self.kernel_kwargs(theta)
        dk = self.to_kernel_order(dtheta)
        tau_max = float(self.spec.max_step_tau)
        threads = allocate_threads(self._weights, self.total_threads)
        futures = [self._pool.submit(self._k.max_step, np.asarray(sol.delta[m.rows], dtype=float),
                                     np.asarray(sol.D[m.t] @ dtheta, dtype=float), m.kernel, th, ph, sp, dk,
                                     tau_max, int(threads[m.t]), **kw) for m in self._mk]
        res = [f.result() for f in futures]
        tau_first = min(float(r["tau_first"]) for r in res)
        tau_vanish = min(float(r["tau_vanish_lb"]) for r in res)
        prof = np.array(res[0]["mass_profile"], dtype=float)
        for r in res[1:]:
            prof[:, 1] = np.maximum(prof[:, 1], np.asarray(r["mass_profile"])[:, 1])
        n_event = sum(int(np.count_nonzero(np.asarray(r["kind_draw"]))) for r in res)
        n_draw = sum(m.R for m in self._mk)
        note = (f"PCM events over {self.T} markets with tau_max {tau_max:g}: tau_first {tau_first:.3g}, "
                f"tau_vanish_lb {tau_vanish:.3g}, {n_event} of {n_draw} draws have an event")
        return StepLimit(tau_cap=1.0, tau_first=tau_first, tau_vanish_lb=tau_vanish, mass_profile=prof, note=note)

    # ---------------------------------------------------------------- warm-start state (DESIGN 2.1, F3)

    def commit(self, theta: np.ndarray) -> None:
        """commit(theta): promote the trial state solved at theta to the accepted state."""
        with self._lock:
            stacked = self.to_kernel_order(np.asarray(theta, dtype=float).ravel())
            al = self.a_lo(theta)
            promoted = 0
            for t, tri in enumerate(self._trial):
                if tri is not None and np.array_equal(tri.theta, stacked) and tri.a_lo == al:
                    self._accepted[t] = tri
                    promoted += 1
            if promoted < self.T:
                log.warning("PCM commit: %d of %d markets have a trial state at this theta; the rest keep "
                            "their accepted state", promoted, self.T)
            self._trial = [None] * self.T

    def restore(self, theta: np.ndarray) -> None:
        """restore(theta): discard the trial state."""
        with self._lock:
            self._trial = [None] * self.T

    def warm_from_delta(self, delta: np.ndarray, theta: np.ndarray) -> int:
        """warm_from_delta(delta, theta) -> markets set: every market whose (n,) row-order ``delta`` entries are finite
        gets that delta and ``theta`` (layout order) as its accepted state, so a run on other draws (e.g. nu 16 from
        a nu 4 stage) starts warm instead of cold (GMY spec WP-G7 item 1). Markets with NaN entries keep theirs."""
        with self._lock:
            delta = np.asarray(delta, dtype=float).ravel()
            if delta.shape != (self.n_products,):
                raise ValueError(f"delta has shape {delta.shape}, expected ({self.n_products},)")
            stacked = self.to_kernel_order(np.asarray(theta, dtype=float).ravel())
            al = self.a_lo(theta)
            done = 0
            for m in self._mk:
                d = delta[m.rows]
                if np.all(np.isfinite(d)):
                    self._accepted[m.t] = _State(delta=d.copy(), theta=stacked.copy(), a_lo=al)
                    done += 1
            self._trial = [None] * self.T
            return done

    def accepted_delta(self) -> np.ndarray | None:
        """(n,) accepted delta in row order, or None while some market has no accepted state."""
        if any(a is None for a in self._accepted):
            return None
        out = np.empty(self.n_products)
        for m, a in zip(self._mk, self._accepted):
            assert a is not None
            out[m.rows] = a.delta
        return out

    def state_dict(self) -> dict[str, Any]:
        """state_dict() -> per-market accepted delta, theta (kernel order) and the bound a_lo it was solved under,
        thread allocation weights."""
        with self._lock:
            return {
                "version": 1,
                "a_lo": np.array([0.0 if a is None else a.a_lo for a in self._accepted]),
                "market_ids": [str(m.market_id) for m in self._mk],
                "has_accepted": np.array([a is not None for a in self._accepted], dtype=bool),
                "delta": [np.zeros(0) if a is None else a.delta.copy() for a in self._accepted],
                "theta": [np.zeros(0) if a is None else a.theta.copy() for a in self._accepted],
                "thread_weights": self._weights.copy(),
                "measured": bool(self._measured),
            }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """load_state_dict(d): restore what ``state_dict`` wrote (also a state written with the old key
        ``alpha_lo`` for the bound). Refuses a state from other markets."""
        with self._lock:
            ids = [str(x) for x in d["market_ids"]]
            if ids != [str(m.market_id) for m in self._mk]:
                raise ValueError(f"state is for markets {ids}, model has {[str(m.market_id) for m in self._mk]}")
            has = np.asarray(d["has_accepted"], dtype=bool).ravel()
            als = np.asarray(d["a_lo"] if "a_lo" in d else d.get("alpha_lo", np.zeros(self.T)), dtype=float).ravel()
            accepted: list[_State | None] = []
            for m, h, dl, th, al in zip(self._mk, has, d["delta"], d["theta"], als):
                if not h:
                    accepted.append(None)
                    continue
                dl, th = np.asarray(dl, dtype=float).ravel(), np.asarray(th, dtype=float).ravel()
                if dl.shape != (m.J,) or th.shape != (self.P_kernel,):
                    raise ValueError(f"market {m.market_id}: stored delta {dl.shape} / theta {th.shape} do not fit "
                                     f"({m.J} products, {self.P_kernel} kernel parameters)")
                accepted.append(_State(delta=dl.copy(), theta=th.copy(), a_lo=float(al)))
            self._accepted = accepted
            self._trial = [None] * self.T
            w = np.asarray(d.get("thread_weights", self._weights), dtype=float).ravel()
            if w.shape == self._weights.shape:
                self._weights = w.copy()
                self._measured = bool(d.get("measured", self._measured))

    # ---------------------------------------------------------------- lifetime

    def close(self) -> None:
        """Shut down the market thread pool."""
        self._pool.shutdown(wait=True)

    def __enter__(self) -> PCMModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

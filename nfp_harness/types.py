"""Shared types of the estimation harness (docs/estimation_harness/DESIGN.md 2.1-2.5, 4.6-4.7, 5.10).

Every module imports its data structures from here. The field lists are frozen after WP0: later
work packages extend behaviour, not fields, unless the owner of this file agrees.

Conventions: theta is the (P,) vector of nonlinear parameters in the natural parameterisation and
in ``ParameterLayout.names`` order; n is the number of products; k is the number of stacked moments.
Dataclasses that hold arrays use ``eq=False``, so equality is identity and never an ambiguous
elementwise array comparison.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .params import ParameterLayout

__all__ = [
    "EvalStatus",
    "Evaluation",
    "StepLimit",
    "Problem",
    "MarketIndex",
    "MicroPartRequest",
    "KernelRequests",
    "InnerSolution",
    "InnerModel",
    "LinearFit",
    "MomentBlock",
    "Moment",
    "ParameterSpec",
    "ConvergenceClass",
    "HistoryRecord",
    "OptimizeResult",
    "BlockStatus",
    "DiagnosticBlock",
    "Report",
    "REPORT_SCHEMA_VERSION",
]


# --------------------------------------------------------------------------------------------
# 2.1 Problem interface (what the optimizer sees)
# --------------------------------------------------------------------------------------------


class EvalStatus(enum.Enum):
    """Outcome of one evaluation. ``evaluate`` never raises for model failures; it returns one of these."""

    OK = "ok"
    INVERSION_FAILED = "inversion_failed"  # some market did not reach the share tolerance
    SINGULAR_INNER_JACOBIAN = "singular_ds_ddelta"
    OUT_OF_DOMAIN = "out_of_domain"  # theta outside hard bounds (e.g. sigma_p <= 0)
    NUMERICAL = "numerical"  # NaN/inf in g or G


@dataclass(frozen=True, eq=False)
class Evaluation:
    """Stacked moments and their total Jacobian at one theta."""

    theta: np.ndarray  # (P,) natural parameterisation
    g: np.ndarray  # (k,) stacked moments
    G: np.ndarray | None  # (k, P) total derivative dg/dtheta (chain rule through delta)
    status: EvalStatus
    blocks: dict[str, slice]  # moment block name -> rows of g, e.g. {"aggregate": 0:35, "micro": 35:45}
    aux: dict[str, Any]  # model outputs: delta, xi, beta, per-market residuals, timings, notes
    seconds: float

    @property
    def ok(self) -> bool:
        return self.status is EvalStatus.OK


@dataclass(frozen=True, eq=False)
class StepLimit:
    """Answer of the optional ``Problem.max_step`` hook (DESIGN 3.3, 4.2)."""

    tau_cap: float  # largest step fraction the model accepts along dtheta (1 = full step)
    tau_first: float  # first frontier event on any draw (exact for the linearised path)
    tau_vanish_lb: float  # lower bound on the step at which some product loses all buyers
    mass_profile: np.ndarray  # (n, 2) pairs (tau, max over products of affected buyer mass)
    note: str


@runtime_checkable
class Problem(Protocol):
    """What the optimizer minimises: q(theta) = n g' W g.

    ``max_step``, ``commit``, ``restore``, ``cue_terms``, ``state_dict`` and ``load_state_dict`` are
    optional hooks; the optimizer checks for them with ``hasattr``. Warm-start semantics: the
    problem keeps an accepted state and a trial state; ``evaluate`` warm-starts from the accepted
    state and writes the trial state; ``commit`` promotes the trial state; ``restore`` discards it.
    """

    layout: ParameterLayout

    def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation: ...

    def covariance(self, ev: Evaluation) -> np.ndarray:  # S(theta), (k, k)
        ...

    def set_weighting(self, W: np.ndarray) -> None:  # re-concentrates linear params
        ...

    # optional hooks
    def max_step(self, ev: Evaluation, dtheta: np.ndarray) -> StepLimit: ...

    def commit(self, theta: np.ndarray) -> None:  # theta accepted: promote trial state
        ...

    def restore(self, theta: np.ndarray) -> None:  # trial rejected: back to accepted state
        ...

    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]:  # (V: C x k, dV: C x k x P)
        ...

    def state_dict(self) -> dict[str, Any]:  # warm starts etc. for checkpoints
        ...

    def load_state_dict(self, d: dict[str, Any]) -> None: ...


# --------------------------------------------------------------------------------------------
# 2.2 The generic NFP-GMM problem: inner model interface
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MarketIndex:
    """One market: its id and the rows of the product table it owns."""

    market_id: Any
    rows: np.ndarray  # (J_t,) integer product rows


@dataclass(frozen=True)
class MicroPartRequest:
    """One micro part, a pair (agent weight h, product weight v with outside value v0) (DESIGN 2.3).

    ``h(i)`` is the product of the named agent columns (empty: h = 1). ``v_j`` is the product of the
    named product columns (empty: v = 1), and ``outside_value`` is v0. The kernel contracts with
    u = v - v0. A part used for a covariance product (``E[v1 v2]``) is requested as the elementwise
    product: concatenated column tuples and ``outside_value = v1_0 * v2_0``.
    """

    name: str
    agent_columns: tuple[str, ...] = ()
    product_columns: tuple[str, ...] = ()
    outside_value: float = 0.0


@dataclass(frozen=True)
class KernelRequests:
    """Extra kernel outputs a moment asks the model for. ``NFPGMMProblem`` merges them once."""

    micro_parts: tuple[MicroPartRequest, ...] = ()
    price_derivatives: bool = False  # ds/dp (postestimation)
    # product columns whose per-household purchase-weighted sums the model returns (GMY micro moments; the
    # grouping is always the agent): sol.outputs["group_sums"], contract in moments.GroupMicroMoments
    group_sums: tuple[str, ...] = ()
    # exact second-choice matrices C[j, k] = P(second = k | first = j, second inside) in the markets
    # ``second_choice_markets`` (ids): sol.outputs["second_choice"], contract in moments.SecondChoiceMoments
    second_choice: bool = False
    second_choice_markets: tuple[Any, ...] = ()

    def merge(self, other: KernelRequests) -> KernelRequests:
        """Union of two requests. Parts are matched by name; one name with two definitions is an error."""
        parts = {p.name: p for p in self.micro_parts}
        merged = list(self.micro_parts)
        for p in other.micro_parts:
            known = parts.get(p.name)
            if known is None:
                parts[p.name] = p
                merged.append(p)
            elif known != p:
                raise ValueError(f"micro part {p.name!r} requested with two definitions: {known} and {p}")
        def union(x: tuple[Any, ...], y: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(x) + tuple(v for v in y if v not in x)

        return KernelRequests(
            micro_parts=tuple(merged),
            price_derivatives=self.price_derivatives or other.price_derivatives,
            group_sums=union(self.group_sums, other.group_sums),
            second_choice=self.second_choice or other.second_choice,
            second_choice_markets=union(self.second_choice_markets, other.second_choice_markets),
        )


@dataclass(eq=False)
class InnerSolution:
    """Result of the inner inversion delta(theta) over all markets."""

    delta: np.ndarray  # (n,)
    D: list[np.ndarray] | None  # per market (J_t, P): d delta_t / d theta
    status: EvalStatus
    per_market: list[dict[str, Any]]  # converged, residual, evaluations, seconds, threads, stages
    outputs: dict[str, object]  # kernel products requested by moments (micro parts ...)


@runtime_checkable
class InnerModel(Protocol):
    """A model with an inner inversion. ``max_step``, ``commit`` and ``restore`` are optional."""

    markets: list[MarketIndex]
    n_products: int

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution: ...

    def max_step(self, sol: InnerSolution, dtheta: np.ndarray) -> StepLimit:  # optional
        ...

    def commit(self, theta: np.ndarray) -> None: ...

    def restore(self, theta: np.ndarray) -> None: ...


# --------------------------------------------------------------------------------------------
# 2.3-2.4 Moments and concentrated linear parameters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinearFit:
    """Concentrated linear parameters at the current delta (DESIGN 2.4).

    xi = M delta with M the annihilator of the concentrated linear part (IV with the aggregate
    block of W, or within-group demeaning for absorbed fixed effects).
    """

    names: tuple[str, ...]  # (K,) linear parameter names; empty when everything is absorbed
    beta: np.ndarray  # (K,)
    xi: np.ndarray  # (n,)
    M: np.ndarray | Any | None  # (n, n) map with xi = M delta: an array or a linear.Annihilator; None if not formed


@dataclass(eq=False)
class MomentBlock:
    """One moment object's contribution. The framework applies the chain rule through delta."""

    name: str
    value: np.ndarray  # (k_b,)
    d_delta: np.ndarray | None  # (k_b, n) partial derivative in delta, theta held fixed
    d_theta: np.ndarray | None  # (k_b, P) partial derivative in theta, delta held fixed
    data: np.ndarray | None  # (k_b,) data values for display (micro), else None


@runtime_checkable
class Moment(Protocol):
    name: str

    def requests(self) -> KernelRequests: ...

    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock: ...

    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray: ...


# --------------------------------------------------------------------------------------------
# 2.5 Parameter specification (ParameterLayout lives in params.py)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterSpec:
    """One nonlinear parameter. The optimizer works in u = (theta - start) / unit."""

    name: str  # e.g. "sigma_sw", "pi_mi_logfs_fv", "phi_low_inc", "sigma_p"
    role: str  # model-specific: "taste", "scale", "sigma_p"
    column: str | None  # characteristic the taste loads on ("sw"); None for scale/sigma_p
    basis: str | None  # draw column: "nu3", "logfs_fv", "low_inc"
    start: float
    lower: float  # hard lower bound (domain or sign normalisation)
    upper: float  # hard upper bound
    cap: bool = False  # True if 'upper' is a stated normalisation, not a domain guard
    unit: float | None = None  # natural unit; None means the automatic rule of DESIGN 2.5
    free: bool = True  # False fixes it at start (used by pre-fits and profiles)
    note: str = ""  # printed in the report next to the parameter

    def __post_init__(self) -> None:
        if not self.lower <= self.start <= self.upper:
            raise ValueError(f"{self.name}: start {self.start} outside [{self.lower}, {self.upper}]")
        if self.unit is not None and not self.unit > 0:
            raise ValueError(f"{self.name}: unit must be positive, got {self.unit}")


# --------------------------------------------------------------------------------------------
# 4.6-4.7 Optimizer results and history
# --------------------------------------------------------------------------------------------


class ConvergenceClass(enum.Enum):
    """Stopping classes of DESIGN 4.6."""

    CONVERGED = "CONVERGED"
    CONVERGED_AT_CAP = "CONVERGED_AT_CAP"
    # stopped on the objective's measured noise floor: the last rejections at a tiny trust radius changed q by
    # less than the jumps measured at the stage start (``OptimizerOptions.noise_probe``). Counts as converged.
    CONVERGED_NOISE = "CONVERGED_NOISE"
    STALLED = "STALLED"
    FLAT_PROGRESS = "FLAT_PROGRESS"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    INVERSION_FAILED = "INVERSION_FAILED"

    @property
    def converged(self) -> bool:
        return self in (ConvergenceClass.CONVERGED, ConvergenceClass.CONVERGED_AT_CAP,
                        ConvergenceClass.CONVERGED_NOISE)


@dataclass(frozen=True, eq=False)
class HistoryRecord:
    """One trial evaluation: one row of ``history.csv`` (DESIGN 4.7), plus the mass guard of 4.4."""

    iteration: int
    accepted: bool
    q: float
    q_fixed: float  # q with the stage's W1
    q_cue: float  # scale-free objective with S from xi
    sd_xi: float
    rho: float  # gain ratio; NaN for the start point and non-OK trials
    Delta: float  # trust radius used for the step, natural units
    tau_cap: float  # step fraction applied by max_step (1 when not truncated)
    status: EvalStatus
    seconds: float
    theta: np.ndarray  # (P,)
    predicted_mass: float = float("nan")  # max_step affected mass at tau_cap
    realised_mass: float = float("nan")  # from frontier_signature at the trial point
    stage: str = ""
    sd_delta: float = float("nan")  # sd of the mean utilities (``Problem.history_extras``)
    mean_alpha_buyers: float = float("nan")  # mean price sensitivity of predicted buyers, accepted points only


@dataclass(eq=False)
class OptimizeResult:
    """Return value of ``optimizer.minimize`` for one stage."""

    theta: np.ndarray  # (P,) accepted point at stop
    q: float
    convergence: ConvergenceClass
    evaluation: Evaluation | None  # accepted evaluation at stop; None if the start failed
    history: list[HistoryRecord]
    accepted_steps: int
    evaluations: int
    wall_seconds: float
    projected_gradient: float  # max_k |pg_k| in natural units
    pg_tol: float
    gn_decrease: float  # predicted decrease of the full Gauss-Newton step
    gn_tol: float
    Delta: float  # final trust radius
    q_mass: float  # final affected-mass threshold of the max_step guard
    kink_limited_steps: int = 0
    stage: str = ""
    weighting: str = ""  # "fixed", "two_step", "cue"
    W: np.ndarray | None = None
    message: str = ""
    noise_floor: float = float("nan")  # largest block change of q over the stage-start probes (NaN: not probed)


# --------------------------------------------------------------------------------------------
# 5.10 Report skeleton
# --------------------------------------------------------------------------------------------

REPORT_SCHEMA_VERSION = 1

BlockStatus = Literal["ok", "flag", "fail", "not_run"]


@dataclass
class DiagnosticBlock:
    """Result of one diagnostic block (DESIGN 5): status, numbers, thresholds and message lines."""

    name: str
    status: BlockStatus = "not_run"
    values: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Structured end-of-stage report; the JSON layout of DESIGN 5.10. ``report.py`` validates and renders it."""

    run_id: str
    spec: dict[str, Any]  # name, hash, normalisations
    stage: str
    created: str  # ISO 8601 UTC
    code_commit: str
    convergence: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    linear: list[dict[str, Any]] = field(default_factory=list)
    identification: dict[str, Any] = field(default_factory=dict)
    collinearity: dict[str, Any] = field(default_factory=dict)
    weak_identification: dict[str, Any] = field(default_factory=dict)
    hansen_j: dict[str, Any] | None = None
    degeneracies: list[dict[str, Any]] = field(default_factory=list)
    economics: dict[str, Any] = field(default_factory=dict)
    integration: dict[str, Any] | None = None
    multistart: dict[str, Any] | None = None
    standard_errors: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""Box trust-region Gauss-Newton for q(theta) = n g' W g (DESIGN 4.1-4.7). Owned by WP5.

Contract with the rest of the harness (only ``types.py`` is assumed):

- ``problem.layout.specs`` gives start, bounds, ``cap`` and ``free`` per parameter. Natural units come
  from ``problem.units`` (an array) when the problem has one, else from
  ``problem.layout.units(problem.data)`` (``data`` may be absent: None is passed).
- n (the q multiplier) is ``problem.n``, else ``problem.n_products``, else ``ev.aux["n"]``, else 1.
- ``weighting`` is a (k, k) array (fixed W), or an object with ``mode`` in {"fixed", "two_step",
  "cue"} or of a class named ``Fixed`` / ``TwoStep`` / ``CUE``. Fixed exposes ``W``, TwoStep exposes
  ``W1``; CUE may expose ``W1`` (used only for the q_fixed column). The string "cue" also works.
- Optional problem hooks, detected with ``hasattr``: ``set_weighting``, ``covariance`` (q_cue column,
  W2, CUE), ``cue_terms``, ``max_step``, ``commit``, ``restore``, ``state_dict``, ``load_state_dict``.
  A hook that returns None or raises ``NotImplementedError`` counts as absent.
- ``ev.aux["xi"]`` (optional) gives sd(xi); ``ev.aux["realised_mass"]`` (optional) is the realised
  affected mass at a trial point for the guard of DESIGN 4.4.
- ``checkpoint`` is a ``checkpoint.RunDirectory`` or None. With a run directory, ``minimize`` resumes
  the stage named ``options.stage`` from the checkpoint, or returns the stored result when that stage
  already finished (one evaluation at its endpoint to rebuild the ``Evaluation``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
from scipy.optimize import lsq_linear

from .diagnostics.convergence import classify, thresholds
from .types import ConvergenceClass, EvalStatus, Evaluation, HistoryRecord, OptimizeResult, Problem

__all__ = [
    "OptimizerOptions",
    "minimize",
    "projected_gradient",
    "cue_objective",
    "cue_gradient",
    "resolve_weighting",
]

log = logging.getLogger("nfp_harness.optimizer")

_BOUND_TOL = 1e-10  # in natural units: a parameter this close to a bound is at the bound


@dataclass(frozen=True)
class OptimizerOptions:
    """Tuning constants of DESIGN 4.2-4.6 with their documented defaults.

    ``stage`` names the stage in the history, checkpoint and report; ``theta0`` overrides the layout's
    start (e.g. step 2 from theta_hat_1). ``two_step_gate(step1_result) -> reason | None`` blocks step 2
    when it returns a reason (DESIGN 4.5: SCALE_ARTIFACT, SD_XI_BLOWUP); step 2 also needs a converged
    step 1 unless ``two_step_require_converged`` is False.

    Noise floor (round-3 analysis 4 (ii)): with ``noise_probe`` set (natural units, e.g. 1e-5), the stage
    start evaluates theta +- noise_probe unit_k for every free k without a Jacobian and records the largest
    change of any block of q (``OptimizeResult.noise_floor``); the stage then ends CONVERGED_NOISE after
    ``noise_rejections`` consecutive rejections tried with Delta < ``noise_Delta`` whose q differs from the
    accepted q by less than that floor (``diagnostics.convergence.noise_stop``). None: no probes, no rule.

    Levenberg-Marquardt damping (4 (iii)): with ``lm_damping``, a rejected trial with rho < 0 multiplies the
    step's damping ``lm`` by ``lm_factor`` (starting at ``lm_start``, at most ``lm_max``); an accepted step
    resets it to 0. The step solves ``min ||r + A d||^2 + (mu + lm) ||diag(d_k) d||^2`` in the box; the
    stopping test keeps the undamped Gauss-Newton decrease.

    Round 4 (SPEC_ANALYSIS v4 section 4, change 9), each off by default:

    - ``more_scaling``: the trust region is the box ``|w_k d_k| <= Delta`` with More's scaling
      ``w_k = dk_k / geomean(dk)``, dk_k = sqrt(diag(G'WG))_k in natural units (the running maximum the
      ridge already uses), clipped to [1 / ``more_scale_clip``, ``more_scale_clip``]: stiff directions get
      short steps, flat ones long steps. Delta keeps its meaning on the geometric-mean direction, and the radius
      updates use ``max_k |w_k d_k|``.
    - ``cauchy_stop``: the stopping test's decrease is the Cauchy decrease (the model's best decrease along
      the scaled steepest-descent direction inside the current box) instead of dq_gn, the decrease of the full
      unconstrained Gauss-Newton step, which on a ridge is not a convergence statistic.
    - ``noise_converge`` (needs the noise floor of ``noise_probe``): a rejected trial whose q differs from the
      accepted q by less than the floor carries no information, so the radius is not shrunk; such a rejection,
      or a trust-region step whose predicted decrease is below ``noise_pred_factor`` (10) x the floor, sends
      the stage to the terminal probe, a compass search: q at +-s natural units along the Gauss-Newton direction
      (scaled to unit max-norm) and along each free coordinate, without Jacobians, for s in ``probe_steps`` from
      the largest down (0.1, 0.03, 0.01), stopping at the first step whose best point lowers q by more than
      ``probe_gain_factor`` (10) x the floor. If no step does, the stage ends CONVERGED_NOISE; else it continues from
      that point (at most ``max_probe_restarts`` times, then the next probe ends it).

    Round 5 (LEARNINGS 32), in noise mode only: a restart from a probe sets the trust radius to the winning
    probe step in scaled units (at least ``Delta_0 / 4``), resets the Levenberg-Marquardt damping and recomputes the
    More scales from the restart point's Jacobian (the running maximum would keep stale scales). The radius is
    floored at ``noise_Delta_floor`` (1e-3 natural units, the supply block's secant step): below it a rejection
    carries no information about the direction, so a rejection at the floor goes straight to the probe.

    GMY round 1 (a trust-region collapse on a kinked objective: step 2 crawled at Delta 1e-3 for 56 steps while
    0.1-unit steps from the same point lowered q by 913), each off by default, noise mode only:

    - ``noise_Delta_floor`` is in natural units, so it is relative to each parameter's scale; a stage whose objective is
      dominated by kinked blocks (second choices, step-function micro moments) sets it to the scale at which the
      Gauss-Newton model is informative, e.g. 0.05, instead of the supply block's 1e-3 secant step.
    - ``probe_restart_at_step``: a restart from a probe sets the trust radius to at least the winning probe step itself
      (in natural units), not only its More-scaled length, so a probe that won at 0.1 along a coordinate whose More
      weight is 0.01 does not restart at a radius of 0.001.
    - ``probe_restart_gain`` (q units): a probe whose gain exceeds it restarts the stage even after
      ``max_probe_restarts``, so the stage is not classified CONVERGED_NOISE while +-0.1-unit compass steps still lower
      q by more than this amount. Each such restart lowers q by more than ``probe_restart_gain``, so the stage ends.

    GMY round 2 (step 1 spent two hours on compass restarts gaining 1 to 3 on q near 3,870), off by default:

    - ``probe_restart_gain_rel`` (a fraction of q at the probe's origin): a probe whose gain is at most
      ``probe_restart_gain_rel`` x q never restarts the stage, also within ``max_probe_restarts``, and a probe whose gain
      exceeds it restarts the stage past ``max_probe_restarts`` (with ``probe_restart_gain`` also set, past the cap the
      gain must exceed both). Each restart lowers q by more than that fraction of q, so the stage ends.

    Outer solver: ``outer`` is ``"nested"`` (this module, the default) or ``"mpec"`` (``mpec.minimize_mpec``, IPOPT on
    (theta, delta) for demand-only problems, LEARNINGS 31) with ``mpec_hessian`` ``"gn"`` or ``"lbfgs"`` and at most
    ``mpec_max_restarts`` lift-and-restart rounds. These three fields and ``probe_restart_gain_rel`` are left out of the
    repr, so the options print, and hash in profile configurations, as before.
    """

    Delta_0: float = 0.3  # initial box radius, natural units
    Delta_max: float = 5.0
    Delta_min: float = 1e-8
    rho_accept: float = 1e-4
    q_mass_0: float = 0.2  # initial affected-mass threshold of the max_step cap
    q_mass_max: float = 0.8
    q_mass_min: float = 1e-3
    ridge: float = 1e-10  # mu = ridge * max(d)^2
    pg_tol: float = 1e-3  # max |pg| <= pg_tol (1 + q)^{1/2}
    gn_tol: float = 1e-6  # dq_gn <= gn_tol (1 + q)
    max_rejections: int = 10
    flat_window: int = 25
    flat_tol: float = 1e-3
    max_iterations: int | None = None  # trial evaluations in this stage
    max_seconds: float | None = None  # wall time in this stage, summed over resumes
    stage: str = ""
    theta0: np.ndarray | None = None
    record_cue: bool = True  # compute q_cue from problem.covariance at accepted points
    two_step_gate: Callable[[OptimizeResult], str | None] | None = None
    two_step_require_converged: bool = True
    noise_probe: float | None = None  # natural units; None: no noise floor and no CONVERGED_NOISE
    noise_Delta: float = 1e-4  # natural units
    noise_rejections: int = 3
    lm_damping: bool = False
    lm_start: float = 1e-3
    lm_factor: float = 10.0
    lm_max: float = 1e8
    more_scaling: bool = False
    more_scale_clip: float = 100.0
    cauchy_stop: bool = False
    noise_converge: bool = False
    noise_pred_factor: float = 10.0
    probe_steps: tuple[float, ...] = (0.01, 0.03, 0.1)
    probe_gain_factor: float = 10.0
    max_probe_restarts: int = 5
    noise_Delta_floor: float = 1e-3  # natural units; noise mode only
    probe_restart_at_step: bool = False
    probe_restart_gain: float | None = None  # q units; None: max_probe_restarts always ends the stage
    probe_restart_gain_rel: float | None = field(default=None, repr=False)  # fraction of q; None: no threshold
    outer: str = field(default="nested", repr=False)
    mpec_hessian: str = field(default="gn", repr=False)
    mpec_max_restarts: int = field(default=3, repr=False)


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------


def _hook(problem: Any, name: str) -> Callable[..., Any] | None:
    return getattr(problem, name, None) if hasattr(problem, name) else None


def _call_hook(problem: Any, name: str, *args: Any) -> Any:
    fn = _hook(problem, name)
    if fn is None:
        return None
    try:
        return fn(*args)
    except NotImplementedError:
        return None


def resolve_weighting(weighting: Any) -> tuple[str, np.ndarray | None]:
    """resolve_weighting(weighting) -> (mode, W) with mode in {"fixed", "two_step", "cue"}.

    W is the fixed W, the two-step W1, or the CUE object's optional W1 (None if absent)."""
    if isinstance(weighting, str):
        if weighting.lower() != "cue":
            raise ValueError(f"weighting string must be 'cue', got {weighting!r}")
        return "cue", None
    if isinstance(weighting, np.ndarray):
        return "fixed", np.asarray(weighting, dtype=float)
    mode = getattr(weighting, "mode", None)
    if mode is None:
        mode = {"fixed": "fixed", "twostep": "two_step", "cue": "cue"}.get(type(weighting).__name__.lower())
    if mode == "fixed":
        return "fixed", np.asarray(weighting.W, dtype=float)
    if mode == "two_step":
        return "two_step", np.asarray(weighting.W1, dtype=float)
    if mode == "cue":
        W1 = getattr(weighting, "W1", None)
        return "cue", None if W1 is None else np.asarray(W1, dtype=float)
    raise TypeError(f"cannot read a weighting mode from {weighting!r}")


def _n_of(problem: Any, ev: Evaluation | None) -> float:
    for name in ("n", "n_products"):
        v = getattr(problem, name, None)
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return float(v)
    if ev is not None and "n" in ev.aux:
        return float(ev.aux["n"])
    return 1.0


def _layout_arrays(problem: Any) -> dict[str, np.ndarray]:
    layout = problem.layout
    specs = list(layout.specs)
    units = getattr(problem, "units", None)
    if units is None or callable(units):
        units = layout.units(getattr(problem, "data", None))
    units = np.asarray(units, dtype=float)
    lower = np.array([s.lower for s in specs], dtype=float)
    upper = np.array([s.upper for s in specs], dtype=float)
    free = np.array([s.free for s in specs], dtype=bool) & (upper > lower)
    if units.shape != lower.shape or not np.all(units > 0):
        raise ValueError(f"natural units must be positive with shape {lower.shape}, got {units}")
    return {
        "start": np.array([s.start for s in specs], dtype=float),
        "lower": lower, "upper": upper, "units": units, "free": free,
        "cap": np.array([s.cap for s in specs], dtype=bool),
        "names": np.array([s.name for s in specs], dtype=object),
    }


def _half(W: np.ndarray) -> np.ndarray:
    """L with W = L L' (Cholesky, or symmetric eigen root for a singular PSD W); r = L' g."""
    W = 0.5 * (W + W.T)
    try:
        return np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(W)
        return V * np.sqrt(np.clip(w, 0.0, None))


def _solve_sym(S: np.ndarray, g: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(S, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(S, g, rcond=None)[0]


def _inv(S: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S)


def projected_gradient(G: np.ndarray, W: np.ndarray, g: np.ndarray, n: int, units: np.ndarray,
                       theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """projected_gradient(G, W, g, n, units, theta, lower, upper) -> pg.

    ``grad_k = 2 n (G' W g)_k unit_k``, set to zero for parameters at a bound whose gradient points outward.
    """
    grad = 2.0 * n * (G.T @ (W @ g)) * units
    return _project(grad, theta, lower, upper, units)


def _project(grad: np.ndarray, theta: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             units: np.ndarray) -> np.ndarray:
    pg = np.array(grad, dtype=float)
    at_lo = theta <= lower + _BOUND_TOL * units
    at_hi = theta >= upper - _BOUND_TOL * units
    pg[at_lo & (pg > 0)] = 0.0  # descent (-grad) would decrease theta below the lower bound
    pg[at_hi & (pg < 0)] = 0.0
    return pg


def cue_objective(ev: Evaluation, S: np.ndarray, n: float) -> float:
    """cue_objective(ev, S, n) -> q = n g' S^{-1} g."""
    return float(n * ev.g @ _solve_sym(S, ev.g))


def cue_gradient(ev: Evaluation, S: np.ndarray, V: np.ndarray, dV: np.ndarray, n: float) -> np.ndarray:
    """cue_gradient(ev, S, V, dV, n) -> dq/dtheta in the natural parameterisation (LEARNINGS 1).

    ``dq/dtheta_k = n [2 G_k' a - (2/n) (dV_k a)' (V a)]`` with ``a = S^{-1} g``, ``S = V'V / n``, V (C, k)
    the cluster sums of the moment contributions and dV (C, k, P) their theta derivatives.
    """
    a = _solve_sym(S, ev.g)
    Va = V @ a
    dVa = np.einsum("ckp,k->cp", dV, a)
    return n * (2.0 * ev.G.T @ a - (2.0 / n) * dVa.T @ Va)


# ------------------------------------------------------------------------------------------------
# local model at the accepted point
# ------------------------------------------------------------------------------------------------


@dataclass
class _Model:
    q: float
    r: np.ndarray  # effective residual: n ||r + A d||^2 has the exact gradient at d = 0
    A: np.ndarray  # (k, P_free) in natural units
    grad: np.ndarray  # (P,) natural units, unprojected
    pg: np.ndarray  # (P,)
    W: np.ndarray  # weighting used for the model (S^{-1} for CUE)


class _Objective:
    """q, the local model and the history columns for one weighting mode."""

    def __init__(self, problem: Any, mode: str, W: np.ndarray | None, W1: np.ndarray | None,
                 options: OptimizerOptions, arr: dict[str, np.ndarray]) -> None:
        self.problem, self.mode, self.W, self.W1, self.options, self.arr = problem, mode, W, W1, options, arr
        self.L = _half(W) if W is not None else None

    def q(self, ev: Evaluation) -> float:
        n = _n_of(self.problem, ev)
        if self.mode == "cue":
            return cue_objective(ev, self.problem.covariance(ev), n)
        return float(n * ev.g @ (self.W @ ev.g))

    def q_fixed(self, ev: Evaluation, q: float) -> float:
        if self.mode != "cue":
            W1 = self.W1 if self.W1 is not None else self.W
            return float(_n_of(self.problem, ev) * ev.g @ (W1 @ ev.g))
        if self.W1 is None:
            return float("nan")
        return float(_n_of(self.problem, ev) * ev.g @ (self.W1 @ ev.g))

    def q_cue(self, ev: Evaluation, q: float) -> float:
        if self.mode == "cue":
            return q
        if not self.options.record_cue or not hasattr(self.problem, "covariance"):
            return float("nan")
        try:
            S = self.problem.covariance(ev)
            return cue_objective(ev, np.asarray(S, dtype=float), _n_of(self.problem, ev))
        except (NotImplementedError, np.linalg.LinAlgError, ValueError):
            return float("nan")

    def model(self, ev: Evaluation, theta: np.ndarray) -> _Model:
        arr, f = self.arr, self.arr["free"]
        n = _n_of(self.problem, ev)
        units = arr["units"]
        if self.mode == "cue":
            S = np.asarray(self.problem.covariance(ev), dtype=float)
            W = _inv(S)
            L = _half(W)
            V, dV = self.problem.cue_terms(ev)
            grad = cue_gradient(ev, S, V, dV, n) * units
            q = cue_objective(ev, S, n)
        else:
            W, L = self.W, self.L
            grad = 2.0 * n * (ev.G.T @ (W @ ev.g)) * units
            q = float(n * ev.g @ (W @ ev.g))
        r = L.T @ ev.g
        A = (L.T @ ev.G[:, f]) * units[f]
        if self.mode == "cue":
            # shift r inside range(A) so that 2 n A' r equals the exact CUE gradient (Hessian stays 2 n A'A)
            gap = grad[f] / (2.0 * n) - A.T @ r
            r = r + np.linalg.lstsq(A.T, gap, rcond=None)[0]
        pg = _project(grad, theta, arr["lower"], arr["upper"], units)
        pg[~f] = 0.0
        return _Model(q=q, r=r, A=A, grad=grad, pg=pg, W=W)


def _box_lsq(A: np.ndarray, r: np.ndarray, dk: np.ndarray, mu: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """argmin ||r + A d||^2 + mu ||diag(dk) d||^2 subject to lo <= d <= hi."""
    P = A.shape[1]
    M = np.vstack([A, math.sqrt(mu) * np.diag(dk)])
    b = np.concatenate([-r, np.zeros(P)])
    if np.all(np.isinf(lo)) and np.all(np.isinf(hi)):
        d = np.linalg.lstsq(M, b, rcond=None)[0]
    else:
        method = "bvls" if P <= 30 else "trf"
        kw = {"lsq_solver": "exact"} if method == "trf" else {}
        d = lsq_linear(M, b, bounds=(lo, hi), method=method, tol=1e-13, **kw).x
    return np.clip(d, lo, hi)


def _ridge(A: np.ndarray, dk: np.ndarray, ridge: float) -> float:
    """mu = ridge max(d)^2 (DESIGN 4.2) when the Jacobi-scaled A is ill conditioned, else 0.

    The ridge exists for rank deficiency. On a well-conditioned A it only biases every step by about
    ``ridge cond^2`` relative, so it is switched off when ``s_min / s_max`` of ``A diag(1/d)`` is at least
    ``1e3 sqrt(ridge)`` (1e-2 at the default ridge, where the bias would be below 1e-6).
    """
    if not dk.size or ridge <= 0:
        return 0.0
    s = np.linalg.svd(A / dk, compute_uv=False)
    if s.size and s[0] > 0 and s[-1] / s[0] >= 1e3 * math.sqrt(ridge) and A.shape[0] >= A.shape[1]:
        return 0.0
    return ridge * float(dk.max()) ** 2


def _gn_pred(model: _Model, d: np.ndarray, n: float) -> float:
    return float(n * (model.r @ model.r - np.sum((model.r + model.A @ d) ** 2)))


def _more_weights(dk: np.ndarray, clip: float) -> np.ndarray:
    """More's trust-region scaling w_k = dk_k / geomean(dk), clipped to [1 / clip, clip]."""
    if not dk.size:
        return dk
    pos = dk[dk > 0]
    ref = float(np.exp(np.mean(np.log(pos)))) if pos.size else 1.0
    return np.clip(dk / ref, 1.0 / clip, clip)


def _cauchy_pred(model: _Model, n: float, w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """The model decrease at the Cauchy point: the minimum of n ||r + t A p||^2 over t in [0, t_box] along the
    scaled steepest-descent direction p = -grad / w^2, with components that would leave [lo, hi] at the current
    point (a bound with zero room) set to zero; t_box keeps t p inside [lo, hi]."""
    g = model.A.T @ model.r
    p = -g / (w * w)
    p[(p < 0) & (lo >= 0)] = 0.0
    p[(p > 0) & (hi <= 0)] = 0.0
    Ap = model.A @ p
    den = float(Ap @ Ap)
    if not np.any(p) or den <= 0:
        return 0.0
    t = -float(model.r @ Ap) / den
    with np.errstate(divide="ignore", invalid="ignore"):
        room = np.where(p > 0, hi / p, np.where(p < 0, lo / p, np.inf))
    t = max(0.0, min(t, float(np.min(room))))
    return max(0.0, _gn_pred(model, t * p, n))


# ------------------------------------------------------------------------------------------------
# one stage
# ------------------------------------------------------------------------------------------------


def _limit_from(limit: Any, q_mass: float) -> tuple[float, float]:
    """(tau_cap, predicted mass at tau_cap) from a StepLimit, DESIGN 4.2."""
    prof = np.asarray(limit.mass_profile, dtype=float).reshape(-1, 2) if limit.mass_profile is not None else \
        np.zeros((0, 2))
    tau_mass = math.inf
    if len(prof):
        hit = np.nonzero(prof[:, 1] >= q_mass)[0]
        if len(hit):
            tau_mass = float(prof[hit[0], 0])
    tau_cap = min(1.0, tau_mass, 0.5 * float(limit.tau_vanish_lb))
    if limit.tau_cap is not None and math.isfinite(limit.tau_cap) and limit.tau_cap > 0:
        tau_cap = min(tau_cap, float(limit.tau_cap))
    tau_cap = max(tau_cap, 0.0)
    pred = float(prof[prof[:, 0] <= tau_cap, 1].max()) if len(prof) and np.any(prof[:, 0] <= tau_cap) else 0.0
    return tau_cap, pred


def _sd_xi(ev: Evaluation) -> float:
    xi = ev.aux.get("xi") if ev.aux else None
    return float(np.std(xi)) if xi is not None and np.size(xi) else float("nan")


def _extras(problem: Any, ev: Evaluation, accepted: bool) -> dict[str, float]:
    """``problem.history_extras(ev, accepted)`` (optional hook): sd_delta and, at accepted points,
    mean_alpha_buyers. A failing hook leaves the columns NaN rather than losing the step."""
    fn = _hook(problem, "history_extras")
    if fn is None:
        return {}
    try:
        return {k: float(v) for k, v in dict(fn(ev, accepted) or {}).items()}
    except NotImplementedError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a diagnostic column must not end the stage
        log.warning("history_extras failed: %s: %s", type(exc).__name__, exc)
        return {}


def _record(stage: str, it: int, accepted: bool, q: float, ev: Evaluation | None, obj: _Objective, rho: float,
            Delta: float, tau_cap: float, status: EvalStatus, seconds: float, theta: np.ndarray,
            pred_mass: float = float("nan"), real_mass: float = float("nan")) -> HistoryRecord:
    ok = ev is not None and status is EvalStatus.OK
    q_fixed = obj.q_fixed(ev, q) if ok else float("nan")
    q_cue = obj.q_cue(ev, q) if (ok and accepted) else float("nan")
    ex = _extras(obj.problem, ev, accepted) if ok else {}
    return HistoryRecord(iteration=it, accepted=accepted, q=q, q_fixed=q_fixed, q_cue=q_cue,
                         sd_xi=_sd_xi(ev) if ok else float("nan"), rho=rho, Delta=Delta, tau_cap=tau_cap,
                         status=status, seconds=seconds, theta=np.array(theta, dtype=float),
                         predicted_mass=pred_mass, realised_mass=real_mass, stage=stage,
                         sd_delta=ex.get("sd_delta", float("nan")),
                         mean_alpha_buyers=ex.get("mean_alpha_buyers", float("nan")))


def _probe_noise(problem: Any, obj: _Objective, theta: np.ndarray, ev: Evaluation, arr: dict[str, np.ndarray],
                 h: float, heartbeat: Any = None, path: Any = None, stage: str = "", it: int = 0) -> tuple[float, int]:
    """(noise floor, evaluations): the largest change of any block of q over evaluations at theta +- h unit_k for
    every free k (no Jacobian), against the accepted point ``ev`` (fixed weighting; NaN under CUE). The model
    is restored to the accepted state after every probe."""
    if obj.mode == "cue" or obj.W is None:
        return float("nan"), 0
    n = _n_of(problem, ev)
    W = obj.W
    blocks = dict(ev.blocks) or {"all": slice(0, ev.g.size)}

    def block_q(g: np.ndarray) -> dict[str, float]:
        return {b: float(n * g[sl] @ W[sl, sl] @ g[sl]) for b, sl in blocks.items()}

    base = block_q(ev.g)
    floor, nev = 0.0, 0
    for k in np.nonzero(arr["free"])[0]:
        for sgn in (1.0, -1.0):
            th = np.array(theta, dtype=float)
            th[k] = min(max(theta[k] + sgn * h * arr["units"][k], arr["lower"][k]), arr["upper"][k])
            if th[k] == theta[k]:
                continue
            e = problem.evaluate(th, False)
            nev += 1
            _call_hook(problem, "restore", theta)
            if heartbeat is not None:
                heartbeat(path, it, stage=stage)
            if not e.ok or not np.all(np.isfinite(e.g)):
                continue
            qb = block_q(e.g)
            floor = max(floor, max(abs(qb[b] - base[b]) for b in blocks))
    return floor, nev


def _probe_steps(steps: tuple[float, ...], fail_cap: float) -> tuple[float, ...]:
    """The probe steps no longer than ``fail_cap`` (a step length at which trials failed to evaluate); when every step
    is longer, the steps scaled so the largest equals the cap."""
    ok = tuple(s for s in steps if s <= fail_cap)
    if ok or not math.isfinite(fail_cap):
        return ok if ok else tuple(steps)
    m = max(steps)
    return tuple(s * fail_cap / m for s in steps)


def _terminal_probe(problem: Any, obj: _Objective, theta: np.ndarray, q: float, d_gn: np.ndarray,
                    arr: dict[str, np.ndarray], steps: tuple[float, ...], heartbeat: Any = None, path: Any = None,
                    stage: str = "", it: int = 0, gain_needed: float = math.inf
                    ) -> tuple[np.ndarray, float, int, str, float]:
    """(best theta, its q, evaluations, summary, winning step) of a compass search from theta: q at theta +- s along
    the Gauss-Newton direction d_gn (free coordinates, natural units, scaled to unit max-norm) and along each free
    coordinate, clipped to the bounds, for s in ``steps`` natural units from the largest down. The search stops at
    the first s whose best point lowers q by more than ``gain_needed``; the winning step is that s (0 if none won,
    in which case the best point over all steps is returned). No Jacobians; the model is restored after every
    probe."""
    f, units = arr["free"], arr["units"]
    dirs: list[tuple[str, np.ndarray]] = []
    if d_gn.size and np.max(np.abs(d_gn)) > 0:
        dirs.append(("gauss-newton", d_gn / np.max(np.abs(d_gn))))
    idx = np.nonzero(f)[0]
    for j, k in enumerate(idx):
        e = np.zeros(idx.size)
        e[j] = 1.0
        dirs.append((str(arr["names"][k]), e))
    best_theta, best_q, nev, best_what, win = np.array(theta), q, 0, "", 0.0
    order = sorted({float(x) for x in steps if x > 0}, reverse=True)
    tried: list[float] = []
    for st in order:
        tried.append(st)
        for what, v in dirs:
            for sgn in (1.0, -1.0):
                th = np.array(theta, dtype=float)
                th[f] = np.clip(theta[f] + sgn * st * v * units[f], arr["lower"][f], arr["upper"][f])
                if np.array_equal(th, theta):
                    continue
                e = problem.evaluate(th, False)
                nev += 1
                _call_hook(problem, "restore", theta)
                if heartbeat is not None:
                    heartbeat(path, it, stage=stage)
                if not e.ok or not np.all(np.isfinite(e.g)):
                    continue
                qp = obj.q(e)
                if qp < best_q:
                    best_theta, best_q, best_what = th, qp, f"{what} {sgn * st:+g}"
        if q - best_q > gain_needed:
            win = st
            break
    summary = (f"terminal probe: compass search, {nev} evaluations at +-{', '.join(f'{x:g}' for x in tried)} units "
               f"along the Gauss-Newton direction and {idx.size} coordinates; best q {best_q:.10g} "
               f"({'at ' + best_what if best_what else 'the current point'}), change {best_q - q:+.3e}")
    return best_theta, best_q, nev, summary, win


def _bound_note(theta: np.ndarray, arr: dict[str, np.ndarray], pg_raw: np.ndarray) -> tuple[bool, str]:
    units, f = arr["units"], arr["free"]
    at_lo = f & (theta <= arr["lower"] + _BOUND_TOL * units)
    at_hi = f & (theta >= arr["upper"] - _BOUND_TOL * units)
    at_cap = bool(np.any(at_hi & arr["cap"]))
    parts = []
    for k in np.nonzero(at_lo | at_hi)[0]:
        b = arr["lower"][k] if at_lo[k] else arr["upper"][k]
        what = "cap" if (at_hi[k] and arr["cap"][k]) else "bound"
        parts.append(f"{arr['names'][k]} at its {what} {b:g} (gradient {pg_raw[k]:+.2e})")
    return at_cap, ("Parameters at bounds: " + "; ".join(parts) + ".") if parts else ""


def _state(stage: str, mode: str, it: int, acc: int, nev: int, elapsed: float, theta: np.ndarray, Delta: float,
           q_mass: float, dk: np.ndarray | None, kink: int, W: np.ndarray | None, W1: np.ndarray | None,
           history: list[HistoryRecord], problem: Any, q: float, done: bool = False,
           result: dict[str, Any] | None = None, lm: float = 0.0,
           noise_floor: float = float("nan"), probe_restarts: int = 0) -> dict[str, Any]:
    from .checkpoint import history_to_state

    ps = _call_hook(problem, "state_dict")
    return {
        "stage": stage, "weighting": mode, "iteration": it, "accepted_steps": acc, "evaluations": nev,
        "elapsed": elapsed, "theta": np.array(theta), "Delta": Delta, "q_mass": q_mass,
        "dk": None if dk is None else np.array(dk), "kink_limited_steps": kink, "q": q,
        "W": None if W is None else np.array(W), "W1": None if W1 is None else np.array(W1),
        "history": history_to_state(history), "problem_state": ps, "done": done, "result": result,
        "lm": float(lm), "noise_floor": float(noise_floor), "probe_restarts": int(probe_restarts),
    }


def _run_stage(problem: Any, mode: str, W: np.ndarray | None, W1: np.ndarray | None, options: OptimizerOptions,
               checkpoint: Any, stage: str, theta0: np.ndarray) -> OptimizeResult:
    from .checkpoint import history_from_state

    arr = _layout_arrays(problem)
    f, units, lower, upper = arr["free"], arr["units"], arr["lower"], arr["upper"]
    names = [str(x) for x in arr["names"]]
    if checkpoint is not None:
        checkpoint.parameter_names = names
    heartbeat = None
    if checkpoint is not None:
        from .runtime import heartbeat as heartbeat  # noqa: PLC0415

    # finished stage: rebuild the result from its stored state
    if checkpoint is not None and checkpoint.stage_done(stage):
        return _rebuild_done(problem, mode, checkpoint.load_stage(stage), options, arr)

    resume = None
    if checkpoint is not None and checkpoint.has_checkpoint():
        st = checkpoint.load()
        if st.get("stage") == stage and not st.get("done"):
            resume = st
            if mode != "cue" and st.get("W") is not None:
                W = np.array(st["W"], dtype=float)  # the stage's own W, bit for bit (two-step W2)

    if mode != "cue" and W is not None:
        _call_hook(problem, "set_weighting", W)
    obj = _Objective(problem, mode, W, W1, options, arr)

    t0 = time.monotonic()
    if resume is not None:
        theta = np.array(resume["theta"], dtype=float)
        it, acc, nev = int(resume["iteration"]), int(resume["accepted_steps"]), int(resume["evaluations"])
        prior = float(resume["elapsed"])
        Delta, q_mass = float(resume["Delta"]), float(resume["q_mass"])
        dk = None if resume["dk"] is None else np.array(resume["dk"], dtype=float)
        kink = int(resume["kink_limited_steps"])
        lm = float(resume.get("lm", 0.0))
        noise_floor = float(resume.get("noise_floor", float("nan")))
        probe_restarts = int(resume.get("probe_restarts", 0))
        history = history_from_state(resume["history"])
        if resume.get("problem_state") is not None:
            _call_hook(problem, "load_state_dict", resume["problem_state"])
        checkpoint.rewrite_history(stage, history)
        log.info("stage %s: resumed at iteration %d, accepted step %d", stage, it, acc)
    else:
        theta = np.clip(np.array(theta0, dtype=float), lower, upper)
        it = acc = nev = kink = probe_restarts = 0
        lm, noise_floor = 0.0, float("nan")
        prior = 0.0
        Delta, q_mass, dk = options.Delta_0, options.q_mass_0, None
        history = []
        if checkpoint is not None:
            checkpoint.rewrite_history(stage, [])

    def elapsed() -> float:
        return prior + time.monotonic() - t0

    # (re-)evaluate the accepted point
    t_e = time.monotonic()
    ev = problem.evaluate(theta, True)
    nev += 1
    if heartbeat is not None:
        heartbeat(checkpoint.path, it, stage=stage)
    if not ev.ok or ev.G is None:
        _call_hook(problem, "restore", theta)
        msg = (f"accepted point could not be evaluated on resume (status {ev.status.value}{_failed_markets(ev)})"
               if resume is not None else f"start point could not be evaluated (status {ev.status.value}"
               f"{_failed_markets(ev)})")
        if resume is None:
            history.append(_record(stage, it, False, float("nan"), None, obj, float("nan"), Delta, 1.0, ev.status,
                                   time.monotonic() - t_e, theta))
        return _finish(problem, ConvergenceClass.INVERSION_FAILED, theta, None, float("nan"), None, history, acc, nev,
                       elapsed(), Delta, q_mass, kink, stage, mode, W, msg, options, checkpoint, arr, it, dk, W1)
    _call_hook(problem, "commit", theta)
    q = obj.q(ev)
    if resume is None:
        rec = _record(stage, it, True, q, ev, obj, float("nan"), Delta, 1.0, ev.status, time.monotonic() - t_e, theta)
        history.append(rec)
        if checkpoint is not None:
            checkpoint.append_history(rec)
    probed = False
    if options.noise_probe is not None and options.noise_probe > 0 and not math.isfinite(noise_floor) and f.any():
        noise_floor, n_probe = _probe_noise(problem, obj, theta, ev, arr, float(options.noise_probe), heartbeat,
                                            None if checkpoint is None else checkpoint.path, stage, it)
        nev += n_probe
        probed = True
        log.info("stage %s: noise floor %.3e from %d probes of +-%g units", stage, noise_floor, n_probe,
                 options.noise_probe)
    if checkpoint is not None and (resume is None or probed):
        checkpoint.save(_state(stage, mode, it, acc, nev, elapsed(), theta, Delta, q_mass, dk, kink, W, W1,
                               history, problem, q, lm=lm, noise_floor=noise_floor))

    def scales(m: _Model, prev: np.ndarray | None) -> np.ndarray:
        """dk: the running maximum of the column norms of A since the stage start (or a restart), floored."""
        colnorm = np.linalg.norm(m.A, axis=0)
        out = colnorm if prev is None else np.maximum(prev, colnorm)
        top = float(out.max()) if out.size and out.max() > 0 else 1.0
        out = np.maximum(out, 1e-8 * top)
        if out.size and not np.any(out > 0):
            out = np.ones_like(out)
        return out

    model: _Model | None = None
    probe_due = ""         # why the next step goes to the terminal probe (a sub-noise rejection, one at the floor)
    # The radius floor in force: noise_Delta_floor, lowered to half of any radius at which a trial failed to
    # evaluate, so the floor never pushes the stage back to a step length known to leave the evaluable region.
    floor = float(options.noise_Delta_floor)
    fail_cap = math.inf    # half the smallest radius at which a trial failed to evaluate; caps the probe steps too
    since_fail = 0         # evaluated trials since the last failed one; every 10 of them double fail_cap again
    probe_notes: list[str] = []

    def noise_mode() -> bool:
        return bool(options.noise_converge) and math.isfinite(noise_floor) and noise_floor > 0

    while True:
        if model is None:
            model = obj.model(ev, theta)
            q = model.q
            dk = scales(model, dk)
        mu = _ridge(model.A, dk, options.ridge)
        u_lo = (lower[f] - theta[f]) / units[f]
        u_hi = (upper[f] - theta[f]) / units[f]
        n = _n_of(problem, ev)
        w = _more_weights(dk, options.more_scale_clip) if options.more_scaling else np.ones(int(f.sum()))
        rad = Delta / w   # per-coordinate trust radius in natural units

        # stopping test at the accepted point
        pg = float(np.max(np.abs(model.pg))) if model.pg.size else 0.0
        if f.any():
            d_gn = _box_lsq(model.A, model.r, dk, mu, u_lo, u_hi)
            dq_gn = max(0.0, _gn_pred(model, d_gn, n))
            if options.cauchy_stop:   # the decrease the model offers at the current radius
                dq_gn = _cauchy_pred(model, n, w, np.maximum(-rad, u_lo), np.minimum(rad, u_hi))
        else:
            d_gn = np.zeros(0)
            dq_gn = 0.0
        at_cap, _ = _bound_note(theta, arr, model.grad)
        limit = None
        if options.max_iterations is not None and it >= options.max_iterations:
            limit = "iterations"
        elif options.max_seconds is not None and elapsed() >= options.max_seconds:
            limit = "time"
        cls = classify(history, pg, dq_gn, q, options, at_cap=at_cap, limit=limit, Delta=Delta,
                       noise_floor=noise_floor)
        if cls is None and not f.any():
            cls = ConvergenceClass.CONVERGED
        if cls is not None:
            return _finish(problem, cls, theta, ev, q, model, history, acc, nev, elapsed(), Delta, q_mass, kink, stage,
                           mode, W, "", options, checkpoint, arr, it, dk, W1, pg=pg, dq_gn=dq_gn,
                           noise_floor=noise_floor, lm=lm)

        # trust-region step in natural units
        d = _box_lsq(model.A, model.r, dk, mu + lm, np.maximum(-rad, u_lo), np.minimum(rad, u_hi))

        # round 4: the terminal probe when the model has nothing left to offer above the noise floor
        if noise_mode() and (probe_due or _gn_pred(model, d, n) < options.noise_pred_factor * noise_floor):
            why = (probe_due if probe_due else
                   f"the predicted decrease at radius {Delta:.2e} is below {options.noise_pred_factor:g} x the "
                   f"noise floor")
            probe_due = ""
            need = options.probe_gain_factor * noise_floor
            pt, pq, n_probe, summary, win = _terminal_probe(
                problem, obj, theta, q, d_gn, arr, _probe_steps(options.probe_steps, fail_cap), heartbeat,
                None if checkpoint is None else checkpoint.path, stage, it, gain_needed=need)
            nev += n_probe
            log.info("stage %s it %d: %s; %s", stage, it, why, summary)
            probe_notes.append(f"{why}; {summary}")
            gain = q - pq
            rel = None if options.probe_restart_gain_rel is None else options.probe_restart_gain_rel * abs(q)
            past = [x for x in (options.probe_restart_gain, rel) if x is not None]
            restart = win > 0 and gain > need and (rel is None or gain > rel) and (
                probe_restarts < options.max_probe_restarts or (bool(past) and gain > max(past)))
            if win > 0 and gain > need and rel is not None and gain <= rel:
                probe_notes.append(f"the probe's gain {gain:.4g} is at most {options.probe_restart_gain_rel:g} x q = "
                                   f"{rel:.4g}, so the stage does not restart from it")
            ev_p = problem.evaluate(pt, True) if restart else None
            if ev_p is not None:
                nev += 1
                it += 1
            if ev_p is None or not ev_p.ok or ev_p.G is None:
                if ev_p is not None:
                    _call_hook(problem, "restore", theta)
                return _finish(problem, ConvergenceClass.CONVERGED_NOISE, theta, ev, q, model, history, acc, nev,
                               elapsed(), Delta, q_mass, kink, stage, mode, W, " ".join(probe_notes), options,
                               checkpoint, arr, it, dk, W1, pg=pg, dq_gn=dq_gn, noise_floor=noise_floor, lm=lm,
                               probe_restarts=probe_restarts)
            # a probe found more than the floor: continue from it, with the trust radius at the scale the probe measured
            # (at least Delta_0 / 4), no damping and the More scales of the restart point (LEARNINGS 32)
            _call_hook(problem, "commit", pt)
            probe_restarts += 1
            d_win = (pt[f] - theta[f]) / units[f]
            theta, ev = pt, ev_p
            model = obj.model(ev, theta)
            q = model.q
            dk = scales(model, None)
            w_new = _more_weights(dk, options.more_scale_clip) if options.more_scaling else np.ones(int(f.sum()))
            Delta = min(options.Delta_max, max(float(np.max(np.abs(w_new * d_win))) if d_win.size else 0.0,
                                               (float(win) if options.probe_restart_at_step else 0.0),
                                               0.25 * options.Delta_0, floor))
            lm = 0.0
            acc += 1
            rec = _record(stage, it, True, q, ev, obj, float("nan"), Delta, 1.0, EvalStatus.OK, 0.0, theta)
            history.append(rec)
            if checkpoint is not None:
                checkpoint.append_history(rec)
                checkpoint.save(_state(stage, mode, it, acc, nev, elapsed(), theta, Delta, q_mass, dk, kink, W, W1,
                                       history, problem, q, lm=lm, noise_floor=noise_floor,
                                       probe_restarts=probe_restarts))
            log.info("stage %s it %d: restart %d from the best probe (step %g), q %.10g (gain %.3e); Delta -> %.3e, "
                     "lm reset, More scales recomputed", stage, it, probe_restarts, win, q, gain, Delta)
            continue
        tau_cap, pred_mass = 1.0, float("nan")
        dtheta = np.zeros_like(theta)
        dtheta[f] = d * units[f]
        lim = _call_hook(problem, "max_step", ev, dtheta) if np.any(d) else None
        if lim is not None:
            tau_cap, pred_mass = _limit_from(lim, q_mass)
            if tau_cap < 1.0:
                d = tau_cap * d
        dinf = float(np.max(np.abs(w * d))) if d.size else 0.0   # scaled step length (w = 1 without scaling)
        if dinf == 0.0:
            log.info("stage %s: zero step at iteration %d (Delta %.3e); stalled", stage, it, Delta)
            Delta = 0.0
            continue
        trial_theta = theta.copy()
        trial_theta[f] = np.clip(theta[f] + d * units[f], lower[f], upper[f])
        pred = _gn_pred(model, d, n)

        t_e = time.monotonic()
        trial = problem.evaluate(trial_theta, True)
        nev += 1
        it += 1
        secs = time.monotonic() - t_e
        if heartbeat is not None:
            heartbeat(checkpoint.path, it, stage=stage)
        real_mass = float(trial.aux.get("realised_mass", float("nan"))) if trial.aux else float("nan")
        Delta_used = Delta

        if not trial.ok or trial.G is None or not np.all(np.isfinite(trial.g)):
            _call_hook(problem, "restore", theta)
            status = trial.status if not trial.ok else EvalStatus.NUMERICAL
            # A failed evaluation (e.g. an inversion that does not converge) says the step left the region where the
            # model evaluates, which is information about the step length: the radius shrinks below the noise floor
            # (down to Delta_min) instead of being reset to it. Flooring it here repeated the same failing step, and
            # the terminal probe at +-0.1 units failed the same way. The floor still applies to rejections of
            # evaluated trials below.
            Delta = max(0.25 * dinf, options.Delta_min)
            if noise_mode():
                fail_cap = min(fail_cap, max(0.5 * Delta_used, options.Delta_min))
                floor = min(floor, fail_cap)
                since_fail = 0
            rec = _record(stage, it, False, float("nan"), trial, obj, float("nan"), Delta_used, tau_cap, status, secs,
                          trial_theta, pred_mass, real_mass)
            history.append(rec)
            if checkpoint is not None:
                checkpoint.append_history(rec)
            log.info("stage %s it %d: rejected, status %s; Delta -> %.3e", stage, it, status.value, Delta)
            continue

        q_trial = obj.q(trial)
        if pred > 0:
            rho = (q - q_trial) / pred
        else:
            rho = 1.0 if q_trial < q else -math.inf
        accept = rho >= options.rho_accept
        if options.lm_damping:
            if accept:
                lm = 0.0
            elif rho < 0:
                lm = options.lm_start if lm <= 0 else min(lm * options.lm_factor, options.lm_max)
        sub_noise = (noise_mode() and not accept and abs(q_trial - q) < noise_floor)
        if sub_noise:
            probe_due = "a rejection changed q by less than the noise floor"   # no information: keep the radius
        elif rho < 0.25:
            Delta = 0.25 * dinf
        elif rho > 0.75 and dinf >= 0.9 * Delta:
            Delta = min(2.0 * Delta, options.Delta_max)
        if math.isfinite(fail_cap):
            # growth stays below a step length that failed to evaluate; the cap relaxes after 10 evaluated trials,
            # since a failure says the region was left at that point, not everywhere
            since_fail += 1
            if since_fail >= 10:
                fail_cap, since_fail = 2.0 * fail_cap, 0
                floor = min(float(options.noise_Delta_floor), fail_cap)
            Delta = min(Delta, max(fail_cap, options.Delta_min))
        if noise_mode() and Delta < floor:
            # below the secant step a rejection says nothing about the direction: floor the radius, and a rejection
            # already at the floor goes to the probe
            if not accept and not probe_due and Delta_used <= floor * (1 + 1e-9):
                probe_due = f"a rejection at the radius floor {floor:g}"
            Delta = floor
        if tau_cap < 1.0 and rho > 0.75:
            q_mass = min(options.q_mass_max, 1.5 * q_mass)
        if math.isfinite(real_mass) and math.isfinite(pred_mass) and real_mass > 2.0 * pred_mass:
            q_mass = max(options.q_mass_min, 0.5 * q_mass)

        rec = _record(stage, it, accept, q_trial, trial, obj, rho, Delta_used, tau_cap, EvalStatus.OK, secs,
                      trial_theta, pred_mass, real_mass)
        history.append(rec)
        if checkpoint is not None:
            checkpoint.append_history(rec)
        if accept:
            _call_hook(problem, "commit", trial_theta)
            theta, ev, q = trial_theta, trial, q_trial
            acc += 1
            kink += int(tau_cap < 1.0)
            model = None
            if checkpoint is not None:
                checkpoint.save(_state(stage, mode, it, acc, nev, elapsed(), theta, Delta, q_mass, dk, kink, W, W1,
                                       history, problem, q, lm=lm, noise_floor=noise_floor,
                                       probe_restarts=probe_restarts))
        else:
            _call_hook(problem, "restore", theta)
        log.info("stage %s it %d: %s q %.10g rho %.3f Delta %.3e tau_cap %.3g lm %.1e", stage, it,
                 "accepted" if accept else "rejected", q_trial, rho, Delta, tau_cap, lm)


def _finish(problem: Any, cls: ConvergenceClass, theta: np.ndarray, ev: Evaluation | None, q: float,
            model: _Model | None, history: list[HistoryRecord], acc: int, nev: int, wall: float, Delta: float,
            q_mass: float, kink: int, stage: str, mode: str, W: np.ndarray | None, msg: str,
            options: OptimizerOptions, checkpoint: Any, arr: dict[str, np.ndarray], it: int, dk: np.ndarray | None,
            W1: np.ndarray | None, pg: float = float("nan"), dq_gn: float = float("nan"),
            noise_floor: float = float("nan"), lm: float = 0.0, probe_restarts: int = 0) -> OptimizeResult:
    pg_thr, gn_thr = thresholds(q, options)
    notes = [msg] if msg else []
    if model is not None:
        _, bound_msg = _bound_note(theta, arr, model.grad)
        if bound_msg:
            notes.append(bound_msg)
    if cls is ConvergenceClass.CONVERGED_NOISE and msg:
        notes.append(f"Stopped on the noise floor {noise_floor:.3e} after the terminal probe found no decrease above "
                     f"{options.probe_gain_factor:g} x the floor ({probe_restarts} restarts from a probe).")
    elif cls is ConvergenceClass.CONVERGED_NOISE:
        notes.append(f"Stopped on the noise floor: the last {options.noise_rejections} rejections, tried with a trust "
                     f"radius below {options.noise_Delta:g} natural units, changed q by less than the floor "
                     f"{noise_floor:.3e} measured at the stage start.")
    if options.cauchy_stop:
        notes.append("The predicted decrease in the stopping test is the Cauchy decrease at the current radius.")
    if cls is ConvergenceClass.STALLED:
        notes.append(f"Stopped: trust radius {Delta:.2e} or {options.max_rejections} consecutive rejections.")
    elif cls is ConvergenceClass.FLAT_PROGRESS:
        notes.append(f"Stopped: the last {options.flat_window} accepted steps lowered q by less than "
                     f"{options.flat_tol:g} (1 + q) while the convergence test failed.")
    W_out = W if mode != "cue" else (model.W if model is not None else None)
    res = OptimizeResult(theta=np.array(theta), q=q, convergence=cls, evaluation=ev, history=list(history),
                         accepted_steps=acc, evaluations=nev, wall_seconds=wall, projected_gradient=pg, pg_tol=pg_thr,
                         gn_decrease=dq_gn, gn_tol=gn_thr, Delta=Delta, q_mass=q_mass, kink_limited_steps=kink,
                         stage=stage, weighting=mode, W=None if W_out is None else np.array(W_out),
                         message=" ".join(notes), noise_floor=noise_floor)
    log.info("stage %s finished: %s q %.10g pg %.3e dq_gn %.3e", stage, cls.value, q, pg, dq_gn)
    if checkpoint is not None:
        summary = {"convergence": cls.value, "q": q, "projected_gradient": pg, "pg_tol": pg_thr, "gn_decrease": dq_gn,
                   "gn_tol": gn_thr, "wall_seconds": wall, "message": res.message, "noise_floor": noise_floor}
        state = _state(stage, mode, it, acc, nev, wall, theta, Delta, q_mass, dk, kink, res.W, W1, history,
                       problem, q, done=True, result=summary, lm=lm, noise_floor=noise_floor,
                       probe_restarts=probe_restarts)
        checkpoint.save_stage(stage, state)
        checkpoint.save(state)
    return res


def _failed_markets(ev: Any) -> str:
    """'; markets <id>: <kernel status> (<stages>), ...' for the markets of a failed evaluation that did not solve
    (the model's ``per_market`` records), or '' when the evaluation carries none."""
    pm = [m for m in (ev.aux.get("per_market") or []) if isinstance(m, dict) and m.get("status", "ok") != "ok"]
    if not pm:
        return ""
    return "; markets " + ", ".join(f"{m.get('market_id')}: {m.get('kernel_status', m.get('status'))}"
                                    + (f" ({m['stages']})" if m.get("stages") else "") for m in pm)


def _rebuild_done(problem: Any, mode: str, st: dict[str, Any], options: OptimizerOptions,
                  arr: dict[str, np.ndarray]) -> OptimizeResult:
    from .checkpoint import history_from_state

    theta = np.array(st["theta"], dtype=float)
    W = None if st["W"] is None else np.array(st["W"], dtype=float)
    if st.get("problem_state") is not None:
        _call_hook(problem, "load_state_dict", st["problem_state"])
    if mode != "cue" and W is not None:
        _call_hook(problem, "set_weighting", W)
    ev = problem.evaluate(theta, True)
    if ev.ok:
        _call_hook(problem, "commit", theta)
    else:
        _call_hook(problem, "restore", theta)
        ev = None
    r = st["result"]
    log.info("stage %s already finished (%s); skipped", st["stage"], r["convergence"])
    return OptimizeResult(theta=theta, q=float(r["q"]), convergence=ConvergenceClass(r["convergence"]), evaluation=ev,
                          history=history_from_state(st["history"]), accepted_steps=int(st["accepted_steps"]),
                          evaluations=int(st["evaluations"]), wall_seconds=float(r["wall_seconds"]),
                          projected_gradient=float(r["projected_gradient"]), pg_tol=float(r["pg_tol"]),
                          gn_decrease=float(r["gn_decrease"]), gn_tol=float(r["gn_tol"]), Delta=float(st["Delta"]),
                          q_mass=float(st["q_mass"]), kink_limited_steps=int(st["kink_limited_steps"]),
                          stage=st["stage"], weighting=mode, W=W, message=r["message"],
                          noise_floor=float(r.get("noise_floor", float("nan"))))


# ------------------------------------------------------------------------------------------------
# public entry point
# ------------------------------------------------------------------------------------------------


def minimize(problem: Problem, weighting: Any, options: OptimizerOptions | None = None,
             checkpoint: Any = None) -> OptimizeResult:
    """minimize(problem, weighting, options, checkpoint) -> OptimizeResult.

    Minimises ``q(theta) = n g' W g`` over the free parameters within ``[lower, upper]``. Steps solve
    ``min ||r + A d||^2 + mu ||diag(d_k) d||^2`` in a box of radius Delta (natural units) with
    ``scipy.optimize.lsq_linear``; ``max_step`` truncates steps by affected mass; acceptance on the
    gain ratio rho; non-OK trials are rejected and ``restore`` is called; a checkpoint is written
    after every accepted step. Stopping classes are those of ``types.ConvergenceClass``.

    Two-step weighting runs stage ``<stage>.step1`` with W1, then (if step 1 converged and
    ``options.two_step_gate`` does not object) sets ``W2 = S(theta_hat_1)^{-1}`` with
    ``problem.covariance``, calls ``set_weighting(W2)`` and runs ``<stage>.step2`` from theta_hat_1. The
    returned result is the last stage that ran; its history holds both stages (``record.stage``).
    CUE minimises ``n g' S(theta)^{-1} g`` with the exact gradient from ``cue_terms`` and the
    Gauss-Newton Hessian ``2 n G' S^{-1} G``. ``options.outer == "mpec"`` hands the call to ``mpec.minimize_mpec``.
    """
    options = options or OptimizerOptions()
    if options.outer == "mpec":
        from .mpec import minimize_mpec

        return minimize_mpec(problem, weighting, options, checkpoint)
    if options.outer != "nested":
        raise ValueError(f"outer must be 'nested' or 'mpec', got {options.outer!r}")
    mode, W = resolve_weighting(weighting)
    arr = _layout_arrays(problem)
    theta0 = arr["start"] if options.theta0 is None else np.asarray(options.theta0, dtype=float)
    if mode == "fixed":
        return _run_stage(problem, "fixed", W, None, options, checkpoint, options.stage or "fixed", theta0)
    if mode == "cue":
        return _run_stage(problem, "cue", None, W, options, checkpoint, options.stage or "cue", theta0)

    base = options.stage or "two_step"
    r1 = _run_stage(problem, "fixed", W, W, options, checkpoint, f"{base}.step1", theta0)
    reason = None
    if options.two_step_require_converged and not r1.convergence.converged:
        reason = f"step 1 ended {r1.convergence.value}, not converged"
    elif options.two_step_gate is not None:
        reason = options.two_step_gate(r1)
    if reason:
        r1.message = (r1.message + " " if r1.message else "") + f"Step 2 not started: {reason}."
        r1.weighting = "two_step"
        return r1
    W2 = _inv(np.asarray(problem.covariance(r1.evaluation), dtype=float))
    W2 = 0.5 * (W2 + W2.T)
    r2 = _run_stage(problem, "fixed", W2, W, replace(options, theta0=None), checkpoint, f"{base}.step2", r1.theta)
    r2.history = list(r1.history) + list(r2.history)
    r2.weighting = "two_step"
    return r2

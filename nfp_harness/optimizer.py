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
from dataclasses import dataclass, replace
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


def _record(stage: str, it: int, accepted: bool, q: float, ev: Evaluation | None, obj: _Objective, rho: float,
            Delta: float, tau_cap: float, status: EvalStatus, seconds: float, theta: np.ndarray,
            pred_mass: float = float("nan"), real_mass: float = float("nan")) -> HistoryRecord:
    ok = ev is not None and status is EvalStatus.OK
    q_fixed = obj.q_fixed(ev, q) if ok else float("nan")
    q_cue = obj.q_cue(ev, q) if (ok and accepted) else float("nan")
    return HistoryRecord(iteration=it, accepted=accepted, q=q, q_fixed=q_fixed, q_cue=q_cue,
                         sd_xi=_sd_xi(ev) if ok else float("nan"), rho=rho, Delta=Delta, tau_cap=tau_cap,
                         status=status, seconds=seconds, theta=np.array(theta, dtype=float),
                         predicted_mass=pred_mass, realised_mass=real_mass, stage=stage)


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
           result: dict[str, Any] | None = None) -> dict[str, Any]:
    from .checkpoint import history_to_state

    ps = _call_hook(problem, "state_dict")
    return {
        "stage": stage, "weighting": mode, "iteration": it, "accepted_steps": acc, "evaluations": nev,
        "elapsed": elapsed, "theta": np.array(theta), "Delta": Delta, "q_mass": q_mass,
        "dk": None if dk is None else np.array(dk), "kink_limited_steps": kink, "q": q,
        "W": None if W is None else np.array(W), "W1": None if W1 is None else np.array(W1),
        "history": history_to_state(history), "problem_state": ps, "done": done, "result": result,
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
        history = history_from_state(resume["history"])
        if resume.get("problem_state") is not None:
            _call_hook(problem, "load_state_dict", resume["problem_state"])
        checkpoint.rewrite_history(stage, history)
        log.info("stage %s: resumed at iteration %d, accepted step %d", stage, it, acc)
    else:
        theta = np.clip(np.array(theta0, dtype=float), lower, upper)
        it = acc = nev = kink = 0
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
        msg = (f"accepted point could not be evaluated on resume (status {ev.status.value})" if resume is not None
               else f"start point could not be evaluated (status {ev.status.value})")
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
            checkpoint.save(_state(stage, mode, it, acc, nev, elapsed(), theta, Delta, q_mass, dk, kink, W, W1,
                                   history, problem, q))

    model: _Model | None = None
    while True:
        if model is None:
            model = obj.model(ev, theta)
            q = model.q
            colnorm = np.linalg.norm(model.A, axis=0)
            dk = colnorm if dk is None else np.maximum(dk, colnorm)
            top = float(dk.max()) if dk.size and dk.max() > 0 else 1.0
            dk = np.maximum(dk, 1e-8 * top)
            if dk.size and not np.any(dk > 0):
                dk = np.ones_like(dk)
        mu = _ridge(model.A, dk, options.ridge)
        u_lo = (lower[f] - theta[f]) / units[f]
        u_hi = (upper[f] - theta[f]) / units[f]
        n = _n_of(problem, ev)

        # stopping test at the accepted point
        pg = float(np.max(np.abs(model.pg))) if model.pg.size else 0.0
        if f.any():
            d_gn = _box_lsq(model.A, model.r, dk, mu, u_lo, u_hi)
            dq_gn = max(0.0, _gn_pred(model, d_gn, n))
        else:
            dq_gn = 0.0
        at_cap, _ = _bound_note(theta, arr, model.grad)
        limit = None
        if options.max_iterations is not None and it >= options.max_iterations:
            limit = "iterations"
        elif options.max_seconds is not None and elapsed() >= options.max_seconds:
            limit = "time"
        cls = classify(history, pg, dq_gn, q, options, at_cap=at_cap, limit=limit, Delta=Delta)
        if cls is None and not f.any():
            cls = ConvergenceClass.CONVERGED
        if cls is not None:
            return _finish(problem, cls, theta, ev, q, model, history, acc, nev, elapsed(), Delta, q_mass, kink, stage,
                           mode, W, "", options, checkpoint, arr, it, dk, W1, pg=pg, dq_gn=dq_gn)

        # trust-region step in natural units
        d = _box_lsq(model.A, model.r, dk, mu, np.maximum(-Delta, u_lo), np.minimum(Delta, u_hi))
        tau_cap, pred_mass = 1.0, float("nan")
        dtheta = np.zeros_like(theta)
        dtheta[f] = d * units[f]
        lim = _call_hook(problem, "max_step", ev, dtheta) if np.any(d) else None
        if lim is not None:
            tau_cap, pred_mass = _limit_from(lim, q_mass)
            if tau_cap < 1.0:
                d = tau_cap * d
        dinf = float(np.max(np.abs(d))) if d.size else 0.0
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
            Delta = 0.25 * dinf
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
        if rho < 0.25:
            Delta = 0.25 * dinf
        elif rho > 0.75 and dinf >= 0.9 * Delta:
            Delta = min(2.0 * Delta, options.Delta_max)
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
                                       history, problem, q))
        else:
            _call_hook(problem, "restore", theta)
        log.info("stage %s it %d: %s q %.10g rho %.3f Delta %.3e tau_cap %.3g", stage, it,
                 "accepted" if accept else "rejected", q_trial, rho, Delta, tau_cap)


def _finish(problem: Any, cls: ConvergenceClass, theta: np.ndarray, ev: Evaluation | None, q: float,
            model: _Model | None, history: list[HistoryRecord], acc: int, nev: int, wall: float, Delta: float,
            q_mass: float, kink: int, stage: str, mode: str, W: np.ndarray | None, msg: str,
            options: OptimizerOptions, checkpoint: Any, arr: dict[str, np.ndarray], it: int, dk: np.ndarray | None,
            W1: np.ndarray | None, pg: float = float("nan"), dq_gn: float = float("nan")) -> OptimizeResult:
    pg_thr, gn_thr = thresholds(q, options)
    notes = [msg] if msg else []
    if model is not None:
        _, bound_msg = _bound_note(theta, arr, model.grad)
        if bound_msg:
            notes.append(bound_msg)
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
                         message=" ".join(notes))
    log.info("stage %s finished: %s q %.10g pg %.3e dq_gn %.3e", stage, cls.value, q, pg, dq_gn)
    if checkpoint is not None:
        summary = {"convergence": cls.value, "q": q, "projected_gradient": pg, "pg_tol": pg_thr, "gn_decrease": dq_gn,
                   "gn_tol": gn_thr, "wall_seconds": wall, "message": res.message}
        state = _state(stage, mode, it, acc, nev, wall, theta, Delta, q_mass, dk, kink, res.W, W1, history,
                       problem, q, done=True, result=summary)
        checkpoint.save_stage(stage, state)
        checkpoint.save(state)
    return res


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
                          stage=st["stage"], weighting=mode, W=W, message=r["message"])


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
    Gauss-Newton Hessian ``2 n G' S^{-1} G``.
    """
    options = options or OptimizerOptions()
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

"""Optional demand-only MPEC outer solver for the exact PCM (LEARNINGS 31): IPOPT on (theta, delta).

Reached only through ``OptimizerOptions(outer="mpec")`` (``--outer mpec`` on the command line); the nested
trust-region Gauss-Newton of ``optimizer.py`` stays the default and is not changed by this module.

Problem. Variables x = (theta_free / unit, delta) with delta in row order. Constraints per market,
``log s_t(delta_t, theta) - log s_data_t = 0``: a zero share is log 0 = -inf, which IPOPT treats as an evaluation
error and backtracks from (no clipping; the level form lets theta run away, LEARNINGS 31). Objective
``q = n g' W g`` with g from the problem's own moment objects at (delta, theta), so q equals
``NFPGMMProblem.evaluate`` on the constraint set. Exact first derivatives from ``pcm_kernel.evaluate_market``
(per-market dense J_t x J_t delta and J_t x P_free theta blocks of the constraint Jacobian; d g / d delta and
d g / d theta from the moments). Lagrangian Hessian: Gauss-Newton ``2 n G' W G`` with G = [dg/dtheta | dg/ddelta]
(default, ``mpec_hessian="gn"``) or IPOPT's limited-memory BFGS (``mpec_hessian="lbfgs"``, history 50). Never
finite differences: a kink inside a difference step gives spurious curvature w f / (gap h).

Start and restoration. delta_0 comes from one nested solve per market at theta_0 (warm when the model has an
accepted state), so every product is on some frontier. A given ``delta0`` with zero-share products, and the
final iterate after IPOPT stops on an invalid number (-13) or a failed restoration (-2), are repaired market by
market with ``pcm_kernel.solve_market`` warm-started from the current delta at the current theta: its warm stage
applies the frontier lift with the margin schedule of ``invert_from`` and then inverts. IPOPT restarts from the
repaired point with ``warm_start_init_point yes`` and the carried multipliers, at most ``mpec_max_restarts``
times; then the stage ends INVERSION_FAILED. (``frontier_lift`` itself has no Python binding, and this module
needs no C++ change.)

Stopping classes: IPOPT 0 and 1 -> CONVERGED; -1 (max_iter) -> FLAT_PROGRESS; -2 / -13 after the restarts ->
INVERSION_FAILED; -4 and wall-time limits -> TIME_LIMIT; anything else (2 infeasible, 3 step too small, ...) ->
STALLED. The message names the IPOPT status. After IPOPT stops, ``problem.evaluate(theta)`` runs once (nested
inversion warm-started from the MPEC delta), so W2, the reports and every diagnostic come from the same code as
the nested outer. ``OptimizeResult.mpec`` (an extra attribute) holds the IPOPT status, iterations, restarts,
kernel passes and ``max |log s / s_data|`` at the MPEC solution.

Run directory. One ``history.csv`` row per IPOPT iteration (``accepted`` 1, ``Delta`` = IPOPT's ||d||, ``rho``
NaN, ``q_cue`` NaN), ``ipopt_<stage>.log``, x and multipliers every 10 iterations in
``stages/<stage>.mpec.{npz,json}`` (a rerun of the same command resumes from them with a warm start), and the
finished stage in ``stages/<stage>`` as the nested outer writes it.

Limits. Demand only: supply and margin blocks are refused (``StageRefused``) before any kernel call, because Nash
markups are discontinuous in delta at finite draws (jumps of 2e-3 to 4e-2 in log mc per frontier event on Petrin
1984) and IPOPT has no noise-aware stop. CUE is refused (use the nested outer). Gauss-Newton is quadratically
convergent only when q* is near zero; at a non-zero-residual optimum it is linear or stalls (150 iterations
without convergence on the shifted-micro synthetic problem), where ``mpec_hessian="lbfgs"`` is slow but steady.
An exact active-piece Lagrangian Hessian needs a kernel routine that does not exist yet. The dense Hessian has
(P + n)^2 / 2 entries (2.9 million at n = 2,400). The model must be ``models.pcm.PCMModel``.

Install. cyipopt is imported at call time, never at import, so the main anaconda python imports this module
unchanged. IPOPT needs its own environment (the brew IPOPT needs glibc 2.38; anaconda's cyipopt finds no
libipopt), and the native module loads in it without a rebuild:

    conda create -p /local/home/istomin/pcm_inversion/.envs/mpec -c conda-forge --override-channels \\
        python=3.12 cyipopt numpy scipy pandas pytest
    /local/home/istomin/pcm_inversion/.envs/mpec/bin/python -m pytest nfp_harness/tests/test_mpec.py
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np

from .diagnostics.convergence import thresholds
from .moments import MICRO_OUTPUTS
from .optimizer import (
    _BOUND_TOL,
    OptimizerOptions,
    _inv,
    _layout_arrays,
    _rebuild_done,
    _state,
    projected_gradient,
    resolve_weighting,
)
from .types import (
    ConvergenceClass,
    EvalStatus,
    Evaluation,
    HistoryRecord,
    InnerSolution,
    KernelRequests,
    OptimizeResult,
)

__all__ = ["MPECProblem", "minimize_mpec", "refusal", "SUPPLY_REFUSAL"]

log = logging.getLogger("nfp_harness.mpec")

SUPPLY_REFUSAL = ("MPEC outer does not support supply/margin moments: their markups are discontinuous in delta at "
                  "finite draws; use the nested outer")

_INF = 2e19   # beyond IPOPT's nlp_{lower,upper}_bound_inf (1e19): no bound
_CHECKPOINT_EVERY = 10
_DEFAULT_MAX_ITER = 300

# IPOPT ApplicationReturnStatus
_SUCCESS = (0, 1)
_RESTART = (-2, -13)
_TIME = (-4, -5)


def refusal(problem: Any, weighting: Any = None) -> str | None:
    """The reason the MPEC outer cannot run ``problem`` (None when it can). Reads only the problem's structure, so
    it never calls the kernel."""
    if getattr(problem, "supply", None) is not None or getattr(problem, "calibration", None):
        return SUPPLY_REFUSAL
    if weighting is not None and resolve_weighting(weighting)[0] == "cue":
        return ("MPEC outer does not support CUE: the objective is a ratio of quadratics in delta; use the nested "
                "outer")
    model = getattr(problem, "model", None)
    if not all(hasattr(model, a) for a in ("kernel_market", "kernel_params", "_mk", "_pool")):
        return "MPEC outer needs the exact PCM model (nfp_harness.models.pcm.PCMModel)"
    return None


def _refuse(reason: str) -> None:
    from .pipeline import StageRefused

    raise StageRefused(reason)


def _import_cyipopt() -> Any:
    try:
        import cyipopt
    except Exception as exc:  # noqa: BLE001 - ImportError or a missing libipopt (OSError)
        raise RuntimeError(f"--outer mpec needs cyipopt; see docs (the nfp_harness.mpec module docstring has the "
                           f"install line): {type(exc).__name__}: {exc}") from None
    return cyipopt


class MPECProblem:
    """The IPOPT problem object for one stage: ``objective``, ``gradient``, ``constraints``, ``jacobian``,
    ``jacobianstructure``, ``hessian``, ``hessianstructure`` and ``intermediate`` (module docstring).

    ``MPECProblem(problem, W, W1=None, threads=None)``; ``x_of(theta, delta)`` and ``theta_of(x)`` convert between
    x and the layout theta. Kernel results are cached by ``x.tobytes()`` (the last 4 points); ``share_passes`` and
    ``derivative_passes`` count passes over all markets.
    """

    def __init__(self, problem: Any, W: np.ndarray, W1: np.ndarray | None = None, threads: int | None = None) -> None:
        reason = refusal(problem)
        if reason:
            _refuse(reason)
        self.problem = problem
        self.model = problem.model
        self.W = np.asarray(W, dtype=float)
        self.W1 = None if W1 is None else np.asarray(W1, dtype=float)
        self.arr = _layout_arrays(problem)
        self.free = self.arr["free"]
        self.units_f = self.arr["units"][self.free]
        self.base = self.arr["start"].copy()
        self.n = int(problem.n)
        self.Pf = int(self.free.sum())
        self.N = self.Pf + self.n
        self.k = int(problem.k)
        self.threads = threads
        self.parts = tuple(problem.requests.micro_parts)
        mk = self.model._mk
        self.rows = [np.asarray(m.rows) for m in mk]
        self.s_data = np.empty(self.n)
        for m in mk:
            self.s_data[m.rows] = m.s_data
        self.log_sd = np.log(self.s_data)
        # constraint Jacobian sparsity: per market the J_t x P_free theta block, then the J_t x J_t delta block
        ri, ci = [], []
        for r in self.rows:
            J = r.size
            ri.append(np.repeat(r, self.Pf))
            ci.append(np.tile(np.arange(self.Pf), J))
            ri.append(np.repeat(r, J))
            ci.append(self.Pf + np.tile(r, J))
        self._jac_rows = np.concatenate(ri).astype(np.int64)
        self._jac_cols = np.concatenate(ci).astype(np.int64)
        self._tril = np.tril_indices(self.N)
        self._cache: collections.OrderedDict[tuple[bytes, bool], dict[str, Any]] = collections.OrderedDict()
        self.share_passes = 0
        self.derivative_passes = 0
        self.nlp: Any = None
        self.on_iteration: Any = None   # callback(iter_count, info) -> bool, set by the stage driver

    # ------------------------------------------------------------------ coordinates

    def theta_of(self, x: np.ndarray) -> np.ndarray:
        """The layout theta of x: fixed parameters at their layout start, free ones at x / unit, clipped to the
        hard bounds (IPOPT relaxes bounds by 1e-8)."""
        theta = self.base.copy()
        theta[self.free] = np.asarray(x[:self.Pf], dtype=float) * self.units_f
        return np.clip(theta, self.arr["lower"], self.arr["upper"])

    def x_of(self, theta: np.ndarray, delta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return np.r_[theta[self.free] / self.units_f, np.asarray(delta, dtype=float)]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = self.arr["lower"][self.free] / self.units_f
        hi = self.arr["upper"][self.free] / self.units_f
        lo = np.where(np.isfinite(lo), lo, -_INF)
        hi = np.where(np.isfinite(hi), hi, _INF)
        return np.r_[lo, np.full(self.n, -_INF)], np.r_[hi, np.full(self.n, _INF)]

    def with_fixed(self, theta: np.ndarray) -> None:
        """Take the fixed parameters (band values, alpha_lo, phi_y) from ``theta`` instead of the layout start."""
        theta = np.asarray(theta, dtype=float)
        self.base[~self.free] = theta[~self.free]

    # ------------------------------------------------------------------ kernel

    def _threads(self) -> np.ndarray:
        from .models.pcm import allocate_threads

        total = int(self.threads or self.model.total_threads)
        return allocate_threads(self.model._weights, total)

    def _kernel(self, x: np.ndarray, derivatives: bool) -> dict[str, Any]:
        key = (x.tobytes(), derivatives)
        hit = self._cache.get(key) or (None if derivatives else self._cache.get((x.tobytes(), True)))
        if hit is not None:
            return hit
        model = self.model
        theta = self.theta_of(x)
        delta = np.array(x[self.Pf:], dtype=float)   # a copy: IPOPT may reuse the buffer of x
        th, ph, sp = model.kernel_params(theta)
        kw = model.kernel_kwargs(theta)
        threads = self._threads()
        micro = [model._micro_for(m, self.parts) if self.parts else None for m in model._mk]

        def one(t: int) -> dict[str, Any]:
            m = model._mk[t]
            return model._k.evaluate_market(delta[m.rows], m.kernel, th, ph, sp, None if micro[t] is None else
                                            micro[t][0], derivatives=derivatives, n_threads=int(threads[t]), **kw)

        out: dict[str, Any] = {"theta": theta, "delta": delta, "ok": True}
        try:
            res = list(model._pool.map(one, range(len(model._mk))))
        except ValueError as exc:   # bad kernel input (non-finite delta, ...): an evaluation error for IPOPT
            out.update(ok=False, note=str(exc))
            res = []
        if derivatives:
            self.derivative_passes += 1
        else:
            self.share_passes += 1
        if out["ok"]:
            s = np.empty(self.n)
            table: dict[str, list[dict[str, Any]]] = {p.name: [] for p in self.parts}
            Jd, Jt = [], []
            for t, (m, r) in enumerate(zip(model._mk, res)):
                s[m.rows] = r["s"]
                if derivatives:
                    Jd.append(np.asarray(r["ds_ddelta"], dtype=float))
                    dth = model.from_kernel_order(np.asarray(r["ds_dtheta"], dtype=float))
                    Jt.append(dth[:, self.free] * self.units_f)
                if self.parts:
                    _, h_sum, w_sum = micro[t]
                    mic = np.asarray(r["micro"])
                    md = np.asarray(r["micro_d_delta"]) if derivatives else None
                    mt = np.asarray(r["micro_d_theta"]) if derivatives else None
                    for a, p in enumerate(self.parts):
                        table[p.name].append({
                            "sum": float(mic[a]), "h_sum": float(h_sum[a]), "w_sum": w_sum,
                            "d_delta": None if md is None else md[a].copy(),
                            "d_theta": None if mt is None else model.from_kernel_order(mt[a]),
                        })
            out.update(s=s, Jd=Jd, Jt=Jt, table=table)
            out.update(self._moments(out, derivatives))
        self._cache[key] = out
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return out

    def _moments(self, k: dict[str, Any], derivatives: bool) -> dict[str, Any]:
        """g, d g / d delta (k, n) and d g / d theta_free in x units (k, P_free) from the problem's moment objects."""
        problem = self.problem
        delta = k["delta"]
        fit = problem.linear.fit(delta)
        sol = InnerSolution(delta=delta, D=None, status=EvalStatus.OK, per_market=[],
                            outputs={MICRO_OUTPUTS: k["table"]} if self.parts else {})
        g = np.empty(self.k)
        Gd = np.zeros((self.k, self.n)) if derivatives else None
        Gt = np.zeros((self.k, self.Pf)) if derivatives else None
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                for m in problem.moments:
                    rows = problem.blocks[m.name]
                    blk = m.evaluate(sol, fit, derivatives)
                    g[rows] = blk.value
                    if derivatives:
                        assert Gd is not None and Gt is not None
                        if blk.d_delta is not None:
                            Gd[rows] = blk.d_delta
                        if blk.d_theta is not None:
                            Gt[rows] = np.asarray(blk.d_theta)[:, self.free] * self.units_f
        except (FloatingPointError, ZeroDivisionError, ValueError) as exc:
            return {"ok": False, "note": f"moments: {exc}", "g": np.full(self.k, np.nan), "Gd": Gd, "Gt": Gt,
                    "xi": fit.xi, "beta": fit.beta}
        ok = bool(np.all(np.isfinite(g)) and (not derivatives or (np.all(np.isfinite(Gd)) and np.all(np.isfinite(Gt)))))
        return {"ok": ok, "g": g, "Gd": Gd, "Gt": Gt, "xi": fit.xi, "beta": fit.beta}

    # ------------------------------------------------------------------ IPOPT callbacks

    def objective(self, x: np.ndarray) -> float:
        k = self._kernel(x, False)
        if not k["ok"]:
            return float("nan")
        g = k["g"]
        return float(self.n * g @ self.W @ g)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        k = self._kernel(x, True)
        if not k["ok"]:
            return np.full(self.N, np.nan)
        Wg = self.W @ k["g"]
        return np.r_[2.0 * self.n * k["Gt"].T @ Wg, 2.0 * self.n * k["Gd"].T @ Wg]

    def constraints(self, x: np.ndarray) -> np.ndarray:
        k = self._kernel(x, False)
        if "s" not in k:
            return np.full(self.n, np.nan)
        with np.errstate(divide="ignore"):
            return np.log(k["s"]) - self.log_sd   # -inf at a zero share: an evaluation error for IPOPT

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._jac_rows, self._jac_cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        k = self._kernel(x, True)
        if "s" not in k:
            return np.full(self._jac_rows.size, np.nan)
        vals = []
        for r, Jd, Jt in zip(self.rows, k["Jd"], k["Jt"]):
            s = k["s"][r]
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(s > 0, 1.0 / s, np.nan)
            vals.append((inv[:, None] * Jt).ravel())
            vals.append((inv[:, None] * Jd).ravel())
        return np.concatenate(vals)

    def jacobian_dense(self, x: np.ndarray) -> np.ndarray:
        """(n, P_free + n) constraint Jacobian as a dense matrix (tests and diagnostics)."""
        J = np.zeros((self.n, self.N))
        J[self._jac_rows, self._jac_cols] = self.jacobian(x)
        return J

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._tril

    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        """Gauss-Newton Lagrangian Hessian ``obj_factor 2 n G' W G`` (no constraint curvature), lower triangle."""
        k = self._kernel(x, True)
        if not k["ok"]:
            return np.full(self._tril[0].size, np.nan)
        G = np.hstack([k["Gt"], k["Gd"]])
        H = (obj_factor * 2.0 * self.n) * (G.T @ (self.W @ G))
        return H[self._tril]

    def intermediate(self, alg_mod: int, iter_count: int, obj_value: float, inf_pr: float, inf_du: float, mu: float,
                     d_norm: float, regularization_size: float, alpha_du: float, alpha_pr: float,
                     ls_trials: int) -> bool:
        if self.on_iteration is None:
            return True
        info = {"obj": obj_value, "inf_pr": inf_pr, "inf_du": inf_du, "mu": mu, "d_norm": d_norm,
                "alpha_pr": alpha_pr, "ls_trials": ls_trials, "restoration": alg_mod == 1}
        return bool(self.on_iteration(iter_count, info))

    # ------------------------------------------------------------------ helpers

    def evaluate(self, x: np.ndarray, derivatives: bool = False) -> dict[str, Any]:
        """Kernel and moment results at x (cached): s, g, xi, beta and, with derivatives, Jd, Jt, Gd, Gt."""
        return self._kernel(np.ascontiguousarray(x, dtype=float), derivatives)

    def q(self, g: np.ndarray, W: np.ndarray | None = None) -> float:
        W = self.W if W is None else W
        return float(self.n * g @ W @ g)

    def zero_share_markets(self, s: np.ndarray) -> list[int]:
        return [t for t, r in enumerate(self.rows) if not np.all(s[r] > 0)]

    def repair(self, theta: np.ndarray, delta: np.ndarray, markets: list[int]) -> tuple[np.ndarray, list[str]]:
        """delta with the given markets re-solved by ``solve_market`` warm-started from (delta_t, theta): its warm
        stage lifts zero-share products onto a frontier with the margin schedule of ``invert_from`` and inverts.
        Returns the new delta and one note per market (kernel status and stages)."""
        model = self.model
        spec = model.spec
        th, ph, sp = model.kernel_params(theta)
        al = model.alpha_lo(theta)
        threads = self._threads()
        delta = np.array(delta, dtype=float)

        def one(t: int) -> dict[str, Any]:
            m = model._mk[t]
            return model._k.solve_market(m.s_data, m.kernel, th, ph, sp, delta[m.rows], None, None, tol=spec.tol,
                                         warm_evals=spec.warm_evals, cold_evals=spec.cold_evals,
                                         path_evals=spec.path_evals, path_seconds=spec.path_seconds,
                                         n_threads=int(threads[t]), derivatives=False, rcond_min=spec.rcond_min,
                                         alpha_lo=al)

        notes = []
        for t, r in zip(markets, model._pool.map(one, markets)):
            m = model._mk[t]
            d = np.asarray(r["delta"], dtype=float)
            if d.shape == (m.J,) and np.all(np.isfinite(d)):
                delta[m.rows] = d
            notes.append(f"{m.market_id}: {r['status']} ({r['stages']})")
        self._cache.clear()
        return delta, notes


# ------------------------------------------------------------------------------------------------
# stage driver
# ------------------------------------------------------------------------------------------------


def _seed_model(problem: Any, theta: np.ndarray, delta: np.ndarray) -> None:
    """Make (delta, theta) the model's accepted warm start, so the closing nested evaluation takes 1-2 passes."""
    model = problem.model
    st = model.state_dict()
    stacked = model.to_kernel_order(theta)
    st["has_accepted"] = np.ones(model.T, dtype=bool)
    st["delta"] = [np.array(delta[m.rows], dtype=float) for m in model._mk]
    st["theta"] = [stacked.copy() for _ in model._mk]
    st["alpha_lo"] = np.full(model.T, model.alpha_lo(theta))
    model.load_state_dict(st)


def _status_name(info: dict[str, Any]) -> str:
    msg = info.get("status_msg", "")
    return msg.decode() if isinstance(msg, bytes) else str(msg)


def _classify(status: int) -> ConvergenceClass:
    """IPOPT's final status -> stopping class (module docstring); -2 and -13 reach here only after the restarts."""
    if status in _SUCCESS:
        return ConvergenceClass.CONVERGED
    if status == -1:
        return ConvergenceClass.FLAT_PROGRESS
    if status in _RESTART:
        return ConvergenceClass.INVERSION_FAILED
    if status in _TIME:
        return ConvergenceClass.TIME_LIMIT
    return ConvergenceClass.STALLED


def _start(problem: Any, mp: MPECProblem, theta0: np.ndarray, delta0: np.ndarray | None
           ) -> tuple[np.ndarray | None, list[str], int]:
    """(x0 or None when infeasible, notes, zero-share products found): delta from one nested solve per market at
    theta0 unless ``delta0`` is given, then every market with a zero-share product lifted and re-solved."""
    notes: list[str] = []
    if delta0 is None:
        sol = problem.model.solve(theta0, KernelRequests(), False)
        problem.model.restore(theta0)
        delta = np.asarray(sol.delta, dtype=float)
        bad = [pm for pm in sol.per_market if pm["status"] != "ok"]
        if bad:
            notes.append("nested start inversion not converged in " + "; ".join(
                f"{pm['market_id']}: {pm['kernel_status']}" for pm in bad))
    else:
        delta = np.array(delta0, dtype=float)
    if delta.shape != (mp.n,):
        raise ValueError(f"delta0 has shape {delta.shape}, expected ({mp.n},)")
    delta = np.where(np.isfinite(delta), delta, 0.0)
    k = mp.evaluate(mp.x_of(theta0, delta))
    zero = mp.zero_share_markets(k["s"]) if "s" in k else list(range(len(mp.rows)))
    n_zero = int(np.sum(~(k["s"] > 0))) if "s" in k else -1
    if zero:
        delta, rnotes = mp.repair(theta0, delta, zero)
        notes.append(f"start: {n_zero} zero-share products in {len(zero)} markets lifted and re-solved "
                     f"({'; '.join(rnotes)})")
        k = mp.evaluate(mp.x_of(theta0, delta))
    if "s" not in k or mp.zero_share_markets(k["s"]) or not k["ok"]:
        notes.append("the start is infeasible after the lift" if "s" in k else f"kernel failed: {k.get('note')}")
        return None, notes, n_zero
    return mp.x_of(theta0, delta), notes, n_zero


def _run_stage(problem: Any, W: np.ndarray, W1: np.ndarray | None, options: OptimizerOptions, checkpoint: Any,
               stage: str, theta0: np.ndarray, delta0: np.ndarray | None) -> OptimizeResult:
    from .checkpoint import history_from_state, history_to_state

    arr = _layout_arrays(problem)
    if checkpoint is not None:
        checkpoint.parameter_names = [str(x) for x in arr["names"]]
        if checkpoint.stage_done(stage):
            return _rebuild_done(problem, "fixed", checkpoint.load_stage(stage), options, arr)
    cyipopt = _import_cyipopt()

    problem.set_weighting(W)
    mp = MPECProblem(problem, W, W1)
    theta0 = np.clip(np.asarray(theta0, dtype=float), arr["lower"], arr["upper"])
    mp.with_fixed(theta0)
    t0 = time.monotonic()
    run = {"prior": 0.0, "row": -1, "steps": 0, "base": 0, "iter_count": 0, "last": t0}
    history: list[HistoryRecord] = []
    mult: dict[str, np.ndarray] | None = None
    notes: list[str] = []
    n_zero = 0
    resume_key = f"{stage}.mpec"
    x: np.ndarray | None = None
    if checkpoint is not None and checkpoint.stage_done(resume_key):
        st = checkpoint.load_stage(resume_key)
        if np.array_equal(np.asarray(st["W"], dtype=float), W):
            x = np.asarray(st["x"], dtype=float)
            mult = {k: np.asarray(st[k], dtype=float) for k in ("mult_g", "mult_x_L", "mult_x_U")}
            run.update(prior=float(st["elapsed"]), row=int(st["row"]), steps=int(st["steps"]))
            run["base"] = run["row"] + 1
            mp.share_passes, mp.derivative_passes = int(st["share_passes"]), int(st["derivative_passes"])
            history = history_from_state(st["history"])
            checkpoint.rewrite_history(stage, history)
            notes.append(f"resumed after IPOPT iteration {run['steps']}")
            log.info("stage %s (mpec): resumed after iteration %d", stage, run["steps"])

    def elapsed() -> float:
        return run["prior"] + time.monotonic() - t0

    if x is None:
        if checkpoint is not None:
            checkpoint.rewrite_history(stage, [])
        x, snotes, n_zero = _start(problem, mp, theta0, delta0)
        notes += snotes
        if x is None:
            return _finish(problem, mp, ConvergenceClass.INVERSION_FAILED, theta0, history, options, checkpoint,
                           stage, W, W1, "MPEC start failed: " + "; ".join(notes),
                           {"status": None, "iterations": 0, "restarts": 0, "start_zero_share_products": n_zero},
                           elapsed())

    def on_iteration(iter_count: int, info: dict[str, Any]) -> bool:
        now = time.monotonic()
        row = run["base"] + iter_count
        run["iter_count"] = iter_count
        try:
            cur = mp.nlp.get_current_iterate()
            xc = np.asarray(cur["x"], dtype=float)
        except Exception:  # noqa: BLE001 - an iterate we cannot read is not worth ending the stage for
            return True
        k = mp.evaluate(xc)
        ok = bool(k.get("ok")) and "s" in k
        rec = HistoryRecord(iteration=row, accepted=True, q=float(info["obj"]),
                            q_fixed=mp.q(k["g"], W1 if W1 is not None else W) if ok else float("nan"),
                            q_cue=float("nan"), sd_xi=float(np.std(k["xi"])) if ok else float("nan"),
                            rho=float("nan"), Delta=float(info["d_norm"]), tau_cap=1.0,
                            status=EvalStatus.OK if ok else EvalStatus.NUMERICAL, seconds=now - run["last"],
                            theta=mp.theta_of(xc), stage=stage, sd_delta=float(np.std(xc[mp.Pf:])))
        run["last"], run["row"] = now, row
        history.append(rec)
        steps = run["steps"] + iter_count
        if checkpoint is not None:
            from .runtime import heartbeat

            checkpoint.append_history(rec)
            heartbeat(checkpoint.path, row, stage=stage)
            if iter_count > 0 and steps % _CHECKPOINT_EVERY == 0:
                checkpoint.save_stage(resume_key, {
                    "stage": stage, "x": xc, "mult_g": np.asarray(cur["mult_g"], dtype=float),
                    "mult_x_L": np.asarray(cur["mult_x_L"], dtype=float),
                    "mult_x_U": np.asarray(cur["mult_x_U"], dtype=float), "W": np.asarray(W, dtype=float),
                    "row": row, "steps": steps, "elapsed": elapsed(), "share_passes": mp.share_passes,
                    "derivative_passes": mp.derivative_passes, "history": history_to_state(history)})
        log.info("stage %s (mpec) it %d: q %.10g inf_pr %.2e inf_du %.2e mu %.1e ||d|| %.2e alpha_pr %.2g%s", stage,
                 steps, info["obj"], info["inf_pr"], info["inf_du"], info["mu"], info["d_norm"], info["alpha_pr"],
                 " (restoration)" if info["restoration"] else "")
        return True

    mp.on_iteration = on_iteration
    lb, ub = mp.bounds()
    max_iter = int(options.max_iterations) if options.max_iterations is not None else _DEFAULT_MAX_ITER
    restarts = 0
    while True:
        nlp = cyipopt.Problem(n=mp.N, m=mp.n, problem_obj=mp, lb=lb, ub=ub, cl=np.zeros(mp.n), cu=np.zeros(mp.n))
        mp.nlp = nlp
        for name, value in (("tol", 1e-8), ("constr_viol_tol", 1e-10), ("mu_strategy", "adaptive"),
                            ("max_iter", max(0, max_iter - run["steps"])), ("print_level", 0), ("sb", "yes")):
            nlp.add_option(name, value)
        if options.max_seconds is not None:
            nlp.add_option("max_wall_time", max(1.0, float(options.max_seconds) - elapsed()))
        if checkpoint is not None:
            nlp.add_option("output_file", str(checkpoint.path / f"ipopt_{stage}.log"))
            nlp.add_option("file_print_level", 5)
        if options.mpec_hessian == "lbfgs":
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 50)
        else:
            nlp.add_option("hessian_approximation", "exact")
        run["iter_count"] = 0
        if mult is not None:
            nlp.add_option("warm_start_init_point", "yes")
            x, info = nlp.solve(x, lagrange=mult["mult_g"], zl=mult["mult_x_L"], zu=mult["mult_x_U"])
        else:
            x, info = nlp.solve(x)
        x = np.asarray(x, dtype=float)
        status = int(info["status"])
        run["steps"] += run["iter_count"]
        run["base"] = run["row"] + 1
        mult = {k: np.asarray(info[k], dtype=float) for k in ("mult_g", "mult_x_L", "mult_x_U")}
        log.info("stage %s (mpec): IPOPT status %d (%s) after %d iterations", stage, status, _status_name(info),
                 run["steps"])
        if status not in _RESTART or restarts >= options.mpec_max_restarts or run["steps"] >= max_iter:
            break
        theta = mp.theta_of(x)
        k = mp.evaluate(x)
        zero = mp.zero_share_markets(k["s"]) if "s" in k else list(range(len(mp.rows)))
        n_zero_it = int(np.sum(~(k["s"] > 0))) if "s" in k else -1
        if not zero:   # nothing to lift: project onto the constraint set at this theta instead
            zero = list(range(len(mp.rows)))
        delta, rnotes = mp.repair(theta, x[mp.Pf:], zero)
        restarts += 1
        notes.append(f"restart {restarts} after IPOPT status {status} ({_status_name(info)}): {n_zero_it} zero-share "
                     f"products, {len(zero)} markets lifted and re-solved")
        log.info("stage %s (mpec): %s; %s", stage, notes[-1], "; ".join(rnotes))
        x = mp.x_of(theta, delta)

    cls = _classify(status)
    theta = mp.theta_of(x)
    k = mp.evaluate(x)
    viol = float(np.max(np.abs(np.log(k["s"]) - mp.log_sd))) if "s" in k and np.all(k["s"] > 0) else float("inf")
    notes.insert(0, f"MPEC outer (IPOPT, {options.mpec_hessian} Hessian): status {status} ({_status_name(info)}) "
                    f"after {run['steps']} iterations, {restarts} restarts, {mp.derivative_passes} derivative and "
                    f"{mp.share_passes} share passes; max |log s/s_data| {viol:.2e}.")
    if status == -1:
        notes.append(f"Stopped: IPOPT reached max_iter {max_iter}.")
    if status in _RESTART:
        notes.append(f"Stopped: zero shares or invalid numbers persisted after {restarts} lift-and-restart rounds.")
    info_out = {"status": status, "status_msg": _status_name(info), "iterations": run["steps"], "restarts": restarts,
                "share_passes": mp.share_passes, "derivative_passes": mp.derivative_passes,
                "max_log_share_violation": viol, "obj": float(info["obj_val"]), "x": x.copy(),
                "beta": np.asarray(k["beta"]) if "beta" in k else None, "start_zero_share_products": n_zero,
                "seconds": elapsed()}
    if "s" in k and np.all(k["s"] > 0):
        _seed_model(problem, theta, x[mp.Pf:])
    return _finish(problem, mp, cls, theta, history, options, checkpoint, stage, W, W1, " ".join(notes), info_out,
                   elapsed())


def _finish(problem: Any, mp: MPECProblem, cls: ConvergenceClass, theta: np.ndarray,
            history: list[HistoryRecord], options: OptimizerOptions, checkpoint: Any, stage: str, W: np.ndarray,
            W1: np.ndarray | None, msg: str, info: dict[str, Any], wall: float) -> OptimizeResult:
    """The closing nested evaluation at theta, the OptimizeResult and the finished-stage state (as the nested outer
    writes it, so a rerun rebuilds the result with ``optimizer._rebuild_done``)."""
    arr = mp.arr
    ev: Evaluation | None = problem.evaluate(theta, True)
    if ev is not None and ev.ok and ev.G is not None:
        problem.commit(theta)
        q = float(problem.n * ev.g @ W @ ev.g)
        pg_vec = projected_gradient(ev.G, W, ev.g, problem.n, arr["units"], theta, arr["lower"], arr["upper"])
        pg_vec[~arr["free"]] = 0.0
        pg = float(np.max(np.abs(pg_vec))) if pg_vec.size else 0.0
    else:
        problem.restore(theta)
        if ev is not None and cls.converged:
            msg += f" The closing nested evaluation failed ({ev.status.value})."
            cls = ConvergenceClass.INVERSION_FAILED
        ev = None
        q, pg = float("nan"), float("nan")
    pg_thr, gn_thr = thresholds(q, options)
    at = [f"{arr['names'][j]} at a bound" for j in np.nonzero(arr["free"] & (
        (theta <= arr["lower"] + _BOUND_TOL * arr["units"]) | (theta >= arr["upper"] - _BOUND_TOL * arr["units"])))[0]]
    if at:
        msg += " Parameters at bounds: " + "; ".join(at) + "."
    res = OptimizeResult(theta=np.array(theta), q=q, convergence=cls, evaluation=ev, history=list(history),
                         accepted_steps=int(info.get("iterations") or 0),
                         evaluations=mp.share_passes + mp.derivative_passes, wall_seconds=wall, projected_gradient=pg,
                         pg_tol=pg_thr, gn_decrease=float("nan"), gn_tol=gn_thr, Delta=float("nan"),
                         q_mass=float("nan"), stage=stage, weighting="fixed", W=np.array(W), message=msg)
    res.mpec = info  # type: ignore[attr-defined]
    log.info("stage %s (mpec) finished: %s q %.10g pg %.3e", stage, cls.value, q, pg)
    if checkpoint is not None:
        summary = {"convergence": cls.value, "q": q, "projected_gradient": pg, "pg_tol": pg_thr,
                   "gn_decrease": float("nan"), "gn_tol": gn_thr, "wall_seconds": wall, "message": msg,
                   "noise_floor": float("nan")}
        st = _state(stage, "fixed", int(info.get("iterations") or 0), res.accepted_steps, res.evaluations, wall, theta,
                    float("nan"), float("nan"), None, 0, W, W1, history, problem, q, done=True, result=summary)
        checkpoint.save_stage(stage, st)
        checkpoint.save(st)
    return res


def minimize_mpec(problem: Any, weighting: Any, options: OptimizerOptions | None = None, checkpoint: Any = None, *,
                  delta0: np.ndarray | None = None) -> OptimizeResult:
    """minimize_mpec(problem, weighting, options, checkpoint, *, delta0=None) -> OptimizeResult.

    Same contract as ``optimizer.minimize`` for fixed and two-step weighting (step 2 with
    ``W2 = S(theta_hat_1)^{-1}`` from the closing nested evaluation of step 1). ``delta0`` (row order) replaces the
    nested start inversion of the first stage; zero-share products in it are lifted. Refuses supply, margin and
    CUE problems with ``StageRefused`` before any kernel call; raises RuntimeError when cyipopt is missing.
    """
    options = options or OptimizerOptions(outer="mpec")
    reason = refusal(problem, weighting)
    if reason:
        _refuse(reason)
    if options.mpec_hessian not in ("gn", "lbfgs"):
        raise ValueError(f"mpec_hessian must be 'gn' or 'lbfgs', got {options.mpec_hessian!r}")
    _import_cyipopt()
    mode, W = resolve_weighting(weighting)
    assert W is not None
    arr = _layout_arrays(problem)
    theta0 = arr["start"] if options.theta0 is None else np.asarray(options.theta0, dtype=float)
    if mode == "fixed":
        return _run_stage(problem, W, None, options, checkpoint, options.stage or "fixed", theta0, delta0)
    base = options.stage or "two_step"
    r1 = _run_stage(problem, W, W, options, checkpoint, f"{base}.step1", theta0, delta0)
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
    r2 = _run_stage(problem, W2, W, replace(options, theta0=None), checkpoint, f"{base}.step2", r1.theta, None)
    r2.history = list(r1.history) + list(r2.history)
    r2.weighting = "two_step"
    return r2


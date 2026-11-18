"""End-of-stage diagnostics (DESIGN 5). ``run_diagnostics`` is owned by WP6.

Each block returns a ``types.DiagnosticBlock`` with status in {"ok", "flag", "fail", "not_run"},
the numbers, the thresholds (from ``spec.diagnostics`` with the documented defaults) and its
message lines.

``run_diagnostics`` collects its inputs from the problem, the optimizer result, the spec and the
options into a ``DiagnosticInputs`` and calls ``build_report``, which is pure and is what the tests
drive with synthetic evaluations and histories.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import DiagnosticBlock, OptimizeResult, ParameterSpec, Problem, Report
from . import collinearity, degeneracies, identification, inference, overid, report, stability
from .collinearity import collinearity_block
from .degeneracies import detect_all, layout_specs
from .identification import identification_block, order_condition_block, sqrt_psd, weak_identification_block
from .inference import se_gate, standard_errors
from .overid import hansen_j
from .report import block_to_dict, render_text, to_jsonable, validate, write_report
from .stability import integration_stability, multistart_summary

__all__ = ["DiagnosticInputs", "at_bound_parameters", "build_report", "run_diagnostics", "render_text", "validate", "write_report",
           "collinearity", "degeneracies", "identification", "inference", "overid", "report", "stability"]

NOT_INTERPRETED = "not interpreted: objective did not converge"


@dataclass(eq=False)
class DiagnosticInputs:
    """Everything the diagnostic blocks read. Arrays over parameters follow ``specs`` order."""

    result: OptimizeResult
    specs: Sequence[ParameterSpec]
    units: np.ndarray  # (P,) natural units
    n: int  # number of products (q = n g' W g)
    run_id: str = ""
    spec_name: str = ""
    spec_hash: str = ""
    normalisations: Sequence[str] = ()
    stage: str = ""
    code_commit: str = ""
    created: str = ""
    G: np.ndarray | None = None  # (k, P) concentrated Jacobian; default result.evaluation.G
    g: np.ndarray | None = None  # (k,); default result.evaluation.g
    W: np.ndarray | None = None  # (k, k); default result.W
    S: np.ndarray | None = None  # (k, k) moment covariance at the estimate
    blocks: Mapping[str, slice] | None = None  # default result.evaluation.blocks
    X1: np.ndarray | None = None
    Z: np.ndarray | None = None  # aggregate instruments
    linear_names: Sequence[str] = ()
    beta: np.ndarray | None = None
    xi: np.ndarray | None = None
    G_full: np.ndarray | None = None  # (k, K + P) Jacobian over (beta, theta) for SEs (DESIGN 2.4)
    gradient: np.ndarray | None = None  # (P,) dq/dtheta; default the Gauss-Newton gradient 2 n G'Wg
    named_directions: Mapping[str, np.ndarray] | None = None
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    economics: DiagnosticBlock | Mapping[str, Any] | None = None
    integration: DiagnosticBlock | None = None
    multistart: Mapping[str, Any] | None = None
    price_range: float | None = None
    cross_group_diversion: float | None = None
    binary_columns: Sequence[str] = ()
    files: Mapping[str, str] = field(default_factory=dict)


def _convergence(result: OptimizeResult) -> dict[str, Any]:
    cls = result.convergence
    try:
        from .convergence import convergence_block  # WP5

        cb = convergence_block(result)
        messages = list(cb.messages)
    except NotImplementedError:
        messages = []
    if not messages:
        messages = [f"Convergence: {cls.value}. Projected gradient {result.projected_gradient:.2e} (tolerance "
                    f"{result.pg_tol:.2e}); predicted Gauss-Newton decrease {result.gn_decrease:.2e} (tolerance "
                    f"{result.gn_tol:.2e}); {result.accepted_steps} accepted steps, {result.evaluations} "
                    f"evaluations, {result.wall_seconds / 3600:.1f} h."]
    if cls.converged:
        status = "ok"
    else:
        status = "fail" if cls.value == "INVERSION_FAILED" else "flag"
        mark = "FAIL: " if status == "fail" else "FLAG: "
        messages = [m if m.startswith(("FLAG:", "FAIL:")) else mark + m for m in messages]
    return {"class": cls.value, "projected_gradient": result.projected_gradient, "pg_tol": result.pg_tol,
            "gn_decrease": result.gn_decrease, "gn_tol": result.gn_tol, "accepted_steps": result.accepted_steps,
            "evaluations": result.evaluations, "wall_seconds": result.wall_seconds,
            "kink_limited_steps": result.kink_limited_steps, "message": messages[0], "status": status,
            "messages": messages}


def _zero_spread(s: ParameterSpec) -> bool:
    """A taste spread (``sigma_*``, role taste) bounded below at 0: the bound is the sign normalisation of a
    symmetric draw, so an estimate of 0 is an interior answer (no dispersion), not a bound the data push past."""
    return s.role == "taste" and s.name.startswith("sigma_") and s.lower == 0.0


def _at_bound(theta: np.ndarray, specs: Sequence[ParameterSpec], units: np.ndarray, tol: float):
    lo = np.array([s.lower for s in specs], dtype=float)
    hi = np.array([s.upper for s in specs], dtype=float)
    zero_spread = np.array([_zero_spread(s) for s in specs], dtype=bool)
    at_lo = np.isfinite(lo) & ~zero_spread & (np.abs(theta - lo) <= tol * units)
    at_hi = np.isfinite(hi) & (np.abs(theta - hi) <= tol * units)
    return at_lo, at_hi


def at_bound_parameters(theta: np.ndarray, specs: Sequence[ParameterSpec], units: np.ndarray,
                        tol: float | None = None) -> list[str]:
    """Names of the free parameters within ``tol`` (default the degeneracy ``bound_tol``, 1e-3) natural units of a
    layout bound, a cap at its cap and a taste spread at 0 excepted: the AT_BOUND flag of the report's
    convergence block."""
    tol = float(degeneracies.DEGENERACY_DEFAULTS["bound_tol"] if tol is None else tol)
    theta = np.asarray(theta, dtype=float)
    lo, hi = _at_bound(theta, specs, np.asarray(units, dtype=float), tol)
    return [s.name for i, s in enumerate(specs) if s.free and (lo[i] or (hi[i] and not s.cap))]


def build_report(inp: DiagnosticInputs) -> Report:
    """Run every WP6 block on ``inp`` and assemble the ``Report`` of DESIGN 5.10."""
    res = inp.result
    ev = res.evaluation
    specs = list(inp.specs)
    names = [s.name for s in specs]
    units = np.asarray(inp.units, dtype=float)
    theta = np.asarray(res.theta, dtype=float)
    th = dict(inp.thresholds or {})
    G = inp.G if inp.G is not None else (ev.G if ev is not None else None)
    g = inp.g if inp.g is not None else (ev.g if ev is not None else None)
    W = inp.W if inp.W is not None else res.W
    blocks = inp.blocks if inp.blocks is not None else (ev.blocks if ev is not None else {})
    n = int(inp.n)
    k = int(np.size(g)) if g is not None else 0
    free = np.array([s.free for s in specs], dtype=bool)
    k_linear = len(inp.linear_names)
    converged = res.convergence.converged

    conv = _convergence(res)

    bound_tol = float(th.get("bound_tol", degeneracies.DEGENERACY_DEFAULTS["bound_tol"]))
    at_lo, at_hi = _at_bound(theta, specs, units, bound_tol)
    cap_active = np.array([s.cap for s in specs], dtype=bool) & at_hi & free
    # AT_BOUND (ROUND5 5.1 item 4): a free parameter within bound_tol natural units of a layout bound (a cap at its
    # cap is a stated normalisation, NORMALISATION_CAP_ACTIVE, not this) is a failed estimate, not a result
    at_bound = free & (at_lo | (at_hi & ~cap_active))
    conv["at_bound"] = [names[i] for i in np.nonzero(at_bound)[0]]
    if conv["at_bound"]:
        what = "; ".join(f"{names[i]} = {theta[i]:.6g} at its {'lower' if at_lo[i] else 'upper'} bound "
                         f"{specs[i].lower if at_lo[i] else specs[i].upper:g}" for i in np.nonzero(at_bound)[0])
        conv["messages"] = list(conv["messages"]) + [
            f"FLAG: AT_BOUND: {what} (within {bound_tol:g} natural units). The objective wants to go past the "
            f"layout bound, so this point is a failed estimate and must not be reported as a headline."]
        if conv["status"] == "ok":
            conv["status"] = "flag"

    parts: dict[str, float] = {}
    if g is not None and W is not None:
        Wg = np.asarray(W) @ np.asarray(g)
        for b, sl in (blocks or {}).items():
            parts[b] = float(n * np.asarray(g)[sl] @ Wg[sl])
    objective = {"q": res.q, "parts": parts, "weighting": res.weighting or "unknown", "n": n, "k": k,
                 "p_free": int(free.sum()), "k_linear": k_linear}

    # identification and collinearity on the free parameters
    order = order_condition_block(k, int(free.sum()), k_linear)
    A_jac = None
    if G is not None and W is not None and np.all(np.isfinite(G)):
        Gf = np.asarray(G, dtype=float)[:, free]
        named = {nm: np.asarray(v, dtype=float)[free] for nm, v in (inp.named_directions or {}).items()}
        ident = identification_block(Gf, W, units[free], [nm for nm, f in zip(names, free) if f], n, named, th)
        A_nat = sqrt_psd(W) @ Gf * units[free][None, :]
        c = np.linalg.norm(A_nat, axis=0)
        A_jac = A_nat / np.where(c > 0, c, 1.0)[None, :]
    else:
        ident = DiagnosticBlock(name="identification", status="not_run",
                                messages=["Identification not computed: no Jacobian or weighting at the final point."],
                                values={"rc_natural": None, "rc_jacobi": None, "directions": []})
    ident_d = block_to_dict(ident)
    ident_d["order_condition"] = block_to_dict(order)
    if order.status == "fail":
        ident_d["status"] = "fail"

    coll = collinearity_block(inp.X1, inp.Z, A_jac, [nm for nm, f in zip(names, free) if f], th,
                              x1_names=list(inp.linear_names) or None)
    coll_d = block_to_dict(coll)

    if G is not None and inp.S is not None:
        weak = weak_identification_block(np.asarray(G)[:, free], inp.S, n, th, units=units[free])
    else:
        weak = DiagnosticBlock(name="weak_identification", status="not_run",
                               values={"statistic": None, "dof": k - int(free.sum()) + 1, "critical_95": None,
                                       "verdict": "not_run"},
                               messages=["Weak identification not computed: no Jacobian or covariance S."])

    gradient = inp.gradient
    if gradient is None and G is not None and W is not None and g is not None:
        gradient = 2.0 * n * (np.asarray(G).T @ (np.asarray(W) @ np.asarray(g)))
    econ_values: Mapping[str, Any] | None = None
    if isinstance(inp.economics, DiagnosticBlock):
        econ_values = inp.economics.values
    elif isinstance(inp.economics, Mapping):
        econ_values = inp.economics
    final = {"theta": theta, "layout": specs, "units": units, "gradient": gradient, "xi": inp.xi,
             "economics": econ_values, "price_range": inp.price_range,
             "cross_group_diversion": inp.cross_group_diversion, "binary_columns": tuple(inp.binary_columns)}
    degen = detect_all(res.history, final, None, th)

    dof = k - (int(free.sum()) + k_linear) - int(cap_active.sum())
    if g is not None and inp.S is not None:
        hj = hansen_j(np.asarray(g), inp.S, n, dof, res.weighting, converged=converged)
    else:
        hj = DiagnosticBlock(name="hansen_j", status="not_run", values={"J": None, "dof": dof, "p_value": None},
                             messages=["Hansen J not computed: no covariance S at the final point."])

    if isinstance(inp.economics, DiagnosticBlock):
        econ = block_to_dict(inp.economics)
    elif isinstance(inp.economics, Mapping):
        econ = dict(inp.economics)
        econ.setdefault("status", "ok")
        econ.setdefault("messages", [])
    else:
        econ = {"status": "not_run", "messages": ["Economic sanity block not run."]}
    if not converged and econ.get("status") != "not_run":
        econ["messages"] = list(econ["messages"]) + [f"Economics {NOT_INTERPRETED}."]

    # standard errors
    computed, reason = se_gate(res.convergence, ident.values.get("rc_jacobi"), degen)
    se_by_name: dict[str, float | None] = {}
    se_d: dict[str, Any] = {"computed": False, "reason": reason, "status": "not_run", "messages": []}
    if computed and (inp.S is None or W is None or G is None):
        computed, reason = False, "Standard errors omitted: no covariance S at the final point."
    if computed:
        fixed = ~free | cap_active
        try:
            if inp.G_full is not None:
                full_names = list(inp.linear_names) + names
                out = standard_errors(inp.G_full, W, inp.S, n, full_names, np.r_[np.zeros(k_linear, bool), fixed])
                note = "Sandwich over (beta, theta) with the full Jacobian (DESIGN 2.4)."
            else:
                out = standard_errors(G, W, inp.S, n, names, fixed)
                note = ("Sandwich over theta with the concentrated Jacobian; linear coefficients have no SE "
                        "because the full Jacobian was not supplied.")
            se_by_name = out["se"]
            capped = [nm for nm, c in zip(names, cap_active) if c]
            msgs = [f"Standard errors computed. {note}"]
            if capped:
                msgs.append(f"Cap parameters treated as fixed, no SE: {', '.join(capped)}.")
            se_d = {"computed": True, "reason": None, "status": "ok", "messages": msgs}
        except np.linalg.LinAlgError as exc:
            computed, reason = False, f"Standard errors omitted: G'WG is singular ({exc})."
    if not computed:
        se_d = {"computed": False, "reason": reason, "status": "not_run", "messages": [reason]}

    parameters = []
    for i, s in enumerate(specs):
        parameters.append({"name": s.name, "value": float(theta[i]), "unit": float(units[i]), "lower": s.lower,
                           "upper": s.upper, "cap": s.cap, "at_bound": bool(free[i] and (at_lo[i] or at_hi[i])),
                           "free": s.free, "se": se_by_name.get(s.name), "note": s.note})
    linear = []
    beta = np.asarray(inp.beta, dtype=float) if inp.beta is not None else np.full(k_linear, np.nan)
    for j, nm in enumerate(inp.linear_names):
        linear.append({"name": nm, "value": float(beta[j]), "se": se_by_name.get(nm)})

    integ = block_to_dict(inp.integration) if inp.integration is not None else None
    created = inp.created or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rep = Report(
        run_id=inp.run_id, spec={"name": inp.spec_name, "hash": inp.spec_hash,
                                 "normalisations": list(inp.normalisations)},
        stage=inp.stage, created=created, code_commit=inp.code_commit,
        convergence=conv, objective=objective, parameters=parameters, linear=linear,
        identification=ident_d, collinearity=coll_d, weak_identification=block_to_dict(weak),
        hansen_j=block_to_dict(hj), degeneracies=degen, economics=econ, integration=integ,
        multistart=dict(inp.multistart) if inp.multistart is not None else None,
        standard_errors=se_d, files=dict(inp.files))
    # store plain JSON values so the Report round-trips through json without loss
    for f in ("convergence", "objective", "parameters", "linear", "identification", "collinearity",
              "weak_identification", "hansen_j", "degeneracies", "economics", "integration", "multistart",
              "standard_errors"):
        setattr(rep, f, to_jsonable(getattr(rep, f)))
    return rep


def _pick(key: str, options: Mapping[str, Any], spec: Any, aux: Mapping[str, Any], default: Any = None) -> Any:
    if key in options and options[key] is not None:
        return options[key]
    v = getattr(spec, key, None) if spec is not None else None
    if v is not None:
        return v
    v = aux.get(key)
    return default if v is None else v


def run_diagnostics(problem: Problem, result: OptimizeResult, spec: Any, options: Any = None) -> Report:
    """run_diagnostics(problem, result, spec, options) -> Report.

    Runs, in order: convergence class (5.1), flat directions and units versus identification (5.2),
    collinearity and weak instruments (5.3), named degeneracies (5.4), overidentification (5.5),
    economic sanity (5.6), integration stability (5.7), multi-start (5.8), standard errors (5.9).
    Blocks that need a stationary point print "not interpreted: objective did not converge" unless
    the class is CONVERGED or CONVERGED_AT_CAP.

    Inputs are read, in this order of precedence, from ``options`` (a mapping), attributes of
    ``spec`` and ``result.evaluation.aux``: ``units``, ``n``, ``X1``, ``Z``, ``linear_names``, ``beta``,
    ``xi``, ``G_full``, ``gradient``, ``named_directions``, ``price_range``, ``cross_group_diversion``,
    ``binary_columns``, ``normalisations``, ``run_id``, ``code_commit``, ``files``. ``spec.diagnostics``
    holds threshold overrides. ``S`` defaults to ``problem.covariance(result.evaluation)``. Options
    ``reevaluate`` (callable theta -> OptimizeResult with doubled draws) and ``multistart_results``
    run the stability blocks.
    """
    options = dict(options or {})
    ev = result.evaluation
    aux = dict(ev.aux) if ev is not None and ev.aux else {}
    layout = problem.layout
    specs = layout_specs(layout)

    units = _pick("units", options, None, {})
    if units is None:
        try:
            units = layout.units(getattr(spec, "data", None))
        except Exception:  # noqa: BLE001 - fall back to declared units
            units = np.array([s.unit if s.unit is not None else 1.0 for s in specs], dtype=float)
    xi = _pick("xi", options, None, aux)
    n = _pick("n", options, spec, aux)
    if n is None:
        n = int(np.size(xi)) if xi is not None else 1

    S = options.get("S")
    if S is None and ev is not None and ev.ok:
        try:
            S = problem.covariance(ev)
        except Exception:  # noqa: BLE001 - reported as a missing S
            S = None

    economics = options.get("economics")
    if economics is None:
        try:
            from .economics import economics_block  # WP8

            economics = economics_block(problem, result, spec)
        except NotImplementedError:
            economics = None

    th = dict(getattr(spec, "diagnostics", None) or {})
    th.update(options.get("thresholds") or {})

    integ = None
    if callable(options.get("reevaluate")):
        integ = integration_stability(result, options["reevaluate"], np.asarray(units),
                                      se=options.get("se"), names=[s.name for s in specs],
                                      headlines=options.get("headlines"), thresholds=th)
    ms = None
    if options.get("multistart_results"):
        ms = multistart_summary(options["multistart_results"], np.asarray(units))

    inp = DiagnosticInputs(
        result=result, specs=specs, units=np.asarray(units, dtype=float), n=int(n),
        run_id=str(_pick("run_id", options, spec, aux, "")), spec_name=str(getattr(spec, "name", "") or ""),
        spec_hash=str(_pick("hash", options, spec, aux, "")),
        normalisations=list(_pick("normalisations", options, spec, aux, ()) or ()),
        stage=result.stage, code_commit=str(_pick("code_commit", options, spec, aux, "")),
        S=S, X1=_pick("X1", options, spec, aux), Z=_pick("Z", options, spec, aux),
        linear_names=list(_pick("linear_names", options, spec, aux, ()) or ()),
        beta=_pick("beta", options, None, aux), xi=xi, G_full=_pick("G_full", options, None, aux),
        gradient=_pick("gradient", options, None, aux),
        named_directions=_pick("named_directions", options, spec, aux),
        thresholds=th, economics=economics, integration=integ, multistart=ms,
        price_range=_pick("price_range", options, spec, aux),
        cross_group_diversion=_pick("cross_group_diversion", options, spec, aux),
        binary_columns=list(_pick("binary_columns", options, spec, aux, ()) or ()),
        files=dict(_pick("files", options, None, aux, {}) or {}))
    return build_report(inp)

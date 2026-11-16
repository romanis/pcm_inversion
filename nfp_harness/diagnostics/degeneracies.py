"""Named degeneracy detectors (DESIGN 5.4). Owned by WP6.

Each detector reads the final point and the accepted-step history and returns
``{"name", "detected", "evidence", "message"}``.

``layout`` may be a ``ParameterLayout`` (anything with a ``specs`` attribute) or a sequence of
``ParameterSpec``. Natural units come from the ``units`` argument when given, else from
``ParameterSpec.unit``, else 1.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from ..types import EvalStatus, HistoryRecord, ParameterSpec

__all__ = [
    "DEGENERACIES",
    "DEGENERACY_DEFAULTS",
    "detect_all",
    "scale_artifact",
    "sd_xi_blowup",
    "run_to_bound",
    "normalisation_cap_active",
    "run_to_infinity",
    "segmented_corner",
    "vertical_collapse",
    "layout_specs",
    "layout_units",
]

DEGENERACIES = (
    "SCALE_ARTIFACT",
    "SD_XI_BLOWUP",
    "RUN_TO_BOUND",
    "NORMALISATION_CAP_ACTIVE",
    "RUN_TO_INFINITY",
    "SEGMENTED_CORNER",
    "VERTICAL_COLLAPSE",
)

DEGENERACY_DEFAULTS: dict[str, float] = {
    "scale_q_fixed_fall": 10.0,
    "scale_q_cue_change": 2.0,
    "scale_sd_xi_fall": 2.0,
    "blowup_q_fall": 0.5,
    "blowup_sd_xi_rise": 2.0,
    "blowup_kurtosis": 10.0,
    "blowup_top_share": 0.25,
    "blowup_top_fraction": 0.01,
    "bound_tol": 1e-3,
    "infinity_window": 10,
    "infinity_factor": 5.0,
    "infinity_q_fall": 0.10,
    "corner_share": 0.2,
    "corner_diversion": 0.01,
    "vertical_spread": 0.05,
    "vertical_cross": 2.0,
    "vertical_own": -50.0,
}


def _merge(thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(DEGENERACY_DEFAULTS)
    if thresholds:
        out.update({k: v for k, v in thresholds.items() if k in DEGENERACY_DEFAULTS})
    return out


def layout_specs(layout: Any) -> list[ParameterSpec]:
    """The parameter specs of a layout object or a plain sequence of specs."""
    specs = getattr(layout, "specs", layout)
    return list(specs)


def layout_units(layout: Any, units: np.ndarray | None = None) -> np.ndarray:
    if units is not None:
        return np.asarray(units, dtype=float)
    return np.array([s.unit if s.unit is not None else 1.0 for s in layout_specs(layout)], dtype=float)


def _is_spread(s: ParameterSpec) -> bool:
    """A taste spread: a taste parameter with a sign normalisation (lower bound 0) or a sigma_ name."""
    return s.role == "taste" and (s.lower >= 0 or s.name.startswith("sigma"))


def _accepted(history: Sequence[HistoryRecord]) -> list[HistoryRecord]:
    return [h for h in history if h.accepted and h.status is EvalStatus.OK and np.isfinite(h.q)]


def _result(name: str, detected: bool, evidence: dict[str, Any], message: str) -> dict[str, Any]:
    return {"name": name, "detected": bool(detected), "evidence": evidence, "message": message if detected else ""}


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return math.inf if a > 0 else 1.0
    return a / b


def scale_artifact(history: Sequence[HistoryRecord], thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SCALE_ARTIFACT (LEARNINGS 1): q_fixed falls by more than 10x while q_cue changes by less than 2x
    and sd(xi) falls by more than 2x. Compares the first and the last accepted point."""
    th = _merge(thresholds)
    acc = [h for h in _accepted(history) if np.isfinite(h.q_fixed) and np.isfinite(h.q_cue) and np.isfinite(h.sd_xi)]
    if len(acc) < 2:
        return _result("SCALE_ARTIFACT", False, {"accepted_points": len(acc)}, "")
    a, b = acc[0], acc[-1]
    qf = _ratio(a.q_fixed, b.q_fixed)
    qc = max(_ratio(a.q_cue, b.q_cue), _ratio(b.q_cue, a.q_cue))
    sd = _ratio(a.sd_xi, b.sd_xi)
    ev = {"q_fixed": [a.q_fixed, b.q_fixed], "q_cue": [a.q_cue, b.q_cue], "sd_xi": [a.sd_xi, b.sd_xi],
          "q_fixed_fall_factor": qf, "q_cue_change_factor": qc, "sd_xi_fall_factor": sd}
    det = qf > th["scale_q_fixed_fall"] and qc < th["scale_q_cue_change"] and sd > th["scale_sd_xi_fall"]
    msg = (f"q fell with sd(xi)^2 while the scale-free objective did not: the fixed-W objective is rewarding a "
           f"small xi through the utility scale (q_fixed {a.q_fixed:.3g} to {b.q_fixed:.3g}, q_cue {a.q_cue:.3g} "
           f"to {b.q_cue:.3g}, sd(xi) {a.sd_xi:.3g} to {b.sd_xi:.3g}).")
    return _result("SCALE_ARTIFACT", det, ev, msg)


def sd_xi_blowup(history: Sequence[HistoryRecord], xi: np.ndarray | None,
                 thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """SD_XI_BLOWUP (LEARNINGS 13): q falls by more than 50% while sd(xi) rises by more than 2x; or
    kurtosis(xi) > 10; or the top 1% of products carry more than 25% of sum xi^2.

    Kurtosis is Pearson's (3 for a normal)."""
    th = _merge(thresholds)
    ev: dict[str, Any] = {}
    reasons: list[str] = []
    acc = [h for h in _accepted(history) if np.isfinite(h.sd_xi)]
    if len(acc) >= 2:
        a, b = acc[0], acc[-1]
        q_ratio = _ratio(b.q, a.q)
        sd_rise = _ratio(b.sd_xi, a.sd_xi)
        ev.update(q=[a.q, b.q], sd_xi=[a.sd_xi, b.sd_xi], q_ratio=q_ratio, sd_xi_rise_factor=sd_rise)
        if q_ratio < th["blowup_q_fall"] and sd_rise > th["blowup_sd_xi_rise"]:
            reasons.append(f"q {a.q:.3g} to {b.q:.3g} while sd(xi) rose {sd_rise:.1f}x")
    if xi is not None and np.size(xi) >= 4:
        x = np.asarray(xi, dtype=float)
        x = x[np.isfinite(x)]
        kurt = float(stats.kurtosis(x, fisher=False)) if x.std() > 0 else 0.0
        sq = np.sort(x ** 2)[::-1]
        m = max(1, int(math.ceil(th["blowup_top_fraction"] * x.size)))
        top = float(sq[:m].sum() / sq.sum()) if sq.sum() > 0 else 0.0
        ev.update(kurtosis=kurt, top_share=top, top_products=m)
        if kurt > th["blowup_kurtosis"]:
            reasons.append(f"kurtosis(xi) = {kurt:.1f}")
        if top > th["blowup_top_share"]:
            reasons.append(f"the top {m} products carry {100 * top:.0f}% of sum xi^2")
    msg = ("q fell while xi grew and became lumpy: the objective is being lowered by noisy moments, not by fit ("
           + "; ".join(reasons) + ").")
    return _result("SD_XI_BLOWUP", bool(reasons), ev, msg)


def _bound_hits(theta: np.ndarray, specs: list[ParameterSpec], units: np.ndarray, tol: float):
    for i, s in enumerate(specs):
        for side, b in (("lower", s.lower), ("upper", s.upper)):
            if np.isfinite(b) and abs(theta[i] - b) <= tol * units[i]:
                yield i, s, side, b


def run_to_bound(theta: np.ndarray, gradient: np.ndarray | None, layout: Any,
                 thresholds: Mapping[str, Any] | None = None, units: np.ndarray | None = None) -> dict[str, Any]:
    """RUN_TO_BOUND: a non-cap parameter within 1e-3 units of a bound with outward gradient.

    ``gradient`` is dq/dtheta (any positive scaling). Outward means q falls when moving past the
    bound: gradient > 0 at a lower bound, < 0 at an upper bound. Without a gradient every bound hit
    counts. The upper bound of a cap parameter is excluded (NORMALISATION_CAP_ACTIVE)."""
    th = _merge(thresholds)
    specs = layout_specs(layout)
    u = layout_units(layout, units)
    theta = np.asarray(theta, dtype=float)
    hits = []
    for i, s, side, b in _bound_hits(theta, specs, u, th["bound_tol"]):
        if not s.free or (s.cap and side == "upper"):
            continue
        g = None if gradient is None else float(gradient[i])
        outward = g is None or (g > 0 if side == "lower" else g < 0)
        if outward:
            hits.append({"name": s.name, "side": side, "bound": float(b), "value": float(theta[i]), "gradient": g})
    msg = " ".join(f"{h['name']} ended at its {h['side']} bound {h['bound']:.4g}; the objective wants to go further."
                   for h in hits)
    return _result("RUN_TO_BOUND", bool(hits), {"parameters": hits}, msg)


def normalisation_cap_active(theta: np.ndarray, layout: Any, units: np.ndarray | None = None,
                             thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """NORMALISATION_CAP_ACTIVE: a cap parameter at its cap (a stated normalisation, not a failure)."""
    th = _merge(thresholds)
    specs = layout_specs(layout)
    u = layout_units(layout, units)
    theta = np.asarray(theta, dtype=float)
    hits = [{"name": s.name, "cap": float(b), "value": float(theta[i])}
            for i, s, side, b in _bound_hits(theta, specs, u, th["bound_tol"]) if s.cap and side == "upper"]
    msg = " ".join(f"{h['name']} is at its cap {h['cap']:.4g}, a stated normalisation of a direction the data "
                   f"leave quiet." for h in hits)
    return _result("NORMALISATION_CAP_ACTIVE", bool(hits), {"parameters": hits}, msg)


def run_to_infinity(history: Sequence[HistoryRecord], layout: Any,
                    thresholds: Mapping[str, Any] | None = None, units: np.ndarray | None = None) -> dict[str, Any]:
    """RUN_TO_INFINITY: over the last 10 accepted steps |theta_k| rose monotonically to above 5 times
    its start in natural units while q fell by less than 10%.

    "Above 5 times its start" is ``|theta_k| > 5 max(|start_k|, unit_k)``, so a parameter starting at
    zero must travel five natural units. "q fell by less than 10%" is measured against (1 + q) at the
    start of the window, the scale of the stopping rules in DESIGN 4.6, so a q that is already near
    zero and keeps halving still counts as little gain."""
    th = _merge(thresholds)
    specs = layout_specs(layout)
    u = layout_units(layout, units)
    acc = _accepted(history)
    w = int(th["infinity_window"])
    if len(acc) < w + 1:
        return _result("RUN_TO_INFINITY", False, {"accepted_points": len(acc), "window": w}, "")
    win = acc[-(w + 1):]
    T = np.abs(np.array([np.ravel(h.theta) for h in win], dtype=float))
    if T.ndim != 2 or T.shape[1] != len(specs):
        return _result("RUN_TO_INFINITY", False, {"note": "history theta does not match the layout"}, "")
    q0, q1 = win[0].q, win[-1].q
    small_gain = (q0 - q1) < th["infinity_q_fall"] * (1.0 + q0)
    hits = []
    for i, s in enumerate(specs):
        if not s.free:
            continue
        col = T[:, i]
        mono = bool(np.all(np.diff(col) >= 0) and col[-1] > col[0])
        far = col[-1] > th["infinity_factor"] * max(abs(s.start), u[i])
        if mono and far:
            hits.append({"name": s.name, "from": float(win[0].theta[i]), "to": float(win[-1].theta[i])})
    det = small_gain and bool(hits)
    msg = " ".join(f"{h['name']} is running away ({h['from']:.4g} to {h['to']:.4g}) with little gain in q "
                   f"({q0:.4g} to {q1:.4g})." for h in hits)
    return _result("RUN_TO_INFINITY", det, {"parameters": hits, "q": [q0, q1], "window": w}, msg)


def segmented_corner(theta: np.ndarray, layout: Any, price_range: float | None, cross_group_diversion: float | None,
                     thresholds: Mapping[str, Any] | None = None,
                     binary_columns: Sequence[str] = ()) -> dict[str, Any]:
    """SEGMENTED_CORNER (LEARNINGS 13): for a taste spread on a binary column,
    ``2 (1 - Phi(range / sigma)) > 0.2``, or cross-group diversion below 1%.

    ``binary_columns`` names the 0/1 characteristics (body dummies for Petrin)."""
    th = _merge(thresholds)
    specs = layout_specs(layout)
    theta = np.asarray(theta, dtype=float)
    hits = []
    if price_range is not None and price_range > 0:
        for i, s in enumerate(specs):
            if _is_spread(s) and s.column in set(binary_columns) and theta[i] != 0:
                share = float(2 * stats.norm.sf(price_range / abs(theta[i])))
                if share > th["corner_share"]:
                    hits.append({"name": s.name, "column": s.column, "value": float(theta[i]), "share_beyond": share})
    msgs = [f"{h['name']} = {h['value']:.4g} splits consumers into separate {h['column']} pools "
            f"({100 * h['share_beyond']:.0f}% have taste beyond the price range {price_range:.4g})." for h in hits]
    div_hit = cross_group_diversion is not None and cross_group_diversion < th["corner_diversion"]
    if div_hit:
        msgs.append(f"Cross-group diversion is {100 * cross_group_diversion:.2f}%, below "
                    f"{100 * th['corner_diversion']:.0f}%: groups are separate markets.")
    ev = {"parameters": hits, "price_range": price_range, "cross_group_diversion": cross_group_diversion}
    return _result("SEGMENTED_CORNER", bool(hits) or div_hit, ev, " ".join(msgs))


def _econ_median(economics: Mapping[str, Any] | None, *keys: str) -> float | None:
    if not economics:
        return None
    for k in keys:
        v = economics.get(k)
        if isinstance(v, Mapping) and v.get("median") is not None:
            return float(v["median"])
        if isinstance(v, (int, float)):
            return float(v)
    return None


def vertical_collapse(theta: np.ndarray, layout: Any, economics: Mapping[str, Any] | None,
                      thresholds: Mapping[str, Any] | None = None, units: np.ndarray | None = None) -> dict[str, Any]:
    """VERTICAL_COLLAPSE: all non-cap taste spreads below 0.05 natural units, or median nonzero
    cross-elasticities per product <= 2, or median own elasticity below -50.

    ``economics`` keys read: ``own_elasticity`` and ``nonzero_cross`` (or ``nonzero_cross_elasticities``),
    each a number or a mapping with ``median``."""
    th = _merge(thresholds)
    specs = layout_specs(layout)
    u = layout_units(layout, units)
    theta = np.asarray(theta, dtype=float)
    spreads = [(s.name, abs(theta[i]) / u[i]) for i, s in enumerate(specs) if _is_spread(s) and not s.cap]
    reasons = []
    if spreads and all(v < th["vertical_spread"] for _, v in spreads):
        reasons.append(f"all {len(spreads)} non-cap taste spreads are below {th['vertical_spread']} natural units")
    own = _econ_median(economics, "own_elasticity")
    cross = _econ_median(economics, "nonzero_cross", "nonzero_cross_elasticities")
    if cross is not None and cross <= th["vertical_cross"]:
        reasons.append(f"median nonzero cross-elasticities per product is {cross:.3g}")
    if own is not None and own < th["vertical_own"]:
        reasons.append(f"median own elasticity is {own:.3g}")
    el = f"{own:.3g}" if own is not None else "not computed"
    msg = (f"Taste spreads collapsed: the model is close to vertical (median own elasticity {el}). Evidence: "
           + "; ".join(reasons) + ".")
    ev = {"spreads_natural_units": dict(spreads), "median_own_elasticity": own, "median_nonzero_cross": cross}
    return _result("VERTICAL_COLLAPSE", bool(reasons), ev, msg)


def detect_all(history: Sequence[HistoryRecord], final: Mapping[str, Any], spec: Any,
               thresholds: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """detect_all(history, final, spec, thresholds=None) -> one entry per name in ``DEGENERACIES``.

    ``final`` keys: ``theta`` (required), ``layout`` (else ``spec.layout``), ``units``, ``gradient``
    (dq/dtheta), ``xi``, ``economics``, ``price_range``, ``cross_group_diversion``, ``binary_columns``.
    Missing keys fall back to attributes of ``spec`` with the same name, then to None.
    """
    def get(key: str, default: Any = None) -> Any:
        if key in final and final[key] is not None:
            return final[key]
        return getattr(spec, key, default) if spec is not None else default

    theta = np.asarray(final["theta"], dtype=float)
    layout = get("layout")
    units = get("units")
    return [
        scale_artifact(history, thresholds),
        sd_xi_blowup(history, get("xi"), thresholds),
        run_to_bound(theta, get("gradient"), layout, thresholds, units),
        normalisation_cap_active(theta, layout, units, thresholds),
        run_to_infinity(history, layout, thresholds, units),
        segmented_corner(theta, layout, get("price_range"), get("cross_group_diversion"), thresholds,
                         get("binary_columns", ()) or ()),
        vertical_collapse(theta, layout, get("economics"), thresholds, units),
    ]

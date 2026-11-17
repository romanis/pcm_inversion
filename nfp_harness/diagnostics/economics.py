"""Economic sanity at the estimate against spec ranges (DESIGN 5.6). Owned by WP8."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..types import DiagnosticBlock, OptimizeResult, Problem

__all__ = ["economics_block", "economics_values", "plausible_flags", "summary", "PLAUSIBLE_STATS",
           "OWN_ELASTICITY_EXTREME"]

OWN_ELASTICITY_EXTREME = -50.0

#: Statistic names ``spec.plausible`` may declare ranges for. ``spread_buyer_dollars`` takes one range for
#: every taste spread, or a mapping parameter name -> range; ``spread_buyer_dollars:<name>`` targets one.
#: A bound may be the string ``"price_range"`` (max p - min p pooled, or ``spec.price_range``).
PLAUSIBLE_STATS = (
    "own_elasticity_median",
    "share_own_elasticity_below_-50",
    "nonzero_cross_median",
    "diversion_leaving_group_median",
    "margin_median",
    "inside_share_elasticity_median",
    "spread_buyer_dollars",
    "sd_xi_over_sd_delta",
)


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        v = spec.get(key, default)
    else:
        v = getattr(spec, key, default)
    return default if v is None else v


def summary(v: np.ndarray) -> dict[str, float | None]:
    """summary(v) -> median, p05, p95, mean and count of the finite entries (None when there are none)."""
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": None, "p05": None, "p95": None, "mean": None, "n": 0}
    return {"median": float(np.median(a)), "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "n": int(a.size)}


def _bound(b: Any, price_range: float | None) -> float:
    if b is None:
        return math.nan
    if isinstance(b, str):
        if b != "price_range":
            raise ValueError(f"plausible bound {b!r}: only numbers, None and 'price_range' are allowed")
        return math.nan if price_range is None else float(price_range)
    return float(b)


def _fmt(b: float, side: str) -> str:
    if math.isnan(b):
        return "-inf" if side == "lo" else "inf"
    return f"{b:.4g}"


def plausible_flags(stats: Mapping[str, float | None], plausible: Mapping[str, Any],
                    price_range: float | None = None) -> tuple[list[str], list[str], dict[str, Any]]:
    """plausible_flags(stats, plausible, price_range=None) -> (flags, notes, checked).

    ``stats`` maps a statistic name to its value. ``plausible`` maps a name of ``PLAUSIBLE_STATS`` to
    ``(lo, hi)`` (None for an open side). Each value outside its range gives
    ``"FLAG: {stat} = {v} is outside the declared plausible range [{lo}, {hi}]."``; a declared range whose
    statistic was not computed gives a note. ``checked`` records every range applied.
    """
    flags: list[str] = []
    notes: list[str] = []
    checked: dict[str, Any] = {}
    ranges: dict[str, Any] = {}
    for key, rng in plausible.items():
        if key == "spread_buyer_dollars":
            names = [k.split(":", 1)[1] for k in stats if k.startswith("spread_buyer_dollars:")]
            for nm in names:
                r = rng.get(nm) if isinstance(rng, Mapping) else rng
                if r is not None:
                    ranges.setdefault(f"spread_buyer_dollars:{nm}", r)
            if not names:
                notes.append("Plausible range for spread_buyer_dollars not checked: no taste spread was computed.")
        else:
            ranges[key] = rng
    for key, rng in ranges.items():
        lo_s, hi_s = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) else (None, rng)
        lo, hi = _bound(lo_s, price_range), _bound(hi_s, price_range)
        v = stats.get(key)
        checked[key] = {"lo": None if math.isnan(lo) else lo, "hi": None if math.isnan(hi) else hi,
                        "value": v}
        if v is None or not np.isfinite(v):
            notes.append(f"Plausible range for {key} not checked: statistic not computed.")
            continue
        if (not math.isnan(lo) and v < lo) or (not math.isnan(hi) and v > hi):
            flags.append(f"FLAG: {key} = {v:.4g} is outside the declared plausible range "
                         f"[{_fmt(lo, 'lo')}, {_fmt(hi, 'hi')}].")
    return flags, notes, checked


def economics_values(model: Any, theta: np.ndarray, delta: np.ndarray, spec: Any = None,
                     xi: np.ndarray | None = None) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """economics_values(model, theta, delta, spec=None, xi=None) -> (values, messages, post).

    The numbers of the economics block at a fixed estimate: ``postestimation`` on every market plus
    ``spread_in_buyer_dollars``. ``post`` is the ``postestimation`` output (for ``product_level.csv``).
    Spec keys read: ``firm_column`` (default ``"firm_ids"``), ``groups``, ``group_other`` (default
    ``"other"``), ``price_range`` (default pooled max p - min p).
    """
    from ..models import pcm_post

    post = pcm_post.postestimation(model, theta, delta, firm_ids=_get(spec, "firm_column", "firm_ids"),
                                   groups=_get(spec, "groups"), group_other=_get(spec, "group_other", "other"))
    pr = post["products"]
    own = pr["own_elasticity"]
    nz = pr["nonzero_cross"]
    margin = pr["margin"]
    msgs: list[str] = []
    values: dict[str, Any] = {
        "own_elasticity": summary(own),
        "share_own_elasticity_below_-50": float(np.mean(own[np.isfinite(own)] < OWN_ELASTICITY_EXTREME))
        if np.isfinite(own).any() else None,
        "nonzero_cross": summary(nz),
        "nonzero_cross_ratio": summary(pr["nonzero_cross_ratio"]),
        "outside_diversion": summary(pr["outside_diversion"]),
        "margin": summary(margin),
        "negative_margins": int(np.sum(margin < 0)),
        "margins_above_one": int(np.sum(margin > 1)),
        "nash_singular_markets": [str(m) for m in post["nash_singular"]],
        "inside_share": {str(m["market_id"]): m["inside_share"] for m in post["markets"]},
        "inside_share_elasticity": summary(np.array([m["inside_share_elasticity"] for m in post["markets"]])),
    }
    if post["group_names"]:
        values["diversion_leaving_group"] = summary(pr["diversion_leaving_group"])
        values["cross_group_diversion"] = values["diversion_leaving_group"]["median"]
        gm = post["group_matrix"]
        values["group_diversion"] = {
            g: {**{h: float(gm[a, b]) for b, h in enumerate(post["group_names"])}, "outside": float(gm[a, -1])}
            for a, g in enumerate(post["group_names"])}
    if post["nash_singular"]:
        msgs.append(f"FLAG: NASH_JACOBIAN_SINGULAR in markets {', '.join(values['nash_singular_markets'])}: the "
                    f"ownership-weighted ds/dp has rcond below 1e-12, so costs and any price counterfactual in "
                    f"those markets cannot be computed.")
    if values["negative_margins"] or values["margins_above_one"]:
        msgs.append(f"FLAG: {values['negative_margins']} products have a negative margin and "
                    f"{values['margins_above_one']} a margin above 1.")
    if not np.isfinite(pr["markup"]).any() and not post["nash_singular"]:
        msgs.append("Margins not computed: no firm ids.")

    prices = pr["prices"]
    price_range = _get(spec, "price_range")
    if price_range is None and np.isfinite(prices).any():
        price_range = float(np.nanmax(prices) - np.nanmin(prices))
    values["price_range"] = price_range
    sb = pcm_post.spread_in_buyer_dollars(model, theta, delta)
    values["mean_alpha_buyers"] = sb["mean_alpha_buyers"]
    values["spread_buyer_dollars"] = sb["spreads"]
    if xi is not None and np.size(xi) == np.size(delta):
        sd_d = float(np.std(delta))
        values["sd_xi_over_sd_delta"] = float(np.std(xi)) / sd_d if sd_d > 0 else None
    else:
        values["sd_xi_over_sd_delta"] = None
    return values, msgs, post


def economics_block(problem: Problem, result: OptimizeResult, spec: Any) -> DiagnosticBlock:
    """economics_block(problem, result, spec) -> DiagnosticBlock.

    Own-price elasticities, nonzero cross-elasticities per product, diversion (within spec groups and
    to the outside), Nash margins (NASH_JACOBIAN_SINGULAR when ownership-weighted ds/dp has rcond
    below 1e-12), inside share semi-elasticity, taste spreads in buyer dollars. Out-of-range values
    against ``spec.plausible`` are flags: "{stat} = {v} is outside the declared plausible range [{lo}, {hi}]."

    Reads the PCM model from ``problem.model``, theta from ``result.theta`` and delta and xi from
    ``result.evaluation.aux``. Evaluates the kernel at that delta (no inversion). Spec keys: see
    ``economics_values`` and ``PLAUSIBLE_STATS``.
    """
    ev = result.evaluation
    blk = DiagnosticBlock(name="economics")
    if ev is None or not ev.ok or ev.aux.get("delta") is None:
        blk.messages = ["Economic sanity not computed: no successful evaluation with delta at the final point."]
        return blk
    model = getattr(problem, "model", None)
    if model is None or not hasattr(model, "kernel_market"):
        blk.messages = ["Economic sanity not computed: the problem's model is not a PCM model."]
        return blk
    values, msgs, _ = economics_values(model, np.asarray(result.theta, dtype=float), np.asarray(ev.aux["delta"]),
                                       spec, ev.aux.get("xi"))
    stats: dict[str, float | None] = {
        "own_elasticity_median": values["own_elasticity"]["median"],
        "share_own_elasticity_below_-50": values["share_own_elasticity_below_-50"],
        "nonzero_cross_median": values["nonzero_cross"]["median"],
        "diversion_leaving_group_median": (values.get("diversion_leaving_group") or {}).get("median"),
        "margin_median": values["margin"]["median"],
        "inside_share_elasticity_median": values["inside_share_elasticity"]["median"],
        "sd_xi_over_sd_delta": values["sd_xi_over_sd_delta"],
    }
    for nm, v in values["spread_buyer_dollars"].items():
        stats[f"spread_buyer_dollars:{nm}"] = v
    flags, notes, checked = plausible_flags(stats, dict(_get(spec, "plausible", {}) or {}), values["price_range"])
    own = values["own_elasticity"]
    head = (f"Median own-price elasticity {own['median']:.3g} (5th {own['p05']:.3g}, 95th {own['p95']:.3g}); "
            f"median nonzero cross-elasticities per product {values['nonzero_cross']['median']:.3g}; "
            f"median margin {values['margin']['median']:.3g}."
            if own["median"] is not None and values["margin"]["median"] is not None
            else f"Median own-price elasticity {own['median']}; median margin {values['margin']['median']}.")
    blk.messages = [head] + msgs + flags + notes
    blk.values = values
    blk.thresholds = {"plausible": checked, "nash_rcond_min": 1e-12, "cross_tol": 1e-10}
    blk.status = "flag" if (flags or any(m.startswith("FLAG:") for m in msgs)) else "ok"
    return blk

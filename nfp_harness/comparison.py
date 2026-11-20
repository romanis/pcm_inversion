"""Model-agnostic comparison quantities for demand estimates (DESIGN 6.5 item 5).

Both replication scripts of a paper (``estimate_pcm.py`` on the harness, ``estimate_blp.py`` on pyblp)
report the same numbers under the same keys, so one table can put them side by side. Inputs are per
market arrays that any demand model provides: shares ``s`` (J,), the price Jacobian ``ds_dp`` (J, J)
with ``ds_dp[j, k] = ds_j / dp_k``, prices, costs and product group labels; and per-consumer surplus
arrays for the counterfactual scenarios.

Definitions are those of ``models.pcm_post`` (DESIGN 5.6): own elasticity ``e_jj``, nonzero
cross-elasticities ``|e_jk| > 1e-10``, diversion ``D_jk = -(ds_k/dp_j)/(ds_j/dp_j)``, the share of inside
diversion leaving j's group, and the share-weighted group diversion matrix pooled over markets.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .models.pcm_post import cs_by_group, diversion, elasticities, group_diversion, nonzero_cross

__all__ = ["market_quantities", "products_from_markets", "summarise", "cs_gain_by_group", "scalar_items", "gmy_table",
           "weighted_median"]


def market_quantities(s: np.ndarray, ds_dp: np.ndarray, p: np.ndarray, labels: np.ndarray,
                      names: Sequence[str], cost: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """market_quantities(s, ds_dp, p, labels, names, cost=None) -> per-product arrays of one market.

    Keys: ``own_elasticity``, ``nonzero_cross``, ``nonzero_cross_ratio``, ``outside_diversion``,
    ``diversion_leaving_group``, ``to_group`` (J, G), ``margin`` ((p - c)/p, NaN without ``cost``),
    ``shares``, ``prices``, ``group``.
    """
    s = np.asarray(s, dtype=float)
    p = np.asarray(p, dtype=float)
    e = elasticities(s, ds_dp, p)
    D = diversion(ds_dp)
    gd = group_diversion(D, labels, names)
    J = s.size
    nz = nonzero_cross(e)
    return {
        "own_elasticity": np.diag(e).copy(), "nonzero_cross": nz.astype(float),
        "nonzero_cross_ratio": nz / max(J - 1, 1), "outside_diversion": gd["outside"],
        "diversion_leaving_group": gd["leaving"], "to_group": gd["to_group"],
        "margin": np.full(J, np.nan) if cost is None else (p - np.asarray(cost, dtype=float)) / p,
        "shares": s, "prices": p, "group": np.asarray(labels).astype(str),
    }


def products_from_markets(markets: Sequence[Mapping[str, np.ndarray]], names: Sequence[str]
                          ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """products_from_markets(markets, names) -> (stacked per-product arrays, group matrix).

    The group matrix is (G, G + 1): from each source group (rows), the share-weighted mean diversion to
    each group and to the outside good (last column), pooled over markets, as in
    ``pcm_post.postestimation``.
    """
    keys = ("own_elasticity", "nonzero_cross", "nonzero_cross_ratio", "outside_diversion",
            "diversion_leaving_group", "margin", "shares", "prices", "group")
    prod = {k: np.concatenate([np.asarray(m[k]) for m in markets]) for k in keys}
    G = len(names)
    num = np.zeros((G, G + 1))
    den = np.zeros(G)
    for m in markets:
        tg, out, s, lab = np.asarray(m["to_group"]), np.asarray(m["outside_diversion"]), m["shares"], m["group"]
        for a, g in enumerate(names):
            sel = (lab == str(g)) & np.all(np.isfinite(tg), axis=1)
            num[a, :-1] += s[sel] @ tg[sel]
            num[a, -1] += s[sel] @ out[sel]
            den[a] += s[sel].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        gm = num / den[:, None]
    return prod, gm


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    """weighted_median(v, w) -> the smallest v_k with at least half the weight of the finite (v, w > 0) pairs at or
    below it; NaN when there are none. With equal weights and an odd count this is the median."""
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    order = np.argsort(v[ok], kind="stable")
    vs, cw = v[ok][order], np.cumsum(w[ok][order])
    return float(vs[np.searchsorted(cw, 0.5 * cw[-1])])


def _stat(v: np.ndarray, fn: Any) -> float | None:
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else None


def summarise(products: Mapping[str, np.ndarray], group_matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """summarise(products, group_matrix, names) -> comparison keys over all products.

    Scalar keys: ``own_elasticity_median``, ``own_elasticity_mean``, ``share_own_elasticity_below_-50``,
    ``margin_median``, ``margin_mean``, ``nonzero_cross_median``, ``nonzero_cross_ratio_median``,
    ``outside_diversion_median``, ``diversion_leaving_group_median``. Nested: ``group_diversion``
    (source group -> {destination group or "outside": share-weighted mean diversion}).
    """
    own = products["own_elasticity"]
    fin = np.asarray(own, dtype=float)[np.isfinite(own)]
    out: dict[str, Any] = {
        "own_elasticity_median": _stat(own, np.median), "own_elasticity_mean": _stat(own, np.mean),
        "share_own_elasticity_below_-50": float(np.mean(fin < -50.0)) if fin.size else None,
        "margin_median": _stat(products["margin"], np.median), "margin_mean": _stat(products["margin"], np.mean),
        "nonzero_cross_median": _stat(products["nonzero_cross"], np.median),
        "nonzero_cross_ratio_median": _stat(products["nonzero_cross_ratio"], np.median),
        "outside_diversion_median": _stat(products["outside_diversion"], np.median),
        "diversion_leaving_group_median": _stat(products["diversion_leaving_group"], np.median),
    }
    out["group_diversion"] = {
        str(g): {**{str(h): float(group_matrix[a, b]) for b, h in enumerate(names)},
                 "outside": float(group_matrix[a, -1])}
        for a, g in enumerate(names)}
    return out


def cs_gain_by_group(per_draw: Mapping[str, np.ndarray], weights: np.ndarray, labels: np.ndarray,
                     households: float, baseline: str = "observed") -> list[dict[str, Any]]:
    """cs_gain_by_group(per_draw, weights, labels, households, baseline="observed") -> rows by label.

    ``per_draw`` maps scenario -> (R,) surplus per consumer in price units per household. Each row holds
    ``group``, ``mass`` (weight share), ``cs_<scenario>`` (mean per household in the group) and, for each
    scenario other than ``baseline``, ``gain_<scenario>_musd = households x sum_{i in group} w_i
    (CS_baseline_i - CS_scenario_i)``: the loss from moving to the scenario, i.e. the gain from the
    baseline's choice set. With prices in $1000 and households in thousands this is in $ million.
    """
    rows = cs_by_group(dict(per_draw), weights, labels)
    for r in rows:
        for k in per_draw:
            if k != baseline:
                r[f"gain_{k}_musd"] = float(households * (r[f"cs_total_{baseline}"] - r[f"cs_total_{k}"]))
    return rows


def scalar_items(d: Mapping[str, Any]) -> dict[str, Any]:
    """The entries of d whose values are numbers, strings or None (for ``pcm_post.compare_blp``)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, str):
            out[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[k] = None if not math.isfinite(float(v)) else float(v)
    return out


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if not math.isfinite(x) else f"{x:.{digits}f}"


def _year_get(d: Mapping[Any, Any] | None, y: int) -> Any:
    if not d:
        return None
    return d.get(y, d.get(str(y)))


def gmy_table(pcm: Mapping[str, Mapping[str, Any]], blp: Mapping[str, Mapping[str, Any]],
              paper: Mapping[str, Any] | None = None) -> str:
    """gmy_table(pcm, blp, paper=None) -> the PCM-vs-BLP comparison table of the GMY spec (section 4) as text.

    ``pcm`` maps a column label to a PCM summary (``GMYSpec.summary``), ``blp`` a label to an ``estimate_blp.py`` JSON
    (``table5``, ``lerner_median_by_year``, ``second_choice``, ``micro``, ``own_elasticity_mean_by_year``), ``paper``
    the paper's quoted numbers (definitions unverified). Every row uses one definition for every column; a missing
    entry prints as ``-``. Second-choice means are over the years of the first PCM column.
    """
    paper = dict(paper or {})
    cols = list(pcm) + list(blp) + (["paper (definition UNVERIFIED)"] if paper else [])
    first = next(iter(pcm.values()), {})
    years = sorted(int(y) for y in (first.get("lerner_median_by_year") or first.get("own_elasticity_median_by_year")
                                    or {}))
    lines: list[str] = []
    w = max(14, *(len(c) + 2 for c in cols))

    def row(label: str, vals: Sequence[Any], digits: int = 3) -> None:
        lines.append(f"{label:52s}" + "".join(f"{_fmt(v, digits):>{w}s}" for v in vals))

    lines.append(f"{'':52s}" + "".join(f"{c:>{w}s}" for c in cols))
    lines.append("Own-price elasticity by income quintile, mean over products (quintile demand s_jq, e_jq = p_j/s_jq "
                 "ds_jq/dp_j; PCM: mean / 5-95% trimmed mean [median])")
    for y in (1980, 2018):
        for q in range(5):
            vals: list[Any] = []
            for s in pcm.values():
                t = _year_get(s.get("table5"), y)
                trim = t.get("trimmed_mean_over_products") if t else None
                vals.append(f"{_fmt(t['mean_over_products'][q], 2)}"
                            + (f" / {_fmt(trim[q], 2)}" if trim else "")
                            + f" [{_fmt(t['median_over_products'][q], 2)}]"
                            if t and "median_over_products" in t else (t["mean_over_products"][q] if t else None))
            for b in blp.values():
                t = _year_get(b.get("table5"), y)
                vals.append(t["mean_over_products"][q] if t else None)
            if paper:
                t = _year_get(paper.get("table5"), y)
                vals.append(t[q] if t else None)
            row(f"  {y} Q{q + 1}", vals, 2)
    lines.append("Median Lerner index (p - mc)/p, Nash with parent ownership, median over products")
    for y in years:
        vals = [_year_get(s.get("lerner_median_by_year"), y) for s in pcm.values()]
        vals += [_year_get(b.get("lerner_median_by_year"), y) for b in blp.values()]
        if paper:
            vals.append(" / ".join(f"{_fmt(_year_get(v, y))}" for v in paper.get("lerner_median", {}).values())
                        if y in (1980, 2018) else None)
        row(f"  {y}", vals)
    na = [None] * (len(blp) + (1 if paper else 0))
    row("  all years: median over products", [s.get("margin_median") for s in pcm.values()] + na)
    row("  all years: sales-weighted median", [s.get("margin_median_sales_weighted") for s in pcm.values()] + na)
    row("  products with |own elasticity| < 1 (margin > 1)", [s.get("products_own_elasticity_above_-1")
                                                             for s in pcm.values()] + na, 0)
    lines.append("Own-price elasticity, mean over products (PCM: mean / 5-95% trimmed mean [median])")
    for y in years:
        vals: list[Any] = [f"{_fmt(_year_get(s.get('own_elasticity_mean_by_year'), y), 2)} / "
                           f"{_fmt(_year_get(s.get('own_elasticity_trimmed_mean_by_year'), y), 2)} "
                           f"[{_fmt(_year_get(s.get('own_elasticity_median_by_year'), y), 2)}]" for s in pcm.values()]
        vals += [_year_get(b.get("own_elasticity_mean_by_year"), y) for b in blp.values()]
        if paper:
            vals.append(None)
        row(f"  {y}", vals, 2)
    sc_years = sorted(int(y) for y in (first.get("second_choice") or {}))
    if sc_years:
        lines.append(f"Second-choice correlation of first and second choice, mean over {sc_years} (data: merged "
                     f"products; paper: Table 6 data / model)")
        chars = list((paper.get("table6_data") or {}).keys()) or [f"x{k}" for k in range(11)]
        for c_i, c in enumerate(chars):
            def mean_of(sc: Mapping[Any, Any] | None, key: str) -> Any:
                v = [(_year_get(sc, y) or {}).get(key, [None] * len(chars))[c_i] for y in sc_years] if sc else []
                v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
                return sum(v) / len(v) if v else None
            vals = [f"{_fmt(mean_of(s.get('second_choice'), 'pcm'))} ({_fmt(mean_of(s.get('second_choice'), 'data'))})"
                    for s in pcm.values()]
            vals += [mean_of(b.get("second_choice"), "model") for b in blp.values()]
            if paper:
                vals.append(f"{_fmt(paper['table6_data'].get(c), 2)} / {_fmt(paper['table6_model'].get(c), 2)}")
            row(f"  {c} model (data)", vals)
        row("  RMSE model - data (PCM years)", [s.get("second_choice_rmse") for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    mic = first.get("micro") or []
    if mic:
        lines.append("Micro moments (V3): data, then each model")
        bl = [{r["name"]: r.get("model") for r in (b.get("micro") or [])} for b in blp.values()]
        row("  name / data", ["data"] + [""] * (len(cols) - 1))
        for r in mic:
            vals = [f"{_fmt(next((x['pcm'] for x in (s.get('micro') or []) if x['name'] == r['name']), None))}"
                    for s in pcm.values()] + [d.get(r["name"]) for d in bl]
            if paper:
                vals.append(None)
            lines.append(f"  {r['name']:42s}{_fmt(r['data']):>10s}" + "".join(f"{_fmt(v):>{w}s}" for v in vals))
    lines.append("Diversion to the product's own vehicle type (share-weighted, pooled over markets)")
    for g in ("car", "van", "suv", "truck"):
        row(f"  {g}", [(s.get("diversion_to_own_type") or {}).get(g) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    lines.append("Consumer surplus per household, $ (PCM frontier formula, 1/alpha tail bounded by a_lo; not a headline)")
    for y in years:
        row(f"  {y}", [_year_get(s.get("cs_per_household_usd_by_year"), y) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)), 0)
    lines.append("Window anatomy and fit")
    for k in ("frontier_products_per_buying_draw", "log_alpha_range_buyers", "window_width_log_alpha"):
        row(f"  {k}", [(s.get("window_anatomy") or {}).get(k) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    bins = next((list(h) for s in pcm.values() if (h := (s.get("window_anatomy") or {}).get("frontier_size_mass"))),
                [])
    for b in bins:
        row(f"  buyer mass on frontiers of {b} product{'' if b == '1' else 's'}",
            [((s.get("window_anatomy") or {}).get("frontier_size_mass") or {}).get(b) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    row("  own_elasticity_times_window_width", [s.get("own_elasticity_times_window_width") for s in pcm.values()]
        + [None] * (len(blp) + (1 if paper else 0)), 2)
    row("  convergence", [s.get("convergence") for s in pcm.values()] + [None] * (len(blp) + (1 if paper else 0)))
    row("  q", [s.get("q") for s in pcm.values()] + [b.get("q") for b in blp.values()] + ([None] if paper else []), 2)
    for blk in (first.get("q_blocks") or {}):
        row(f"  q:{blk}", [(s.get("q_blocks") or {}).get(blk) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)), 2)
    row("  AT_BOUND", [";".join(s.get("at_bound") or []) or "none" for s in pcm.values()]
        + [None] * (len(blp) + (1 if paper else 0)))
    return "\n".join(lines)

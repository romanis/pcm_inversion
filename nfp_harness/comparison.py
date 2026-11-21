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

__all__ = ["market_quantities", "products_from_markets", "summarise", "cs_gain_by_group", "scalar_items", "weighted_median",
           "format_cell", "year_entry", "trimmed_mean", "near_twins", "load_summaries"]


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


def format_cell(v: Any, digits: int = 3) -> str:
    """A table cell: ``-`` for None or a non-finite number, a number to ``digits`` decimals, else ``str(v)``."""
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if not math.isfinite(x) else f"{x:.{digits}f}"


def year_entry(d: Mapping[Any, Any] | None, y: int) -> Any:
    """``d[y]`` with the year as an int key or (as JSON writes it) a str key; None when absent or ``d`` is empty."""
    if not d:
        return None
    return d.get(y, d.get(str(y)))


def trimmed_mean(v: np.ndarray, trim: tuple[float, float] = (0.05, 0.95)) -> float:
    """Mean of the finite values between the ``trim`` quantiles (5% and 95% by default): an own-elasticity tail of
    near-twin products is a draw artefact, the trimmed mean is not."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    lo, hi = np.quantile(v, trim)
    return float(v[(v >= lo) & (v <= hi)].mean())


def near_twins(products: Any, gap: float, group: str, price: str = "prices",
               market: str = "market_ids") -> dict[int, int]:
    """Per market, the products whose ``price`` is within ``gap`` of another product of the same ``group``."""
    out = {}
    for y, g in products.groupby(market):
        n = 0
        for _, h in g.groupby(group):
            p = np.sort(h[price].to_numpy(float))
            if p.size > 1:
                d = np.diff(p)
                near = np.minimum(np.r_[np.inf, d], np.r_[d, np.inf]) < gap
                n += int(near.sum())
        out[int(y)] = n
    return out


def load_summaries(files: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """{label: the JSON summary at path} for each ``files`` entry whose file exists (reference estimates, e.g. BLP)."""
    import json
    import pathlib

    return {label: json.loads(pathlib.Path(f).read_text()) for label, f in files.items() if pathlib.Path(f).exists()}

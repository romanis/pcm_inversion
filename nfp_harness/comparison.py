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

__all__ = ["market_quantities", "products_from_markets", "summarise", "cs_gain_by_group", "scalar_items"]


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

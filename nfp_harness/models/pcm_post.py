"""PCM postestimation: elasticities, diversion, Nash costs, consumer surplus, counterfactuals
(DESIGN 5.6, 6.5). Owned by WP8.

Two layers. The array functions (``elasticities``, ``nonzero_cross``, ``diversion``,
``group_diversion``, ``nash_markups``, ``foc_residual``, ``inside_share``) work on one market's
shares ``s`` (J,) and price Jacobian ``ds_dp`` (J, J) with ``ds_dp[j, k] = ds_j / dp_k``, the
convention of ``pcm_kernel.evaluate_market``. The model functions take a ``PCMModel``, a layout-order
theta and the (n,) row-order delta of an estimate, and evaluate the kernel at that fixed delta (no
inversion): ``market_post``, ``postestimation``, ``consumer_surplus``, ``spread_in_buyer_dollars``,
``drop_products_counterfactual`` and the CS tabulations.

Units. Prices and delta are in the data's price units ($1000 for Petrin). Consumer surplus is
``pcm_kernel.consumer_surplus`` (WP3), ``CS_i = E_a[max_j (u_ij)^+ / (c_i a)]`` in price units per
household, so every reported object is invariant to the joint utility rescaling
(delta, t, c) -> kappa (delta, t, c) (LEARNINGS 11). The old ``pcm_postestimation.consumer_surplus``
is wrong (LEARNINGS 17) and is not used.

Nash costs come from the multi-product Bertrand first-order condition
``s_j + sum_k Omega_jk (p_k - c_k) ds_k/dp_j = 0``, so ``markup = -(Omega * ds_dp')^{-1} s``. When
the ownership-weighted Jacobian has reciprocal condition number below ``rcond_min`` the result has
status ``NASH_JACOBIAN_SINGULAR`` and NaN markups; nothing raises.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from ..data import break_price_ties

__all__ = [
    "NASH_OK", "NASH_SINGULAR", "NASH_NOT_RUN",
    "elasticities", "nonzero_cross", "diversion", "group_diversion", "nash_markups", "foc_residual",
    "inside_share", "inside_share_elasticity", "group_labels", "firm_labels", "market_position",
    "market_post", "postestimation", "consumer_surplus", "draw_price_scale", "alpha_quintiles",
    "demographic_labels", "cs_by_group", "spread_in_buyer_dollars", "drop_products_counterfactual",
    "quality_dispersion", "relative_quality_dispersion",
    "compare_blp",
]

NASH_OK = "ok"
NASH_SINGULAR = "NASH_JACOBIAN_SINGULAR"
NASH_NOT_RUN = "not_run"
CROSS_TOL = 1e-10  # |e_jk| above this counts as a nonzero cross-elasticity (DESIGN 5.6)


# --------------------------------------------------------------------------------------------
# One market, arrays only
# --------------------------------------------------------------------------------------------


def elasticities(s: np.ndarray, ds_dp: np.ndarray, p: np.ndarray) -> np.ndarray:
    """elasticities(s, ds_dp, p) -> (J, J) matrix ``e_jk = (ds_j/dp_k) p_k / s_j`` for one market.

    Rows of products with zero share are NaN.
    """
    s = np.asarray(s, dtype=float)
    ds_dp = np.asarray(ds_dp, dtype=float)
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = ds_dp * p[None, :] / s[:, None]
    e[s <= 0, :] = np.nan
    return e


def nonzero_cross(e: np.ndarray, tol: float = CROSS_TOL) -> np.ndarray:
    """nonzero_cross(e, tol=1e-10) -> (J,) number of rivals k != j with ``|e_jk| > tol``."""
    e = np.asarray(e, dtype=float)
    big = np.abs(e) > tol
    np.fill_diagonal(big, False)
    return big.sum(axis=1)


def diversion(ds_dp: np.ndarray) -> np.ndarray:
    """diversion(ds_dp) -> (J, J) ``D_jk = -(ds_k/dp_j) / (ds_j/dp_j)`` with a zero diagonal.

    Outside diversion is ``1 - sum_k D_jk``. Rows of products with ``ds_j/dp_j = 0`` are NaN.
    """
    ds_dp = np.asarray(ds_dp, dtype=float)
    own = np.diag(ds_dp).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        D = -ds_dp.T / own[:, None]
    D[own == 0, :] = np.nan
    np.fill_diagonal(D, 0.0)
    return D


def group_diversion(D: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> dict[str, np.ndarray]:
    """group_diversion(D, labels, names) -> per-product diversion summed by group.

    Returns ``to_group`` (J, G) with ``sum_{k in g, k != j} D_jk`` in ``names`` order, ``outside``
    (J,), ``within`` (J,) the share of inside diversion that stays in j's group and ``leaving``
    (J,) = 1 - within. ``within`` is NaN when j diverts nothing to other products.
    """
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels).astype(str)
    names = [str(g) for g in names]
    to_group = np.column_stack([D[:, labels == g].sum(axis=1) for g in names]) if names else np.zeros((D.shape[0], 0))
    inside = D.sum(axis=1)
    own_group = np.array([to_group[j, names.index(labels[j])] for j in range(D.shape[0])]) if names \
        else np.full(D.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        within = np.where(inside > 0, own_group / inside, np.nan)
    return {"to_group": to_group, "outside": 1.0 - inside, "within": within, "leaving": 1.0 - within}


def _rcond(A: np.ndarray) -> float:
    if A.size == 0:
        return 1.0
    if not np.all(np.isfinite(A)):
        return 0.0
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return 0.0
    n1 = np.linalg.norm(A, 1) * np.linalg.norm(inv, 1)
    return float(1.0 / n1) if np.isfinite(n1) and n1 > 0 else 0.0


def nash_markups(s: np.ndarray, ds_dp: np.ndarray, firm_ids: np.ndarray, rcond_min: float = 1e-12) -> dict[str, Any]:
    """nash_markups(s, ds_dp, firm_ids, rcond_min=1e-12) -> markups ``-(Omega * (ds/dp)')^{-1} s``.

    Returns ``status`` (``"ok"`` or ``NASH_JACOBIAN_SINGULAR`` when the ownership-weighted ds/dp has
    rcond below ``rcond_min``), ``markup`` (J,) (NaN when singular), ``rcond`` and ``foc_residual``
    (max |s + (Omega * ds_dp') markup|).
    """
    s = np.asarray(s, dtype=float)
    firm_ids = np.asarray(firm_ids).ravel()
    A = (firm_ids[:, None] == firm_ids[None, :]) * np.asarray(ds_dp, dtype=float).T
    rc = _rcond(A)
    if not rc >= rcond_min:
        return {"status": NASH_SINGULAR, "markup": np.full(s.size, np.nan), "rcond": rc,
                "foc_residual": float("nan"),
                "message": f"Ownership-weighted ds/dp has rcond {rc:.2e} below {rcond_min:.0e}: no Nash costs."}
    markup = -np.linalg.solve(A, s)
    res = float(np.max(np.abs(s + A @ markup))) if s.size else 0.0
    return {"status": NASH_OK, "markup": markup, "rcond": rc, "foc_residual": res, "message": ""}


def foc_residual(s: np.ndarray, ds_dp: np.ndarray, firm_ids: np.ndarray, p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """foc_residual(s, ds_dp, firm_ids, p, c) -> (J,) ``s + (Omega * ds_dp') (p - c)``."""
    firm_ids = np.asarray(firm_ids).ravel()
    A = (firm_ids[:, None] == firm_ids[None, :]) * np.asarray(ds_dp, dtype=float).T
    return np.asarray(s, dtype=float) + A @ (np.asarray(p, dtype=float) - np.asarray(c, dtype=float))


def inside_share(s: np.ndarray) -> float:
    """inside_share(s) -> sum_j s_j."""
    return float(np.sum(s))


def inside_share_elasticity(s: np.ndarray, ds_dp: np.ndarray, p: np.ndarray) -> float:
    """inside_share_elasticity(s, ds_dp, p) -> percent change of the inside share for a common 1% price
    rise, ``sum_jk (ds_j/dp_k) p_k / sum_j s_j``."""
    S = float(np.sum(s))
    return float(np.sum(np.asarray(ds_dp) @ np.asarray(p, dtype=float)) / S) if S > 0 else float("nan")


# --------------------------------------------------------------------------------------------
# Product and agent labels
# --------------------------------------------------------------------------------------------


def _table_col(frame: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(frame[name]).ravel()
    except (KeyError, IndexError, TypeError, ValueError):
        raise KeyError(f"table has no column {name!r}") from None


def _has_col(frame: Any, name: str) -> bool:
    try:
        return name in frame
    except TypeError:
        return False


def group_labels(products: Any, groups: Any, other: str = "other") -> tuple[np.ndarray, list[str]] | None:
    """group_labels(products, groups, other="other") -> ((n,) str labels, group names) or None.

    ``groups`` is None (no groups), a column name holding labels, a mapping label -> 0/1 column, or a
    sequence of 0/1 column names (each its own label). With dummies, a product with none of them is
    ``other`` (Petrin: ``mi, sw, su, pv`` and ``car``); a product with two raises ValueError.
    """
    if groups is None:
        return None
    if isinstance(groups, str):
        lab = _table_col(products, groups).astype(str)
        return lab, sorted(set(lab.tolist()))
    pairs = list(groups.items()) if isinstance(groups, Mapping) else [(g, g) for g in groups]
    n = None
    lab_l: list[str] | None = None
    hits = None
    for label, col in pairs:
        v = np.asarray(_table_col(products, col), dtype=float) != 0
        if lab_l is None:
            n = v.size
            lab_l = [other] * n
            hits = np.zeros(n, dtype=int)
        assert hits is not None
        hits += v
        for j in np.flatnonzero(v):
            lab_l[j] = str(label)
    if lab_l is None or hits is None:
        return None
    if np.any(hits > 1):
        raise ValueError(f"{int(np.sum(hits > 1))} products belong to more than one group {[g for g, _ in pairs]}")
    names = [str(g) for g, _ in pairs]
    if np.any(hits == 0):
        names.append(other)
    return np.array(lab_l, dtype=object).astype(str), names


def firm_labels(model: Any, firm_ids: Any = "firm_ids") -> np.ndarray | None:
    """firm_labels(model, firm_ids="firm_ids") -> (n,) firm ids in row order, or None when unavailable.

    ``firm_ids`` is an (n,) array or a product column name.
    """
    if firm_ids is None:
        return None
    if isinstance(firm_ids, str):
        return _table_col(model.products, firm_ids) if _has_col(model.products, firm_ids) else None
    f = np.asarray(firm_ids).ravel()
    if f.size != model.n_products:
        raise ValueError(f"{f.size} firm ids for {model.n_products} products")
    return f


def market_position(model: Any, market: Any) -> int:
    """market_position(model, market) -> position t of the market with id ``market`` in ``model.markets``."""
    for t, m in enumerate(model.markets):
        if m.market_id == market or str(m.market_id) == str(market):
            return t
    raise KeyError(f"no market {market!r}; markets are {[m.market_id for m in model.markets]}")


# --------------------------------------------------------------------------------------------
# Kernel helpers
# --------------------------------------------------------------------------------------------


def _k() -> Any:
    from .._native import pcm_kernel

    return pcm_kernel


def _fields(km: Any) -> dict[str, Any]:
    return {"x": np.asarray(km.x), "p": np.asarray(km.p), "weights": np.asarray(km.weights),
            "taste_basis": np.asarray(km.taste_basis), "taste_col": list(km.taste_col),
            "scale_basis": np.asarray(km.scale_basis)}


def _market_like(f: Mapping[str, Any], idx: np.ndarray, p: np.ndarray) -> Any:
    """A kernel Market of the products ``idx`` (positions in f's price order) at prices ``p``."""
    return _k().Market(np.ascontiguousarray(f["x"][idx]), np.asarray(p, dtype=float), f["weights"],
                       f["taste_basis"], f["taste_col"], f["scale_basis"])


def _threads(model: Any, n_threads: int | None) -> int:
    return int(n_threads) if n_threads else max(1, min(int(getattr(model, "total_threads", 1)), 32))


def _eval(km: Any, delta: np.ndarray, th: np.ndarray, ph: np.ndarray, sp: float, n_threads: int,
          price_derivatives: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    r = _k().evaluate_market(np.asarray(delta, dtype=float), km, th, ph, sp, derivatives=False,
                             price_derivatives=price_derivatives, n_threads=n_threads)
    return np.asarray(r["s"]), (np.asarray(r["ds_dp"]) if price_derivatives else None)


def _cs(km: Any, delta: np.ndarray, th: np.ndarray, ph: np.ndarray, sp: float, n_threads: int) -> dict[str, Any]:
    if np.size(delta) == 0:
        w = np.asarray(km.weights)
        return {"total": 0.0, "per_draw": np.zeros(w.size)}
    r = _k().consumer_surplus(np.asarray(delta, dtype=float), km, th, ph, sp, n_threads)
    return {"total": float(r["total"]), "per_draw": np.asarray(r["per_draw"])}


# --------------------------------------------------------------------------------------------
# Model level: one market and all markets at an estimate
# --------------------------------------------------------------------------------------------


def market_post(model: Any, theta: np.ndarray, delta: np.ndarray, t: int, *, firm_ids: np.ndarray | None = None,
                groups: tuple[np.ndarray, list[str]] | None = None, rcond_min: float = 1e-12,
                n_threads: int | None = None) -> dict[str, Any]:
    """market_post(model, theta, delta, t, firm_ids=None, groups=None, rcond_min=1e-12, n_threads=None) -> dict.

    Evaluates market t at the fixed (n,) row-order ``delta`` and returns, in the market's price order
    (``rows`` = product rows): ``p``, ``s``, ``ds_dp``, ``elasticity`` (J, J), ``own``, ``nonzero_cross``,
    ``diversion`` (J, J), ``outside_diversion``, ``group`` (``group_diversion`` output, when ``groups`` =
    (row-order labels, names) is given), ``nash`` (``nash_markups`` output with ``cost`` and ``margin``;
    status ``not_run`` without ``firm_ids``), ``inside_share`` and ``inside_share_elasticity``.
    """
    th, ph, sp = model.kernel_params(theta)
    rows = np.asarray(model.markets[t].rows)
    km = model.kernel_market(t)
    p = np.asarray(km.p, dtype=float)
    s, J = _eval(km, np.asarray(delta, dtype=float)[rows], th, ph, sp, _threads(model, n_threads))
    assert J is not None
    e = elasticities(s, J, p)
    D = diversion(J)
    out: dict[str, Any] = {
        "market_id": model.markets[t].market_id, "t": t, "rows": rows, "p": p, "s": s, "ds_dp": J,
        "elasticity": e, "own": np.diag(e).copy(), "nonzero_cross": nonzero_cross(e), "diversion": D,
        "outside_diversion": 1.0 - D.sum(axis=1), "inside_share": inside_share(s),
        "inside_share_elasticity": inside_share_elasticity(s, J, p), "group": None,
    }
    if groups is not None:
        out["group"] = group_diversion(D, np.asarray(groups[0])[rows], groups[1])
    if firm_ids is None:
        nash: dict[str, Any] = {"status": NASH_NOT_RUN, "markup": np.full(s.size, np.nan), "rcond": float("nan"),
                                "foc_residual": float("nan"), "message": "No firm ids: Nash costs not computed."}
    else:
        nash = nash_markups(s, J, np.asarray(firm_ids)[rows], rcond_min)
    nash["cost"] = p - nash["markup"]
    nash["margin"] = nash["markup"] / p
    out["nash"] = nash
    return out


def postestimation(model: Any, theta: np.ndarray, delta: np.ndarray, *, firm_ids: Any = "firm_ids",
                   groups: Any = None, group_other: str = "other", rcond_min: float = 1e-12,
                   keep_matrices: bool = False) -> dict[str, Any]:
    """postestimation(model, theta, delta, firm_ids="firm_ids", groups=None, group_other="other",
    rcond_min=1e-12, keep_matrices=False) -> dict.

    Runs ``market_post`` on every market (markets in parallel, OpenMP threads split by J^2 R) and
    returns ``products``: (n,) row-order arrays ``market_ids, prices, shares, own_elasticity,
    nonzero_cross, nonzero_cross_ratio`` (to J_t - 1), ``outside_diversion, markup, cost, margin`` and,
    with groups, ``group, diversion_within_group, diversion_leaving_group`` and ``diversion_to_<g>``
    per group (the content of ``product_level.csv``); ``markets``: one dict per market with
    ``market_id, inside_share, inside_share_elasticity, nash_status, rcond, foc_residual`` (plus the
    full ``market_post`` output when ``keep_matrices``); ``group_names``; ``group_matrix``: (G, G + 1)
    share-weighted mean diversion from each source group (rows) to each group and the outside (last
    column), pooled over markets; and ``nash_singular``: ids of markets with a singular Nash Jacobian.
    ``firm_ids`` and ``groups`` are as in ``firm_labels`` and ``group_labels``.
    """
    from .pcm import allocate_threads

    firms = firm_labels(model, firm_ids)
    grp = group_labels(model.products, groups, group_other)
    T = len(model.markets)
    w = np.array([float(np.size(m.rows)) ** 2 * np.size(model.kernel_market(t).weights)
                  for t, m in enumerate(model.markets)])
    thr = allocate_threads(w, int(getattr(model, "total_threads", 1)))
    with ThreadPoolExecutor(max_workers=max(1, T)) as pool:
        futs = [pool.submit(market_post, model, theta, delta, t, firm_ids=firms, groups=grp, rcond_min=rcond_min,
                            n_threads=int(thr[t])) for t in range(T)]
        res = [f.result() for f in futs]

    n = int(model.n_products)
    prod: dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in (
        "prices", "shares", "own_elasticity", "nonzero_cross", "nonzero_cross_ratio", "outside_diversion",
        "markup", "cost", "margin")}
    prod["market_ids"] = np.empty(n, dtype=object)
    names = grp[1] if grp is not None else []
    if grp is not None:
        prod["group"] = np.asarray(grp[0]).astype(object)
        for k in ("diversion_within_group", "diversion_leaving_group"):
            prod[k] = np.full(n, np.nan)
        for g in names:
            prod[f"diversion_to_{g}"] = np.full(n, np.nan)
    num = np.zeros((len(names), len(names) + 1))
    den = np.zeros(len(names))
    markets = []
    for r in res:
        rows = r["rows"]
        J = rows.size
        prod["market_ids"][rows] = r["market_id"]
        prod["prices"][rows] = r["p"]
        prod["shares"][rows] = r["s"]
        prod["own_elasticity"][rows] = r["own"]
        prod["nonzero_cross"][rows] = r["nonzero_cross"]
        prod["nonzero_cross_ratio"][rows] = r["nonzero_cross"] / max(J - 1, 1)
        prod["outside_diversion"][rows] = r["outside_diversion"]
        for k in ("markup", "cost", "margin"):
            prod[k][rows] = r["nash"][k]
        if r["group"] is not None:
            gd = r["group"]
            prod["diversion_within_group"][rows] = gd["within"]
            prod["diversion_leaving_group"][rows] = gd["leaving"]
            for i, g in enumerate(names):
                prod[f"diversion_to_{g}"][rows] = gd["to_group"][:, i]
            lab = np.asarray(grp[0])[rows] if grp is not None else None
            for a, g in enumerate(names):
                sel = (lab == g) & np.all(np.isfinite(gd["to_group"]), axis=1)
                wts = r["s"][sel]
                num[a, :-1] += wts @ gd["to_group"][sel]
                num[a, -1] += wts @ gd["outside"][sel]
                den[a] += wts.sum()
        m = {"market_id": r["market_id"], "products": J, "inside_share": r["inside_share"],
             "inside_share_elasticity": r["inside_share_elasticity"], "nash_status": r["nash"]["status"],
             "rcond": r["nash"]["rcond"], "foc_residual": r["nash"]["foc_residual"]}
        if keep_matrices:
            m["post"] = r
        markets.append(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        gm = num / den[:, None] if names else np.zeros((0, 1))
    return {"products": prod, "markets": markets, "group_names": list(names), "group_matrix": gm,
            "nash_singular": [m["market_id"] for m in markets if m["nash_status"] == NASH_SINGULAR]}


# --------------------------------------------------------------------------------------------
# Consumer surplus and its tabulations
# --------------------------------------------------------------------------------------------


def consumer_surplus(model: Any, theta: np.ndarray, delta: np.ndarray, market: Any, *,
                     n_threads: int | None = None) -> dict[str, Any]:
    """consumer_surplus(model, theta, delta, market, n_threads=None) -> dict for the market with id ``market``.

    ``total`` = sum_i w_i CS_i and ``per_draw`` CS_i (unweighted, (R,)) from ``pcm_kernel.consumer_surplus``
    at the fixed (n,) row-order ``delta``, in price units per household; also ``weights`` (R,).
    """
    t = market_position(model, market)
    th, ph, sp = model.kernel_params(theta)
    km = model.kernel_market(t)
    out = _cs(km, np.asarray(delta, dtype=float)[model.markets[t].rows], th, ph, sp, _threads(model, n_threads))
    out["weights"] = np.asarray(km.weights)
    return out


def draw_price_scale(model: Any, theta: np.ndarray, t: int) -> np.ndarray:
    """draw_price_scale(model, theta, t) -> (R,) ``c_i = exp(phi . scale_basis_i)``, the median alpha of draw i."""
    _, ph, _ = model.kernel_params(theta)
    sb = np.asarray(model.kernel_market(t).scale_basis, dtype=float)
    return np.exp(sb @ ph) if sb.shape[1] else np.ones(sb.shape[0])


def alpha_quintiles(model: Any, theta: np.ndarray, t: int) -> np.ndarray:
    """alpha_quintiles(model, theta, t) -> (R,) labels 1..5, weighted quintiles of the draws' median alpha c_i.

    a_i ~ LogNormal(0, sigma_p) is integrated analytically inside each draw's CS, so the draw-level
    price sensitivity is c_i (the demographic part of alpha). Draws with equal c_i share a label; with
    c_i constant every draw is in quintile 1.
    """
    c = draw_price_scale(model, theta, t)
    w = np.asarray(model.kernel_market(t).weights, dtype=float)
    order = np.argsort(c, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    mid = cw - 0.5 * w[order] / w.sum()
    lab_sorted = np.minimum(5, np.floor(mid * 5).astype(int) + 1)
    cs = c[order]
    for k in range(1, cs.size):  # ties take the label of the first draw in the tie
        if cs[k] == cs[k - 1]:
            lab_sorted[k] = lab_sorted[k - 1]
    lab = np.empty(c.size, dtype=int)
    lab[order] = lab_sorted
    return lab


def demographic_labels(model: Any, t: int, column: str, bins: Sequence[float] | None = None) -> np.ndarray:
    """demographic_labels(model, t, column, bins=None) -> (R,) the agent column of each draw of market t
    (via ``market_draws(t)``), or its ``np.digitize`` bin index when ``bins`` is given."""
    md = model.market_draws(t)
    v = np.asarray(model.agents[column], dtype=float).ravel()[np.asarray(md["agent_rows"])][md["agent_of_draw"]]
    return np.digitize(v, np.asarray(bins, dtype=float)) if bins is not None else v


def cs_by_group(per_draw: np.ndarray | Mapping[str, np.ndarray], weights: np.ndarray,
                labels: np.ndarray) -> list[dict[str, Any]]:
    """cs_by_group(per_draw, weights, labels) -> one row per label: ``group``, ``mass`` (sum of weights),
    ``cs`` (weighted mean CS per household in the group) and ``cs_total`` (sum w_i CS_i).

    ``per_draw`` is an (R,) array, or a mapping scenario -> (R,) array, in which case each row holds
    ``cs_<scenario>`` and ``cs_total_<scenario>``.
    """
    w = np.asarray(weights, dtype=float)
    labels = np.asarray(labels)
    scen = dict(per_draw) if isinstance(per_draw, Mapping) else {"": np.asarray(per_draw, dtype=float)}
    rows = []
    for g in np.unique(labels):
        sel = labels == g
        mass = float(w[sel].sum())
        row: dict[str, Any] = {"group": g.item() if hasattr(g, "item") else g, "mass": mass}
        for k, v in scen.items():
            tot = float(w[sel] @ np.asarray(v, dtype=float)[sel])
            sfx = f"_{k}" if k else ""
            row[f"cs{sfx}"] = tot / mass if mass > 0 else float("nan")
            row[f"cs_total{sfx}"] = tot
        rows.append(row)
    return rows


def spread_in_buyer_dollars(model: Any, theta: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """spread_in_buyer_dollars(model, theta, delta) -> taste spreads divided by the mean alpha among
    predicted buyers (LEARNINGS 11, DESIGN 5.6).

    Draw i buys when ``c_i a < A_i = max_j y_ij / p_j`` with ``y_ij = delta_j + x_j . t_i``. With
    a ~ LogNormal(0, s): ``P(buy_i) = Phi(ln b_i / s)`` and ``E[a; buy_i] = exp(s^2/2) Phi((ln b_i - s^2)/s)``,
    ``b_i = A_i / c_i``. ``mean_alpha_buyers = sum_i w_i c_i E[a; buy_i] / sum_i w_i P(buy_i)`` pooled over
    markets. Returns ``mean_alpha_buyers``, ``inside_share`` (the pooled ``sum w P(buy)`` divided by the
    number of markets, a check against the model shares) and ``spreads``: name -> |sigma| / mean alpha for
    every taste parameter whose basis is a standard normal draw.
    """
    th, ph, sp = model.kernel_params(theta)
    delta = np.asarray(delta, dtype=float)
    num = den = 0.0
    for t, m in enumerate(model.markets):
        km = model.kernel_market(t)
        f = _fields(km)
        T = np.zeros((f["weights"].size, f["x"].shape[1]))
        for k, col in enumerate(f["taste_col"]):
            T[:, col] += th[k] * f["taste_basis"][:, k]
        y = delta[np.asarray(m.rows)][None, :] + T @ f["x"].T
        A = np.max(y / f["p"][None, :], axis=1)
        c = draw_price_scale(model, theta, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.where(A > 0, np.log(np.where(A > 0, A, 1.0) / c), -np.inf)
        P = norm.cdf(lb / sp)
        Ea = math.exp(sp * sp / 2) * norm.cdf((lb - sp * sp) / sp)
        num += float(f["weights"] @ (c * Ea))
        den += float(f["weights"] @ P)
    mab = num / den if den > 0 else float("nan")
    spreads = {}
    nu_names = set(getattr(model, "nu_names", []))
    for k, s in enumerate(model.layout.specs):
        if s.role == "taste" and s.basis in nu_names:
            spreads[s.name] = abs(float(np.asarray(theta, dtype=float)[k])) / mab
    return {"mean_alpha_buyers": mab, "inside_share": den / max(len(model.markets), 1), "spreads": spreads}


def _taste_utility(model: Any, theta: np.ndarray, t: int) -> tuple[np.ndarray, dict[str, Any]]:
    """(R, J_t) taste part ``x_j . t_i`` of market t in price order, and the kernel fields."""
    th, _, _ = model.kernel_params(theta)
    f = _fields(model.kernel_market(t))
    T = np.zeros((f["weights"].size, f["x"].shape[1]))
    for k, col in enumerate(f["taste_col"]):
        T[:, col] += th[k] * f["taste_basis"][:, k]
    return T @ f["x"].T, f


def purchase_probability(model: Any, theta: np.ndarray, delta: np.ndarray, t: int) -> np.ndarray:
    """purchase_probability(model, theta, delta, t) -> (R,) P(draw i buys a car) in market t, exact.

    Draw i buys when ``c_i a < A_i = max_j y_ij / p_j`` with ``y_ij = delta_j + x_j . t_i`` and
    a ~ LogNormal(0, sigma_p), so ``P_i = Phi(ln(A_i / c_i) / sigma_p)`` (0 when A_i <= 0). The weighted
    sum over draws is the inside share.
    """
    _, _, sp = model.kernel_params(theta)
    U, f = _taste_utility(model, theta, t)
    y = np.asarray(delta, dtype=float)[np.asarray(model.markets[t].rows)][None, :] + U
    A = np.max(y / f["p"][None, :], axis=1)
    c = draw_price_scale(model, theta, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.where(A > 0, np.log(np.where(A > 0, A, 1.0) / c), -np.inf)
    return norm.cdf(lb / sp)


def quality_dispersion(model: Any, theta: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """quality_dispersion(model, theta, delta) -> per-product dispersion of quality across consumers.

    For product j in market t: the weighted sd over draws of ``x_j . t_i`` (utils of the alpha = 1
    consumer), divided by ``mean_alpha_buyers`` of ``spread_in_buyer_dollars`` to put it in buyer dollars
    (price units). This is the PCM counterpart of the per-product dispersion the logit error supplies in
    BLP (SPEC 1.1). Returns ``per_product`` ((n,) row order), ``median`` and ``mean_alpha_buyers``.
    """
    mab = spread_in_buyer_dollars(model, theta, delta)["mean_alpha_buyers"]
    out = np.full(int(model.n_products), np.nan)
    for t, m in enumerate(model.markets):
        U, f = _taste_utility(model, theta, t)
        w = f["weights"] / f["weights"].sum()
        mu = w @ U
        sd = np.sqrt(np.maximum(w @ (U - mu[None, :]) ** 2, 0.0))
        out[np.asarray(m.rows)] = sd / mab
    fin = out[np.isfinite(out)]
    return {"per_product": out, "median": float(np.median(fin)) if fin.size else float("nan"),
            "mean_alpha_buyers": mab}


def relative_quality_dispersion(model: Any, theta: np.ndarray, delta: np.ndarray,
                                exclude: Sequence[str] = ("const",)) -> dict[str, Any]:
    """relative_quality_dispersion(model, theta, delta, exclude=("const",)) -> per-product dispersion across
    consumers of the quality difference to the next-priced product.

    For product j in market t (price order) and k the next more expensive product (the next cheaper one for
    the most expensive): the weighted sd over draws of ``(x_j - x_k) . t_i`` with the taste loadings on the
    ``exclude`` characteristics left out, divided by ``mean_alpha_buyers`` (buyer dollars, price units). A
    taste on a column every product shares (``const``) shifts all inside goods together and does not move a
    consumer between two of them, so it is excluded. Returns ``per_product`` ((n,) row order, NaN in a market
    with one product), ``median`` and ``mean_alpha_buyers``.
    """
    mab = spread_in_buyer_dollars(model, theta, delta)["mean_alpha_buyers"]
    th, _, _ = model.kernel_params(theta)
    chars = list(getattr(getattr(model, "spec", None), "characteristics", ()))
    skip = {chars.index(c) for c in exclude if c in chars}
    out = np.full(int(model.n_products), np.nan)
    for t, m in enumerate(model.markets):
        f = _fields(model.kernel_market(t))
        J = f["x"].shape[0]
        if J < 2:
            continue
        T = np.zeros((f["weights"].size, f["x"].shape[1]))
        for k, col in enumerate(f["taste_col"]):
            if col not in skip:
                T[:, col] += th[k] * f["taste_basis"][:, k]
        U = T @ f["x"].T
        nxt = np.r_[np.arange(1, J), J - 2]
        dU = U - U[:, nxt]
        w = f["weights"] / f["weights"].sum()
        mu = w @ dU
        sd = np.sqrt(np.maximum(w @ (dU - mu[None, :]) ** 2, 0.0))
        out[np.asarray(m.rows)] = sd / mab
    fin = out[np.isfinite(out)]
    return {"per_product": out, "median": float(np.median(fin)) if fin.size else float("nan"),
            "mean_alpha_buyers": mab}


# --------------------------------------------------------------------------------------------
# Product-removal counterfactual
# --------------------------------------------------------------------------------------------


def _sorted_prices(p: np.ndarray, min_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Order of p and the sorted prices with ties spread to ``min_gap``."""
    order = np.argsort(p, kind="stable")
    ps = break_price_ties(p[order], np.zeros(p.size), min_gap)
    return order, ps


def drop_products_counterfactual(model: Any, theta: np.ndarray, delta: np.ndarray, market: Any, drop: Any, *,
                                 firm_ids: Any = "firm_ids", costs: np.ndarray | None = None,
                                 resolve_prices: bool = False, method: str = "newton", damping: float = 0.5,
                                 tol: float = 1e-7,
                                 max_iterations: int = 500, min_gap: float = 0.01, rcond_min: float = 1e-12,
                                 tabulate: Sequence[Any] = (), n_threads: int | None = None) -> dict[str, Any]:
    """drop_products_counterfactual(model, theta, delta, market, drop, firm_ids="firm_ids", costs=None,
    resolve_prices=False, method="newton", damping=0.5, tol=1e-7, max_iterations=500, min_gap=0.01,
    rcond_min=1e-12, tabulate=(), n_threads=None) -> dict.

    Removes the products ``drop`` from the market with id ``market`` at fixed theta, with delta (so
    xi) of the remaining products unchanged. ``drop`` is a boolean (n,) row mask (intersected with the
    market, so ``products["mi"] == 1`` works) or an array of product rows inside the market.

    Always reported, prices held at their observed values: ``observed`` and ``fixed_prices``, each with
    ``rows`` (row order of the products present), ``prices``, ``shares``, ``inside_share``, ``cs`` (sum
    w_i CS_i, price units per household) and ``per_draw``; ``cs_change_fixed = cs_fixed - cs_observed``.

    With ``resolve_prices``: costs are ``costs[rows]`` if given ((n,) row order) else the Nash costs at the
    observed prices with ``firm_ids``. Prices solve ``r(p) = c + markup(p) - p = 0``, re-sorted and with ties
    spread to ``min_gap`` at every evaluation, until ``max |r| < tol`` or ``max_iterations``.
    ``method="fixed_point"`` is DESIGN 6.5's damped iteration ``p <- p + damping r(p)``. It is locally
    unstable whenever d markup / dp has a real eigenvalue above 1 (then ``1 + damping (lambda - 1) > 1`` for
    every damping), which PCM markets with close price neighbours produce. ``method="newton"`` (default) takes
    Newton steps on r with a forward-difference Jacobian (J extra evaluations per step, run in parallel),
    keeps prices positive and backtracks (down to 1/16 of the step) until max |r| falls; when no backtracked
    step lowers max |r| (ds/dp, so r, jumps where a draw's frontier changes) it takes one damped fixed-point
    step instead. ``fixed_point_steps`` counts those. At finite draws a pure-strategy solution need not
    exist; ``not_converged`` is then the honest outcome.
    ``nash`` holds ``status`` (``converged``, ``not_converged``, ``NASH_JACOBIAN_SINGULAR`` at the observed
    or at an iterated point, ``nonpositive_price``, or for Newton ``jacobian_failed``, ``jacobian_singular``,
    ``no_descent``), ``method``, ``iterations``, ``step`` (max |r|), ``foc_residual`` (max
    |s + (Omega ds_dp')(p - c)| at the last evaluated prices), ``costs_observed`` and the same fields as
    ``fixed_prices``; ``cs_change_nash = cs_nash - cs_observed``. A run that stops early keeps its last
    prices, and its status says why.

    ``tables`` holds ``cs_by_group`` rows by ``alpha_quintile`` and by each entry of ``tabulate`` (an agent
    column name, or a (column, bins) pair), with columns for every computed scenario. A households
    multiplier is the caller's: gain in price units = households x (CS_observed - CS_counterfactual).
    """
    t = market_position(model, market)
    th, ph, sp = model.kernel_params(theta)
    nt = _threads(model, n_threads)
    rows = np.asarray(model.markets[t].rows)
    km = model.kernel_market(t)
    f = _fields(km)
    p0 = f["p"].astype(float)
    d_t = np.asarray(delta, dtype=float)[rows]
    dm = np.asarray(drop)
    if dm.dtype == bool:
        if dm.size != model.n_products:
            raise ValueError(f"boolean drop mask has {dm.size} entries for {model.n_products} products")
        gone = dm[rows]
    else:
        dm = dm.astype(int).ravel()
        bad = np.setdiff1d(dm, rows)
        if bad.size:
            raise ValueError(f"drop rows {bad.tolist()} are not in market {market!r}")
        gone = np.isin(rows, dm)
    keep = np.flatnonzero(~gone)

    def scenario(idx: np.ndarray, p: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        mk = _market_like(f, idx, p)
        s, J = _eval(mk, d_t[idx], th, ph, sp, nt) if idx.size else (np.zeros(0), np.zeros((0, 0)))
        cs = _cs(mk, d_t[idx], th, ph, sp, nt)
        return ({"rows": rows[idx], "prices": np.asarray(p, dtype=float), "shares": s, "inside_share": float(s.sum()),
                 "cs": cs["total"], "per_draw": cs["per_draw"]}, J)

    obs, J0 = scenario(np.arange(rows.size), p0)
    fixed, _ = scenario(keep, p0[keep])
    out: dict[str, Any] = {"market_id": model.markets[t].market_id, "dropped_rows": rows[gone],
                           "observed": obs, "fixed_prices": fixed, "cs_change_fixed": fixed["cs"] - obs["cs"],
                           "nash": None, "cs_change_nash": None}
    per_draw = {"observed": obs["per_draw"], "fixed_prices": fixed["per_draw"]}

    if resolve_prices:
        firms = firm_labels(model, firm_ids)
        if firms is None:
            raise ValueError("resolve_prices needs firm ids (an (n,) array or a product column)")
        fk = np.asarray(firms)[rows]
        nash: dict[str, Any] = {"status": "", "iterations": 0, "step": float("nan"), "foc_residual": float("nan")}
        if costs is None:
            assert J0 is not None
            nm = nash_markups(obs["shares"], J0, fk, rcond_min)
            nash["rcond_observed"] = nm["rcond"]
            c_all = p0 - nm["markup"]
            if nm["status"] != NASH_OK:
                nash["status"] = NASH_SINGULAR
        else:
            c_all = np.asarray(costs, dtype=float)[rows]
        nash["costs_observed"] = c_all
        ck, fkk = c_all[keep], fk[keep]
        p = p0[keep].copy()
        res: dict[str, Any] = {}
        it = 0
        nash["method"] = method
        nash["fixed_point_steps"] = 0

        def F(pk: np.ndarray, n_thr: int = nt) -> tuple[np.ndarray | None, str, float, np.ndarray]:
            """c + markup(p) - p in kept order, its status, the FOC residual and the spread prices."""
            order, ps = _sorted_prices(pk, min_gap)
            pk = np.empty_like(ps)
            pk[order] = ps
            if ps.size and ps[0] <= 0:
                return None, "nonpositive_price", float("nan"), pk
            idx = keep[order]
            s, J = _eval(_market_like(f, idx, ps), d_t[idx], th, ph, sp, n_thr)
            assert J is not None
            nm = nash_markups(s, J, fkk[order], rcond_min)
            if nm["status"] != NASH_OK:
                return None, NASH_SINGULAR, float("nan"), pk
            r = np.empty_like(ps)
            r[order] = ck[order] + nm["markup"] - ps
            foc = float(np.max(np.abs(foc_residual(s, J, fkk[order], ps, ck[order])))) if ps.size else 0.0
            return r, "", foc, pk

        if method not in ("newton", "fixed_point"):
            raise ValueError(f"method must be 'newton' or 'fixed_point', got {method!r}")
        while not nash["status"]:
            r, st, foc, p = F(p)
            if r is None:
                nash["status"] = st
                break
            nash["step"] = float(np.max(np.abs(r))) if r.size else 0.0
            nash["foc_residual"] = foc
            if nash["step"] < tol:
                nash["status"] = "converged"
                break
            if it >= max_iterations:
                nash["status"] = "not_converged"
                break
            it += 1
            if method == "fixed_point":
                nash["fixed_point_steps"] += 1
                p = p + damping * r
                continue
            # Newton on r(p) = 0 with a forward-difference Jacobian (columns in parallel) and backtracking
            h = 1e-6 * np.maximum(1.0, np.abs(p))
            workers = max(1, min(p.size, nt))
            per = max(1, nt // workers)

            def jac_col(k: int) -> np.ndarray | None:
                rk = F(p + h[k] * np.eye(p.size)[k], per)[0]
                return None if rk is None else (rk - r) / h[k]

            with ThreadPoolExecutor(max_workers=workers) as pool:
                cols = list(pool.map(jac_col, range(p.size)))
            if any(c is None for c in cols):
                nash["fixed_point_steps"] += 1
                p = p + damping * r
                continue
            JF = np.column_stack([c for c in cols if c is not None])
            try:
                dp = -np.linalg.solve(JF, r)
            except np.linalg.LinAlgError:
                dp = np.zeros_like(r)
            neg = dp < 0
            lam = min(1.0, float(np.min(0.9 * p[neg] / -dp[neg]))) if neg.any() else 1.0
            best = None
            while lam >= 1.0 / 16:
                r2, _, _, p2 = F(p + lam * dp)
                if r2 is not None and np.max(np.abs(r2)) < (1.0 - 1e-4 * lam) * nash["step"]:
                    best = p2
                    break
                lam /= 2
            if best is None:  # r(p) jumps where a draw's frontier changes: fall back to a damped step
                nash["fixed_point_steps"] += 1
                best = p + damping * r
            p = best
        nash["iterations"] = it
        order, ps = _sorted_prices(p, min_gap)
        if ps.size == 0 or ps[0] > 0:
            res, _ = scenario(keep[order], ps)
            nash.update(res)
            out["cs_change_nash"] = res["cs"] - obs["cs"]
            per_draw["nash"] = res["per_draw"]
        out["nash"] = nash

    w = f["weights"]
    tables = {"alpha_quintile": cs_by_group(per_draw, w, alpha_quintiles(model, theta, t))}
    for spec in tabulate:
        col, bins = (spec, None) if isinstance(spec, str) else (spec[0], spec[1])
        tables[str(col)] = cs_by_group(per_draw, w, demographic_labels(model, t, col, bins))
    out["tables"] = tables
    return out


# --------------------------------------------------------------------------------------------
# Comparison with the BLP summaries (DESIGN 6.5 item 5)
# --------------------------------------------------------------------------------------------

_BLP_ROWS = (("minivan_cs_gain_1984_musd", "minivan CS gain 1984 ($M)"),
             ("own_elasticity_median", "own-price elasticity, median"),
             ("own_elasticity_mean", "own-price elasticity, mean"),
             ("margin_median", "margin (p - c)/p, median"),
             ("margin_mean", "margin (p - c)/p, mean"))


def compare_blp(pcm_summary: Mapping[str, Any], blp_summaries: Mapping[str, Mapping[str, Any]],
                note: str = "") -> str:
    """compare_blp(pcm_summary, blp_summaries, note="") -> text table, PCM next to each BLP summary.

    Rows are the keys of the BLP summary JSON files (``minivan_cs_gain_1984_musd``,
    ``own_elasticity_median``, ``own_elasticity_mean``, ``margin_median``, ``margin_mean``) followed by any
    other key of ``pcm_summary``; a missing value prints as ``-``. ``note`` is appended below the table.
    """
    cols = ["PCM"] + list(blp_summaries)
    rows = list(_BLP_ROWS) + [(k, k) for k in pcm_summary if k not in dict(_BLP_ROWS)]

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    srcs = [pcm_summary] + [blp_summaries[c] for c in cols[1:]]
    body = [[label] + [fmt(s.get(key)) for s in srcs] for key, label in rows]
    head = ["quantity"] + cols
    width = [max(len(r[i]) for r in body + [head]) for i in range(len(head))]
    line = "  ".join(h.ljust(width[0]) if i == 0 else h.rjust(width[i]) for i, h in enumerate(head))
    lines = [line] + ["  ".join(c.ljust(width[0]) if i == 0 else c.rjust(width[i]) for i, c in enumerate(r))
                      for r in body]
    if note:
        lines += ["", note]
    return "\n".join(lines)

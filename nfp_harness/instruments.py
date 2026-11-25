"""Instrument construction and rank screens for the linear IV part of a demand spec (model-agnostic).

Column helpers work on arrays: ``standardise`` (unit variance, not centred), ``keep_rank`` and
``drop_collinear_columns`` / ``drop_collinear`` (greedy left-to-right rank screens). Instrument builders read a product
table with ``market_ids`` and ``firm_ids`` and named characteristic columns: ``blp_sums`` and ``gh_instruments`` (pyblp
1.3.0 ``build_blp_instruments`` / ``build_differentiation_instruments``) and ``excluded_instruments`` (named columns and
Gandhi-Houde instruments, standardised). ``first_stage_table`` screens candidate excluded sets for price relevance.

Supply-side builders: ``firm_product_counts`` and ``markup_shifters`` (Gandhi-Houde columns over the firm's other
products and over rivals', plus the two counts, optionally ``rival_distances`` in characteristic space; they move
Nash markups and not costs), and the within-id first
differences ``within_differences`` / ``difference_instruments``, which write E[dz dr] = 0 as a levels moment Z_d' r so
the IV and supply blocks apply unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .linear import matrix_rank

__all__ = ["standardise", "keep_rank", "drop_collinear_columns", "drop_collinear", "pyblp_frame", "blp_sums",
           "gh_instruments", "excluded_instruments", "first_stage_table", "firm_product_counts", "markup_shifters",
           "rival_distances", "within_differences", "difference_instruments"]


def standardise(A: np.ndarray) -> np.ndarray:
    """Columns scaled to unit variance (not centred, so a column's span with the constant is unchanged)."""
    A = np.asarray(A, dtype=float)
    sd = A.std(axis=0)
    return A / np.where(sd > 0, sd, 1.0)


def keep_rank(A: np.ndarray, tol: float = 1e-10) -> list[int]:
    """Columns of A kept greedily left to right while each raises the rank (a constant column after the first is
    skipped)."""
    keep: list[int] = []
    for k in range(A.shape[1]):
        if np.std(A[:, k]) == 0 and k > 0:
            continue
        if matrix_rank(A[:, keep + [k]], rtol=tol) == len(keep) + 1:
            keep.append(k)
    return keep


def drop_collinear_columns(Z: np.ndarray, keep: int) -> list[int]:
    """Column indices of Z: the first ``keep``, then each later column that adds rank (left to right)."""
    cols = list(range(keep))
    for k in range(keep, Z.shape[1]):
        if matrix_rank(Z[:, cols + [k]]) == len(cols) + 1:
            cols.append(k)
    return cols


def drop_collinear(Z: np.ndarray, keep: int) -> np.ndarray:
    """Z with columns past the first ``keep`` dropped when they add no rank."""
    return Z[:, drop_collinear_columns(Z, keep)]


def pyblp_frame(products: pd.DataFrame, characteristics: Sequence[str]) -> pd.DataFrame:
    """The pyblp product data of the instrument builders: ``market_ids``, ``firm_ids`` and ``characteristics`` as
    floats."""
    d = pd.DataFrame({"market_ids": products.market_ids.to_numpy(), "firm_ids": products.firm_ids.to_numpy()})
    for c in characteristics:
        d[c] = products[c].to_numpy(float)
    return d


def blp_sums(products: pd.DataFrame, characteristics: Sequence[str]) -> np.ndarray:
    """BLP sums of ``characteristics`` over the firm's other products and over rivals (pyblp 1.3.0
    ``build_blp_instruments``), (n, 2 len(characteristics))."""
    import pyblp
    return np.asarray(pyblp.build_blp_instruments(
        pyblp.Formulation("0 + " + " + ".join(characteristics)), pyblp_frame(products, characteristics)), dtype=float)


def gh_instruments(products: pd.DataFrame, characteristics: Sequence[str], version: str = "local") -> np.ndarray:
    """Gandhi-Houde differentiation instruments on continuous ``characteristics`` (pyblp 1.3.0
    ``build_differentiation_instruments``, ``version`` "local" or "quadratic"), own-firm and rival,
    (n, 2 len(characteristics))."""
    import pyblp
    return np.asarray(pyblp.build_differentiation_instruments(
        pyblp.Formulation("0 + " + " + ".join(characteristics)), pyblp_frame(products, characteristics),
        version=version), dtype=float)


def excluded_instruments(products: pd.DataFrame, columns: Sequence[str] = (), gh_characteristics: Sequence[str] = (),
                         gh_version: str = "local") -> tuple[np.ndarray, list[str]]:
    """(Z_ex (n, k), names): the product columns ``columns``, then the Gandhi-Houde ``gh_version`` instruments on
    ``gh_characteristics`` (named ``gh_<version><k>``), every column standardised so the rank tolerance of
    ``drop_collinear`` is scale-free. Raises ValueError when both are empty."""
    cols: list[np.ndarray] = [products[c].to_numpy(float) for c in columns]
    names: list[str] = list(columns)
    if gh_characteristics:
        gh = gh_instruments(products, gh_characteristics, gh_version)
        cols += list(gh.T)
        names += [f"gh_{gh_version}{k}" for k in range(gh.shape[1])]
    if not cols:
        raise ValueError("no excluded instruments: columns and gh_characteristics are both empty")
    return standardise(np.column_stack(cols)), names


def first_stage_table(price: np.ndarray, X1: np.ndarray, sets: Sequence[tuple[str, np.ndarray]]
                      ) -> list[dict[str, Any]]:
    """Relevance screen of candidate excluded sets ``(name, E (n, m))``: columns, rank added to X1 (``E``
    standardised, collinear columns dropped), the homoskedastic first-stage F of price on the set after partialling
    X1, and the R^2 it adds. Data only."""
    price = np.asarray(price, dtype=float)
    n = price.size
    r_x = price - X1 @ np.linalg.lstsq(X1, price, rcond=None)[0]
    tss = float(np.sum((price - price.mean()) ** 2))
    rss_r = float(r_x @ r_x)
    rows = []
    for name, E in sets:
        E = standardise(E)
        keep = drop_collinear_columns(np.column_stack([X1, E]), keep=X1.shape[1])
        q = len(keep) - X1.shape[1]
        A = np.column_stack([X1, E])[:, keep]
        r_u = price - A @ np.linalg.lstsq(A, price, rcond=None)[0]
        rss_u = float(r_u @ r_u)
        F = ((rss_r - rss_u) / q) / (rss_u / (n - A.shape[1])) if q > 0 else float("nan")
        rows.append({"set": name, "columns": int(E.shape[1]), "rank_added": int(q), "F_price": float(F),
                     "R2_added": (rss_r - rss_u) / tss})
    return rows


def firm_product_counts(market_ids: np.ndarray, firm_ids: np.ndarray) -> np.ndarray:
    """firm_product_counts(market_ids, firm_ids) -> (n, 2): per product, the number of the firm's other products in its
    market and the number of rivals' products there."""
    market_ids = np.asarray(market_ids).ravel()
    firm_ids = np.asarray(firm_ids).ravel()
    if market_ids.shape != firm_ids.shape:
        raise ValueError(f"{firm_ids.size} firm ids for {market_ids.size} market ids")
    _, mk = np.unique(market_ids, return_inverse=True)
    _, fm = np.unique(firm_ids, return_inverse=True)
    n_market = np.bincount(mk)[mk]
    _, pk = np.unique(mk * (fm.max() + 1) + fm, return_inverse=True)
    n_firm = np.bincount(pk)[pk]
    return np.column_stack([n_firm - 1, n_market - n_firm]).astype(float)


#: the columns of ``rival_distances``
RIVAL_DISTANCE_NAMES = ("rival_distance_1", "rival_distance_2", "rival_products_within_median")


def rival_distances(products: pd.DataFrame, characteristics: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """rival_distances(products, characteristics) -> ((n, 3), names): per product, the Euclidean distance in
    characteristic space to the nearest and to the second-nearest product of another firm in its market, and the number
    of rival products within the median nearest-rival distance (the median over every product). Each characteristic is
    standardised over the pooled products (``standardise``); a column with no variation is dropped. Characteristics
    only: prices are set in the same equilibrium, so a distance in price would be endogenous. Raises ValueError when a
    product has fewer than two rival products in its market."""
    X = np.column_stack([products[c].to_numpy(float) for c in characteristics])
    X = standardise(X[:, X.std(axis=0) > 0])
    market = products["market_ids"].to_numpy()
    firm = products["firm_ids"].to_numpy()
    n = len(products)
    d1, d2 = np.empty(n), np.empty(n)
    dists: list[tuple[np.ndarray, np.ndarray]] = []
    for m in np.unique(market):
        rows = np.flatnonzero(market == m)
        D = np.sqrt(((X[rows, None, :] - X[None, rows, :]) ** 2).sum(axis=2))
        D[firm[rows][:, None] == firm[rows][None, :]] = np.inf   # the firm's own products are not rivals
        near = np.sort(D, axis=1)[:, :2]
        if not np.all(np.isfinite(near)):
            raise ValueError(f"market {m}: a product has fewer than two rival products")
        d1[rows], d2[rows] = near[:, 0], near[:, 1]
        dists.append((rows, D))
    cut = float(np.median(d1))
    within = np.empty(n)
    for rows, D in dists:
        within[rows] = (D <= cut).sum(axis=1)
    return np.column_stack([d1, d2, within]), list(RIVAL_DISTANCE_NAMES)


def markup_shifters(products: pd.DataFrame, characteristics: Sequence[str], version: str = "local",
                    distance_characteristics: Sequence[str] = ()) -> tuple[np.ndarray, list[str]]:
    """markup_shifters(products, characteristics, version="local", distance_characteristics=()) -> ((n, 2K + 2 [+ 3]),
    names): excluded supply instruments that move Nash markups and not costs. ``gh_instruments`` of the K
    ``characteristics`` over the firm's other products (``own_<x>``) and over rivals' products (``rival_<x>``), then the
    counts of the firm's other products (``own_products``) and of rivals' products (``rival_products``), then, when
    ``distance_characteristics`` is given, the three ``rival_distances`` columns in that space. Unscaled."""
    characteristics = list(characteristics)
    if not characteristics:
        raise ValueError("markup_shifters needs at least one characteristic")
    gh = gh_instruments(products, characteristics, version)
    names = ([f"own_{c}" for c in characteristics] + [f"rival_{c}" for c in characteristics]
             + ["own_products", "rival_products"])
    counts = firm_product_counts(products["market_ids"].to_numpy(), products["firm_ids"].to_numpy())
    if not distance_characteristics:
        return np.column_stack([gh, counts]), names
    dist, dnames = rival_distances(products, distance_characteristics)
    return np.column_stack([gh, counts, dist]), names + dnames


def within_differences(ids: np.ndarray, periods: np.ndarray, max_gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """within_differences(ids, periods, max_gap=1) -> (cur, prev) row indices: every row whose ``ids`` value was also
    observed at an earlier period at most ``max_gap`` periods before, paired with the latest such row. ``max_gap=1``
    pairs consecutive periods only. An id seen twice in one period is refused."""
    ids = np.asarray(ids).ravel()
    periods = np.asarray(periods).ravel()
    if ids.shape != periods.shape:
        raise ValueError(f"{ids.size} ids for {periods.size} periods")
    if not max_gap >= 1:
        raise ValueError(f"max_gap must be at least 1, got {max_gap}")
    order = np.lexsort((periods, ids))
    i_s, t_s = ids[order], periods[order]
    same = i_s[1:] == i_s[:-1]
    if np.any(same & (t_s[1:] == t_s[:-1])):
        raise ValueError("an id is observed twice in one period")
    keep = same & (t_s[1:] - t_s[:-1] <= max_gap)
    return order[1:][keep], order[:-1][keep]


def difference_instruments(cur: np.ndarray, prev: np.ndarray, z: np.ndarray) -> np.ndarray:
    """difference_instruments(cur, prev, z) -> (n, k) Z_d with ``Z_d' r = sum_pairs (z_cur - z_prev) (r_cur - r_prev)``
    for any (n,) residual r: the instruments of a first-difference moment E[dz dr] = 0 written as a levels moment
    Z_d' r, so the IV and supply blocks apply unchanged (Z_d = D' dZ, D the pair differencing matrix)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z must be 2-d, got shape {z.shape}")
    cur = np.asarray(cur, dtype=int).ravel()
    prev = np.asarray(prev, dtype=int).ravel()
    dz = z[cur] - z[prev]
    out = np.zeros_like(z)
    np.add.at(out, cur, dz)
    np.add.at(out, prev, -dz)
    return out

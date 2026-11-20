"""Instrument construction and rank screens for the linear IV part of a demand spec (model-agnostic).

Column helpers work on arrays: ``standardise`` (unit variance, not centred), ``keep_rank`` and
``drop_collinear_columns`` / ``drop_collinear`` (greedy left-to-right rank screens). Instrument builders read a product
table with ``market_ids`` and ``firm_ids`` and named characteristic columns: ``blp_sums`` and ``gh_instruments`` (pyblp
1.3.0 ``build_blp_instruments`` / ``build_differentiation_instruments``) and ``excluded_instruments`` (named columns and
Gandhi-Houde instruments, standardised). ``first_stage_table`` screens candidate excluded sets for price relevance.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .linear import matrix_rank

__all__ = ["standardise", "keep_rank", "drop_collinear_columns", "drop_collinear", "pyblp_frame", "blp_sums",
           "gh_instruments", "excluded_instruments", "first_stage_table"]


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

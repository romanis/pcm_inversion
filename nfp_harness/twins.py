"""Exact twins: products equal on every characteristic the model sees are one product to every consumer.

``merge_twins`` keeps the first row of each twin group with summed shares; ``twin_targets`` maps the rows of the
unmerged table to the merged ones; ``merge_joint_law`` carries a first-choice x second-choice probability matrix of the
unmerged products over to the merged set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["merge_twins", "twin_targets", "merge_joint_law"]


def merge_twins(products: pd.DataFrame, keys: Sequence[str], sum_columns: Sequence[str],
                constant_columns: Sequence[str] = ()) -> tuple[pd.DataFrame, int]:
    """merge_twins(products, keys, sum_columns, constant_columns=()) -> (products, number of rows merged away).

    Rows equal on ``keys`` (market, firm, price and every model characteristic) are one product: the first row of each
    group is kept, in table order, with ``sum_columns`` (shares, quantities) summed over the group. A
    ``constant_columns`` entry that differs within a group raises ValueError (a column outside ``keys`` the merged
    product must still carry consistently, e.g. a cost characteristic)."""
    g = products.groupby(list(keys), sort=False)
    first = (g.cumcount() == 0).to_numpy()
    out = products.copy()
    for c in constant_columns:
        if int(g[c].nunique().max()) > 1:
            raise ValueError(f"merged twins differ in {c}")
    for c in sum_columns:
        out[c] = g[c].transform("sum")
    return out[first].reset_index(drop=True), int((~first).sum())


def twin_targets(raw: pd.DataFrame, merged: pd.DataFrame, keys: Sequence[str], raw_price: str = "prices",
                 merged_price: str = "prices") -> np.ndarray:
    """(len(raw),) row of ``merged`` holding each ``raw`` row, matched on ``keys`` and the price (``raw_price`` in
    ``raw``, ``merged_price`` in ``merged``, so a merged table whose prices were since moved can be matched on its
    data price); -1 where the raw product's merged product is not in ``merged`` (dropped)."""
    index = {tuple(r): k for k, r in enumerate(zip(*[merged[c].to_numpy() for c in keys],
                                                   merged[merged_price].to_numpy()))}
    return np.array([index.get(tuple(r), -1)
                     for r in zip(*[raw[c].to_numpy() for c in keys], raw[raw_price].to_numpy())])


def merge_joint_law(prob: np.ndarray, s_raw: np.ndarray, to: np.ndarray, n_merged: int) -> np.ndarray:
    """(n_merged, n_merged) joint law pi of first and second choice on the merged products.

    ``prob[j, k]`` is P(second choice k | first choice j) on the raw products, ``s_raw`` their shares and ``to`` their
    merged rows (``twin_targets``). The raw joint law (s / S) prob is summed over each twin group in rows and columns,
    and a twin-to-twin pair (now the product with itself) is dropped. A raw product with ``to = -1`` leaves the law, as
    first and second choice. Not renormalised."""
    s_raw = np.asarray(s_raw, dtype=float)
    Mmap = np.zeros((s_raw.size, n_merged))
    kept = to >= 0
    Mmap[np.flatnonzero(kept), to[kept]] = 1.0
    pi = Mmap.T @ ((s_raw / s_raw.sum())[:, None] * prob) @ Mmap
    np.fill_diagonal(pi, 0.0)
    return pi

"""Tests for nfp_harness.twins: merging exact twins and carrying a second-choice law over to the merged set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfp_harness.twins import merge_joint_law, merge_twins, twin_targets

KEYS = ("market_ids", "firm_ids", "prices", "x")


def table() -> pd.DataFrame:
    # rows 0 and 2 are twins (same market, firm, price, x); row 3 has row 0's x but another firm; row 4 another market
    return pd.DataFrame({"market_ids": [1, 1, 1, 1, 2], "firm_ids": [1, 1, 1, 2, 1],
                         "prices": [1.0, 2.0, 1.0, 1.0, 1.0], "x": [0.5, 0.7, 0.5, 0.5, 0.5],
                         "shares": [0.1, 0.2, 0.05, 0.15, 0.3],
                         "q": [10.0, 20.0, 5.0, 15.0, 30.0], "wt": [1.0, 2.0, 1.0, 3.0, 1.0]})


def test_merge_keeps_the_first_row_and_sums() -> None:
    out, n = merge_twins(table(), KEYS, ("shares", "q"), constant_columns=("wt",))
    assert n == 1 and len(out) == 4
    np.testing.assert_array_equal(out.shares, [0.1 + 0.05, 0.2, 0.15, 0.3])
    np.testing.assert_array_equal(out.q, [15.0, 20.0, 15.0, 30.0])
    assert list(out.columns) == list(table().columns) and out.index.tolist() == [0, 1, 2, 3]


def test_merge_refuses_twins_that_differ_in_a_constant_column() -> None:
    t = table()
    t.loc[2, "wt"] = 9.0
    with pytest.raises(ValueError, match="differ in wt"):
        merge_twins(t, KEYS, ("shares",), constant_columns=("wt",))
    merge_twins(t, KEYS, ("shares",))   # unchecked without it


def test_targets_and_joint_law() -> None:
    raw = table()
    raw = raw[raw.market_ids == 1].reset_index(drop=True)
    merged, _ = merge_twins(raw, KEYS, ("shares",))
    merged = merged.iloc[[0, 2]].reset_index(drop=True)          # product 1 (price 2) dropped from the merged set
    merged["prices_data"] = merged.prices
    merged["prices"] = merged.prices + np.array([0.0, 0.01])      # a tie spread after the merge
    to = twin_targets(raw, merged, ("firm_ids", "x"), merged_price="prices_data")
    np.testing.assert_array_equal(to, [0, -1, 0, 1])
    prob = np.array([[0.0, 0.5, 0.3, 0.2], [0.4, 0.0, 0.4, 0.2], [0.6, 0.2, 0.0, 0.2], [0.5, 0.25, 0.25, 0.0]])
    s = raw.shares.to_numpy(float)
    pi = merge_joint_law(prob, s, to, 2)
    joint = (s / s.sum())[:, None] * prob
    want = np.array([[0.0, joint[0, 3] + joint[2, 3]], [joint[3, 0] + joint[3, 2], 0.0]])
    np.testing.assert_allclose(pi, want, rtol=1e-15, atol=0)

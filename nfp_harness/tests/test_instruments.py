"""Tests for nfp_harness.instruments on small synthetic designs (the GMY-scale checks are test_gmy_instruments.py)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from nfp_harness.instruments import (blp_sums, drop_collinear, drop_collinear_columns, excluded_instruments,
                                     first_stage_table, gh_instruments, keep_rank, standardise)

VENDOR = pathlib.Path(__file__).resolve().parents[2] / "nevo_replication" / ".vendor"


def products(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"market_ids": np.repeat([1, 2], n // 2), "firm_ids": np.tile([1, 1, 2, 3], n // 4),
                      "x": rng.normal(size=n), "y": rng.uniform(1.0, 3.0, size=n), "big": rng.normal(size=n) * 6e5})
    d["prices"] = 1.0 + 0.5 * d.x + 0.3 * d.y + rng.normal(scale=0.1, size=n)
    return d


def test_standardise_scales_without_centring() -> None:
    A = np.column_stack([np.full(5, 3.0), np.arange(5.0) * 1e5 + 7.0])
    S = standardise(A)
    np.testing.assert_array_equal(S[:, 0], A[:, 0])        # a constant column has sd 0 and is left as it is
    np.testing.assert_array_equal(S[:, 1], A[:, 1] / A[:, 1].std())
    assert S[:, 1].std() == pytest.approx(1.0, abs=1e-15)


def test_rank_screens_keep_the_first_column_that_adds_rank() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(30, 2)).T
    A = np.column_stack([np.ones(30), a, 2 * a, np.ones(30), b, a + b])
    assert keep_rank(A) == [0, 1, 4]
    assert drop_collinear_columns(A, keep=2) == [0, 1, 4]
    # the first ``keep`` are kept whatever their rank, and a later column must raise the rank to len(kept) + 1, so a
    # rank-deficient kept block lets none in
    assert drop_collinear_columns(A, keep=3) == [0, 1, 2]
    np.testing.assert_array_equal(drop_collinear(A, keep=2), A[:, [0, 1, 4]])


def test_excluded_instruments_from_columns() -> None:
    p = products()
    Z, names = excluded_instruments(p, ("y", "big"))
    assert names == ["y", "big"]
    np.testing.assert_array_equal(Z, standardise(np.column_stack([p.y.to_numpy(float), p.big.to_numpy(float)])))
    with pytest.raises(ValueError, match="no excluded instruments"):
        excluded_instruments(p)


def test_first_stage_table_against_a_direct_regression() -> None:
    p = products()
    X1 = np.column_stack([np.ones(len(p)), p.x])
    E = p[["y"]].to_numpy(float)
    row, dup = first_stage_table(p.prices.to_numpy(float), X1, [("y", E), ("y twice", np.column_stack([E, 2 * E]))])
    price = p.prices.to_numpy(float)
    rss = [float(np.sum((price - A @ np.linalg.lstsq(A, price, rcond=None)[0]) ** 2))
           for A in (X1, np.column_stack([X1, E]))]
    n = len(p)
    assert row["columns"] == 1 and row["rank_added"] == 1
    assert row["F_price"] == pytest.approx((rss[0] - rss[1]) / (rss[1] / (n - 3)), rel=1e-12)
    assert row["R2_added"] == pytest.approx((rss[0] - rss[1]) / np.sum((price - price.mean()) ** 2), rel=1e-12)
    assert dup["columns"] == 2 and dup["rank_added"] == 1 and dup["F_price"] == pytest.approx(row["F_price"])


def test_pyblp_builders_shapes_and_names() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    pytest.importorskip("pyblp")
    p = products()
    assert blp_sums(p, ("x", "y")).shape == (len(p), 4)
    gh = gh_instruments(p, ("x", "y"))
    assert gh.shape == (len(p), 4) and np.isfinite(gh).all()
    Z, names = excluded_instruments(p, ("big",), ("x", "y"), gh_version="quadratic")
    assert names == ["big", "gh_quadratic0", "gh_quadratic1", "gh_quadratic2", "gh_quadratic3"]
    np.testing.assert_array_equal(Z[:, 1:], standardise(gh_instruments(p, ("x", "y"), "quadratic")))

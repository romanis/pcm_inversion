"""Tests for the markup-shifter supply instruments and the within-id first-difference moments."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from nfp_harness.instruments import (difference_instruments, firm_product_counts, gh_instruments, markup_shifters,
                                     within_differences)
from nfp_harness.moments import DifferenceIVMoments, DifferenceSarganIVMoments, IVMoments, SarganIVMoments
from nfp_harness.types import LinearFit

VENDOR = pathlib.Path(__file__).resolve().parents[2] / "nevo_replication" / ".vendor"


@pytest.fixture(scope="module")
def pyblp():   # type: ignore[no-untyped-def]
    if str(VENDOR) not in sys.path:
        sys.path.append(str(VENDOR))
    return pytest.importorskip("pyblp")


def test_firm_product_counts() -> None:
    # market 1: firm a has 2 products, firm b 1; market 2: firm a 1, firm c 2
    mk = np.array([1, 1, 1, 2, 2, 2])
    fm = np.array(["a", "a", "b", "a", "c", "c"])
    np.testing.assert_array_equal(firm_product_counts(mk, fm),
                                  [[1, 1], [1, 1], [0, 2], [0, 2], [1, 1], [1, 1]])
    with pytest.raises(ValueError):
        firm_product_counts(mk, fm[:5])


def test_markup_shifters_are_pyblps_gh_columns_and_counts(pyblp) -> None:   # type: ignore[no-untyped-def]
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 40
    mk = np.repeat([0, 1], n // 2)
    fm = rng.integers(0, 4, n)
    X = rng.normal(size=(n, 2))
    frame = pd.DataFrame({"market_ids": mk, "firm_ids": fm, "a": X[:, 0], "b": X[:, 1]})
    for version in ("local", "quadratic"):
        ref = pyblp.build_differentiation_instruments(pyblp.Formulation("0 + a + b"), frame, version=version)
        np.testing.assert_array_equal(gh_instruments(frame, ("a", "b"), version), ref)
    Z, names = markup_shifters(frame, ("a", "b"))
    assert names == ["own_a", "own_b", "rival_a", "rival_b", "own_products", "rival_products"]
    assert Z.shape == (n, 6)
    np.testing.assert_array_equal(Z[:, 4:], firm_product_counts(mk, fm))
    # local version: a count of the other products within one sd, so own + rival <= products in the market - 1
    assert np.all(Z[:, 0] <= Z[:, 4]) and np.all(Z[:, 2] <= Z[:, 5])
    with pytest.raises(ValueError):
        markup_shifters(frame, ())


def test_within_differences_pairs_and_gaps() -> None:
    ids = np.array(["x", "y", "x", "x", "y", "z"])
    per = np.array([1, 1, 2, 5, 3, 5])
    cur, prev = within_differences(ids, per)
    assert sorted(zip(cur.tolist(), prev.tolist())) == [(2, 0)]
    cur, prev = within_differences(ids, per, max_gap=3)
    assert sorted(zip(cur.tolist(), prev.tolist())) == [(2, 0), (3, 2), (4, 1)]
    with pytest.raises(ValueError):
        within_differences(np.array(["x", "x"]), np.array([1, 1]))
    with pytest.raises(ValueError):
        within_differences(ids, per, max_gap=0)


def test_difference_instruments_form_the_differenced_moment() -> None:
    rng = np.random.default_rng(1)
    n = 12
    ids = np.repeat(np.arange(4), 3)
    per = np.tile([1, 2, 3], 4)
    cur, prev = within_differences(ids, per)
    z = rng.normal(size=(n, 2))
    r = rng.normal(size=n)
    Zd = difference_instruments(cur, prev, z)
    np.testing.assert_allclose(Zd.T @ r, (z[cur] - z[prev]).T @ (r[cur] - r[prev]))
    # a residual constant within id (a model effect) leaves the moment at zero
    np.testing.assert_allclose(Zd.T @ ids.astype(float), 0.0, atol=1e-12)


def test_difference_iv_blocks_are_not_aggregate() -> None:
    rng = np.random.default_rng(2)
    n = 30
    ids = np.repeat(np.arange(10), 3)
    per = np.tile([1, 2, 3], 10)
    cur, prev = within_differences(ids, per)
    Zd = difference_instruments(cur, prev, rng.normal(size=(n, 2)))
    X1 = np.column_stack([np.ones(n), rng.normal(size=n)])
    M = np.eye(n) - X1 @ np.linalg.pinv(X1)
    xi = M @ rng.normal(size=n)
    fit = LinearFit(names=("c", "x"), beta=np.zeros(2), xi=xi, M=M)
    for diff_cls, base_cls in ((DifferenceIVMoments, IVMoments), (DifferenceSarganIVMoments, SarganIVMoments)):
        m = diff_cls(Zd, ids)
        assert m.name == "demand_fd" and not m.is_aggregate and m.is_difference and not m.is_micro
        a, b = m.evaluate(None, fit, True), base_cls(Zd, ids).evaluate(None, fit, True)   # type: ignore[arg-type]
        assert a.d_delta is not None and b.d_delta is not None
        np.testing.assert_array_equal(a.value, b.value)
        np.testing.assert_array_equal(a.d_delta, b.d_delta)

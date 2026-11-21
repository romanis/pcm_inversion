"""Tests for nfp_harness.micro_sets: household-set columns, GroupMomentRow arguments and the direct evaluator, on a
two-market table small enough to compute by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfp_harness.micro_sets import (MicroMomentEvaluator, MicroMoments, add_set_columns, group_moment_rows,
                                    model_micro_moments, set_column)

CHAR_MAP = {"Price": "prices", "Size": "size"}


def products() -> pd.DataFrame:
    return pd.DataFrame({"market_ids": [1, 1, 2, 2, 2], "prices": [1.0, 2.0, 1.5, 2.5, 3.0],
                         "size": [0.2, 0.4, 0.1, 0.3, 0.5]})


def agents() -> pd.DataFrame:
    return pd.DataFrame({"market_ids": [1, 1, 1, 2, 2], "weights": [0.5, 0.3, 0.2, 0.6, 0.4],
                         "inc": [1, 2, 2, 1, 2], "s1": [1.0, 1.0, 0.0, 1.0, 1.0]})


def micro() -> MicroMoments:
    rows = [{"name": "Price inc2-inc1", "survey": "s1", "char": "Price", "kind": "diff", "demographic": "inc",
             "a": 2, "b": 1, "value": 0.1, "variance": 0.04},
            {"name": "Size inc2-inc1", "survey": "s1", "char": "Size", "kind": "diff", "demographic": "inc",
             "a": 2, "b": 1, "value": 0.0, "variance": 0.01},
            {"name": "buy inc2/inc1", "survey": "s1", "char": "PurchaseProb", "kind": "ratio", "demographic": "inc",
             "a": 2, "b": 1, "value": 1.5, "variance": 0.25}]
    return MicroMoments(pd.DataFrame(rows))


def test_table_accessors() -> None:
    m = micro()
    np.testing.assert_array_equal(m.values, [0.1, 0.0, 1.5])
    np.testing.assert_array_equal(m.variances, [0.04, 0.01, 0.25])
    np.testing.assert_array_equal(m.weight, np.diag([25.0, 100.0, 4.0]))


def test_set_columns_and_rows() -> None:
    assert set_column("s1", "inc", 2.0) == "s1_inc_2"
    a = add_set_columns(agents(), ("s1",), {"inc": (1, 2)})
    assert list(a.columns[-2:]) == ["s1_inc_1", "s1_inc_2"]
    np.testing.assert_array_equal(a.s1_inc_1, [1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(a.s1_inc_2, [0.0, 1.0, 0.0, 0.0, 1.0])   # household 2 is outside the survey
    assert group_moment_rows(micro(), CHAR_MAP) == [("diff", "prices", "s1_inc_2", "s1_inc_1"),
                                                    ("diff", "size", "s1_inc_2", "s1_inc_1"),
                                                    ("ratio", None, "s1_inc_2", "s1_inc_1")]


def test_evaluator_matches_a_hand_computation() -> None:
    p, a, m = products(), agents(), micro()
    rng = np.random.default_rng(0)
    P = [rng.uniform(0.05, 0.3, size=(2, 3)), rng.uniform(0.05, 0.3, size=(3, 2))]
    got = MicroMomentEvaluator(p, a, m, CHAR_MAP)(P)
    # per household: purchase probability and expected x given a purchase, households in table order
    x = [p.loc[p.market_ids == t, ["prices", "size"]].to_numpy() for t in (1, 2)]
    buy = np.concatenate([Pt.sum(axis=0) for Pt in P])
    ex = np.concatenate([(xt.T @ Pt) / Pt.sum(axis=0) for xt, Pt in zip(x, P)], axis=1)
    w = a.weights.to_numpy()
    in_a = ((a.inc == 2) & (a.s1 == 1)).to_numpy()
    in_b = ((a.inc == 1) & (a.s1 == 1)).to_numpy()

    def mean(v: np.ndarray, s: np.ndarray) -> float:
        return float((w[s] * v[s]).sum() / w[s].sum())

    want = [mean(ex[0], in_a) - mean(ex[0], in_b), mean(ex[1], in_a) - mean(ex[1], in_b),
            mean(buy, in_a) / mean(buy, in_b)]
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(model_micro_moments(P, p, a, m, CHAR_MAP), got)


def test_evaluator_checks_shapes() -> None:
    ev = MicroMomentEvaluator(products(), agents(), micro(), CHAR_MAP)
    with pytest.raises(ValueError, match="2 markets"):
        ev([np.ones((2, 3))])
    with pytest.raises(ValueError, match="expected"):
        ev([np.ones((2, 3)), np.ones((2, 2))])

"""Tests for the comparison quantities shared by the PCM and BLP scripts (nfp_harness.comparison)."""

from __future__ import annotations

import numpy as np

from nfp_harness import comparison
from nfp_harness.models import pcm_post as pp


def _logit_market(J: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain logit shares and their exact price Jacobian (alpha = 0.5), a market with known diversion."""
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(5.0, 20.0, J))
    v = rng.normal(0.0, 1.0, J) - 0.5 * p + 4.0
    e = np.exp(v)
    s = e / (1.0 + e.sum())
    ds_dp = 0.5 * (np.outer(s, s) - np.diag(s))  # ds_j/dp_k
    return s, ds_dp, p


def test_market_quantities_match_logit_closed_form() -> None:
    s, ds_dp, p = _logit_market(6, 1)
    labels = np.array(["a", "a", "b", "b", "b", "c"])
    m = comparison.market_quantities(s, ds_dp, p, labels, ["a", "b", "c"], cost=0.8 * p)
    np.testing.assert_allclose(m["own_elasticity"], -0.5 * p * (1 - s), rtol=1e-12)
    # logit diversion D_jk = s_k / (1 - s_j); outside diversion s_0 / (1 - s_j)
    s0 = 1 - s.sum()
    np.testing.assert_allclose(m["outside_diversion"], s0 / (1 - s), rtol=1e-12)
    within = np.array([(s[labels == labels[j]].sum() - s[j]) / (s.sum() - s[j]) for j in range(6)])
    np.testing.assert_allclose(m["diversion_leaving_group"], 1 - within, rtol=1e-12)
    np.testing.assert_allclose(m["margin"], 0.2, rtol=1e-12)
    assert np.all(m["nonzero_cross"] == 5) and np.all(m["nonzero_cross_ratio"] == 1.0)


def test_group_matrix_rows_sum_to_one_and_summary() -> None:
    names = ["a", "b"]
    markets = []
    for t in range(3):
        s, ds_dp, p = _logit_market(5, 10 + t)
        markets.append(comparison.market_quantities(s, ds_dp, p, np.array(["a", "b", "a", "b", "b"]), names))
    prod, gm = comparison.products_from_markets(markets, names)
    assert prod["shares"].size == 15 and gm.shape == (2, 3)
    np.testing.assert_allclose(gm.sum(axis=1), 1.0, rtol=1e-12)  # inside groups plus outside
    out = comparison.summarise(prod, gm, names)
    assert out["group_diversion"]["a"]["outside"] == gm[0, -1]
    assert out["own_elasticity_median"] == float(np.median(prod["own_elasticity"]))
    assert out["margin_median"] is None  # no costs given


def test_cs_gain_by_group_adds_up_to_the_total() -> None:
    rng = np.random.default_rng(3)
    R = 40
    w = rng.uniform(0.5, 1.5, R)
    w /= w.sum()
    obs = rng.uniform(1.0, 2.0, R)
    per = {"observed": obs, "fixed_prices": obs - 0.1, "nash": obs - rng.uniform(0.0, 0.3, R)}
    labels = np.where(np.arange(R) % 3 == 0, "low", "high")
    rows = comparison.cs_gain_by_group(per, w, labels, households=1000.0)
    total = sum(r["gain_nash_musd"] for r in rows)
    np.testing.assert_allclose(total, 1000.0 * w @ (per["observed"] - per["nash"]), rtol=1e-12)
    np.testing.assert_allclose(sum(r["gain_fixed_prices_musd"] for r in rows), 1000.0 * 0.1, rtol=1e-12)
    assert {r["group"] for r in rows} == {"low", "high"}
    assert rows == comparison.cs_gain_by_group(per, w, labels, 1000.0)
    assert pp.cs_by_group(per["observed"], w, labels)[0]["mass"] == rows[0]["mass"]


def test_scalar_items_drops_nested_and_non_finite() -> None:
    d = {"a": 1.5, "b": "x", "c": {"d": 1}, "e": float("nan"), "f": None, "g": [1], "h": np.float64(2.0)}
    assert comparison.scalar_items(d) == {"a": 1.5, "b": "x", "e": None, "f": None, "h": 2.0}


def test_weighted_median() -> None:
    from nfp_harness.comparison import weighted_median

    v = np.array([0.1, 0.2, 0.3, 0.4, np.nan])
    assert weighted_median(v[:3], np.ones(3)) == 0.2
    # sales weights: more than half the weight sits on 0.4, so it is the weighted median; at 2.9 of 5.9 it is not
    assert weighted_median(v, np.array([1.0, 1.0, 1.0, 3.1, 5.0])) == 0.4
    assert weighted_median(v, np.array([1.0, 1.0, 1.0, 2.9, 5.0])) == 0.3
    assert weighted_median(v, np.array([0.0, 2.0, 0.0, 1.0, 1.0])) == 0.2   # zero weights and NaN values left out
    assert np.isnan(weighted_median(np.array([np.nan]), np.ones(1)))

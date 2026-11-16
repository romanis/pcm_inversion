"""Tests for params (WP4; DESIGN 2.5) and the data helpers of nfp_harness/data.py (DESIGN 6.1)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nfp_harness.data import break_price_ties, cross_agents, sobol_normal_draws
from nfp_harness.params import ParameterLayout, ParameterSpec

# ------------------------------------------------------------------ params


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"hpwt": rng.normal(0.4, 0.1, 200), "space": rng.normal(1.2, 0.3, 200),
            "logfs_fv": rng.normal(0.5, 0.8, 1000), "low_inc": rng.uniform(0, 0.2, 1000)}


def _layout() -> ParameterLayout:
    return ParameterLayout([
        ParameterSpec("sigma_hpwt", "taste", "hpwt", "nu1", 2.0, 0.0, 100.0),
        ParameterSpec("sigma_sw", "taste", "sw", "nu7", 2.0, 0.0, 66.1, cap=True, unit=1.0, note="cap"),
        ParameterSpec("pi_space", "taste", "space", "logfs_fv", 0.3, -np.inf, np.inf),
        ParameterSpec("phi_low", "scale", None, "low_inc", 0.0, -np.inf, np.inf),
        ParameterSpec("sigma_p", "sigma_p", None, None, 1.0, 0.05, 3.0, free=False),
    ])


def test_automatic_units() -> None:
    data = _data()
    layout = _layout()
    u = layout.units(data)
    np.testing.assert_allclose(u, [1 / np.std(data["hpwt"]), 1.0, 1 / (np.std(data["space"]) * np.std(data["logfs_fv"])),
                                   1 / np.std(data["low_inc"]), 0.1])
    np.testing.assert_allclose(layout.units(), u)  # cached


def test_automatic_unit_errors_are_named() -> None:
    with pytest.raises(ValueError, match="units\\(data\\) first"):
        _layout().units()
    data = _data()
    data["hpwt"] = np.ones(200)
    with pytest.raises(ValueError, match="sigma_hpwt.*sd"):
        _layout().units(data)
    with pytest.raises(ValueError, match="'space'"):
        _layout().units({k: v for k, v in _data().items() if k != "space"})
    with pytest.raises(ValueError, match="role 'other'"):
        ParameterLayout([ParameterSpec("a", "other", None, None, 0.0, -1.0, 1.0)]).units({})


def test_scaled_round_trip_bounds_and_masks() -> None:
    layout = _layout()
    u_nat = layout.units(_data())
    theta = layout.starts() + np.array([0.5, -1.0, 0.2, 0.1, 0.3])
    u = layout.to_scaled(theta)
    np.testing.assert_allclose(u, (theta - layout.starts()) / u_nat)
    np.testing.assert_allclose(layout.from_scaled(u), theta)
    lower, upper = layout.bounds()
    assert lower[0] == 0.0 and upper[1] == 66.1 and np.isinf(lower[2])
    np.testing.assert_array_equal(layout.free_mask(), [True, True, True, True, False])
    np.testing.assert_array_equal(layout.cap_mask(), [False, True, False, False, False])
    assert layout.n_free() == 4 and layout.P == 5 and layout.index("sigma_p") == 4
    assert layout.in_bounds(theta) and not layout.in_bounds(theta * np.array([1, 1, 1, 1, 10]))
    assert not layout.in_bounds(np.r_[theta[:4], np.nan])


def test_replace_and_with_starts_keep_units() -> None:
    layout = _layout()
    u = layout.units(_data())
    fixed = layout.replace(sigma_p={"free": True, "start": 2.0}, phi_low={"free": False})
    assert fixed["sigma_p"].start == 2.0 and fixed.free_mask().tolist() == [True, True, True, False, True]
    np.testing.assert_allclose(fixed.units(), u)
    moved = layout.with_starts(layout.starts() + 0.1)
    np.testing.assert_allclose(moved.starts(), layout.starts() + 0.1)
    with pytest.raises(KeyError):
        layout.replace(nope={"free": False})
    assert layout.describe()[1]["cap"] is True and layout.describe()[0]["unit"] == pytest.approx(u[0])


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ParameterLayout([ParameterSpec("a", "sigma_p", None, None, 1.0, 0.0, 2.0)] * 2)
    with pytest.raises(ValueError, match="outside"):
        ParameterSpec("a", "sigma_p", None, None, 3.0, 0.0, 2.0)


# ------------------------------------------------------------------ data helpers


def _old_break_price_ties(p: np.ndarray, min_gap: float) -> np.ndarray:
    """The rule in the local petrin_replication/pcm_petrin.py, one market at a time."""
    p = p.copy()
    order = np.argsort(p, kind="stable")
    for k in range(1, len(p)):
        i, j = order[k - 1], order[k]
        if p[j] < p[i] + min_gap:
            p[j] = p[i] + min_gap
    return p


def test_break_price_ties_matches_old_rule_per_market() -> None:
    rng = np.random.default_rng(1)
    market_ids = rng.integers(0, 3, size=60)
    prices = np.round(rng.uniform(5, 10, size=60), 1)  # many exact ties
    out = break_price_ties(prices, market_ids, 0.01)
    for t in range(3):
        rows = market_ids == t
        np.testing.assert_array_equal(out[rows], _old_break_price_ties(prices[rows], 0.01))
        sp = np.sort(out[rows])
        assert np.all(np.diff(sp) >= 0.01 - 1e-12)
        # order preserved: a strictly cheaper product stays cheaper
        a, b = np.meshgrid(prices[rows], prices[rows])
        oa, ob = np.meshgrid(out[rows], out[rows])
        assert np.all(oa[a < b] < ob[a < b])
    assert np.all(out >= prices)


def test_sobol_normal_draws() -> None:
    x = sobol_normal_draws(4096, 6, seed=3)
    assert x.shape == (4096, 6)
    assert np.allclose(x.mean(0), 0.0, atol=0.01) and np.allclose(x.std(0), 1.0, atol=0.01)
    np.testing.assert_array_equal(x, sobol_normal_draws(4096, 6, seed=3))
    assert not np.array_equal(x, sobol_normal_draws(4096, 6, seed=4))
    with pytest.raises(ValueError, match="power of two"):
        sobol_normal_draws(4000, 6, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sobol_normal_draws(4000, 2, seed=3, require_power_of_two=False).shape == (4000, 2)


def test_cross_agents() -> None:
    A = 1000
    w = np.full(A, 0.001)
    demo = np.column_stack([np.arange(A), np.arange(A) % 2]).astype(float)
    nu = sobol_normal_draws(4096, 3, seed=0)
    w_r, demo_r, nu_r = cross_agents(w, demo, nu)
    assert w_r.shape == (4096,) and demo_r.shape == (4096, 2) and nu_r is not None
    assert w_r.sum() == pytest.approx(1.0)
    counts = np.bincount(demo_r[:, 0].astype(int))
    assert set(counts) == {4, 5} and counts.sum() == 4096
    np.testing.assert_allclose(np.bincount(demo_r[:, 0].astype(int), weights=w_r), w)  # each agent keeps its weight
    w4, d4, _ = cross_agents(w, demo, nu[:4000], nu_per_agent=4)
    np.testing.assert_allclose(w4, 0.00025)
    np.testing.assert_array_equal(d4[:8, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="expected 1000 agents x 4"):
        cross_agents(w, demo, nu, nu_per_agent=4)

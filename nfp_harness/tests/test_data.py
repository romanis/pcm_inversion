"""Tests for nfp_harness.data: Sobol normal draws."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, qmc

from nfp_harness.data import sobol_normal_draws


def test_sobol_draws_are_finite_where_a_point_is_exactly_zero() -> None:
    """Seed 8673321 (GMY 1991 at nu 16) puts one coordinate of 16,384 points at exactly 0; its quantile was -inf. That
    coordinate moves to 0.5 / n and every other draw keeps its value bit for bit."""
    n, dim, seed = 16384, 11, 8673321
    u = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(14)
    zero = u <= 0.0
    assert zero.sum() == 1
    d = sobol_normal_draws(n, dim, seed)
    assert np.isfinite(d).all()
    assert d[zero][0] == norm.ppf(0.5 / n)
    np.testing.assert_array_equal(d[~zero], norm.ppf(u[~zero]))


def test_drop_small_shares_keeps_rows_at_or_above_the_threshold_and_one_per_market() -> None:
    from nfp_harness.data import drop_small_shares

    s = np.array([0.2, 1e-6, 0.05, 2e-7, 3e-7, 1e-5])
    m = np.array([1, 1, 1, 2, 2, 3])
    keep = drop_small_shares(s, m, 1e-5)
    assert keep.tolist() == [True, False, True, False, True, True]   # market 2 keeps its largest row
    assert drop_small_shares(s, m, 0.0).all()
    with pytest.raises(ValueError):
        drop_small_shares(s, m, -1.0)


def test_income_law_basis_is_petrins_clipped_log_ratio() -> None:
    from nfp_harness.data import INCOME_LAW_CLIP, income_law_basis

    y = np.array([0.2, 1.0, 5.78, 40.0, 150.0, 900.0])
    b = income_law_basis(y, 5.78)
    np.testing.assert_array_equal(b, np.log(5.78 / np.array([1.0, 1.0, 5.78, 40.0, 150.0, 150.0])))
    assert b[2] == 0.0 and INCOME_LAW_CLIP == (1.0, 150.0)
    assert np.ptp(b) == np.log(150.0)   # c_i spans at most the factor hi / lo
    np.testing.assert_array_equal(income_law_basis(y, 2.0, (2.0, 4.0)), np.log(2.0 / np.clip(y, 2.0, 4.0)))
    with pytest.raises(ValueError):
        income_law_basis(y, 0.0)
    with pytest.raises(ValueError):
        income_law_basis(y, 1.0, (5.0, 1.0))


def test_antithetic_draws_pair_each_point_with_its_negation() -> None:
    """Columns listed in ``antithetic`` come as (u, -u) pairs of the n / 2 Sobol points, so they sum to exactly zero;
    the other columns repeat within a pair; without the option the draws are the plain ones, bit for bit."""
    import math

    n, dim, seed = 1024, 5, 8673325
    plain = sobol_normal_draws(n, dim, seed)
    np.testing.assert_array_equal(sobol_normal_draws(n, dim, seed, antithetic=()), plain)
    d = sobol_normal_draws(n, dim, seed, antithetic=(1, 3))
    half = sobol_normal_draws(n // 2, dim, seed)
    np.testing.assert_array_equal(d[0::2], half)
    np.testing.assert_array_equal(d[1::2, [1, 3]], -half[:, [1, 3]])
    np.testing.assert_array_equal(d[1::2, [0, 2, 4]], half[:, [0, 2, 4]])
    for c in (1, 3):
        assert math.fsum(d[:, c]) == 0.0   # exactly zero mean in the paired dimensions
        assert np.all(d[0::2, c] + d[1::2, c] == 0.0)
    assert abs(math.fsum(d[:, 0])) > 0.0
    full = sobol_normal_draws(n, dim, seed, antithetic=range(dim))
    assert all(math.fsum(full[:, c]) == 0.0 for c in range(dim))
    with pytest.raises(ValueError, match="even"):
        sobol_normal_draws(7, dim, seed, require_power_of_two=False, antithetic=(0,))
    with pytest.raises(ValueError, match="antithetic columns"):
        sobol_normal_draws(n, dim, seed, antithetic=(dim,))


def test_zscore_uses_the_n_minus_1_sd() -> None:
    from nfp_harness.data import zscore

    v = np.array([1.0, 2.0, 4.0, 9.0])
    z = zscore(v)
    np.testing.assert_array_equal(z, (v - v.mean()) / v.std(ddof=1))
    assert abs(z.mean()) < 1e-15 and z.std(ddof=1) == pytest.approx(1.0, abs=1e-15)

"""Tests for nfp_harness.second_choice on synthetic data."""

from __future__ import annotations

import numpy as np

from nfp_harness.second_choice import first_second_correlations, inside_mass_without, logit_second_choice_matrix


def test_logit_matrix_is_iia_for_one_household():
    P = np.array([[0.1], [0.2], [0.3]])
    C = logit_second_choice_matrix(P, np.array([1.0]))
    assert np.allclose(np.diag(C), 0)
    assert np.allclose(C.sum(axis=1), 1)
    S = P.sum()
    for j in range(3):
        for k in range(3):
            if j != k:
                assert np.isclose(C[j, k], P[k, 0] / (S - P[j, 0]))


def test_logit_matrix_weights_buyers():
    rng = np.random.default_rng(0)
    P = rng.uniform(0.01, 0.1, size=(5, 4))
    w = np.array([0.1, 0.2, 0.3, 0.4])
    C = logit_second_choice_matrix(P, w)
    j = 2
    buyers = w * P[j] / (w @ P[j])
    expect = np.array([sum(buyers[h] * P[k, h] / (P[:, h].sum() - P[j, h]) for h in range(4)) if k != j else 0.0
                       for k in range(5)])
    assert np.allclose(C[j], expect)
    assert np.allclose(C.sum(axis=1), 1)


def test_correlations_limits():
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    s = np.full(4, 0.1)
    same = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    assert np.isclose(first_second_correlations(s, same, x)[0], 1.0)
    other = same[:, ::-1]
    assert np.isclose(first_second_correlations(s, other, x)[0], -1.0)
    indep = np.full((4, 4), 0.25)
    assert np.isclose(first_second_correlations(s, indep, x)[0], 0.0)
    # a first choice without respondents (zero row) drops out after renormalisation
    z = same.copy()
    z[0] = 0
    assert np.isclose(first_second_correlations(s, z, x)[0], 1.0)


def test_logit_matrix_when_one_product_holds_almost_all_inside_mass():
    # a household that buys product 0 with probability 1 - 2e-13: R_h - P_0h cancels to 2e-13 +- 1.1e-16, so the
    # shipped difference gave C[0, 1] = 0.5 +- 3e-4 and a row sum off 1 by as much, and the diagonal (B_0h P_0h =
    # 5e12 subtracted from itself) kept its rounding; both jumped as theta moved
    P = np.array([[1.0 - 2e-13, 0.3], [1e-13, 0.3], [1e-13, 0.3]])
    C = logit_second_choice_matrix(P, np.array([1.0, 1e-30]))
    assert np.allclose(C.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(np.diag(C) == 0.0)
    assert np.isclose(C[0, 1], 0.5, rtol=1e-12, atol=0)
    assert np.isclose(C[0, 2], 0.5, rtol=1e-12, atol=0)
    rest = inside_mass_without(P)
    assert np.isclose(rest[0, 0], 2e-13, rtol=1e-12, atol=0)
    assert np.allclose(rest[1:, 0], 1.0 - 1e-13, rtol=1e-15, atol=0)

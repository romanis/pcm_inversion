"""Tests for nfp_harness.second_choice on synthetic data."""

from __future__ import annotations

import numpy as np

from nfp_harness.second_choice import first_second_correlations, logit_second_choice_matrix


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

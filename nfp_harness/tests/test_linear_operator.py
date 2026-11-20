"""Tests for the annihilator operator and two-way absorption (GMY spec WP-G1)."""

from __future__ import annotations

import resource
import time

import numpy as np
import pytest

from nfp_harness.linear import AbsorbTwoWay, Annihilator, LinearIV
from nfp_harness.moments import IVMoments, SarganIVMoments
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.tests.test_problem import LogitRCModel, make_data, make_layout, make_micro


def _design(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X1 = [1, 4 continuous, dummies of 3 groups (first level dropped)], Z = [X1, 5 excluded]."""
    rng = np.random.default_rng(seed)
    g1, g2, g3 = rng.integers(0, 4, n), rng.integers(0, 3, n), rng.integers(0, 3, n)
    Xc = rng.normal(size=(n, 4))
    dums = [(g == v).astype(float) for g, levels in ((g1, 4), (g2, 3), (g3, 3)) for v in range(1, levels)]
    X1 = np.column_stack([np.ones(n), Xc] + dums)
    Z = np.column_stack([X1, rng.normal(size=(n, 5)) + 0.3 * Xc[:, :1]])
    return X1, Z, Xc, g1, g2


def _spd(k: int, seed: int) -> np.ndarray:
    A = np.random.default_rng(seed).normal(size=(k, k))
    return A @ A.T / k + np.eye(k)


def test_operator_matches_dense_products() -> None:
    X1, Z, _, _, _ = _design()
    n = X1.shape[0]
    assert X1.shape[1] == 12
    lin = LinearIV(X1, Z, W_aa=_spd(Z.shape[1], 1), dense_m=False)
    M = lin.M
    assert isinstance(M, Annihilator) and M.shape == (n, n)
    Md = np.eye(n) - X1 @ lin.projection()
    np.testing.assert_allclose(M.dense(), Md, rtol=0, atol=1e-13)
    rng = np.random.default_rng(2)
    V, v = rng.normal(size=(n, 7)), rng.normal(size=n)
    A = rng.normal(size=(4, n))
    for got, want in ((M @ V, Md @ V), (M @ v, Md @ v), (A @ M, A @ Md), (v @ M, v @ Md), (Z.T @ M, Z.T @ Md)):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert lin.M is M   # one operator per weighting (the identity cache of IVMoments)
    lin.set_weighting(_spd(Z.shape[1], 3))
    assert lin.M is not M


def test_dense_rule_by_size() -> None:
    X1, Z, _, _, _ = _design(200)
    assert isinstance(LinearIV(X1, Z).M, np.ndarray)
    assert isinstance(LinearIV(X1, Z, dense_m=False).M, Annihilator)


@pytest.mark.parametrize("sargan", [False, True])
def test_problem_identical_with_operator(sargan: bool) -> None:
    d = make_data(0)
    out = []
    for dense in (True, False):
        model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
        iv = (SarganIVMoments if sargan else IVMoments)(d["Z"], d["clusters"])
        prob = NFPGMMProblem(model, [iv, make_micro()], LinearIV(d["X1"], d["Z"], dense_m=dense), make_layout())
        evs = [prob.evaluate(th) for th in (np.array([0.6, 0.2, 0.5]), np.array([0.8, 0.3, 0.6]),
                                             np.array([0.4, -0.1, 0.3]))]
        assert all(ev.ok for ev in evs)
        cue = None if sargan else prob.cue_terms(evs[0])
        out.append((evs, cue))
    for a, b in zip(out[0][0], out[1][0]):
        np.testing.assert_allclose(a.g, b.g, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.G, b.G, rtol=0, atol=1e-12)
    if not sargan:
        for x, y in zip(out[0][1], out[1][1]):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


def test_frisch_waugh_lovell_two_way() -> None:
    rng = np.random.default_rng(4)
    n = 500
    ga, gb = rng.integers(0, 6, n), rng.integers(0, 5, n)
    Xc = rng.normal(size=(n, 3))
    G = np.column_stack([(ga == v).astype(float) for v in range(6)] + [(gb == v).astype(float) for v in range(1, 5)])
    X1 = np.column_stack([Xc, G])
    Z = np.column_stack([X1, rng.normal(size=(n, 4))])
    delta = Xc @ np.array([1.0, -0.5, 0.2]) + G @ rng.normal(size=G.shape[1]) + rng.normal(size=n)
    xi = LinearIV(X1, Z, dense_m=False).fit(delta).xi
    ab = AbsorbTwoWay(ga, gb)
    Xt, dt = ab.demean(Xc), ab.demean(delta)
    beta_c = np.linalg.lstsq(Xt, dt, rcond=None)[0]
    np.testing.assert_allclose(xi, dt - Xt @ beta_c, rtol=0, atol=1e-10)
    np.testing.assert_allclose(xi, ab.fit(delta - Xc @ beta_c).xi, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ab.M.dense(), ab.M.dense().T, atol=1e-13)


def test_gmy_size_operator_is_cheap() -> None:
    gmy_data = pytest.importorskip("nfp_harness.gmy_data")
    try:
        p = gmy_data.load_products()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    cols = [np.ones(len(p))] + [p[c].to_numpy(float) for c in gmy_data.STAGE2_X]
    for c in ("market_ids", "make2_do"):
        for v in sorted(p[c].unique())[1:]:
            cols.append((p[c] == v).to_numpy(float))
    X1 = np.column_stack(cols)
    excl = [p[c].to_numpy(float) for c in ("lag_pl_con", "iv_prod", "steel_weight", "IV_N", "IV_N_weight")]
    Z = np.column_stack([X1] + [(e - e.mean()) / e.std() for e in excl]
                        + [np.random.default_rng(0).normal(size=(len(p), 5))])
    assert X1.shape == (9694, 107) and Z.shape == (9694, 117)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    lin = LinearIV(X1, Z)
    assert time.perf_counter() - t0 < 5.0
    xi = lin.fit(np.random.default_rng(1).normal(size=len(p))).xi
    t1 = time.perf_counter()
    ZtM = Z.T @ lin.M / len(p)
    xM = xi @ lin.M
    assert time.perf_counter() - t1 < 0.2
    assert ZtM.shape == (117, 9694) and xM.shape == (9694,)
    assert (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024 < 100   # kB -> MB

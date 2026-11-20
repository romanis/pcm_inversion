"""Tests for the PCM second-choice moments (GMY spec WP-G4): window algebra on hand cases (oracle), the correlation
adjoint, the moment on a logit second-choice matrix, and (test 6, WP-G6) the kernel against the oracle."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from nfp_harness.moments import SECOND_CHOICE_OUTPUTS, SecondChoiceMoments
from nfp_harness.second_choice import correlation_adjoint, first_second_correlations, logit_second_choice_matrix
from nfp_harness.tests.second_choice_oracle import second_choices, truncated_cdf
from nfp_harness.types import EvalStatus, InnerSolution, MarketIndex

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("sigma", [0.5, 2.5])
def test_middle_product_removed(sigma: float) -> None:
    p, d = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.5, 4.2])
    F = lambda a: float(truncated_cdf(np.array([a]), sigma, 0.0)[0])   # noqa: E731
    s, q, _ = second_choices(d, p, sigma)
    den = F(1.5) - F(0.7)
    assert abs(s[1] - den) < 1e-12
    assert abs(q[1, 0] - (F(1.5) - F(1.1)) / den) < 1e-10
    assert abs(q[1, 2] - (F(1.1) - F(0.7)) / den) < 1e-10
    assert q[1, 1] == 0.0


def test_cheapest_product_removed() -> None:
    p, d, sigma = np.array([1.0, 2.0]), np.array([1.5, 2.2]), 1.0
    F = lambda a: float(truncated_cdf(np.array([a]), sigma, 0.0)[0])   # noqa: E731
    s, q, m = second_choices(d, p, sigma)
    assert abs(q[0, 1] - 1.0) < 1e-12
    assert abs((s[0] - m[0, 1]) - (F(1.5) - F(1.1))) < 1e-10   # buyers of 1 with a in (1.1, 1.5) buy nothing else


def test_bound_cuts_the_windows() -> None:
    p, d, sigma = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.5, 4.2]), 1.5
    s, q, _ = second_choices(d, p, sigma, a_lo=1.2)
    assert s[1] > 0 and abs(q[1, 0] - 1.0) < 1e-12 and q[1, 2] == 0.0
    s, q, _ = second_choices(d, p, sigma, a_lo=1.6)
    assert s[1] == 0.0 and not q[1].any()


def test_correlation_adjoint_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    J, K = 6, 3
    s = rng.uniform(0.01, 0.1, J)
    C = rng.uniform(0.0, 1.0, (J, J))
    np.fill_diagonal(C, 0.0)
    C /= C.sum(axis=1, keepdims=True)
    x = np.column_stack([rng.normal(size=J), (rng.random(J) < 0.5).astype(float), rng.normal(size=J)])
    corr, A = correlation_adjoint(s, C, x)
    np.testing.assert_allclose(corr, first_second_correlations(s, C, x), rtol=1e-14)
    h = 1e-6
    for j in range(J):
        for k in range(J):
            E = np.zeros((J, J))
            E[j, k] = h
            fd = (first_second_correlations(s, C + E, x) - first_second_correlations(s, C - E, x)) / (2 * h)
            np.testing.assert_allclose(A[:, j, k], fd, rtol=1e-6, atol=1e-9)


def test_moment_on_a_logit_matrix_reproduces_the_blp_correlations() -> None:
    from nfp_harness import gmy_data
    try:
        data = gmy_data.load(n_households=300, years=(1999,))
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    p = data.products
    x = p[list(gmy_data.SECOND_CHOICE_CHARS)].to_numpy(float)
    s = p.shares.to_numpy(float)
    data_corr = first_second_correlations(s, data.second_choices[1999]["prob"], x)
    blp = ROOT / "gmy_replication" / "results" / "blp_fit_h1000.json"
    if blp.exists():
        ref = np.asarray(json.loads(blp.read_text())["second_choice"]["1999"]["data"], dtype=float)
        np.testing.assert_allclose(data_corr, ref, rtol=1e-12, atol=1e-14, equal_nan=True)
    rng = np.random.default_rng(3)
    H = 300
    u = rng.normal(size=(len(p), H)) - 5.0
    P = np.exp(u) / (1.0 + np.exp(u).sum(axis=0, keepdims=True))
    C = logit_second_choice_matrix(P, data.agents.weights.to_numpy(float))
    mom = SecondChoiceMoments("second_choice", (1999,), gmy_data.SECOND_CHOICE_CHARS, {1999: data_corr},
                              {1999: x}, {1999: s})
    assert mom.k == 10   # electric has no variance in 1999
    mom.bind([MarketIndex(1999, np.arange(len(p)))], len(p), 3)
    sol = InnerSolution(delta=np.zeros(len(p)), D=None, status=EvalStatus.OK, per_market=[],
                        outputs={SECOND_CHOICE_OUTPUTS: [{"C": C, "s": P @ data.agents.weights.to_numpy(float)}]})
    keep = np.isfinite(data_corr)
    np.testing.assert_allclose(mom.predicted(sol), first_second_correlations(s, C, x)[keep], rtol=1e-12, atol=1e-14)
    blk = mom.evaluate(sol, None, False)   # type: ignore[arg-type]
    np.testing.assert_allclose(blk.value, first_second_correlations(s, C, x)[keep] - data_corr[keep], atol=1e-14)
    np.testing.assert_allclose(mom.fixed_covariance, np.eye(10) / 900.0)


def test_gmy_second_choice_moment_has_41_rows() -> None:
    from nfp_harness import gmy_data
    from nfp_harness.tests.test_gmy_instruments import gmy_module
    try:
        gmy_data.package_root()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    gmy = gmy_module()
    spec = gmy.GMYSpec(gmy_data.SECOND_CHOICE_YEARS)
    (mom,) = spec.second_choice_moment()
    assert mom.k == 41 and mom.weight == 30.0
    raw = spec.second_choice_data()
    for y in gmy_data.SECOND_CHOICE_YEARS:
        merged = mom.data_corr[y]
        assert np.isfinite(merged).sum() == (10 if y < 2015 else 11)
        # the twin merge moves the data correlations by little (the unmerged ones are BLP's data side)
        np.testing.assert_allclose(merged, np.asarray(raw[y]), atol=0.02, equal_nan=True)
        assert mom.s_data[y].size == spec.market_rows(y).size


def _kernel() -> object:
    native = pytest.importorskip("nfp_harness._native")
    if not hasattr(native.pcm_kernel, "second_choice"):
        pytest.skip("kernel built without second_choice")
    return native.pcm_kernel


@pytest.mark.parametrize("bound", ["a_lo", "alpha_lo"])
def test_kernel_matches_the_oracle(bound: str) -> None:
    k = _kernel()
    from nfp_harness.tests.second_choice_oracle import market_second_choices
    rng = np.random.default_rng(8)
    J, R = 20, 50
    x = np.column_stack([np.ones(J), rng.normal(size=J), (rng.random(J) < 0.5).astype(float)])
    p = 1.0 + 0.2 * np.arange(J) + 0.05 * rng.random(J)
    delta = 1.6 * p - 0.08 * p ** 2 - 0.3 + 0.05 * rng.normal(size=J)
    w = rng.uniform(0.5, 1.5, R) / R
    tb, sb = rng.normal(size=(R, 2)), 0.5 * rng.normal(size=(R, 1))
    th, ph, sp = np.array([0.3, 0.2]), np.array([0.4]), 1.4
    mk = k.Market(x, p, w, tb, [1, 2], sb)
    got = k.second_choice(delta, mk, th, ph, sp, n_threads=4, **{bound: 0.2})
    c = np.exp(sb @ ph)
    dc = (delta[None, :] + (tb * th) @ x[:, 1:].T) / c[:, None]
    a_lo = np.full(R, 0.2) if bound == "a_lo" else 0.2 / c
    s, C = market_second_choices(dc, p, w, sp, a_lo, n_grid=20_000)
    np.testing.assert_allclose(got["s"], s, rtol=0, atol=1e-10)
    np.testing.assert_allclose(got["C"], C, rtol=0, atol=1e-6)


def test_moment_jacobian_through_the_kernel() -> None:
    _kernel()
    from nfp_harness.moments import SECOND_CHOICE_ADJOINT
    from nfp_harness.tests import test_group_micro as tg
    model, delta, data = tg.model_at()
    theta = tg.layout().starts()
    pr = data["products"]
    xcols = np.column_stack([pr["x1"], pr["prices"], (pr["x1"] > 0).astype(float)])
    years = tuple(m.market_id for m in model.markets)
    xs = {m.market_id: xcols[m.rows] for m in model.markets}
    sd = {m.market_id: pr["shares"][m.rows] for m in model.markets}
    mom = SecondChoiceMoments("second_choice", years, ("x1", "p", "top"), {y: np.array([0.3, 0.5, 0.2]) for y in years},
                              xs, sd)
    mom.bind(model.markets, model.n_products, model.P)

    def sol_at(d: np.ndarray, th: np.ndarray) -> InnerSolution:
        outs = [model.second_choice(t, d[m.rows], th) for t, m in enumerate(model.markets)]

        def adj(t: int, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            o = model.second_choice(t, d[model.markets[t].rows], th, {"A": A, "B": B})
            return o["out_delta"], o["out_theta"]

        return InnerSolution(delta=d, D=None, status=EvalStatus.OK, per_market=[],
                             outputs={SECOND_CHOICE_OUTPUTS: outs, SECOND_CHOICE_ADJOINT: adj})

    val, dd, dt = mom.correlations(sol_at(delta, theta), True)
    assert dd is not None and dt is not None and val.shape == (6,)
    h = 1e-7
    for l in range(model.n_products):
        e = np.zeros_like(delta)
        e[l] = h
        fd = (mom.predicted(sol_at(delta + e, theta)) - mom.predicted(sol_at(delta - e, theta))) / (2 * h)
        np.testing.assert_allclose(dd[:, l], fd, rtol=1e-6, atol=1e-7)
    for k in range(4):
        e = np.zeros_like(theta)
        e[k] = h
        fd = (mom.predicted(sol_at(delta, theta + e)) - mom.predicted(sol_at(delta, theta - e))) / (2 * h)
        np.testing.assert_allclose(dt[:, k], fd, rtol=1e-6, atol=1e-7)
    model.close()

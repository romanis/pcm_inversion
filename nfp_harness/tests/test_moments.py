"""Tests for moments and linear concentration (WP4; DESIGN 2.3, 2.4, WP4 acceptance)."""

from __future__ import annotations

import numpy as np
import pytest

from nfp_harness.linear import Absorb, LinearIV
from nfp_harness.moments import MICRO_OUTPUTS, IVMoments, MicroDataset, MicroPart, product_part
from nfp_harness.tests.test_problem import THETA_TRUE, LogitRCModel, central_fd, make_data, make_micro
from nfp_harness.types import EvalStatus, InnerSolution, KernelRequests


def solution_at(model: LogitRCModel, delta: np.ndarray, theta: np.ndarray, requests: KernelRequests,
                need_jacobian: bool = True) -> InnerSolution:
    """Kernel outputs at a given (delta, theta) without inversion, for partial derivatives."""
    parts: dict[str, list] = {p.name: [] for p in requests.micro_parts}
    for t, mkt in enumerate(model.markets):
        B, mu = model.basis(t, theta)
        s_ij = model.choice_probs(delta[mkt.rows], mu)
        for p in requests.micro_parts:
            parts[p.name].append(model.micro_part(t, p, s_ij, B, need_jacobian))
    return InnerSolution(delta=delta, D=None, status=EvalStatus.OK, per_market=[], outputs={MICRO_OUTPUTS: parts})


def fixture():
    d = make_data()
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
    mom = make_micro()
    mom.bind(model.markets, model.n_products, 3)
    return d, model, mom


# ------------------------------------------------------------------ linear and IV


def test_iv_with_full_Z_reproduces_2sls_beta() -> None:
    """WP4 acceptance: LinearIV with Z = [X1, Z_excl] and W = (Z'Z/n)^{-1} is 2SLS."""
    d = make_data()
    X1, Z = d["X1"], d["Z"]
    rng = np.random.default_rng(7)
    y = X1 @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=X1.shape[0])
    fit = LinearIV(X1, Z, names=["const", "x", "mi"]).fit(y)
    # textbook 2SLS by two least-squares stages, independent of the LinearIV algebra
    X_hat = Z @ np.linalg.lstsq(Z, X1, rcond=None)[0]
    beta_2sls = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    np.testing.assert_allclose(fit.beta, beta_2sls, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.xi, y - X1 @ beta_2sls, atol=1e-12)
    np.testing.assert_allclose(fit.M @ y, fit.xi, atol=1e-12)
    # X1 lies in the span of Z, so the X1 rows of Z'xi are zero at the 2SLS beta
    g = IVMoments(Z).evaluate(None, fit, False).value  # type: ignore[arg-type]
    np.testing.assert_allclose(g[:3], 0.0, atol=1e-12)
    assert np.max(np.abs(g[3:])) > 1e-3
    print(f"2SLS beta {beta_2sls}; LinearIV beta {fit.beta}")


def test_iv_moment_d_delta_is_ZtM_over_n() -> None:
    d = make_data()
    linear = LinearIV(d["X1"], d["Z"])
    iv = IVMoments(d["Z"])
    delta = d["delta_true"]
    blk = iv.evaluate(None, linear.fit(delta), True)  # type: ignore[arg-type]
    assert blk.d_theta is None and blk.d_delta is not None
    fd = central_fd(lambda dl: iv.evaluate(None, linear.fit(dl), False).value, delta)  # type: ignore[arg-type]
    np.testing.assert_allclose(blk.d_delta, fd, atol=1e-10)


def test_iv_clustered_covariance_by_explicit_loop() -> None:
    d = make_data()
    fit = LinearIV(d["X1"], d["Z"]).fit(d["delta_true"])
    iv = IVMoments(d["Z"], d["clusters"])
    S = iv.covariance_block(None, fit)  # type: ignore[arg-type]
    n = d["Z"].shape[0]
    S_loop = np.zeros_like(S)
    for c in np.unique(d["clusters"]):
        rows = d["clusters"] == c
        v = d["Z"][rows].T @ fit.xi[rows]
        S_loop += np.outer(v, v)
    np.testing.assert_allclose(S, S_loop / n, rtol=1e-12)
    S_robust = IVMoments(d["Z"]).covariance_block(None, fit)  # type: ignore[arg-type]
    u = d["Z"] * fit.xi[:, None]
    np.testing.assert_allclose(S_robust, u.T @ u / n, rtol=1e-12)


def test_absorb_matches_dummy_regression() -> None:
    d = make_data()
    groups = d["market_ids"]
    delta = d["delta_true"]
    fit = Absorb(groups).fit(delta)
    dummies = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    resid = delta - dummies @ np.linalg.lstsq(dummies, delta, rcond=None)[0]
    np.testing.assert_allclose(fit.xi, resid, atol=1e-12)
    np.testing.assert_allclose(fit.M @ delta, resid, atol=1e-12)
    assert fit.beta.size == 0 and fit.names == ()


def test_set_weighting_changes_beta_but_not_with_exact_fit() -> None:
    d = make_data()
    linear = LinearIV(d["X1"], d["Z"])
    y = d["X1"] @ np.array([0.3, 0.2, 0.1])  # exactly linear: every weighting returns the same beta
    b1 = linear.fit(y).beta
    linear.set_weighting(np.diag(np.arange(1.0, 6.0)))
    np.testing.assert_allclose(linear.fit(y).beta, b1, atol=1e-12)


# ------------------------------------------------------------------ micro moments


def test_micro_values_match_hand_pooled_ratios() -> None:
    d, model, mom = fixture()
    theta = np.array([0.6, 0.2, 0.5])
    delta = d["delta_true"]
    sol = solution_at(model, delta, theta, mom.requests(), need_jacobian=False)
    num = {"age_mi": 0.0, "fs_mi": 0.0, "mi": 0.0, "mid_in": 0.0, "mid": 0.0}
    for t, mkt in enumerate(model.markets):
        _, mu = model.basis(t, theta)
        s = model.choice_probs(delta[mkt.rows], mu)
        a = d["agents"][t]
        mi = d["products"]["mi"][mkt.rows]
        num["age_mi"] += np.sum(a["w"] * a["age"] * (s @ mi))
        num["fs_mi"] += np.sum(a["w"] * a["fs"] * (s @ mi))
        num["mi"] += np.sum(a["w"] * (s @ mi))
        num["mid_in"] += np.sum(a["w"] * a["mid"] * s.sum(1))
        num["mid"] += np.sum(a["w"] * a["mid"])  # outside value 1 plus inside value 1
    expected = np.array([num["age_mi"] / num["mi"], num["fs_mi"] / num["mi"], num["mid_in"] / num["mid"]])
    np.testing.assert_allclose(mom.predicted(sol), expected, rtol=1e-13)
    blk = mom.evaluate(sol, None, False)  # type: ignore[arg-type]
    np.testing.assert_allclose(blk.value, expected - mom.values, rtol=1e-13)
    np.testing.assert_array_equal(blk.data, mom.values)


def test_ratio_micro_partials_match_finite_differences() -> None:
    """WP4 acceptance: ratio micro moment gradient (partials in delta and theta) against central FD."""
    d, model, mom = fixture()
    theta = np.array([0.6, 0.2, 0.5])
    delta = d["delta_true"] + 0.1
    req = mom.requests()
    blk = mom.evaluate(solution_at(model, delta, theta, req), None, True)  # type: ignore[arg-type]

    def g_delta(dl):
        return mom.evaluate(solution_at(model, dl, theta, req, False), None, False).value  # type: ignore[arg-type]

    def g_theta(th):
        return mom.evaluate(solution_at(model, delta, th, req, False), None, False).value  # type: ignore[arg-type]

    fd_delta = central_fd(g_delta, delta, h=1e-5)
    fd_theta = central_fd(g_theta, theta, h=1e-5)
    err_d = np.max(np.abs(blk.d_delta - fd_delta)) / np.max(np.abs(fd_delta))
    err_t = np.max(np.abs(blk.d_theta - fd_theta)) / np.max(np.abs(fd_theta))
    print(f"micro d_delta rel err {err_d:.2e}, d_theta rel err {err_t:.2e}")
    assert err_d < 1e-8 and err_t < 1e-8


def test_dataset_market_restriction() -> None:
    d = make_data()
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
    mom_all = make_micro()
    mom_all.bind(model.markets, model.n_products, 3)
    mom_one = make_micro(MicroDataset("survey", 500, market_ids=[1]))
    mom_one.bind(model.markets, model.n_products, 3)
    theta = THETA_TRUE
    sol = solution_at(model, d["delta_true"], theta, mom_all.requests())
    assert mom_all.denominator(sol) == pytest.approx(3.0)
    assert mom_one.denominator(sol) == pytest.approx(1.0)
    blk = mom_one.evaluate(sol, None, True)  # type: ignore[arg-type]
    rows_out = np.concatenate([model.markets[0].rows, model.markets[2].rows])
    np.testing.assert_array_equal(blk.d_delta[:, rows_out], 0.0)
    # E[inside | mid] in market 1 alone
    _, mu = model.basis(1, theta)
    s = model.choice_probs(d["delta_true"][model.markets[1].rows], mu)
    a = d["agents"][1]
    assert mom_one.predicted(sol)[2] == pytest.approx(np.sum(a["w"] * a["mid"] * s.sum(1)) / np.sum(a["w"] * a["mid"]))
    with pytest.raises(ValueError, match="covers none"):
        make_micro(MicroDataset("x", 10, market_ids=[99])).bind(model.markets, model.n_products)


def test_requests_and_product_parts() -> None:
    mom = make_micro()
    names = [p.name for p in mom.requests().micro_parts]
    assert names == ["E[age mi]", "E[fs mi]", "E[mid inside]", "E[mi]", "E[mid]"]  # shared denominator once
    cov_names = [p.name for p in mom.covariance_requests().micro_parts]
    assert len(cov_names) == 5 + 15 and len(set(cov_names)) == 20
    pp = product_part(MicroPart("a", ("age",), ("mi",), 0.0), MicroPart("b", ("mid",), (), 1.0))
    assert pp == MicroPart("a*b", ("age", "mid"), ("mi",), 0.0)
    merged = KernelRequests(micro_parts=mom.parts).merge(mom.covariance_requests())
    assert len(merged.micro_parts) == 20
    with pytest.raises(ValueError, match="two definitions"):
        KernelRequests(micro_parts=(MicroPart("E[mi]", (), ("mi",), 0.0),)).merge(
            KernelRequests(micro_parts=(MicroPart("E[mi]", (), ("sw",), 0.0),)))


def test_missing_kernel_outputs_are_named() -> None:
    d, model, mom = fixture()
    sol = InnerSolution(delta=d["delta_true"], D=None, status=EvalStatus.OK, per_market=[], outputs={})
    with pytest.raises(KeyError, match="E\\[age mi\\]"):
        mom.evaluate(sol, None, False)  # type: ignore[arg-type]

"""Tests for nfp_harness.models.blp.MixedLogitModel: the BLP contraction inverts shares, the outputs follow the
moments' contracts, and a problem on it evaluates through the same NFPGMMProblem path as the PCM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.models.blp import MixedLogitModel
from nfp_harness.moments import GroupMicroMoments, GroupMomentRow, IVMoments, MicroDataset, SecondChoiceMoments
from nfp_harness.params import ParameterLayout, ParameterSpec
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.second_choice import first_second_correlations, logit_second_choice_matrix
from nfp_harness.types import KernelRequests, MarketIndex

THETA = np.array([-0.8, 1.2])   # pi on price x income, sigma on x1


def synthetic(J: tuple[int, ...] = (4, 5, 6), H: int = 60, seed: int = 0) -> dict:
    """Markets with X = [const, x1, price], one demographic (income), one taste draw on x1; shares from a known
    delta at THETA."""
    rng = np.random.default_rng(seed)
    rows, X, D, w, nu, delta, start = [], [], [], [], [], [], 0
    for j in J:
        rows.append(np.arange(start, start + j))
        start += j
        X.append(np.column_stack([np.ones(j), rng.normal(size=j), rng.uniform(1, 3, size=j)]))
        D.append(rng.lognormal(size=(H, 1)))
        w.append(np.full(H, 1.0 / H))
        nu.append(rng.normal(size=(H, 1)))
        delta.append(rng.normal(-1.0, 0.5, size=j))
    markets = [MarketIndex(1990 + t, r) for t, r in enumerate(rows)]
    probe = MixedLogitModel(markets, X, D, w, nu, [(2, 0)], [1], [np.full(j, 0.01) for j in J])
    shares = [probe.probabilities_at(d, X[t] @ probe.coefficients(THETA, t)) @ w[t] for t, d in enumerate(delta)]
    probe.close()
    return {"markets": markets, "X": X, "D": D, "w": w, "nu": nu, "shares": shares, "delta": delta, "rows": rows,
            "n": start}


def model_of(d: dict, **kw) -> MixedLogitModel:
    return MixedLogitModel(d["markets"], d["X"], d["D"], d["w"], d["nu"], [(2, 0)], [1], d["shares"], threads=2,
                           **kw)


def test_contraction_inverts_shares_and_warm_starts() -> None:
    d = synthetic()
    m = model_of(d)
    sol = m.solve(THETA, KernelRequests(), need_jacobian=False)
    assert sol.status.value == "ok" and sol.D is None
    np.testing.assert_allclose(sol.delta, np.concatenate(d["delta"]), atol=1e-10)
    np.testing.assert_allclose(sol.outputs["shares"], np.concatenate(d["shares"]), rtol=1e-12)
    assert [P.shape for P in sol.outputs["probabilities"]] == [(r.size, 60) for r in d["rows"]]
    # accepted state: commit keeps the solution as the warm start, the next solve needs one iteration per market
    m.commit(THETA)
    for t in range(len(d["markets"])):
        np.testing.assert_array_equal(m.delta[t], sol.delta[d["rows"][t]])
        assert m.solve_market(THETA, t, m.delta[t])[2] == 1
    m.warm_from_delta(np.zeros(d["n"]))
    assert all(np.all(x == 0.0) for x in m.delta)
    m.close()


def test_second_choice_and_group_sum_outputs() -> None:
    d = synthetic()
    x1 = np.concatenate([X[:, 1] for X in d["X"]])
    m = model_of(d, product_columns={"x1": x1})
    sol = m.solve(THETA, KernelRequests(second_choice=True, second_choice_markets=(1991,), group_sums=("x1",)), False)
    sc = sol.outputs["second_choice"]
    assert sc[0] is None and sc[2] is None
    P, w = sol.outputs["probabilities"][1], d["w"][1]
    np.testing.assert_array_equal(sc[1]["C"], logit_second_choice_matrix(P, w))
    np.testing.assert_array_equal(sc[1]["s"], P @ w)
    gs = sol.outputs["group_sums"][1]
    assert gs["columns"] == ["x1"] and gs["S"].shape == (60, 2) and gs["d_delta"] is None
    np.testing.assert_allclose(gs["S"][:, 0], w * P.sum(axis=0), rtol=1e-14)
    np.testing.assert_allclose(gs["S"][:, 1], w * (d["X"][1][:, 1] @ P), rtol=1e-14)
    with pytest.raises(KeyError):
        m.solve(THETA, KernelRequests(group_sums=("x2",)), False)
    m.close()


def test_one_problem_path_with_aggregate_second_choice_and_group_micro_moments() -> None:
    d = synthetic()
    n = d["n"]
    rng = np.random.default_rng(1)
    X1 = np.column_stack([np.ones(n), np.concatenate([X[:, 1] for X in d["X"]])])
    Z = np.column_stack([X1, rng.normal(size=(n, 2))])
    x1 = np.concatenate([X[:, 1] for X in d["X"]])
    agents = pd.DataFrame({"market_ids": np.repeat([m.market_id for m in d["markets"]], 60),
                           "rich": np.concatenate([(D[:, 0] > 1.0).astype(float) for D in d["D"]]),
                           "poor": np.concatenate([(D[:, 0] <= 1.0).astype(float) for D in d["D"]])})
    agent_rows = [np.arange(60 * t, 60 * (t + 1)) for t in range(len(d["markets"]))]
    model = model_of(d, product_columns={"x1": x1}, agents=agents, agent_rows=agent_rows)
    sc_year = 1991
    r = d["rows"][1]
    x_sc = d["X"][1][:, 1:2]
    s_sc = d["shares"][1]
    corr = first_second_correlations(s_sc, np.full((r.size, r.size), 1.0 / (r.size - 1)) - np.eye(r.size) / (r.size - 1),
                                     x_sc)
    micro = GroupMicroMoments("micro", MicroDataset("survey", 1), [GroupMomentRow("diff", "x1", "rich", "poor")],
                              np.array([0.1]), np.array([0.01]))
    moments = [IVMoments(Z, None), micro,
               SecondChoiceMoments("second_choice", [sc_year], ("x1",), {sc_year: corr}, {sc_year: x_sc},
                                   {sc_year: s_sc})]
    layout = ParameterLayout([ParameterSpec("pi_price_income", "taste", "price", "income", THETA[0], -5.0, 5.0, unit=0.1),
                              ParameterSpec("sigma_x1", "taste", "x1", "nu0", THETA[1], 0.0, 5.0, unit=0.1)])
    prob = NFPGMMProblem(model, moments, LinearIV(X1, Z), layout, max_step=False)
    ev = prob.evaluate(THETA, need_jacobian=False)
    assert ev.ok, ev.status
    assert list(ev.blocks) == ["aggregate", "micro", "second_choice"] and np.all(np.isfinite(ev.g))
    np.testing.assert_allclose(np.asarray(ev.aux["delta"]), np.concatenate(d["delta"]), atol=1e-10)
    # the second-choice block is the logit model's correlation minus the data's
    C = logit_second_choice_matrix(ev.aux["solution"].outputs["probabilities"][1], d["w"][1])
    want = first_second_correlations(s_sc, C, x_sc) - corr
    np.testing.assert_allclose(ev.g[ev.blocks["second_choice"]], want, rtol=1e-10, atol=1e-14)
    model.close()


def test_gmy_blp_objective_through_the_problem_path_equals_the_scripts() -> None:
    """GMY's BLP (gmy_replication/estimate_blp.py) at Table 4's theta: the objective of an NFPGMMProblem on the same
    MixedLogitModel with the harness's GroupMicroMoments and SecondChoiceMoments equals the script's sum of squared
    residuals (the authors' ExjCondH.m micro moments and 30 x the second-choice residuals)."""
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    for p in (str(root / "nevo_replication" / ".vendor"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if not (root / "gmy_replication" / "data" / "GMY_Replication" / "data" / "data_blp.csv").exists():
        pytest.skip("GMY replication package not unpacked")
    name = "test_blp_gmy_estimate_blp"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "gmy_replication" / "estimate_blp.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    blp = sys.modules[name]

    data = blp.gmy_pcm.load_data(n_households=200, seed=blp.SEED, years=(1991, 1999))
    logit = blp.GMYLogit(data, threads=2)
    r = logit.residuals(blp.TABLE4_THETA, commit=True)
    tab = data.micro.table
    prob = logit.problem()
    m = logit.model
    ev = prob.evaluate(blp.TABLE4_THETA, need_jacobian=False)
    assert ev.ok, ev.status
    # q under GMY's weight (D6): the micro block's diag(1 / variance) and 30 per second-choice row, W1 of the problem
    W1 = prob.initial_weighting(blp.TABLE4_THETA)
    q = float(prob.n * ev.g @ W1 @ ev.g)
    k = tab.shape[0]
    assert float(np.sum(ev.g[ev.blocks["micro_gmy"]] ** 2 / tab.variance.to_numpy(float))) == pytest.approx(
        float(r[:k] @ r[:k]), rel=1e-11)
    assert float(np.sum((blp.SC_WEIGHT * ev.g[ev.blocks["second_choice"]]) ** 2)) == pytest.approx(
        float(r[k:] @ r[k:]), rel=1e-11)
    assert q == pytest.approx(float(r @ r), rel=1e-11)
    m.close()


# ---------------------------------------------------------------- analytic Jacobian (refactor step 9, task B)

def central_fd(f, x: np.ndarray, h: np.ndarray | float) -> np.ndarray:
    """(m, P) central differences of f at x with steps h."""
    x = np.asarray(x, dtype=float)
    hs = np.broadcast_to(np.asarray(h, dtype=float), x.shape)
    cols = []
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = hs[k]
        cols.append((np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * hs[k]))
    return np.column_stack(cols)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max over columns of ||a_c - b_c|| / ||b_c|| (columns of b that are zero: absolute)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    nb = np.linalg.norm(b, axis=0)
    return float(np.max(np.linalg.norm(a - b, axis=0) / np.where(nb > 0, nb, 1.0)))


def test_ddelta_dtheta_and_share_partials_match_finite_differences() -> None:
    d = synthetic()
    m = model_of(d, tol=1e-15)
    sol = m.solve(THETA, KernelRequests(), need_jacobian=True)
    assert sol.status.value == "ok"
    D = np.zeros((d["n"], 2))
    for mk, D_t in zip(d["markets"], sol.D):
        D[mk.rows] = D_t

    def delta_at(th: np.ndarray) -> np.ndarray:
        return m.solve(th, KernelRequests(), False).delta

    assert rel_err(D, central_fd(delta_at, THETA, 1e-5)) < 1e-6
    # the partials themselves, at fixed delta
    t = 1
    delta_t = sol.delta[d["rows"][t]]
    ds_dd, ds_dt = m.share_derivatives(THETA, t, m.probabilities(THETA, t, delta_t))
    w = d["w"][t]
    assert rel_err(ds_dd, central_fd(lambda x: m.probabilities(THETA, t, x) @ w, delta_t, 1e-6)) < 1e-8
    assert rel_err(ds_dt, central_fd(lambda th: m.probabilities(th, t, delta_t) @ w, THETA, 1e-6)) < 1e-8
    m.close()


def test_price_derivatives_match_finite_differences() -> None:
    """ds/dp with the price column 2 of X and a linear price coefficient alpha in delta."""
    d = synthetic()
    alpha = -1.5
    m = model_of(d, price_column=2, price_coefficient=alpha)
    t = 2
    delta_t = d["delta"][t]
    X0 = d["X"][t].copy()

    def shares_at_prices(p: np.ndarray) -> np.ndarray:
        m.X[t] = X0.copy()
        m.X[t][:, 2] = p
        return m.probabilities(THETA, t, delta_t + alpha * (p - X0[:, 2])) @ d["w"][t]

    fd = central_fd(shares_at_prices, X0[:, 2], 1e-6)
    m.X[t] = X0
    assert rel_err(m.price_derivatives(THETA, t, m.probabilities(THETA, t, delta_t)), fd) < 1e-8
    sol = m.solve(THETA, KernelRequests(price_derivatives=True), False)
    assert [a.shape for a in sol.outputs["ds_dp"]] == [(r.size, r.size) for r in d["rows"]]
    m.close()


def test_exponential_price_law_derivatives_match_finite_differences() -> None:
    """A lognormal price coefficient -exp(theta_0 + theta_inc income + theta_p nu) on the price column: shares, the
    group sums and the second-choice adjoint keep their analytic theta derivatives through the theta-dependent basis,
    ds/dp uses each agent's coefficient, and the model carries what supply moments bind to."""
    d = synthetic()
    nu2 = [np.column_stack([v, np.random.default_rng(9 + t).normal(size=v.shape[0])]) for t, v in enumerate(d["nu"])]
    x1 = np.concatenate([X[:, 1] for X in d["X"]])
    m = MixedLogitModel(d["markets"], d["X"], d["D"], d["w"], nu2, [], [1], d["shares"], threads=2, price_column=2,
                        exp_column=2, exp_terms=(("D", 0), ("D", 0), ("nu", 1)),
                        exp_offset=[np.full(h, 0.1) for h in (60, 60, 60)], product_columns={"x1": x1})
    theta = np.array([1.1, -0.2, 0.3, 0.4])   # sigma_x1, then log-coefficient terms (two on income: one is a check)
    assert m.P == 4 and m.theta_columns.tolist() == [1, 2, 2, 2]
    t = 1
    delta_t = d["delta"][t]
    a = m.price_coefficients(theta, t)
    np.testing.assert_allclose(a, -np.exp(0.1 + (-0.2 + 0.3) * d["D"][t][:, 0] + 0.4 * nu2[t][:, 1]))
    w = d["w"][t]
    _, ds_dt = m.share_derivatives(theta, t, m.probabilities(theta, t, delta_t))
    assert rel_err(ds_dt, central_fd(lambda th: m.probabilities(th, t, delta_t) @ w, theta, 1e-6)) < 1e-7

    def S(th: np.ndarray) -> np.ndarray:
        return m.group_sums(th, t, m.probabilities(th, t, delta_t), ["x1"], False)["S"].ravel()

    gs = m.group_sums(theta, t, m.probabilities(theta, t, delta_t), ["x1"], True)
    assert rel_err(gs["d_theta"].reshape(-1, 4), central_fd(S, theta, 1e-6)) < 1e-7
    X0 = d["X"][t].copy()

    def shares_at_prices(p: np.ndarray) -> np.ndarray:
        m.X[t] = X0.copy()
        m.X[t][:, 2] = p
        return m.probabilities(theta, t, delta_t) @ w

    fd = central_fd(shares_at_prices, X0[:, 2], 1e-6)
    m.X[t] = X0
    assert rel_err(m.price_derivatives(theta, t, m.probabilities(theta, t, delta_t)), fd) < 1e-8
    sol = m.solve(theta, KernelRequests(price_derivatives=True), True)
    assert sol.status.value == "ok" and len(sol.D) == 3
    np.testing.assert_array_equal(m.kernel_market(t).p, X0[:, 2])
    assert m.spec.share_column == "shares" and m.products["shares"].shape == (d["n"],)
    np.testing.assert_array_equal(sol.outputs["theta"], theta)
    with pytest.raises(ValueError):
        m.basis(t)
    m.close()


def test_group_sum_and_second_choice_partials_match_finite_differences() -> None:
    d = synthetic()
    x1 = np.concatenate([X[:, 1] for X in d["X"]])
    m = model_of(d, product_columns={"x1": x1})
    t = 1
    delta_t = d["delta"][t]
    J = delta_t.size

    def S(th: np.ndarray, dl: np.ndarray) -> np.ndarray:
        return m.group_sums(th, t, m.probabilities(th, t, dl), ["x1"], False)["S"].ravel()

    gs = m.group_sums(THETA, t, m.probabilities(THETA, t, delta_t), ["x1"], True)
    assert rel_err(gs["d_delta"].reshape(-1, J), central_fd(lambda x: S(THETA, x), delta_t, 1e-6)) < 1e-7
    assert rel_err(gs["d_theta"].reshape(-1, 2), central_fd(lambda th: S(th, delta_t), THETA, 1e-6)) < 1e-7

    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(2, J, J)), rng.normal(size=(2, J))
    w = d["w"][t]

    def F(th: np.ndarray, dl: np.ndarray) -> np.ndarray:
        P = m.probabilities(th, t, dl)
        s = P @ w
        N = logit_second_choice_matrix(P, w) * s[:, None]
        return np.einsum("cjk,jk->c", A, N) - B @ s

    od, ot = m.second_choice_adjoint(THETA, t, m.probabilities(THETA, t, delta_t), A, B)
    assert rel_err(od, central_fd(lambda x: F(THETA, x), delta_t, 1e-6)) < 1e-7
    assert rel_err(ot, central_fd(lambda th: F(th, delta_t), THETA, 1e-6)) < 1e-7
    m.close()


def test_problem_jacobian_with_iv_group_micro_and_second_choice_matches_finite_differences() -> None:
    """G of an NFPGMMProblem on the mixed logit (aggregate IV, household micro and second-choice blocks) against
    central differences of g, through D = d delta / d theta."""
    d = synthetic()
    n = d["n"]
    rng = np.random.default_rng(1)
    x1 = np.concatenate([X[:, 1] for X in d["X"]])
    X1 = np.column_stack([np.ones(n), x1])
    Z = np.column_stack([X1, rng.normal(size=(n, 2))])
    agents = pd.DataFrame({"market_ids": np.repeat([mk.market_id for mk in d["markets"]], 60),
                           "rich": np.concatenate([(D[:, 0] > 1.0).astype(float) for D in d["D"]]),
                           "poor": np.concatenate([(D[:, 0] <= 1.0).astype(float) for D in d["D"]])})
    model = model_of(d, product_columns={"x1": x1}, agents=agents,
                     agent_rows=[np.arange(60 * t, 60 * (t + 1)) for t in range(3)], tol=1e-15)
    r = d["rows"][1]
    x_sc, s_sc = d["X"][1][:, 1:2], d["shares"][1]
    corr = first_second_correlations(s_sc, np.full((r.size, r.size), 1.0 / (r.size - 1)) - np.eye(r.size) / (r.size - 1),
                                     x_sc)
    moments = [IVMoments(Z, None),
               GroupMicroMoments("micro", MicroDataset("survey", 1), [GroupMomentRow("diff", "x1", "rich", "poor"),
                                                                      GroupMomentRow("ratio", None, "rich", "poor")],
                                 np.array([0.1, 1.2]), np.array([0.01, 0.01])),
               SecondChoiceMoments("second_choice", [1991], ("x1",), {1991: corr}, {1991: x_sc}, {1991: s_sc})]
    layout = ParameterLayout([ParameterSpec("pi_price_income", "taste", "price", "income", THETA[0], -5.0, 5.0, unit=0.1),
                              ParameterSpec("sigma_x1", "taste", "x1", "nu0", THETA[1], 0.0, 5.0, unit=0.1)])
    prob = NFPGMMProblem(model, moments, LinearIV(X1, Z), layout, max_step=False)
    ev = prob.evaluate(THETA, need_jacobian=True)
    assert ev.ok, ev.status
    fd = central_fd(lambda th: prob.evaluate(th, need_jacobian=False).g, THETA, 1e-5)
    for name, rows in ev.blocks.items():
        assert rel_err(ev.G[rows], fd[rows]) < 1e-6, name
    model.close()


def recovery_data(T: int = 20, J: int = 8, H: int = 100, seed: int = 4) -> dict:
    """Markets whose data shares come from a known theta, beta and xi orthogonal to Z in the sample, so g(true) = 0:
    X = [const, x1, price] with price endogenous, pi on price x income, sigma on x1."""
    rng = np.random.default_rng(seed)
    n = T * J
    x1, z1, z2 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    xi = rng.normal(0.0, 0.3, size=n)
    price = 2.0 + 0.5 * z1 + 0.3 * xi + 0.1 * rng.normal(size=n)
    mkt = np.repeat(np.arange(T), J)
    rival = np.array([x1[mkt == mkt[j]].sum() - x1[j] for j in range(n)])
    Z = np.column_stack([np.ones(n), x1, z1, z2, z1 ** 2, rival, x1 * z1])
    xi = xi - Z @ np.linalg.lstsq(Z, xi, rcond=None)[0]
    X1 = np.column_stack([np.ones(n), x1, price])
    beta = np.array([-1.0, 0.8, -1.2])
    delta = X1 @ beta + xi
    rows = [np.flatnonzero(mkt == t) for t in range(T)]
    X = [X1[r] for r in rows]
    D = [rng.lognormal(size=(H, 1)) for _ in rows]
    nu = [rng.normal(size=(H, 1)) for _ in rows]
    w = [np.full(H, 1.0 / H) for _ in rows]
    markets = [MarketIndex(t, r) for t, r in enumerate(rows)]
    probe = MixedLogitModel(markets, X, D, w, nu, [(2, 0)], [1], [np.full(J, 0.01)] * T)
    shares = [probe.probabilities(THETA, t, delta[r]) @ w[t] for t, r in enumerate(rows)]
    probe.close()
    return {"markets": markets, "X": X, "D": D, "w": w, "nu": nu, "shares": shares, "X1": X1, "Z": Z, "beta": beta}


def test_harness_fit_recovers_a_known_theta() -> None:
    from nfp_harness.optimizer import OptimizerOptions, minimize

    d = recovery_data()
    model = MixedLogitModel(d["markets"], d["X"], d["D"], d["w"], d["nu"], [(2, 0)], [1], d["shares"], threads=4,
                            tol=1e-14)
    layout = ParameterLayout([ParameterSpec("pi_price_income", "taste", None, None, -0.4, -5.0, 5.0, unit=0.1),
                              ParameterSpec("sigma_x1", "taste", None, None, 0.6, 0.0, 5.0, unit=0.1)])
    prob = NFPGMMProblem(model, [IVMoments(d["Z"], None)], LinearIV(d["X1"], d["Z"]), layout, max_step=False)
    start = layout.starts()
    res = minimize(prob, prob.initial_weighting(start), OptimizerOptions(stage="fit", theta0=start,
                                                                          max_iterations=100))
    assert res.convergence.converged, res.convergence
    print(f"recovery: theta {res.theta}, q {res.q:.3e}, {res.convergence.value}")
    np.testing.assert_allclose(res.theta, THETA, atol=1e-5)
    np.testing.assert_allclose(res.evaluation.aux["beta"], d["beta"], atol=1e-5)
    assert res.q < 1e-10
    model.close()


def test_gmy_year_problem_jacobian_matches_finite_differences() -> None:
    """GMY's BLP step-1 problem (household micro and second-choice moments) on 1999 with 200 households: G from the
    analytic Jacobian against central differences of g."""
    blp = _gmy_blp()
    data = blp.gmy_pcm.load_data(n_households=200, seed=blp.SEED, years=(1999,))
    logit = blp.GMYLogit(data, threads=4, tol=1e-14)
    prob = logit.problem()
    theta = blp.TABLE4_THETA
    ev = prob.evaluate(theta, need_jacobian=True)
    assert ev.ok, ev.status
    prob.model.commit(theta)
    h = 1e-5 * np.maximum(1.0, np.abs(theta))
    fd = central_fd(lambda th: prob.evaluate(th, need_jacobian=False).g, theta, h)
    for name, rows in ev.blocks.items():   # relative to the block's Frobenius norm: a column of order 1e-7 holds
        err = float(np.linalg.norm(ev.G[rows] - fd[rows]) / np.linalg.norm(fd[rows]))   # only difference noise
        print(f"GMY 1999 {name}: relative error of G vs central differences {err:.2e}")
        assert err < 1e-6, name
    prob.model.close()


def _gmy_blp():
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    for p in (str(root / "nevo_replication" / ".vendor"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if not (root / "gmy_replication" / "data" / "GMY_Replication" / "data" / "data_blp.csv").exists():
        pytest.skip("GMY replication package not unpacked")
    name = "test_blp_gmy_estimate_blp"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "gmy_replication" / "estimate_blp.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]

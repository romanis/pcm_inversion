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
    with pytest.raises(NotImplementedError):
        m.solve(THETA, KernelRequests(), need_jacobian=True)
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

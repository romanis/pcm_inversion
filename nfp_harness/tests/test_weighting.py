"""Tests for weighting and covariance algebra (WP4; DESIGN 4.5, 2.3, WP4 acceptance)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import MICRO_OUTPUTS, MicroDataset, MicroPart, RatioMicroMoment
from nfp_harness.tests.test_problem import central_fd, make_data, make_layout, make_problem
from nfp_harness.types import EvalStatus, InnerSolution, MarketIndex
from nfp_harness.weighting import (
    CUE,
    Fixed,
    TwoStep,
    clustered_covariance,
    cue_objective,
    initial_weighting,
    stacked_covariance,
    weight_from_covariance,
)

VENDOR = pathlib.Path(__file__).resolve().parents[2] / "nevo_replication" / ".vendor"


# ------------------------------------------------------------------ micro covariance, hand calculation


def test_micro_covariance_hand_calculation_3_agents_2_products() -> None:
    """WP4 acceptance: V_micro on a 3-agent, 2-product toy against an explicit enumeration of the
    (agent, choice) table, including the observations scaling and the n V block."""
    w = np.array([0.2, 0.3, 0.5])
    s = np.array([[0.1, 0.3], [0.2, 0.2], [0.05, 0.6]])  # s_ij, outside = 1 - row sum
    age = np.array([1.0, 0.0, 1.0])
    fs = np.array([2.0, 4.0, 3.0])
    mi = np.array([1.0, 0.0])
    N_obs = 250

    # explicit (agent, choice) table: choices 0 (outside), 1, 2
    P = np.column_stack([w * (1 - s.sum(1)), w[:, None] * s])  # (3, 3) joint probabilities
    mi_full = np.r_[0.0, mi]
    v_age_mi = np.outer(age, mi_full)
    v_fs_mi = np.outer(fs, mi_full)
    v_mi = np.outer(np.ones(3), mi_full)
    vals = [v_age_mi, v_fs_mi, v_mi]
    E = np.array([np.sum(P * v) for v in vals])
    Sigma = np.array([[(np.sum(P * a * b) - np.sum(P * a) * np.sum(P * b)) / N_obs for b in vals] for a in vals])
    # moments E[age|mi] = E0/E2, E[fs|mi] = E1/E2; delta-method gradient by hand
    grad = np.array([[1 / E[2], 0.0, -E[0] / E[2] ** 2], [0.0, 1 / E[2], -E[1] / E[2] ** 2]])
    V_hand = grad @ Sigma @ grad.T

    parts = {"age": MicroPart("E[age mi]", ("age",), ("mi",), 0.0), "fs": MicroPart("E[fs mi]", ("fs",), ("mi",), 0.0),
             "mi": MicroPart("E[mi]", (), ("mi",), 0.0)}
    mom = RatioMicroMoment("micro", MicroDataset("toy", N_obs), [parts["age"], parts["fs"]], [parts["mi"], parts["mi"]],
                           np.array([0.5, 3.0]))
    mom.bind([MarketIndex(0, np.array([0, 1]))], 2, 1)
    agent = {"age": age, "fs": fs}
    product = {"mi": mi}

    def output(p: MicroPart) -> dict:
        h = np.prod([agent[c] for c in p.agent_columns], axis=0) if p.agent_columns else np.ones(3)
        v = np.prod([product[c] for c in p.product_columns], axis=0) if p.product_columns else np.ones(2)
        return {"sum": float((w * h) @ (s @ (v - p.outside_value))), "h_sum": float(w @ h), "w_sum": float(w.sum()),
                "d_delta": None, "d_theta": None}

    req = mom.covariance_requests()
    sol = InnerSolution(delta=np.zeros(2), D=None, status=EvalStatus.OK, per_market=[],
                        outputs={MICRO_OUTPUTS: {p.name: [output(p)] for p in req.micro_parts}})
    np.testing.assert_allclose(mom.predicted(sol), [E[0] / E[2], E[1] / E[2]], rtol=1e-14)
    V = mom.covariance_block(sol, None)  # type: ignore[arg-type]
    print("V_micro hand:\n", V_hand, "\nV_micro harness:\n", V)
    np.testing.assert_allclose(mom.part_covariance(sol), Sigma, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(V, V_hand, rtol=1e-12)
    # stacked S carries n V in the micro block
    n = 7
    S = stacked_covariance(np.eye(3), V, n)
    np.testing.assert_allclose(S[3:, 3:], n * V_hand, rtol=1e-12)
    np.testing.assert_allclose(S[:3, 3:], 0.0)


# ------------------------------------------------------------------ against pyblp 1.3.0


def _pyblp():
    if VENDOR.exists() and str(VENDOR) not in sys.path:
        sys.path.append(str(VENDOR))
    return pytest.importorskip("pyblp")


def test_covariance_matches_pyblp_formula() -> None:
    """WP4 acceptance: on the logit fake, delta, beta, micro values, V_micro and the stacked
    S = blockdiag(S_clustered, n V_micro) equal pyblp 1.3.0's (results.micro_covariances and
    inv(results.updated_W) with W_type='clustered', center_moments=False)."""
    pyblp = _pyblp()
    pyblp.options.verbose = False
    d = make_data()
    theta = np.array([0.6, 0.2, 0.5])
    n = d["shares"].size
    product_data = {"market_ids": d["market_ids"], "shares": d["shares"], "prices": d["products"]["prices"],
                    "x": d["products"]["x"], "mi": d["products"]["mi"], "clustering_ids": d["clusters"],
                    "demand_instruments0": d["Z_excl"][:, 0], "demand_instruments1": d["Z_excl"][:, 1]}
    agent_data: dict[str, list] = {k: [] for k in ["market_ids", "weights", "nodes0", "nodes1", "nodes2", "fs", "age",
                                                   "mid"]}
    for t, a in enumerate(d["agents"]):
        R = a["w"].size
        agent_data["market_ids"] += [t] * R
        agent_data["weights"] += list(a["w"])
        agent_data["nodes0"] += list(a["nu"])
        agent_data["nodes1"] += [0.0] * R
        agent_data["nodes2"] += [0.0] * R
        for c in ("fs", "age", "mid"):
            agent_data[c] += list(a[c])
    agent_arrays = {k: np.asarray(v) for k, v in agent_data.items()}
    problem_py = pyblp.Problem((pyblp.Formulation("1 + x + mi"), pyblp.Formulation("0 + x + mi + prices")),
                               product_data, pyblp.Formulation("0 + fs + age + mid"), agent_arrays)
    ds = pyblp.MicroDataset("survey", observations=500,
                            compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)))

    def part(name, dcol, xcol, v0):
        def values(t, p, a):
            h = np.ones(a.size) if dcol is None else a.demographics[:, dcol]
            v = np.ones(p.size) if xcol is None else p.X2[:, xcol]
            return np.outer(h, np.r_[v0, v])
        return pyblp.MicroPart(name, ds, values)

    mi, age_mi, fs_mi = part("E[mi]", None, 1, 0.0), part("E[age mi]", 1, 1, 0.0), part("E[fs mi]", 0, 1, 0.0)
    mid_in, mid = part("E[mid inside]", 2, None, 0.0), part("E[mid]", 2, None, 1.0)

    def ratio(v):
        return v[0] / v[1]

    def ratio_grad(v):
        return [1 / v[1], -v[0] / v[1] ** 2]

    micro_py = [pyblp.MicroMoment("age|mi", 0.55, [age_mi, mi], ratio, ratio_grad),
                pyblp.MicroMoment("fs|mi", 3.1, [fs_mi, mi], ratio, ratio_grad),
                pyblp.MicroMoment("in|mid", 0.6, [mid_in, mid], ratio, ratio_grad)]
    ZD = problem_py.products.ZD  # pyblp order: [excluded, X1]
    W = np.eye(8)
    W[:5, :5] = np.linalg.inv(ZD.T @ ZD / n)
    res = problem_py.solve(
        sigma=np.diag([theta[0], 0.0, 0.0]), pi=np.array([[0, 0, 0], [theta[1], 0, 0], [0, -theta[2], 0]]),
        optimization=pyblp.Optimization("return"), method="1s", W=W, micro_moments=micro_py,
        center_moments=False, W_type="clustered", iteration=pyblp.Iteration("squarem", {"atol": 1e-14}))
    np.testing.assert_allclose(ZD, np.column_stack([d["Z_excl"], d["X1"]]))

    problem, _, _ = make_problem()
    ev = problem.evaluate(theta)
    np.testing.assert_allclose(ev.aux["delta"], res.delta.ravel(), atol=1e-11)
    np.testing.assert_allclose(ev.aux["beta"], res.beta.ravel(), rtol=1e-10)
    np.testing.assert_allclose(ev.aux["micro_predicted"]["micro"], res.micro_values.ravel(), rtol=1e-12)
    V = problem.micro_covariance(ev)
    rel_V = np.max(np.abs(V - res.micro_covariances)) / np.max(np.abs(V))
    perm = np.r_[3, 4, 0, 1, 2, 5, 6, 7]  # our [X1, excluded] -> pyblp [excluded, X1]
    S = problem.covariance(ev)[np.ix_(perm, perm)]
    S_py = np.linalg.inv(res.updated_W)
    rel_S = np.max(np.abs(S - S_py)) / np.max(np.abs(S))
    print(f"pyblp: V_micro rel diff {rel_V:.2e}; stacked S rel diff {rel_S:.2e}")
    assert rel_V < 1e-10 and rel_S < 1e-10


# ------------------------------------------------------------------ algebra


def test_initial_weighting_and_stacking() -> None:
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 4))
    A = rng.normal(size=(2, 2))
    V = A @ A.T + 0.1 * np.eye(2)
    W1 = initial_weighting(Z, V)
    np.testing.assert_allclose(W1[:4, :4], np.linalg.inv(Z.T @ Z / 40), rtol=1e-10)
    np.testing.assert_allclose(W1[4:, 4:], np.linalg.inv(40 * V), rtol=1e-10)
    np.testing.assert_allclose(W1[:4, 4:], 0.0)
    np.testing.assert_allclose(initial_weighting(Z, None), np.linalg.inv(Z.T @ Z / 40), rtol=1e-10)
    xi = rng.normal(size=40)
    np.testing.assert_allclose(clustered_covariance(Z, xi, None), (Z * xi[:, None]).T @ (Z * xi[:, None]) / 40)


def test_problem_initial_weighting_is_pyblp_W1() -> None:
    problem, _, d = make_problem()
    theta0 = make_layout().starts()
    W1 = problem.initial_weighting(theta0)
    ev = problem.evaluate(theta0)
    np.testing.assert_allclose(W1, initial_weighting(d["Z"], problem.micro_covariance(ev)), rtol=1e-8)


def test_weight_from_covariance_refuses_singular() -> None:
    S = np.diag([1.0, 1e-20])
    with pytest.raises(np.linalg.LinAlgError, match="lambda_min/lambda_max"):
        weight_from_covariance(S)
    np.testing.assert_allclose(weight_from_covariance(np.diag([2.0, 4.0])), np.diag([0.5, 0.25]))


def test_cue_gradient_matches_finite_differences() -> None:
    """CUE terms: exact gradient of q = n g' (V'V/n + S_fixed)^{-1} g against central FD (clustered S, micro held)."""
    problem, _, _ = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    ev = problem.evaluate(theta)
    S_fixed = problem.cue_fixed_covariance(ev)
    V, dV = problem.cue_terms(ev)
    rows = problem.agg_rows
    np.testing.assert_allclose(V.sum(0)[rows] / problem.n, ev.g[rows], atol=1e-14)
    np.testing.assert_allclose(V[:, problem.blocks["micro"]], 0.0)
    q, grad, S = cue_objective(ev.g, ev.G, V, dV, problem.n, S_fixed)
    np.testing.assert_allclose(S, problem.covariance(ev), rtol=1e-10)

    def q_at(th):
        e = problem.evaluate(th)
        Vt, _ = problem.cue_terms(e)
        return np.array([cue_objective(e.g, None, Vt, None, problem.n, S_fixed)[0]])

    fd = central_fd(q_at, theta, h=1e-5).ravel()
    err = np.max(np.abs(grad - fd)) / np.max(np.abs(fd))
    print(f"CUE gradient rel err {err:.2e}")
    assert err < 1e-6

    cue = CUE()
    q2, grad2, _ = cue.objective(problem, ev)
    assert q2 == pytest.approx(q) and np.allclose(grad2, grad)
    with pytest.raises(NotImplementedError):
        CUE(hold_micro_block=False)


def test_two_step_weight_and_gate() -> None:
    problem, _, _ = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    W1 = problem.initial_weighting(theta)
    problem.set_weighting(W1)
    ev = problem.evaluate(theta)
    ts = TwoStep(W1)
    W2 = ts.second_step_weight(problem, ev)
    np.testing.assert_allclose(W2 @ problem.covariance(ev), np.eye(problem.k), atol=1e-8)
    ok, _ = ts.gate(["RUN_TO_BOUND"])
    assert ok
    ok, why = ts.gate(["SD_XI_BLOWUP", "RUN_TO_BOUND"])
    assert not ok and "SD_XI_BLOWUP" in why
    problem.set_weighting(W2)  # S is block diagonal, so W2 is accepted
    assert Fixed(W2).mode == "fixed" and ts.mode == "two_step" and CUE().mode == "cue"


def test_two_step_on_linear_iv_equals_efficient_gmm() -> None:
    """With delta fixed by the data (no nonlinear dependence), beta with W2 = S^{-1} is the efficient GMM beta."""
    d = make_data()
    rng = np.random.default_rng(4)
    X1, Z = d["X1"], d["Z"]
    n = X1.shape[0]
    y = X1 @ np.array([1.0, 0.5, -0.2]) + rng.normal(size=n) * (1 + np.abs(d["products"]["x"]))
    linear = LinearIV(X1, Z)
    xi1 = linear.fit(y).xi
    S = clustered_covariance(Z, xi1, d["clusters"])
    W2 = weight_from_covariance(S)
    linear.set_weighting(W2)
    beta2 = linear.fit(y).beta
    beta_eff = np.linalg.solve(X1.T @ Z @ W2 @ Z.T @ X1, X1.T @ Z @ W2 @ Z.T @ y)
    np.testing.assert_allclose(beta2, beta_eff, rtol=1e-10)

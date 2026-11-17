"""Tests for the log-cost supply moments (``moments.SupplyMoments``) and their assembly in ``NFPGMMProblem``.

Synthetic PCM markets from the WP7 generator (``test_pcm_model.make_synthetic``), with prices shifted up so
that the Nash costs implied by the model's own ds/dp are positive. The supply Jacobian, taken by central
differences at fixed inversions, is checked against central differences of g with a full re-inversion at
every step (both with kink-free steps), for block-diagonal W and for a W that couples the aggregate and
supply rows (joint concentration of beta and gamma).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import IVMoments, SupplyMoments
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.types import EvalStatus
from nfp_harness.weighting import weight_from_covariance

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, central_fd, make_synthetic, synthetic_layout  # noqa: E402


def _supply_inputs(syn: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pr = syn["data"]["products"]
    n = pr["prices"].size
    w = np.column_stack([np.ones(n), pr["x1"], pr["x2"]])
    mk = pr["market_ids"]
    rival = np.array([pr["x2"][mk == m].sum() for m in mk]) - pr["x2"]
    return np.column_stack([w, rival, pr["x1"] ** 2, pr["x1"] * pr["x2"]]), w


def _problem(syn: dict[str, Any], firms: np.ndarray) -> NFPGMMProblem:
    syn["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
    Zs, w = _supply_inputs(syn)
    return NFPGMMProblem(model, [IVMoments(syn["Z"]), SupplyMoments(Zs, w, firms, w_names=("c0", "c1", "c2"))],
                         LinearIV(syn["X1"], syn["Z"]), layout)


@pytest.fixture(scope="module")
def priced() -> dict[str, Any]:
    """2 markets of 8 candidates, 256 draws, prices + 12 and delta + 2: all Nash costs positive."""
    return make_synthetic(2, 8, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-12))


def test_supply_jacobian_matches_reinverted_finite_differences(priced: dict[str, Any]) -> None:
    n = priced["data"]["products"]["prices"].size
    prob = _problem(priced, np.arange(n) // 2)
    try:
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        sup = ev.aux["supply"]
        assert sup["share_mc_nonpositive"] == 0.0 and sup["nash_rcond_min"] > 1e-12
        assert 0.0 < sup["margin_median"] < 1.0
        prob.commit(THETA_TRUE)
        fd, _ = central_fd(prob, THETA_TRUE, ev.aux["delta"])
        rows = prob.blocks["supply"]
        errs = [_rel(ev.G[rows, k], fd[rows, k]) for k in range(THETA_TRUE.size)]
        print("supply block, block-diagonal W:", [f"{e:.1e}" for e in errs])
        assert max(errs) < 1e-5, errs

        # step-2 W with the aggregate-supply cross terms of S: (beta, gamma) concentrated jointly
        S = prob.covariance(ev)
        assert np.abs(S[prob.agg_rows, rows]).max() > 0
        prob.set_weighting(weight_from_covariance(S))
        assert prob._joint is not None
        ev2 = prob.evaluate(THETA_TRUE)
        assert ev2.ok
        prob.commit(THETA_TRUE)
        fd2, _ = central_fd(prob, THETA_TRUE, ev2.aux["delta"])
        errs2 = [_rel(ev2.G[:, k], fd2[:, k]) for k in range(THETA_TRUE.size)]
        print("all rows, coupled W:", [f"{e:.1e}" for e in errs2])
        assert max(errs2) < 1e-4, errs2
        # the joint concentration minimises q over (beta, gamma): perturbing them raises q
        J, names = prob.full_jacobian(ev2)
        assert J.shape == (prob.k, 3 + 3 + THETA_TRUE.size) and names[3:6] == ["c0", "c1", "c2"]
        gl = J[:, :6].T @ prob.W @ ev2.g
        assert np.max(np.abs(gl)) < 1e-10 * max(1.0, float(np.max(np.abs(ev2.g))))
    finally:
        prob.model.close()


def test_initial_weighting_has_the_supply_block(priced: dict[str, Any]) -> None:
    n = priced["data"]["products"]["prices"].size
    prob = _problem(priced, np.arange(n))
    try:
        W1 = prob.initial_weighting(THETA_TRUE)
        Zs = prob.supply.Z
        s = prob.blocks["supply"]
        assert np.allclose(W1[s, s], np.linalg.inv(Zs.T @ Zs / n))
        assert np.all(W1[prob.agg_rows, s] == 0.0)
        assert prob.identification["k_linear"] == 3 + 3
    finally:
        prob.model.close()


def test_nonpositive_marginal_cost_is_a_named_numerical_status() -> None:
    syn = make_synthetic(2, 8, 32, 8, seed=2)   # unshifted prices: one firm per market prices below its markup
    prob = _problem(syn, np.zeros(syn["data"]["products"]["prices"].size, dtype=int))
    try:
        ev = prob.evaluate(THETA_TRUE)
        assert ev.status is EvalStatus.NUMERICAL
        assert "mc <= 0" in ev.aux["note"]
        assert ev.aux["supply"]["share_mc_nonpositive"] > 0.0
    finally:
        prob.model.close()


def test_mc_floor_clips_instead_of_failing() -> None:
    """With mc_floor set, a markup above price gives mc = floor (reported), not SupplyDomainError."""
    import numpy as np
    from nfp_harness.moments import SupplyMoments

    n = 6
    rng = np.random.default_rng(0)
    w = np.column_stack([np.ones(n), rng.normal(size=n)])
    Z = np.column_stack([w, rng.normal(size=(n, 2))])
    sm = SupplyMoments(Z, w, np.arange(n), mc_floor=0.5)
    s = np.full(3, 0.1)
    p = np.array([1.0, 2.0, 3.0])
    ds_dp = -np.diag([0.05, 0.3, 0.3])   # product 0: markup s/|ds/dp| = 2 > price 1
    mc, nash = sm._market_costs(0, s, p, np.arange(3), ds_dp)
    assert mc[0] == 0.5 and bool(nash["clipped"][0]) and not nash["clipped"][1:].any()
    assert np.all(mc > 0)


def test_margin_row_value_jacobian_and_weights(priced: dict[str, Any]) -> None:
    """MarginMoment: the trimmed mean of (p - mc)/p from the supply block's markups, its theta derivative
    against re-inverted central differences, the declared W1 entry and the binding entry."""
    from nfp_harness.moments import MarginMoment

    n = priced["data"]["products"]["prices"].size
    firms = np.arange(n) // 2
    priced["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), priced["data"], layout, priced["nu"], priced["seed"], total_threads=8)
    Zs, w = _supply_inputs(priced)
    sup = SupplyMoments(Zs, w, firms, w_names=("c0", "c1", "c2"))
    mm = MarginMoment(sup, 0.2, sd=0.01, binding_weight=1e4)
    prob = NFPGMMProblem(model, [IVMoments(priced["Z"]), sup, mm], LinearIV(priced["X1"], priced["Z"]), layout)
    try:
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        mc = np.exp(ev.aux["log_mc"])
        p = np.empty(n)
        for t, m in enumerate(model.markets):
            p[np.asarray(m.rows)] = np.asarray(model.kernel_market(t).p)
        margin = np.sort(1.0 - mc / p)
        want = float(np.mean(margin[int(np.floor(0.25 * n)):int(np.ceil(0.75 * n))]))
        row = prob.blocks["margin"]
        assert ev.g[row][0] == pytest.approx(want - 0.2, abs=1e-14)
        assert ev.aux["margin"]["margin_trimmed_mean"] == pytest.approx(want, abs=1e-14)
        prob.commit(THETA_TRUE)
        fd, _ = central_fd(prob, THETA_TRUE, ev.aux["delta"])
        errs = [_rel(ev.G[row, k], fd[row, k]) for k in range(THETA_TRUE.size)]
        print("margin row:", [f"{e:.1e}" for e in errs])
        assert max(errs) < 1e-4, errs
        W1 = prob.initial_weighting(THETA_TRUE)
        assert W1[row, row][0, 0] == pytest.approx(1.0 / (n * 1e-4))
        assert mm.binding_block() == pytest.approx(1e4 / (n * 1e-4))
        assert prob.identification["k_micro"] == 1
    finally:
        model.close()


def test_supply_unit_step_rule(priced: dict[str, Any]) -> None:
    """fd_scale='unit': h = fd_step x natural unit, no shrink; the frontier-change count is reported."""
    n = priced["data"]["products"]["prices"].size
    firms = np.arange(n) // 2
    priced["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), priced["data"], layout, priced["nu"], priced["seed"], total_threads=8)
    Zs, w = _supply_inputs(priced)
    with pytest.raises(ValueError, match="natural units"):
        SupplyMoments(Zs, w, firms, fd_scale="unit").bind(model, THETA_TRUE.size)
    sup = SupplyMoments(Zs, w, firms, fd_step=1e-3, fd_scale="unit")
    prob = NFPGMMProblem(model, [IVMoments(priced["Z"]), sup], LinearIV(priced["X1"], priced["Z"]), layout)
    try:
        assert np.array_equal(sup.units, layout.units())
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok and np.all(np.isfinite(ev.G))
        assert ev.aux["supply"]["fd_step_shrinks"] == 0 and ev.aux["supply"]["fd_kink_columns"] >= 0
    finally:
        model.close()

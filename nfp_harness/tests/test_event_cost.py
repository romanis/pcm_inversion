"""Tests for ``moments.EventCostContinuityMoments``: the difference in differences of Nash log marginal cost across an
ownership change. Synthetic PCM markets as in ``test_supply`` (market 0 the period before, market 1 the period
after, products paired by position); the ownership of the treated pairs changes between the markets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import CostEvent, EventCostContinuityMoments, IVMoments, SupplyMoments
from nfp_harness.problem import NFPGMMProblem

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, central_fd, make_synthetic, synthetic_layout  # noqa: E402
from nfp_harness.tests.test_supply import _supply_inputs  # noqa: E402


@pytest.fixture(scope="module")
def priced() -> dict[str, Any]:
    return make_synthetic(2, 8, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)


def _pairs(syn: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mk = np.asarray(syn["data"]["products"]["market_ids"])
    r0, r1 = np.flatnonzero(mk == np.unique(mk)[0]), np.flatnonzero(mk == np.unique(mk)[1])
    J = min(r0.size, r1.size)
    return r1[:J], r0[:J]


def _event(syn: dict[str, Any], match: bool = False) -> tuple[CostEvent, np.ndarray]:
    cur, prev = _pairs(syn)
    n = np.asarray(syn["data"]["products"]["prices"]).size
    firms = np.arange(n)
    firms[cur[:2]] = firms[cur[0]]      # after the change products 0 and 1 share an owner
    t, c = slice(0, 2), slice(2, None)
    A = None
    if match:
        nc = cur[c].size
        A = np.zeros((2, nc))
        A[0, 0] = 1.0
        A[1, 1:] = 1.0 / (nc - 1)
    return CostEvent("merger", (cur[t], prev[t]), (cur[c], prev[c]), A), firms


def test_event_block_value_variance_and_jacobian(priced: dict[str, Any]) -> None:
    ev_def, firms = _event(priced)
    priced["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), priced["data"], layout, priced["nu"], priced["seed"], total_threads=8)
    Zs, w = _supply_inputs(priced)
    sup = SupplyMoments(Zs, w, firms, w_names=("c0", "c1", "c2"))
    blk = EventCostContinuityMoments(sup, [ev_def])
    prob = NFPGMMProblem(model, [IVMoments(priced["Z"]), sup, blk], LinearIV(priced["X1"], priced["Z"]), layout)
    try:
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        y = ev.aux["log_mc"]
        (tc, tp), (cc, cp) = ev_def.treated, ev_def.control
        g = np.mean(y[tc] - y[tp]) - np.mean(y[cc] - y[cp])
        assert np.isclose(ev.g[prob.blocks["event_cost"]][0], g, rtol=0, atol=1e-13)
        v = np.var(y[tc] - y[tp], ddof=1) / tc.size + np.var(y[cc] - y[cp], ddof=1) / cc.size
        assert np.isclose(blk.variances(y)[0], v)
        row = blk.summary(y)[0]
        assert row["treated"] == 2 and np.isclose(row["t"], g / np.sqrt(v))
        S = prob.covariance(ev)
        r = prob.blocks["event_cost"]
        assert np.isclose(S[r, r][0, 0], prob.n * v)
        W1 = prob.initial_weighting(THETA_TRUE)
        assert np.isclose(prob.n * g * W1[r, r][0, 0] * g, g * g / v)   # q of the block = t^2
        prob.commit(THETA_TRUE)
        fd, _ = central_fd(prob, THETA_TRUE, ev.aux["delta"])
        err = np.max(np.abs(ev.G[r] - fd[r])) / max(float(np.max(np.abs(fd[r]))), 1e-12)
        assert err < 1e-5, err
    finally:
        prob.model.close()


def test_event_block_without_a_supply_block_and_matched_controls(priced: dict[str, Any]) -> None:
    ev_def, firms = _event(priced, match=True)
    priced["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), priced["data"], layout, priced["nu"], priced["seed"], total_threads=8)
    Zs, w = _supply_inputs(priced)
    costs = SupplyMoments(Zs, w, firms)   # not a block of the problem: the event block binds it
    blk = EventCostContinuityMoments(costs, [ev_def])
    prob = NFPGMMProblem(model, [IVMoments(priced["Z"]), blk], LinearIV(priced["X1"], priced["Z"]), layout)
    try:
        assert costs.model is model
        ev = prob.evaluate(THETA_TRUE, need_jacobian=False)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        y, _, _ = costs.log_costs(ev.aux["solution"], False)
        (tc, tp), (cc, cp) = ev_def.treated, ev_def.control
        dt, dc = y[tc] - y[tp], y[cc] - y[cp]
        g = 0.5 * ((dt[0] - dc[0]) + (dt[1] - dc[1:].mean()))
        assert np.isclose(ev.g[prob.blocks["event_cost"]][0], g, rtol=0, atol=1e-13)
    finally:
        prob.model.close()


def test_event_definitions_are_checked() -> None:
    sm = SupplyMoments(np.column_stack([np.ones(6), np.arange(6.0)]), np.ones((6, 1)), np.arange(6))
    with pytest.raises(ValueError, match="both treated and control"):
        EventCostContinuityMoments(sm, [CostEvent("x", (np.array([3]), np.array([0])), (np.array([3]), np.array([1])))])
    with pytest.raises(ValueError, match="rows summing to one"):
        EventCostContinuityMoments(sm, [CostEvent("x", (np.array([3]), np.array([0])),
                                                  (np.array([4, 5]), np.array([1, 2])), np.array([[0.5, 0.2]]))])
    with pytest.raises(ValueError, match="no events"):
        EventCostContinuityMoments(sm, [])

"""Tests for the joint product-level weighting of ``NFPGMMProblem.set_joint``: one clustered covariance over the
aggregate, first-difference, supply and ownership-event blocks, (beta, gamma) concentrated jointly in one GLS step, and
CUE with ``W(theta) = S(theta)^-1``. Synthetic PCM markets as in ``test_event_cost`` (market 0 the period before,
market 1 the period after, products paired by position, a pair one cluster)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.instruments import difference_instruments
from nfp_harness.linear import LinearIV
from nfp_harness.moments import CostEvent, DifferenceIVMoments, EventCostContinuityMoments, IVMoments, SupplyMoments
from nfp_harness.optimizer import cue_gradient, cue_objective
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.weighting import weight_from_covariance

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, central_fd, make_synthetic, synthetic_layout  # noqa: E402
from nfp_harness.tests.test_supply import _supply_inputs  # noqa: E402


@pytest.fixture(scope="module")
def priced() -> dict[str, Any]:
    return make_synthetic(2, 60, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)


def _pairs(syn: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mk = np.asarray(syn["data"]["products"]["market_ids"])
    r0, r1 = np.flatnonzero(mk == np.unique(mk)[0]), np.flatnonzero(mk == np.unique(mk)[1])
    J = min(r0.size, r1.size)
    return r1[:J], r0[:J]


def _clusters(syn: dict[str, Any]) -> np.ndarray:
    """A pair (the same position in both markets) is one cluster; an unpaired row its own."""
    cur, prev = _pairs(syn)
    n = np.asarray(syn["data"]["products"]["prices"]).size
    ids = np.arange(n) + 10_000
    ids[cur] = np.arange(cur.size)
    ids[prev] = np.arange(cur.size)
    return ids


def _build(syn: dict[str, Any], *, fd: bool = True, single: bool = False, scale: dict[str, float] | None = None,
           event: bool = True) -> NFPGMMProblem:
    """Aggregate (clustered), [demand first differences], supply levels, [event] blocks. ``single`` adds to X1 and Z a
    dummy carried by one cluster (a make with one model); ``scale`` multiplies a block's instruments (aggregate,
    demand_fd, supply) or the event contrast by a constant."""
    scale = scale or {}
    pr = syn["data"]["products"]
    n = pr["prices"].size
    cur, prev = _pairs(syn)
    firms = np.arange(n)
    firms[cur[:2]] = firms[cur[0]]      # after the change products 0 and 1 share an owner
    pr["firm_ids"] = firms
    ids = _clusters(syn)
    X1, Z = syn["X1"], syn["Z"][:, :6]
    if single:
        d = (ids == ids[cur[3]]).astype(float)
        X1, Z = np.column_stack([X1, d]), np.column_stack([Z, d])
    Z = Z * scale.get("aggregate", 1.0)
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
    Zs, w = _supply_inputs(syn)
    sup = SupplyMoments(Zs * scale.get("supply", 1.0), w, firms, ids, w_names=("c0", "c1", "c2"))
    ms: list[Any] = [IVMoments(Z, ids), sup]
    if fd:
        Zd = difference_instruments(cur, prev, np.column_stack([pr["x1"] ** 2, np.sin(pr["x2"])]))
        ms.insert(1, DifferenceIVMoments(Zd * scale.get("demand_fd", 1.0), ids))
    if event:
        t, c = slice(0, 2), slice(2, None)
        blk = EventCostContinuityMoments(sup, [CostEvent("merger", (cur[t], prev[t]), (cur[c], prev[c]))])
        blk.L = blk.L * scale.get("event_cost", 1.0)
        ms.append(blk)
    return NFPGMMProblem(model, ms, LinearIV(X1, Z), layout)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-12))


def _cue_q(prob: NFPGMMProblem, ev: Any) -> float:
    return cue_objective(ev, prob.covariance(ev), prob.n)


def test_joint_step_with_block_diagonal_W_is_the_separate_concentration(priced: dict[str, Any]) -> None:
    sep = _build(priced, fd=False)
    jnt = _build(priced, fd=False)
    try:
        W = sep.initial_weighting(THETA_TRUE)
        sep.set_weighting(W)
        jnt.set_weighting(W)
        jnt.set_joint("fixed")
        assert jnt._jointcfg is not None and jnt._jointcfg["K0"].size == 0
        e1, e2 = sep.evaluate(THETA_TRUE), jnt.evaluate(THETA_TRUE)
        assert e1.ok and e2.ok, (e1.status, e2.status, e2.aux.get("note"))
        assert np.allclose(e2.aux["beta"], e1.aux["beta"], rtol=1e-10, atol=1e-12)
        assert np.allclose(e2.aux["gamma"], e1.aux["gamma"], rtol=1e-10, atol=1e-12)
        assert np.allclose(e2.g, e1.g, rtol=1e-9, atol=1e-10)
        assert np.allclose(e2.G, e1.G, rtol=1e-8, atol=1e-9)
        assert np.isclose(e2.aux["q"], e1.aux["q"], rtol=1e-9)
        assert np.isclose(sum(e2.aux["q_blocks"].values()), e2.aux["q"], rtol=1e-12)
        # S_P: the separate covariance blocks on the diagonal, cross terms between every pair of product blocks
        S1, S2 = sep.covariance(e1), jnt.covariance(e2)
        a, s = jnt.agg_rows, jnt.blocks["supply"]
        assert np.allclose(S2[a, a], S1[a, a], rtol=1e-10) and np.allclose(S2[s, s], S1[s, s], rtol=1e-10)
        assert np.abs(S2[a, s]).max() > 0 and np.abs(S2[jnt.blocks["event_cost"], a]).max() > 0
        r = jnt.blocks["event_cost"]   # the event's clustered variance against covariance_block's n Var(g_e)
        assert 0.5 < S2[r, r][0, 0] / S1[r, r][0, 0] < 1.5
    finally:
        sep.model.close()
        jnt.model.close()


def test_joint_fixed_W_jacobian_matches_finite_differences(priced: dict[str, Any]) -> None:
    prob = _build(priced, single=True)
    try:
        prob.set_joint("fixed")
        ev0 = prob.evaluate(THETA_TRUE)
        assert ev0.ok, (ev0.status, ev0.aux.get("note"))
        j = prob._jointcfg
        assert j is not None and j["K0"].size == 1
        prob.set_weighting(weight_from_covariance(prob.covariance(ev0)))   # a joint step-2 W, frozen
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok
        k0 = j["P_idx"][j["K0"]]
        assert np.all(ev.g[k0] == 0.0)   # a single-cluster row is an exact constraint
        J, names = prob.full_jacobian(ev)   # the GLS step minimises q over (beta, gamma) under W, given the constraint
        nb = len(names) - prob.P
        lam = np.linalg.lstsq(J[k0, :nb].T, J[:, :nb].T @ prob._joint_W(ev.aux["joint"]) @ ev.g, rcond=None)[0]
        resid = J[:, :nb].T @ prob._joint_W(ev.aux["joint"]) @ ev.g - J[k0, :nb].T @ lam
        assert np.max(np.abs(resid)) < 1e-9 * max(1.0, float(np.max(np.abs(ev.g))))
        prob.commit(THETA_TRUE)
        fd, _ = central_fd(prob, THETA_TRUE, ev.aux["delta"])
        errs = [_rel(ev.G[:, k], fd[:, k]) for k in range(THETA_TRUE.size)]
        assert max(errs) < 1e-5, errs
    finally:
        prob.model.close()


def test_joint_cue_gradient_matches_finite_differences(priced: dict[str, Any]) -> None:
    prob = _build(priced, single=True)
    try:
        prob.set_joint("cue")
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        S = prob.covariance(ev)
        V, dV = prob.cue_terms(ev)
        j = prob._jointcfg
        assert j is not None
        S_P = (V.T @ V / prob.n + prob.cue_fixed_covariance(ev))
        assert np.allclose(S_P, S, rtol=1e-12, atol=1e-14)   # S = V'V/n + the held part, as cue_gradient assumes
        q = _cue_q(prob, ev)
        assert np.isclose(q, ev.aux["q"], rtol=1e-6)   # two inverses of an S estimated from 23 clusters
        assert np.isclose(sum(ev.aux["q_blocks"].values()), ev.aux["q"], rtol=1e-12)
        grad = cue_gradient(ev, S, V, dV, prob.n)
        prob.commit(THETA_TRUE)
        model = prob.model
        sig0 = model.frontier_signatures(THETA_TRUE, ev.aux["delta"])
        errs = []
        for k in range(THETA_TRUE.size):
            h = 1e-5 * max(1.0, abs(THETA_TRUE[k]))
            for _ in range(5):
                qs, same = [], True
                for sgn in (1.0, -1.0):
                    th = THETA_TRUE.copy()
                    th[k] += sgn * h
                    e = prob.evaluate(th, need_jacobian=False)
                    prob.restore(th)
                    assert e.ok
                    qs.append(_cue_q(prob, e))
                    sig = model.frontier_signatures(th, e.aux["delta"])
                    same &= all(np.array_equal(a, b) for a, b in zip(sig, sig0))
                if same:
                    break
                h /= 10.0
            errs.append(abs(grad[k] - (qs[0] - qs[1]) / (2 * h)) / max(abs(grad[k]), 1e-6 * max(1.0, q)))
        assert max(errs) < 1e-4, (errs, grad)
    finally:
        prob.model.close()


def test_joint_cue_is_invariant_to_the_scale_of_each_block(priced: dict[str, Any]) -> None:
    base = _build(priced, single=True)
    try:
        base.set_joint("cue")
        e0 = base.evaluate(THETA_TRUE, need_jacobian=False)
        assert e0.ok
        q0 = _cue_q(base, e0)
    finally:
        base.model.close()
    for block, c in (("aggregate", 3.0), ("demand_fd", 0.2), ("supply", 7.0), ("event_cost", 5.0)):
        prob = _build(priced, single=True, scale={block: c})
        try:
            prob.set_joint("cue")
            e = prob.evaluate(THETA_TRUE, need_jacobian=False)
            assert e.ok
            assert np.isclose(_cue_q(prob, e), q0, rtol=1e-8), (block, _cue_q(prob, e), q0)
        finally:
            prob.model.close()

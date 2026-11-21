"""Tests for several supply blocks in one ``NFPGMMProblem``: one log mc, one gamma over the union of the blocks' named
cost columns, each block with its own 2SLS weight (``problem._multi_projection``). Synthetic PCM markets as in
``test_supply``: a levels block on [1, x1, x2] and a second block on the cost column x1 alone (shared with the first)
with its own instruments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import IVMoments, SupplyMoments
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.weighting import weight_from_covariance

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, central_fd, make_synthetic, synthetic_layout  # noqa: E402
from nfp_harness.tests.test_supply import _supply_inputs  # noqa: E402


@pytest.fixture(scope="module")
def priced() -> dict[str, Any]:
    return make_synthetic(2, 8, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)


def _blocks(syn: dict[str, Any], firms: np.ndarray) -> tuple[SupplyMoments, SupplyMoments]:
    pr = syn["data"]["products"]
    Zs, w = _supply_inputs(syn)
    lev = SupplyMoments(Zs, w, firms, w_names=("c0", "c1", "c2"), name="supply")
    Z2 = np.column_stack([pr["x1"], pr["x1"] ** 3, np.sin(3.0 * pr["x2"])])
    second = SupplyMoments(Z2, w[:, [1]], firms, w_names=("c1",), name="supply_b")
    return lev, second


def _problem(syn: dict[str, Any], firms: np.ndarray, two: bool = True) -> NFPGMMProblem:
    syn["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
    lev, second = _blocks(syn, firms)
    ms = [IVMoments(syn["Z"]), lev] + ([second] if two else [])
    return NFPGMMProblem(model, ms, LinearIV(syn["X1"], syn["Z"]), layout)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-12))


def test_two_supply_blocks_share_gamma_and_match_finite_differences(priced: dict[str, Any]) -> None:
    n = priced["data"]["products"]["prices"].size
    prob = _problem(priced, np.arange(n) // 2)
    try:
        assert prob.gamma_names == ("c0", "c1", "c2") and prob.K_supply == 3
        assert prob.identification["k_linear"] == 3 + 3
        assert prob.identification["rank_supply"] == 6 + 3
        assert prob.coupled_blocks == ((prob.agg.name, "supply", "supply_b"),)
        ev = prob.evaluate(THETA_TRUE)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        # gamma by hand: stacked 2SLS over both blocks with blockdiag((Z_b'Z_b/n)^-1)
        lev, second = prob.supplies
        y = ev.aux["log_mc"]
        X = np.zeros((lev.k + second.k, 3))
        X[:lev.k] = lev.Z.T @ lev.w / n
        X[lev.k:, [1]] = second.Z.T @ second.w / n
        m = np.r_[lev.Z.T @ y / n, second.Z.T @ y / n]
        Wr = np.zeros((X.shape[0], X.shape[0]))
        Wr[:lev.k, :lev.k] = np.linalg.inv(lev.Z.T @ lev.Z / n)
        Wr[lev.k:, lev.k:] = np.linalg.inv(second.Z.T @ second.Z / n)
        gamma = np.linalg.solve(X.T @ Wr @ X, X.T @ Wr @ m)
        assert np.allclose(ev.aux["gamma"], gamma, rtol=1e-10, atol=1e-12)
        assert np.allclose(ev.g[prob.blocks["supply"]], lev.Z.T @ (y - lev.w @ gamma) / n, atol=1e-12)
        assert np.allclose(ev.g[prob.blocks["supply_b"]], second.Z.T @ (y - second.w @ gamma[[1]]) / n, atol=1e-12)
        assert np.allclose(ev.aux["omega_blocks"]["supply_b"], y - gamma[1] * second.w[:, 0])
        # one log mc: the same as a single-block problem's
        single = _problem(priced, np.arange(n) // 2, two=False)
        try:
            ev1 = single.evaluate(THETA_TRUE, need_jacobian=False)
            assert np.array_equal(ev1.aux["log_mc"], y)
            assert single._multi is None and single.coupled_blocks == ((single.agg.name, "supply"),)
        finally:
            single.model.close()

        prob.commit(THETA_TRUE)
        fd, _ = central_fd(prob, THETA_TRUE, ev.aux["delta"])
        errs = [_rel(ev.G[:, k], fd[:, k]) for k in range(THETA_TRUE.size)]
        assert max(errs) < 1e-5, errs

        # step 2 with cross terms between the aggregate and both supply blocks: (beta, gamma) concentrated jointly
        S = prob.covariance(ev)
        assert np.abs(S[prob.blocks["supply"], prob.blocks["supply_b"]]).max() > 0
        assert np.abs(S[prob.agg_rows, prob.blocks["supply_b"]]).max() > 0
        # 16 products carry fewer clusters than moments, so S is singular: a ridge keeps W positive definite
        ridge = 1e-3 * float(np.trace(S)) / S.shape[0]
        prob.set_weighting(weight_from_covariance(S + ridge * np.eye(S.shape[0])))
        assert prob._multi is not None and prob._multi["include_agg"]
        ev2 = prob.evaluate(THETA_TRUE)
        assert ev2.ok
        prob.commit(THETA_TRUE)
        fd2, _ = central_fd(prob, THETA_TRUE, ev2.aux["delta"])
        errs2 = [_rel(ev2.G[:, k], fd2[:, k]) for k in range(THETA_TRUE.size)]
        assert max(errs2) < 1e-4, errs2
        J, names = prob.full_jacobian(ev2)
        assert J.shape == (prob.k, 3 + 3 + THETA_TRUE.size) and names[3:6] == ["c0", "c1", "c2"]
        gl = J[:, :6].T @ prob.W @ ev2.g   # the joint concentration minimises q over (beta, gamma)
        assert np.max(np.abs(gl)) < 1e-10 * max(1.0, float(np.max(np.abs(ev2.g))))
        W1 = prob.initial_weighting(THETA_TRUE)
        s2 = prob.blocks["supply_b"]
        assert np.allclose(W1[s2, s2], np.linalg.inv(second.Z.T @ second.Z / n))
    finally:
        prob.model.close()


def test_supply_blocks_must_compute_one_log_mc(priced: dict[str, Any]) -> None:
    n = priced["data"]["products"]["prices"].size
    firms = np.arange(n) // 2
    priced["data"]["products"]["firm_ids"] = firms
    layout = synthetic_layout(THETA_TRUE)
    lev, second = _blocks(priced, firms)
    pr = priced["data"]["products"]
    bad_col = SupplyMoments(second.Z, pr["x2"][:, None], firms, w_names=("c1",), name="supply_b")
    bad_floor = SupplyMoments(second.Z, second.w, firms, w_names=("c1",), name="supply_b", mc_floor=0.1)
    for bad, msg in ((bad_col, "cost column"), (bad_floor, "mc_floor")):
        model = PCMModel(PCMSpec(CHARS, tol=1e-12), priced["data"], layout, priced["nu"], priced["seed"],
                         total_threads=2)
        try:
            with pytest.raises(ValueError, match=msg):
                NFPGMMProblem(model, [IVMoments(priced["Z"]), lev, bad], LinearIV(priced["X1"], priced["Z"]), layout)
        finally:
            model.close()

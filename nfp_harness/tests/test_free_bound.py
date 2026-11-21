"""A free PCM bound a_lo: ``PCMModel.fd_columns`` names it, ``NFPGMMProblem`` takes its Jacobian column (and, under
``set_joint``, its CUE terms) by a one-sided difference of g, and the model's trial state at theta survives the extra
solve. Checked against central differences of g and of q_CUE on synthetic PCM markets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import IVMoments, SupplyMoments
from nfp_harness.optimizer import cue_gradient, cue_objective
from nfp_harness.params import ParameterLayout, ParameterSpec
from nfp_harness.problem import NFPGMMProblem

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, make_synthetic, synthetic_layout  # noqa: E402
from nfp_harness.tests.test_supply import _supply_inputs  # noqa: E402

A_LO = 0.3


@pytest.fixture(scope="module")
def priced() -> dict[str, Any]:
    return make_synthetic(2, 60, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)


def _layout(free: bool) -> ParameterLayout:
    return ParameterLayout(list(synthetic_layout(THETA_TRUE).specs) + [
        ParameterSpec("a_lo", "bound", None, None, A_LO, 0.05 if free else A_LO, 1.0 if free else A_LO, unit=0.01,
                      free=free)])


def _build(syn: dict[str, Any], free: bool) -> NFPGMMProblem:
    pr = syn["data"]["products"]
    n = pr["prices"].size
    layout = _layout(free)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
    Zs, w = _supply_inputs(syn)
    ids = np.arange(n)
    sup = SupplyMoments(Zs, w, np.arange(n), ids, w_names=("c0", "c1", "c2"))
    return NFPGMMProblem(model, [IVMoments(syn["Z"][:, :6], ids), sup], LinearIV(syn["X1"], syn["Z"][:, :6]), layout)


def _g(prob: NFPGMMProblem, th: np.ndarray) -> Any:
    ev = prob.evaluate(th, need_jacobian=False)
    prob.restore(th)
    assert ev.ok, (ev.status, ev.aux.get("note"))
    return ev


def test_free_bound_column_matches_central_differences_and_keeps_the_trial(priced: dict[str, Any]) -> None:
    theta = np.r_[THETA_TRUE, A_LO]
    fixed, free = _build(priced, False), _build(priced, True)
    try:
        assert fixed.fd_columns == () and free.fd_columns == (5,)
        ef, eF = fixed.evaluate(theta), free.evaluate(theta)
        assert ef.ok and eF.ok, (eF.status, eF.aux.get("note"))
        np.testing.assert_array_equal(eF.g, ef.g)                 # the extra solve leaves g alone
        np.testing.assert_array_equal(eF.G[:, :5], ef.G[:, :5])   # and every other column
        assert np.all(ef.G[:, 5] == 0.0) and np.abs(eF.G[:, 5]).max() > 0
        h = eF.aux["fd_columns"]["a_lo"]
        assert np.isclose(h, 0.02 * A_LO)
        free.commit(theta)   # the trial at theta was put back: every market is promoted
        assert all(a is not None and a.a_lo == A_LO for a in free.model._accepted)
        hc = 1e-3 * A_LO
        gp, gm = _g(free, theta + hc * np.eye(6)[5]).g, _g(free, theta - hc * np.eye(6)[5]).g
        central = (gp - gm) / (2 * hc)
        scale = np.abs(central).max()
        assert np.max(np.abs(eF.G[:, 5] - central)) < 0.05 * scale, (eF.G[:, 5], central)
    finally:
        fixed.model.close()
        free.model.close()


def test_free_bound_cue_gradient_matches_central_differences(priced: dict[str, Any]) -> None:
    theta = np.r_[THETA_TRUE, A_LO]
    prob = _build(priced, True)
    # a step of 0.2% of a_lo: the check is of the CUE terms, not of the production step's secant bias (q is curved
    # in a_lo on the scale of the 2% secant, and kinked below 0.05%)
    prob.model.fd_step = lambda th, k: 0.002 * float(th[k])   # type: ignore[method-assign]
    try:
        prob.set_joint("cue")
        ev = prob.evaluate(theta)
        assert ev.ok, (ev.status, ev.aux.get("note"))
        S = prob.covariance(ev)
        V, dV = prob.cue_terms(ev)
        grad = cue_gradient(ev, S, V, dV, prob.n)
        prob.commit(theta)
        hc = 1e-3 * A_LO
        qs = []
        for sgn in (1.0, -1.0):
            e = _g(prob, theta + sgn * hc * np.eye(6)[5])
            qs.append(cue_objective(e, prob.covariance(e), prob.n))
        fd = (qs[0] - qs[1]) / (2 * hc)
        assert abs(grad[5] - fd) < 0.05 * max(abs(fd), 1e-6), (grad[5], fd)
    finally:
        prob.model.close()

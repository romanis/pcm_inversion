"""The shared counterfactual engine (nfp_harness/counterfactuals.py) on a mixed logit and on a PCM market."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness import counterfactuals as cf
from nfp_harness.models.blp import MixedLogitModel
from nfp_harness.types import KernelRequests, MarketIndex


def _logit(J: int = 5, H: int = 400, seed: int = 1) -> tuple[MixedLogitModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = [np.column_stack([np.ones(J), rng.normal(size=J), rng.uniform(1.0, 2.0, size=J)])]
    D = [rng.lognormal(size=(H, 1)) * 0.2]
    nu = [rng.normal(size=(H, 1))]
    w = [np.full(H, 1.0 / H)]
    shares = [np.full(J, 0.1)]
    m = MixedLogitModel([MarketIndex(1, np.arange(J))], X, D, w, nu, [(2, 0)], [1], shares, price_column=2,
                        price_coefficient=-2.0)
    theta = np.array([-0.5, 0.8])
    sol = m.solve(theta, KernelRequests(), False)
    return m, theta, np.asarray(sol.delta)


def test_logit_nash_reproduces_observed_prices_and_merger_raises_prices() -> None:
    m, theta, delta = _logit()
    firms = np.arange(5)
    dem = cf.LogitMarketDemand(m, theta, delta, 0, firms)
    s = dem.shares(dem.p0)
    np.testing.assert_allclose(s, 0.1, atol=1e-10)   # the inversion's delta reproduces the data shares
    ic = cf.implied_costs(dem)
    assert ic["status"] == "ok" and np.all(ic["costs"] > 0)
    eq = cf.nash_prices(dem, ic["costs"], firms)
    assert eq["status"] == "converged"
    np.testing.assert_allclose(eq["prices"], dem.p0, rtol=1e-8)
    mg = cf.merger(dem, ic["costs"], 0, 1)
    assert mg["status"] == "converged"
    assert mg["prices"][0] > dem.p0[0] and mg["prices"][1] > dem.p0[1]
    assert mg["price_change_vs_baseline"]["affected"] > mg["price_change_vs_baseline"]["others"] > 0 and mg["affected"] == 2
    assert mg["cs_change"] < 0
    sh = cf.cost_shock(dem, ic["costs"], np.array([True, False, False, False, False]), 0.10)
    assert 0 < sh["pass_through_own"] < 2 and sh["rival_price_change"] > 0
    m.close()


def test_logit_surplus_is_the_integral_of_demand() -> None:
    """CS(p) - CS(p + dp e_j) = int s_j dp along the path (Roy's identity with the log-sum)."""
    m, theta, delta = _logit()
    dem = cf.LogitMarketDemand(m, theta, delta, 0, np.arange(5))
    j, hi = 2, 0.5
    grid = np.linspace(0.0, hi, 401)
    sj = []
    for g in grid:
        p = dem.p0.copy()
        p[j] += g
        sj.append(dem.shares(p)[j])
    p1 = dem.p0.copy()
    p1[j] += hi
    integral = float(np.trapezoid(sj, grid)) if hasattr(np, "trapezoid") else float(np.trapz(sj, grid))
    assert np.isclose(dem.surplus(dem.p0) - dem.surplus(p1), integral, rtol=1e-5)
    m.close()


def test_removal_diversion_sums_to_one() -> None:
    m, theta, delta = _logit()
    dem = cf.LogitMarketDemand(m, theta, delta, 0, np.arange(5))
    (r,) = cf.removal_diversion(dem, [3])
    assert np.isnan(r["to"][3]) and np.isclose(np.nansum(r["to"]) + r["outside"], 1.0)
    assert np.all(r["to"][np.isfinite(r["to"])] > 0) and r["cs_change"] < 0
    lt = cf.lerner_table(dem.p0, cf.implied_costs(dem)["costs"], dem.shares(dem.p0),
                         {"segment": np.array(["a", "a", "b", "b", "b"])})
    assert set(lt["segment"]) == {"a", "b"} and 0 < lt["all"]["median"] < 1
    m.close()


def test_pcm_engine_on_a_synthetic_market() -> None:
    pytest.importorskip("nfp_harness._native")
    from nfp_harness.models.pcm import PCMModel, PCMSpec
    from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, make_synthetic, synthetic_layout

    syn: dict[str, Any] = make_synthetic(1, 12, 16, 8, seed=2, price_shift=12.0, delta_shift=2.0)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], synthetic_layout(THETA_TRUE), syn["nu"], syn["seed"],
                     total_threads=2)
    try:
        sol = model.solve(THETA_TRUE, KernelRequests(), False)
        J = np.size(model.markets[0].rows)
        firms = np.arange(J)
        dem = cf.PCMMarketDemand(model, THETA_TRUE, np.asarray(sol.delta), 0, np.arange(model.n_products))
        np.testing.assert_allclose(dem.shares(dem.p0),
                                   np.asarray(syn["data"]["products"]["shares"])[dem.rows], atol=1e-9)
        ic = cf.implied_costs(dem)
        assert ic["status"] == "ok"
        eq = cf.nash_prices(dem, ic["costs"], firms, box=0.05, max_sweeps=4)
        assert eq["status"] in ("converged", "cycle", "not_converged") and eq["prices"].shape == (J,)
        assert np.all(np.abs(eq["prices"] / dem.p0 - 1) <= 0.05 + 1e-12)
        top = int(np.argmax(ic["shares"]))
        (r,) = cf.removal_diversion(dem, [top])
        assert np.isclose(np.nansum(r["to"]) + r["outside"], 1.0)
        assert r["cs_change"] <= 0
    finally:
        model.close()

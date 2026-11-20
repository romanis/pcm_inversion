"""Cold starts, thread budgets and warm starts under another bound (kernel ``solve_market``).

A cold inversion at a large draw count spends thousands of share evaluations in the heterogeneity continuation of
``pcm_demand::invert`` (the vertical-model guess leaves most products off every frontier); the same path on a thinned
market costs 1 / k as much per evaluation, and a warm start from it needs tens of evaluations on all draws
(``SolveOptions::cold_draws``). A ``ThreadBudget`` lets a finished market's threads go to the markets still running
without changing any result, and ``lo_warm`` lets a delta solved under one bound warm-start another.
"""

from __future__ import annotations

import pathlib
import re
import threading
import time

import numpy as np
import pytest

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel

from nfp_harness.tests.test_kernel_solve import data_shares, make_market  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "gmy_cold_theta.npz"


def full_cost(stages: str, R: int) -> float:
    """Share evaluations of a solve in units of one evaluation on all R draws (``draws R=r`` stages cost r / R)."""
    cost = 0.0
    for m in re.finditer(r"(?:draws R=(\d+)|cold|draws warm|warm|path) (?:ok|failed) \S+s (\d+)ev", stages):
        cost += int(m.group(2)) * (int(m.group(1)) / R if m.group(1) else 1.0)
    return cost


def test_draw_homotopy_converges_to_the_same_delta() -> None:
    mk, delta, theta, phi, sp = make_market(5, n=30, r=8192)
    s_data = data_shares(mk, delta, theta, phi, sp)
    plain = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=4, derivatives=False)
    homo = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=4, derivatives=False, cold_draws=1024)
    assert plain["status"] == "ok" and homo["status"] == "ok", (plain["stages"], homo["stages"])
    assert plain["stages"].startswith("cold ok")
    # 8192 draws, factor 8: 1024 draws cold, then all 8192 warm-started; no plain cold stage
    assert homo["stages"].startswith("draws R=1024 ok") and "draws warm ok" in homo["stages"], homo["stages"]
    assert "cold" not in homo["stages"]
    assert homo["max_abs_residual"] < 1e-9
    np.testing.assert_allclose(homo["delta"], plain["delta"], atol=1e-7)
    # off (the default) and fewer draws than one level: the cold stage as before, bit for bit
    for kw in ({}, {"cold_draws": 0}, {"cold_draws": 2000}):
        again = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=4, derivatives=False, **kw)
        np.testing.assert_array_equal(again["delta"], plain["delta"])
        assert again["stages"].startswith("cold ok")
    bad = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, cold_draws=1024, cold_draw_factor=1)
    assert bad["status"] == "invalid_input" and "cold_draw_factor" in bad["stages"]


def test_thread_budget_changes_mid_solve_and_results_do_not() -> None:
    mk, delta, theta, phi, sp = make_market(2, n=40, r=6000)
    s_data = data_shares(mk, delta, theta, phi, sp)
    ref = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=1)
    assert ref["status"] == "ok", ref["stages"]
    budget = pcm_kernel.ThreadBudget(1)
    assert budget.value == 1
    stop = threading.Event()

    def flip() -> None:   # raise and lower the team size while the solve runs
        n = 1
        while not stop.is_set():
            n = 8 if n == 1 else 1
            budget.value = n
            time.sleep(0.002)

    t = threading.Thread(target=flip)
    t.start()
    try:
        out = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=1, thread_budget=budget)
    finally:
        stop.set()
        t.join()
    for k in ("delta", "s", "ds_ddelta", "ddelta_dtheta"):
        np.testing.assert_array_equal(out[k], ref[k])
    with pytest.raises(ValueError):
        pcm_kernel.ThreadBudget(0)
    with pytest.raises(ValueError):
        budget.value = 0


def test_warm_start_under_another_bound_follows_the_bound() -> None:
    """A delta solved at a_lo 0.3 warm-starts a_lo 0.2: the theta path moves the bound with lo_warm."""
    mk, delta, theta, phi, sp = make_market(7, n=30, r=4000)
    s_data = np.asarray(pcm_kernel.evaluate_market(delta, mk, theta, phi, sp, derivatives=False, a_lo=0.2)["s"])
    cold = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=4, a_lo=0.2)
    at3 = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=4, a_lo=0.3)
    assert cold["status"] == "ok" and at3["status"] == "ok"
    stacked = np.r_[theta, phi, sp]
    # warm stage starved (one evaluation), no cold stage: only the path in a_lo can solve it
    out = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, at3["delta"], stacked, n_threads=4, warm_evals=1,
                                  cold_evals=0, a_lo=0.2, lo_warm=0.3)
    assert out["status"] == "ok", out["stages"]
    assert "path ok" in out["stages"], out["stages"]
    np.testing.assert_allclose(out["delta"], cold["delta"], atol=1e-7)
    # lo_warm equal to the bound, or NaN (the default), is the warm start under the same bound as before
    same = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, cold["delta"], stacked, n_threads=4, a_lo=0.2)
    again = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, cold["delta"], stacked, n_threads=4, a_lo=0.2,
                                    lo_warm=0.2)
    np.testing.assert_array_equal(same["delta"], again["delta"])


@pytest.fixture(scope="module")
def gmy_markets() -> dict:
    from nfp_harness.models.pcm import PCMModel
    from nfp_harness.tests.test_gmy_instruments import gmy_module

    gmy = gmy_module()
    try:
        gmy.package_root()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    spec = gmy.GMYSpec(gmy.DEV_YEARS)
    model = PCMModel(spec.pcm_spec(), {"products": spec.products, "agents": spec.agents}, spec.layout(), 8, gmy.SEED,
                     total_threads=8, antithetic=True)
    out = {int(m.market_id): (m.kernel, m.s_data, m.R) for m in model._mk}
    model.close()
    return out


def test_gmy_cold_start_at_the_round5_step1_theta(gmy_markets: dict) -> None:
    """GMY 1980 (161 products, 1,000 households x 8 draws = 8,192 draws) at the step-1 theta of the round-5 run
    gmy_pcm_d889d4ccea (a_lo 0.2).

    Cold, the vertical-model guess leaves most products off every frontier at full heterogeneity (112 of 161 with
    2,000 households at nu 4), the repair cycles for 60 rounds, the continuation halves its step and Newton starts from
    shares 1e-15 of their targets: the plain cold stage is still without a feasible start after 1,500 share
    evaluations. With the draw homotopy the same path runs on 1,024 draws (an eighth of the cost per evaluation) and the
    warm start on all draws takes about a hundred evaluations: converged at a cost below 1,500 evaluations on all
    draws."""
    z = np.load(FIXTURE)
    km, s, R = gmy_markets[1980]
    th, ph, sp, a_lo = z["theta"], z["phi"], float(z["sigma_p"]), float(z["a_lo"])
    budget = 1500
    plain = pcm_kernel.solve_market(s, km, th, ph, sp, cold_evals=budget, n_threads=8, derivatives=False, a_lo=a_lo)
    assert plain["status"] != "ok", plain["stages"]
    homo = pcm_kernel.solve_market(s, km, th, ph, sp, n_threads=8, derivatives=False, a_lo=a_lo, cold_draws=1024)
    assert homo["status"] == "ok", homo["stages"]
    assert homo["max_abs_residual"] < 1e-9
    warm = re.search(r"draws warm ok \S+s (\d+)ev", homo["stages"])
    assert warm is not None and int(warm.group(1)) < 200, homo["stages"]
    assert full_cost(homo["stages"], R) < budget, homo["stages"]

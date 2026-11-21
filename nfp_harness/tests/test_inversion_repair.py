"""Warm inversions that need a repair (GMY round-1 step 1, market 2015).

At the round-1 step-1 point of the six-year nu-4 run, a step of 1.9e-5 natural units took two 2015 products off every
consumer's frontier. Products 260 and 269 (shares 3.5e-6 and 1.1e-6) are bought by one draw each and sit next to each
other on that draw's hull, so the repair lifted one, which dropped the other below the new chord, for 60 rounds; the
warm stage, the theta path and the cold stage all failed. The fixture holds the kernel parameters at that point and at
the failing trial, and the accepted deltas; the market is rebuilt from the GMY package (skipped without it).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel

from nfp_harness.tests.test_gmy_instruments import gmy_module  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "gmy_inversion_repair.npz"


@pytest.fixture(scope="module")
def markets() -> dict:
    try:
        gmy_module().package_root()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    gmy = gmy_module()
    # the fixture comes from a round-1 run: every product kept (before D9), no pi_const_inc (D8) and no round-4 age taste
    spec = gmy.GMYSpec(gmy.DEV_YEARS, min_share=0.0, robustness=("no_pi_const_inc", "no_age_taste"))
    spec.set_prefit_variant("level")   # the variant that run's prefit kept
    from nfp_harness.models.pcm import PCMModel
    model = PCMModel(spec.pcm_spec(), {"products": spec.products, "agents": spec.agents}, spec.layout(), 4,
                     gmy.SEED, total_threads=8)
    out = {int(m.market_id): (m.kernel, m.s_data) for m in model._mk}
    model.close()
    return out


def _solve(km, s, delta, z, theta, phi, sigma_p):
    return pcm_kernel.solve_market(s, km, theta, phi, sigma_p, delta, np.r_[z["theta"], z["phi"], float(z["sigma_p"])],
                                   None, tol=1e-9, warm_evals=2500, cold_evals=0, path_evals=1500, path_seconds=90.0,
                                   n_threads=8, derivatives=False, a_lo=float(z["a_lo"]))


@pytest.mark.parametrize("scale", [1.0, None])   # the failing trial (1.9e-5) and a 5e-5 step along it
@pytest.mark.parametrize("year", [2015, 2018])
def test_warm_step_inverts(markets: dict, year: int, scale: float | None) -> None:
    z = np.load(FIXTURE)
    km, s = markets[year]
    np.testing.assert_array_equal(s, z[f"s_{year}"])   # the same market as the run
    if year == 2015:
        np.testing.assert_array_equal(np.asarray(km.p), z["p_2015"])
    d = np.r_[z["trial_theta"] - z["theta"], z["trial_phi"] - z["phi"], float(z["trial_sigma_p"]) - float(z["sigma_p"])]
    f = 5e-5 / np.abs(d).max() if scale is None else scale
    M, Q = z["theta"].size, z["phi"].size
    theta, phi, sp = z["theta"] + f * d[:M], z["phi"] + f * d[M:M + Q], float(z["sigma_p"]) + f * d[-1]
    delta = np.asarray(z[f"delta_{year}"], dtype=float)
    s_acc = np.asarray(pcm_kernel.evaluate_market(delta, km, theta, phi, sp, derivatives=False, n_threads=8,
                                                  a_lo=float(z["a_lo"]))["s"])
    if year == 2015:
        assert (s_acc <= 0).sum() >= 2   # the step takes products off every frontier: a repair is needed
    r = _solve(km, s, delta, z, theta, phi, sp)
    assert r["status"] == "ok", r["stages"]
    assert r["stages"].startswith("warm ok")
    assert r["max_abs_residual"] < 1e-9
    assert r["share_evaluations"] < 500

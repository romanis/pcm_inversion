"""Tests for the compiled solve_market binding (WP1; DESIGN 3.2, 3.5, 7 WP1 acceptance).

The C++ behaviour is covered by demand_model/pcm_kernel_test.cpp. These tests check what only the
binding can break: the returned dict carries every SolveResult field, and the call releases the GIL
so markets solved in Python threads run in parallel.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel

SOLVE_RESULT_FIELDS = {
    "status", "delta", "s", "ds_ddelta", "ds_dtheta", "ddelta_dtheta", "ds_dp", "micro",
    "micro_d_delta", "micro_d_theta", "max_abs_residual", "rcond", "seconds", "share_evaluations", "stages",
}


def make_market(seed: int, n: int = 12, r: int = 400) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, float]:
    """The pcm_demand_test fixture shape: x = [1, z1, z2], 5 taste and 2 scale parameters."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), 10 * rng.random(n), (rng.random(n) < 0.5).astype(float)])
    step = 0.12 / n  # 0.01 for n = 12; prices stay where delta is increasing
    p = 0.05 + step * np.arange(n) + 0.3 * step * rng.random(n)
    delta = 2.0 * p - 5.0 * p**2 + 5.0 * step**2 * rng.standard_normal(n)  # noise below the curvature
    d0 = rng.standard_normal(r)
    taste_basis = np.column_stack([rng.standard_normal((r, 3)), d0, d0])
    scale_basis = np.column_stack([d0, (rng.random(r) < 0.3).astype(float)])
    mk = pcm_kernel.Market(x, p, np.full(r, 1.0 / r), taste_basis, [0, 1, 2, 0, 1], scale_basis)
    theta = np.array([0.02, 0.002, 0.01, 0.01, -0.001])
    phi = np.array([0.3, -0.2])
    return mk, delta, theta, phi, 0.6


def data_shares(mk: object, delta: np.ndarray, theta: np.ndarray, phi: np.ndarray, sigma_p: float) -> np.ndarray:
    s = pcm_kernel.evaluate_market(delta, mk, theta, phi, sigma_p, derivatives=False)["s"]
    assert (s > 0).all() and s.sum() < 1
    return s


def test_solve_market_returns_every_field() -> None:
    mk, delta, theta, phi, sp = make_market(1)
    s_data = data_shares(mk, delta, theta, phi, sp)
    n, r = len(delta), len(mk.weights)
    micro = {"H": np.column_stack([np.ones(r), np.linspace(0, 1, r)]),
             "U": np.column_stack([np.ones(n), np.eye(n)[:, 0]]), "h_of_part": [0, 1]}
    out = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, delta + 1e-4, None, micro,
                                  price_derivatives=True, n_threads=2)
    assert set(out) == SOLVE_RESULT_FIELDS
    assert out["status"] == "ok", out["stages"]
    P = len(theta) + len(phi) + 1
    assert out["ds_ddelta"].shape == (n, n) and out["ds_dp"].shape == (n, n)
    assert out["ds_dtheta"].shape == (n, P) and out["ddelta_dtheta"].shape == (n, P)
    assert out["micro"].shape == (2,) and out["micro_d_delta"].shape == (2, n) and out["micro_d_theta"].shape == (2, P)
    assert out["max_abs_residual"] < 1e-9 and out["rcond"] > 0 and out["share_evaluations"] > 0
    np.testing.assert_allclose(out["micro"][0], out["s"].sum(), rtol=1e-13)
    np.testing.assert_allclose(out["ddelta_dtheta"], -np.linalg.solve(out["ds_ddelta"], out["ds_dtheta"]), rtol=1e-8, atol=1e-12)

    # theta_warm is the stacked [theta, phi, sigma_p]; a dict market works too
    as_dict = {k: getattr(mk, k) for k in ("x", "p", "weights", "taste_basis", "taste_col", "scale_basis")}
    again = pcm_kernel.solve_market(s_data, as_dict, theta, phi, sp, delta, np.r_[theta, phi, sp])
    np.testing.assert_array_equal(again["delta"], pcm_kernel.solve_market(s_data, mk, theta, phi, sp, delta)["delta"])
    with pytest.raises(ValueError, match="theta_warm"):
        pcm_kernel.solve_market(s_data, mk, theta, phi, sp, delta, np.r_[theta, sp])

    bad = pcm_kernel.solve_market(np.zeros(n), mk, theta, phi, sp)
    assert bad["status"] == "invalid_input" and "invalid input" in bad["stages"]
    assert bad["ds_ddelta"].shape == (0, 0)


def test_solve_market_releases_the_gil() -> None:
    """Two markets in two Python threads take less than 0.7 of the serial wall time."""
    markets = []
    for seed in (2, 3):
        mk, delta, theta, phi, sp = make_market(seed, n=40, r=6000)
        markets.append((data_shares(mk, delta, theta, phi, sp), mk, theta, phi, sp))

    def solve(m: tuple) -> dict:
        s_data, mk, theta, phi, sp = m
        out = pcm_kernel.solve_market(s_data, mk, theta, phi, sp, n_threads=1)
        assert out["status"] == "ok", out["stages"]
        return out

    for m in markets:  # warm up caches and the OpenMP runtime
        solve(m)
    t0 = time.perf_counter()
    for m in markets:
        solve(m)
    serial = time.perf_counter() - t0

    results: list[dict] = []
    threads = [threading.Thread(target=lambda m=m: results.append(solve(m))) for m in markets]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parallel = time.perf_counter() - t0
    assert len(results) == 2, "a threaded solve failed"
    print(f"serial {serial:.3f}s, two threads {parallel:.3f}s, ratio {parallel / serial:.2f}")
    assert serial > 0.2, "markets too small to time"
    assert parallel < 0.7 * serial


def _all_outputs(mk: object, delta: np.ndarray, theta: np.ndarray, phi: np.ndarray, sp: float,
                 **kw: float) -> dict[str, np.ndarray]:
    """Every kernel binding at one point; ``kw`` is passed to each (``alpha_lo``)."""
    out: dict[str, np.ndarray] = {}
    ev = pcm_kernel.evaluate_market(delta, mk, theta, phi, sp, derivatives=True, price_derivatives=True, **kw)
    out.update({f"ev.{k}": np.asarray(ev[k]) for k in ("s", "ds_ddelta", "ds_dtheta", "ds_dp")})
    cs = pcm_kernel.consumer_surplus(delta, mk, theta, phi, sp, 1, **kw)
    out["cs"] = np.asarray(cs["per_draw"])
    s = data_shares(mk, delta, theta, phi, sp)
    sol = pcm_kernel.solve_market(s, mk, theta * 1.01, phi, sp, delta, np.r_[theta, phi, sp], None, **kw)
    out.update({f"solve.{k}": np.asarray(sol[k]) for k in ("delta", "ds_dtheta", "ddelta_dtheta")})
    P = len(theta) + len(phi) + 1
    ms = pcm_kernel.max_step(delta, np.linspace(-1, 1, len(delta)), mk, theta, phi, sp, np.ones(P), 2.0, 1, **kw)
    out.update({f"ms.{k}": np.asarray(ms[k]) for k in ("tau_draw", "kind_draw", "mass_profile", "buyer_mass")})
    out["sig"] = np.asarray(pcm_kernel.frontier_signature(delta, mk, theta, phi, sp, 1, **kw)).astype(float)
    return out


def test_alpha_lo_zero_is_the_untruncated_model() -> None:
    """alpha_lo = 0 (default or explicit) runs the code that existed before the bound, bit for bit; a bound
    of 1e-300 goes through the truncated formulas and agrees to 1e-12 (LEARNINGS 29)."""
    mk, delta, theta, phi, sp = make_market(3)
    base = _all_outputs(mk, delta, theta, phi, sp)
    zero = _all_outputs(mk, delta, theta, phi, sp, alpha_lo=0.0)
    tiny = _all_outputs(mk, delta, theta, phi, sp, alpha_lo=1e-300)
    for k in base:
        assert np.array_equal(base[k], zero[k], equal_nan=True), k
        if k != "sig":
            assert np.allclose(base[k], tiny[k], rtol=0, atol=1e-12 * max(1.0, np.abs(base[k]).max())), k
    assert np.array_equal(base["sig"], tiny["sig"])


def test_alpha_lo_bounds_surplus_and_changes_shares() -> None:
    mk, delta, theta, phi, sp = make_market(4)
    s0 = pcm_kernel.evaluate_market(delta, mk, theta, phi, sp, derivatives=False)["s"]
    s1 = pcm_kernel.evaluate_market(delta, mk, theta, phi, sp, derivatives=False, alpha_lo=1.0)["s"]
    assert s1.sum() < s0.sum()
    cs0 = pcm_kernel.consumer_surplus(delta, mk, theta, phi, sp, 1)["total"]
    cs1 = pcm_kernel.consumer_surplus(delta, mk, theta, phi, sp, 1, alpha_lo=1.0)["total"]
    assert 0 < cs1 < cs0
    with pytest.raises(ValueError):
        pcm_kernel.evaluate_market(delta, mk, theta, phi, sp, alpha_lo=-1.0)

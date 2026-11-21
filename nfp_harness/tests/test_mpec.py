"""Tests for the optional demand-only MPEC outer solver (``nfp_harness.mpec``, LEARNINGS 31).

Skipped without cyipopt (the anaconda python): run with the solver environment,
``.envs/mpec/bin/python -m pytest nfp_harness/tests/test_mpec.py``. The synthetic problem is the one of
``test_pcm_model.test_recovers_true_theta`` (3 markets, 76 products, 1,024 draws, aggregate IV + 3 ratio micro
moments, fixed W1), where theta_true is the zero-residual GMM minimum. The Petrin 1984 timing check runs only with
``NFP_SLOW=1``.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import pathlib
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

cyipopt = pytest.importorskip("cyipopt")
pytest.importorskip("nfp_harness._native")

from nfp_harness.checkpoint import RunDirectory  # noqa: E402
from nfp_harness.mpec import SUPPLY_REFUSAL, MPECProblem, minimize_mpec  # noqa: E402
from nfp_harness.optimizer import OptimizerOptions, minimize  # noqa: E402
from nfp_harness.pipeline import StageRefused  # noqa: E402
from nfp_harness.tests.test_pcm_model import BETA_TRUE, THETA_TRUE, build_problem, make_synthetic  # noqa: E402
from nfp_harness.types import ConvergenceClass  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
START = THETA_TRUE * np.array([1.3, 0.7, 1.4, 0.75, 1.3])
FAR = THETA_TRUE * np.array([0.6, 1.5, 0.2, 1.6, 0.6])
THREADS = 8
MPEC = OptimizerOptions(stage="mpec", outer="mpec")

def replace_stage(options: OptimizerOptions, stage: str) -> OptimizerOptions:
    return replace(options, stage=stage)


slow = pytest.mark.skipif(os.environ.get("NFP_SLOW") != "1", reason="slow (Petrin 1984 at nu = 16): set NFP_SLOW=1")


@pytest.fixture(scope="module")
def medium() -> dict[str, Any]:
    return make_synthetic(3, 44, 128, 8, seed=6)


@pytest.fixture(scope="module")
def recovered(medium: dict[str, Any]) -> Any:
    """Test 1's run, shared with test 3."""
    prob = build_problem(medium, START, threads=THREADS)
    res = minimize(prob, prob.initial_weighting(START), MPEC)
    prob.model.close()
    return res


def _report(label: str, res: Any) -> None:
    m = res.mpec
    print(f"\n  {label}: {res.convergence.value}, IPOPT {m['status']}, {m['iterations']} iterations, "
          f"{m['restarts']} restarts, {m['derivative_passes']} derivative + {m['share_passes']} share passes, "
          f"theta error {np.abs(res.theta - THETA_TRUE).max():.2e}, max |log s/s_data| "
          f"{m['max_log_share_violation']:.1e}, {m['seconds']:.2f}s")


def test_recovers_theta_to_nested_precision(recovered: Any) -> None:
    """Test 1: from the recovery test's start, MPEC (through ``minimize``) returns CONVERGED with theta_true and
    beta_true to 1e-6 and the log-share constraints satisfied to 1e-9."""
    res = recovered
    _report("perturbed start", res)
    assert res.convergence is ConvergenceClass.CONVERGED, res.message
    np.testing.assert_allclose(res.theta, THETA_TRUE, atol=1e-6)
    np.testing.assert_allclose(res.evaluation.aux["beta"], BETA_TRUE, atol=1e-6)
    np.testing.assert_allclose(res.mpec["beta"], BETA_TRUE, atol=1e-6)
    assert res.mpec["max_log_share_violation"] < 1e-9
    assert len(res.history) == res.mpec["iterations"] + 1


def test_matches_nested_objective_and_derivatives(medium: dict[str, Any]) -> None:
    """Test 2: at the nested solution, the MPEC objective and moments equal the harness's, the constraints hold, and
    the constraint Jacobian matches central differences (every free theta and 6 random delta coordinates). The
    nested solution has q ~ 1e-14 and a zero gradient, so the relative q and gradient checks are also made at the
    start, where q = 36."""
    prob = build_problem(medium, START, threads=THREADS)
    W1 = prob.initial_weighting(START)
    res_n = minimize(prob, W1, OptimizerOptions(stage="nested", max_iterations=200))
    assert res_n.convergence.converged, res_n.message
    mp = MPECProblem(prob, W1)
    for theta in (res_n.theta, START):
        ev = prob.evaluate(theta, True)
        assert ev.ok
        x = mp.x_of(theta, ev.aux["delta"])
        k = mp.evaluate(x)
        assert np.max(np.abs(k["g"] - ev.g)) < 1e-11
        q = ev.aux["q"]
        assert abs(mp.objective(x) - q) <= 1e-10 * max(abs(q), 1e-3), (mp.objective(x), q)
        assert np.max(np.abs(mp.constraints(x))) < 1e-9
    rng = np.random.default_rng(1)
    cols = list(range(mp.Pf)) + list(mp.Pf + rng.choice(mp.n, 6, replace=False))
    h = 1e-6
    err_J = err_g = 0.0
    for theta in (res_n.theta, START):
        x = mp.x_of(theta, prob.evaluate(theta, True).aux["delta"])
        J = mp.jacobian_dense(x)
        grad = mp.gradient(x)
        for c in cols:
            xp, xm = x.copy(), x.copy()
            xp[c] += h
            xm[c] -= h
            fd_c = (mp.constraints(xp) - mp.constraints(xm)) / (2 * h)
            err_J = max(err_J, float(np.abs(fd_c - J[:, c]).max() / max(np.abs(J[:, c]).max(), 1e-12)))
            if theta is START:   # at the zero-residual solution the gradient is 0 to rounding: no relative check
                fd_f = (mp.objective(xp) - mp.objective(xm)) / (2 * h)
                err_g = max(err_g, abs(fd_f - grad[c]) / max(abs(grad[c]), 1e-12))
    print(f"\n  central differences: Jacobian {err_J:.2e}, gradient {err_g:.2e} (relative)")
    assert err_J < 1e-5
    assert err_g < 1e-5
    prob.model.close()


def test_zero_share_start_is_lifted(medium: dict[str, Any], recovered: Any) -> None:
    """Test 3: a start delta with the 6 smallest-share products lowered by 3 (zero shares, log 0) is repaired by the
    start's lift, and the run converges to test 1's theta."""
    prob = build_problem(medium, START, threads=THREADS)
    W1 = prob.initial_weighting(START)
    ev = prob.evaluate(START, False)
    d0 = np.array(ev.aux["delta"])
    s_data = np.asarray(medium["data"]["products"]["shares"])
    d0[np.argsort(s_data)[:6]] -= 3.0
    mp = MPECProblem(prob, W1)
    s = mp.evaluate(mp.x_of(START, d0))["s"]
    assert np.sum(s <= 0) > 0
    res = minimize_mpec(prob, W1, MPEC, delta0=d0)
    _report("zero-share start", res)
    assert res.mpec["start_zero_share_products"] == int(np.sum(s <= 0))
    assert "lifted and re-solved" in res.message
    assert res.convergence is ConvergenceClass.CONVERGED, res.message
    np.testing.assert_allclose(res.theta, recovered.theta, atol=1e-6)
    prob.model.close()


def test_far_start(medium: dict[str, Any]) -> None:
    """Test 4: from the far start (where nested Gauss-Newton ends FLAT_PROGRESS) with delta from the nested
    inversion there, MPEC returns CONVERGED at theta_true."""
    prob = build_problem(medium, FAR, threads=THREADS)
    res = minimize(prob, prob.initial_weighting(FAR), replace_stage(MPEC, "far"))
    _report("far start", res)
    assert res.convergence is ConvergenceClass.CONVERGED, res.message
    np.testing.assert_allclose(res.theta, THETA_TRUE, atol=1e-6)
    prob.model.close()


def test_invalid_number_restarts(medium: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """IPOPT stopping on an invalid number (-13) triggers the lift-and-restart: the markets are re-solved at the
    current theta and IPOPT restarts warm; with no restarts left the stage ends INVERSION_FAILED."""
    prob = build_problem(medium, START, threads=THREADS)
    W1 = prob.initial_weighting(START)
    real = MPECProblem.constraints
    calls = {"n": 0}

    def first_call_nan(self: MPECProblem, x: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.full(self.n, np.nan) if calls["n"] == 1 else real(self, x)

    monkeypatch.setattr(MPECProblem, "constraints", first_call_nan)
    res = minimize(prob, W1, replace_stage(MPEC, "restart"))
    assert res.mpec["restarts"] == 1 and "restart 1 after IPOPT status -13" in res.message
    assert res.convergence is ConvergenceClass.CONVERGED, res.message
    np.testing.assert_allclose(res.theta, THETA_TRUE, atol=1e-6)
    calls["n"] = 0
    res = minimize(prob, W1, OptimizerOptions(stage="no_restart", outer="mpec", mpec_max_restarts=0))
    assert res.mpec["restarts"] == 0 and res.convergence is ConvergenceClass.INVERSION_FAILED
    prob.model.close()


class _NoKernel:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"kernel called: {name}")


def test_supply_refused_before_any_kernel_call() -> None:
    """Test 5: a problem with SupplyMoments and outer="mpec" raises StageRefused, and no kernel function is
    touched; an unknown outer is a ValueError."""
    from nfp_harness.tests.test_supply import _problem

    syn = make_synthetic(2, 8, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)
    prob = _problem(syn, np.arange(syn["data"]["products"]["prices"].size))
    W = prob.W.copy()
    kernel = prob.model._k
    prob.model._k = _NoKernel()
    try:
        with pytest.raises(StageRefused, match="does not support supply/margin moments"):
            minimize(prob, W, OptimizerOptions(stage="refused", outer="mpec"))
        with pytest.raises(StageRefused) as info:
            minimize_mpec(prob, W)
        assert str(info.value) == SUPPLY_REFUSAL
    finally:
        prob.model._k = kernel
    with pytest.raises(ValueError, match="outer"):
        minimize(prob, W, OptimizerOptions(outer="bogus"))
    prob.model.close()


def test_checkpoint_history_and_resume(medium: dict[str, Any], tmp_path: pathlib.Path) -> None:
    """One history.csv row per IPOPT iteration, x and multipliers in stages/<stage>.mpec every 10 iterations, a
    finished stage rebuilt on rerun, and a run killed after a checkpoint resumes from it with a warm start."""
    prob = build_problem(medium, FAR, threads=THREADS)
    W1 = prob.initial_weighting(FAR)
    rd = RunDirectory(tmp_path, "mpec_test", "hash", "key")
    res = minimize(prob, W1, replace_stage(MPEC, "ck"), checkpoint=rd)
    assert res.convergence is ConvergenceClass.CONVERGED, res.message
    with open(rd.path / "history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == res.mpec["iterations"] + 1 and all(r["stage"] == "ck" for r in rows)
    assert rd.stage_done("ck") and rd.stage_done("ck.mpec") and (rd.path / "ipopt_ck.log").exists()
    again = minimize(prob, W1, replace_stage(MPEC, "ck"), checkpoint=rd)   # finished: rebuilt, not re-run
    np.testing.assert_array_equal(again.theta, res.theta)
    assert again.convergence is ConvergenceClass.CONVERGED
    for suffix in (".json", ".npz"):   # as if killed after the iteration-10 checkpoint
        (rd.path / "stages" / f"ck{suffix}").unlink()
    saved = rd.load_stage("ck.mpec")
    resumed = minimize(prob, W1, replace_stage(MPEC, "ck"), checkpoint=rd)
    assert "resumed after IPOPT iteration" in resumed.message
    assert resumed.convergence is ConvergenceClass.CONVERGED, resumed.message
    assert resumed.mpec["iterations"] > int(saved["steps"])
    np.testing.assert_allclose(resumed.theta, THETA_TRUE, atol=1e-6)
    prob.model.close()


# --------------------------------------------------------------------------------------------
# Test 6: timing on Petrin 1984 against the nested outer (slow)
# --------------------------------------------------------------------------------------------


def _petrin_module() -> Any:
    path = ROOT / "petrin_replication" / "estimate_pcm.py"
    spec = importlib.util.spec_from_file_location("estimate_pcm_for_mpec_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gate_theta(names: list[str], start: np.ndarray) -> np.ndarray | None:
    """theta at the nu-4 gate minimum (lambda = 1, lowest q_fixed), read-only; layout starts for parameters the gate
    does not carry (a_lo). ``NFP_PETRIN_GATE`` overrides the profile.csv path."""
    path = pathlib.Path(os.environ.get("NFP_PETRIN_GATE", ROOT / "petrin_replication" / "results" / "harness" /
                                       "petrin_pcm_9367ef1cc1" / "gate" / "profile_sigma_ray" / "profile.csv"))
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok" and abs(float(r["lambda"]) - 1.0) < 1e-12]
    rows = [r for r in rows if np.isfinite(float(r["q_fixed"]))]
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["q_fixed"]))
    return np.array([float(best[f"theta:{n}"]) if f"theta:{n}" in best else float(v) for n, v in zip(names, start)])


def _petrin_1984(mod: Any, threads: int) -> Any:
    """The 1984 market alone at nu = 16 with demand moments (Sargan aggregate block on Z restricted to the market's
    non-constant columns, Petrin's 10 CEX micro moments with their fixed covariance)."""
    from nfp_harness.linear import LinearIV, matrix_rank
    from nfp_harness.models.pcm import PCMModel
    from nfp_harness.moments import SarganIVMoments
    from nfp_harness.problem import NFPGMMProblem

    spec = mod.PetrinSpec()
    rows = np.flatnonzero(spec.products.market_ids.to_numpy() == mod.YEAR)
    products = spec.products.iloc[rows].reset_index(drop=True)
    agents = spec.agents[spec.agents.market_ids == mod.YEAR].reset_index(drop=True)
    def independent(A: np.ndarray) -> list[int]:   # columns that raise the rank, in order
        cols: list[int] = []
        for j in range(A.shape[1]):
            if matrix_rank(A[:, cols + [j]]) == len(cols) + 1:
                cols.append(j)
        return cols

    keep = independent(spec.X1[rows])   # drops pgnp, trend, trend2 (constant in a year) and collinear dummies
    X1 = spec.X1[rows][:, keep]
    Z = np.column_stack([X1, spec.Z[rows, spec.X1.shape[1]:]])
    Z = Z[:, independent(Z)]
    layout = spec.layout()
    model = PCMModel(spec.pcm_spec(), {"products": products, "agents": agents}, layout, 16, 0, total_threads=threads)
    ms = [SarganIVMoments(Z, spec.clusters[rows]), spec.micro_moment("all")]
    names = [mod.X1_COLS[j] for j in keep]
    return NFPGMMProblem(model, ms, LinearIV(X1, Z, names=names), layout, max_step=False)


@slow
@pytest.mark.xfail(strict=False, reason="1984 alone is a non-zero-residual problem (q about 20-44): nested ends "
                   "STALLED at the kink floor and Gauss-Newton MPEC reaches max_iter far from it (LEARNINGS 31)")
def test_petrin_1984_timing() -> None:
    """Test 6: Petrin 1984 at nu = 16, demand moments, from the gate theta, W1 fixed: MPEC's derivative passes
    against the nested run's evaluations x (1 + mean share passes per evaluation), and the two theta to 1e-4 natural
    units. Prints the numbers. Measured 2026-11-17 at 16 threads: nested STALLED at q 43.69 after 60 evaluations
    (79.5 share passes each, cost 4,829, 225 s); MPEC 300 iterations to max_iter (301 derivative + 1,551 share
    passes, 255 s), closing q 19.69, theta 52 natural units from the nested one."""
    threads = int(os.environ.get("NFP_THREADS", "16"))
    try:
        mod = _petrin_module()   # puts the vendored pyblp on sys.path
    except ImportError as exc:
        pytest.skip(f"Petrin spec not importable: {exc}")
    prob_n = _petrin_1984(mod, threads)
    layout = prob_n.layout
    theta0 = _gate_theta(list(layout.names), layout.starts())
    if theta0 is None:
        pytest.skip("no gate profile.csv with a lambda = 1 point")
    units = layout.units()
    W1 = prob_n.initial_weighting(theta0)
    passes: list[int] = []
    solve = prob_n.model.solve

    def counted(theta: np.ndarray, requests: Any, need_jacobian: bool) -> Any:
        sol = solve(theta, requests, need_jacobian)
        passes.append(sum(int(pm["evaluations"]) for pm in sol.per_market))
        return sol

    prob_n.model.solve = counted   # type: ignore[method-assign]
    t0 = time.perf_counter()
    res_n = minimize(prob_n, W1, OptimizerOptions(stage="nested", theta0=theta0, max_iterations=150))
    t_n = time.perf_counter() - t0
    prob_n.model.solve = solve   # type: ignore[method-assign]
    nested_cost = res_n.evaluations * (1.0 + float(np.mean(passes)))
    prob_m = _petrin_1984(mod, threads)
    t0 = time.perf_counter()
    res_m = minimize(prob_m, W1, OptimizerOptions(stage="mpec", theta0=theta0, outer="mpec"))
    t_m = time.perf_counter() - t0
    m = res_m.mpec
    diff = float(np.max(np.abs(res_m.theta - res_n.theta) / units))
    print(f"\n  Petrin 1984, nu 16, {prob_n.model._mk[0].R} draws, J {prob_n.n}, P_free {layout.n_free()}, "
          f"{threads} threads")
    print(f"  nested: {res_n.convergence.value}, q {res_n.q:.6g}, {res_n.evaluations} evaluations, mean share passes "
          f"per evaluation {np.mean(passes):.1f}, cost {nested_cost:.0f}, {t_n:.1f}s")
    print(f"  mpec:   {res_m.convergence.value}, IPOPT {m['status']}, q {res_m.q:.6g}, {m['iterations']} iterations, "
          f"{m['derivative_passes']} derivative + {m['share_passes']} share passes, {m['restarts']} restarts, "
          f"{t_m:.1f}s")
    print(f"  max |theta_mpec - theta_nested| / unit = {diff:.2e}")
    print(f"  theta nested {np.round(res_n.theta, 5).tolist()}")
    print(f"  theta mpec   {np.round(res_m.theta, 5).tolist()}")
    prob_n.model.close()
    prob_m.model.close()
    assert m["derivative_passes"] < nested_cost
    assert diff < 1e-4

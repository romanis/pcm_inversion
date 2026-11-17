"""Tests for profile (WP9a; DESIGN 6.4, 7 WP9a acceptance).

Toy problems are ``NFPGMMProblem``s over pure-numpy inner models:

- ``LinearToy``: delta(theta) = d0 + B (theta - theta_star) with M d0 orthogonal to Z, so g is linear
  in theta, g(theta_star) = 0, and the concentrated profile has a closed form;
- ``ScaleToy``: delta(theta) = exp(-theta) (X1 beta + e). Under a fixed W, q falls as exp(-2 theta)
  without end, while the scale-free CUE objective does not move: the planted fixed-W scale artifact
  of LEARNINGS 1.

The kill-and-resume test runs this file as a child process (``python test_profile.py child ...``).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from nfp_harness.linear import LinearIV  # noqa: E402
from nfp_harness.moments import IVMoments  # noqa: E402
from nfp_harness.params import ParameterLayout  # noqa: E402
from nfp_harness.problem import NFPGMMProblem  # noqa: E402
from nfp_harness.profile import (  # noqa: E402
    NotMonotoneDecreasing,
    ParamAxis,
    ProfileMismatchError,
    ProfileShape,
    RayAxis,
    classify_profile,
    grid_profile,
    profile,
    ray_profile,
)
from nfp_harness.types import EvalStatus, InnerSolution, KernelRequests, MarketIndex, ParameterSpec  # noqa: E402

PY = sys.executable


# ------------------------------------------------------------------------------------------------
# toy problems
# ------------------------------------------------------------------------------------------------


class _ToyModel:
    """One market holding every product; subclasses define delta(theta) and D(theta)."""

    def __init__(self, n: int, sleep: float = 0.0) -> None:
        self.n_products = n
        self.markets = [MarketIndex(0, np.arange(n))]
        self.sleep = sleep
        self.calls = 0
        self.fail_above: tuple[int, float] | None = None  # (parameter, value): status INVERSION_FAILED above
        self.raise_after: int | None = None  # raise RuntimeError on this call (simulates a kill)
        self.accepted: np.ndarray | None = None
        self.trial: np.ndarray | None = None

    def delta_D(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        self.calls += 1
        if self.raise_after is not None and self.calls > self.raise_after:
            raise RuntimeError("killed")
        if self.sleep:
            time.sleep(self.sleep)
        if self.fail_above is not None and theta[self.fail_above[0]] > self.fail_above[1]:
            return InnerSolution(np.full(self.n_products, np.nan), None, EvalStatus.INVERSION_FAILED, [{}], {})
        delta, D = self.delta_D(np.asarray(theta, dtype=float))
        self.trial = np.array(theta)
        return InnerSolution(delta, [D] if need_jacobian else None, EvalStatus.OK, [{"converged": True}], {})

    def commit(self, theta: np.ndarray) -> None:
        self.accepted = None if self.trial is None else self.trial.copy()

    def restore(self, theta: np.ndarray) -> None:
        self.trial = None

    def state_dict(self) -> dict[str, Any]:
        return {"accepted": np.zeros(0) if self.accepted is None else self.accepted.copy()}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        a = np.asarray(d["accepted"], dtype=float)
        self.accepted = None if a.size == 0 else a.copy()


class LinearToy(_ToyModel):
    def __init__(self, d0: np.ndarray, B: np.ndarray, theta_star: np.ndarray, **kw: Any) -> None:
        super().__init__(d0.size, **kw)
        self.d0, self.B, self.theta_star = d0, B, theta_star

    def delta_D(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.d0 + self.B @ (theta - self.theta_star), self.B.copy()


class ScaleToy(_ToyModel):
    def __init__(self, base: np.ndarray, **kw: Any) -> None:
        super().__init__(base.size, **kw)
        self.base = base

    def delta_D(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = math.exp(-theta[0]) * self.base
        return d, -d[:, None]


THETA_STAR = np.array([1.0, -0.5, 2.0])


def _design(n: int = 60, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X1 = np.column_stack([np.ones(n), x])
    Z = np.column_stack([X1, rng.normal(size=(n, 4)) + 0.5 * x[:, None]])
    return X1, Z, rng


def linear_toy(sleep: float = 0.0) -> NFPGMMProblem:
    X1, Z, rng = _design()
    e = rng.normal(size=X1.shape[0])
    xi0 = e - Z @ np.linalg.lstsq(Z, e, rcond=None)[0]  # orthogonal to Z in sample: g(theta_star) = 0
    d0 = X1 @ np.array([1.0, 0.5]) + xi0
    B = rng.normal(size=(X1.shape[0], 3))
    layout = ParameterLayout([ParameterSpec(f"t{k}", "taste", None, None, float(v), -50.0, 50.0, unit=1.0)
                              for k, v in enumerate(THETA_STAR)])
    model = LinearToy(d0, B, THETA_STAR.copy(), sleep=sleep)
    return NFPGMMProblem(model, [IVMoments(Z)], LinearIV(X1, Z), layout)


def scale_toy() -> NFPGMMProblem:
    X1, Z, rng = _design(seed=5)
    base = X1 @ np.array([2.0, 1.0]) + rng.normal(size=X1.shape[0]) + 0.3 * Z[:, 3]  # xi correlated with Z
    layout = ParameterLayout([ParameterSpec("sigma_p", "sigma_p", None, None, 1.0, 0.05, 10.0, unit=0.1)])
    return NFPGMMProblem(ScaleToy(base), [IVMoments(Z)], LinearIV(X1, Z), layout)


def _direct_q(prob: NFPGMMProblem, theta: np.ndarray) -> tuple[float, float]:
    ev = prob.evaluate(theta, need_jacobian=False)
    q_fixed = float(prob.n * ev.g @ prob.W @ ev.g)
    q_cue = float(prob.n * ev.g @ np.linalg.solve(prob.covariance(ev), ev.g))
    return q_fixed, q_cue


# ------------------------------------------------------------------------------------------------
# classifier
# ------------------------------------------------------------------------------------------------


def test_classifier_shapes() -> None:
    x = [0.5, 1.0, 1.5, 2.0, 2.5]
    v = classify_profile(x, [5.0, 3.0, 2.0, 1.5, 1.2])
    assert v.shape is ProfileShape.MONOTONE_DECREASING and v.strict and v.x_min == 2.5
    assert v.rise_left == pytest.approx(3.8) and v.rise_right == 0.0
    v = classify_profile(x, [1.0, 2.0, 2.5, 3.0, 7.0])
    assert v.shape is ProfileShape.MONOTONE_INCREASING and v.strict and v.x_min == 0.5
    v = classify_profile(x, [4.0, 1.0, 0.5, 2.0, 6.0])
    assert v.shape is ProfileShape.INTERIOR_MINIMUM and v.x_min == 1.5 and v.strict
    assert v.rise_left == pytest.approx(3.5) and v.rise_right == pytest.approx(5.5)
    assert "INTERIOR_MINIMUM at x = 1.5" in v.message
    v = classify_profile(x, [2.0, 2.0 + 1e-9, 2.0, 2.0 - 1e-9, 2.0])
    assert v.shape is ProfileShape.FLAT
    # a bump does not hide a minimum at the right end; strict says the fall is not monotone
    v = classify_profile(x, [5.0, 3.0, 4.0, 2.0, 1.0])
    assert v.shape is ProfileShape.MONOTONE_DECREASING and not v.strict
    assert v.evidence["steps_up"] == 1


def test_classifier_unsorted_failed_and_short() -> None:
    v = classify_profile([2.0, 0.5, 1.0, 1.5], [3.0, 2.0, 1.0, np.nan], axis="sigma_p")
    assert v.shape is ProfileShape.INTERIOR_MINIMUM and v.x_min == 1.0
    assert v.evidence["failed_x"] == [1.5] and v.evidence["n_failed"] == 1 and "sigma_p = [1.5]" in v.message
    v = classify_profile([1.0, 2.0, 3.0], [1.0, np.nan, np.nan])
    assert v.shape is ProfileShape.FLAT and v.evidence["insufficient_points"]
    with pytest.raises(ValueError):
        classify_profile([1.0, 2.0], [1.0])


# ------------------------------------------------------------------------------------------------
# DESIGN WP9a: the profile matches direct evaluations
# ------------------------------------------------------------------------------------------------


def test_fixed_profile_matches_direct_evaluations() -> None:
    prob = linear_toy()
    values = [0.0, 0.5, 1.0, 1.5, 2.5]
    res = grid_profile(prob, THETA_STAR, {"t0": values})
    assert [p.status for p in res.points] == ["ok"] * 5
    assert res.columns[:3] == ["q_fixed", "q_cue", "q_fixed:aggregate"] and "sd_xi" in res.columns
    for p, v in zip(res.points, values):
        th = THETA_STAR.copy()
        th[0] = v
        np.testing.assert_array_equal(p.theta, th)
        qf, qc = _direct_q(prob, th)
        assert p.values["q_fixed"] == qf and p.values["q_fixed:aggregate"] == qf
        assert p.values["q_cue"] == pytest.approx(qc, rel=1e-12)
        assert p.values["sd_xi"] == pytest.approx(np.std(prob.evaluate(th, False).aux["xi"]), rel=1e-12)
    v = res.classify("t0", "q_fixed")[()]
    assert v.shape is ProfileShape.INTERIOR_MINIMUM and v.x_min == 1.0 and v.q_min < 1e-20


def test_ray_profile_through_theta0_and_from_zero() -> None:
    prob = linear_toy()
    d = {"t0": 1.0, "t2": -2.0}
    res = ray_profile(prob, THETA_STAR, d, [-0.5, 0.0, 0.5], modes=("fixed",))
    for p, lam in zip(res.points, [-0.5, 0.0, 0.5]):
        np.testing.assert_allclose(p.theta, THETA_STAR + lam * np.array([1.0, 0.0, -2.0]), atol=0)
        assert p.values["q_fixed"] == _direct_q(prob, p.theta)[0]
    assert "q_cue" not in res.columns
    assert res.classify("lambda")[()].shape is ProfileShape.INTERIOR_MINIMUM
    # origin 0: theta[ray] = lambda d (DESIGN 6.4 sigma = lambda x SIG_REF), crossed with a grid, grid fastest
    res = ray_profile(prob, THETA_STAR, d, [0.5, 1.0], {"t1": [-1.0, 0.0]}, origin=0.0, modes=("fixed",))
    got = [(p.coords["lambda"], p.coords["t1"], *p.theta) for p in res.points]
    assert got == [(0.5, -1.0, 0.5, -1.0, -1.0), (0.5, 0.0, 0.5, 0.0, -1.0),
                   (1.0, -1.0, 1.0, -1.0, -2.0), (1.0, 0.0, 1.0, 0.0, -2.0)]
    np.testing.assert_array_equal(res.column("t2"), [-1.0, -1.0, -2.0, -2.0])


def test_concentrated_profile_matches_closed_form() -> None:
    """Re-optimising t1, t2 at each t0 gives q_conc(v) = (v - t0*)^2 (H00 - H0r Hrr^{-1} Hr0)."""
    prob = linear_toy()
    ev = prob.evaluate(THETA_STAR)
    H = prob.n * ev.G.T @ prob.W @ ev.G
    schur = H[0, 0] - H[0, 1:] @ np.linalg.solve(H[1:, 1:], H[1:, 0])
    values = [-1.0, 0.0, 1.0, 2.0, 3.0]
    for cold in (True, False):
        res = grid_profile(prob, THETA_STAR + np.array([0.0, 0.7, -0.4]), {"t0": values}, concentrate=True,
                           cold=cold, modes=("fixed",))
        for p, v in zip(res.points, values):
            assert p.convergence == "CONVERGED"
            assert p.values["q_fixed"] == pytest.approx((v - 1.0) ** 2 * schur, rel=1e-8, abs=1e-12)
            assert p.theta[0] == v
            # the optimum of the free parameters given t0 = v
            r = -np.linalg.solve(H[1:, 1:], H[1:, 0] * (v - 1.0))
            np.testing.assert_allclose(p.theta[1:], THETA_STAR[1:] + r, atol=1e-6)
        assert res.classify("t0")[()].shape is ProfileShape.INTERIOR_MINIMUM
    # hold keeps a parameter fixed at theta0 in the concentrated profile
    res = grid_profile(prob, THETA_STAR, {"t0": [0.0, 1.0, 2.0]}, concentrate=True, hold=("t2",), modes=("fixed",))
    assert all(p.theta[2] == THETA_STAR[2] for p in res.points)


def test_concentrated_cue_objective() -> None:
    prob = linear_toy()
    res = grid_profile(prob, THETA_STAR, {"t0": [0.5, 1.0, 1.5]}, concentrate=True, objective="cue")
    assert all(p.convergence in ("CONVERGED", "CONVERGED_AT_CAP") for p in res.points)
    assert res.points[1].values["q_cue"] < 1e-12
    assert all(p.values["q_cue"] > 1e-3 for p in (res.points[0], res.points[2]))


def test_two_dimensional_grid_and_pass_rule() -> None:
    prob = linear_toy()
    res = grid_profile(prob, THETA_STAR, {"t1": [-1.0, -0.5, 0.0], "t0": [0.0, 1.0, 2.0, 3.0]}, modes=("fixed",),
                       pass_rule=NotMonotoneDecreasing("t0", by=("t1",), rise=0.0))
    assert len(res.points) == 12 and [p.index for p in res.points] == list(range(12))
    verdicts = res.classify("t0", by=("t1",))
    assert set(verdicts) == {(-1.0,), (-0.5,), (0.0,)}
    assert verdicts[(-0.5,)].shape is ProfileShape.INTERIOR_MINIMUM and verdicts[(-0.5,)].x_min == 1.0
    assert res.verdict is not None and res.verdict[0]
    assert "q_fixed (t1 = -0.5) along t0 has an INTERIOR_MINIMUM at t0 = 1" in res.table()


def test_planted_scale_artifact_is_monotone_decreasing(tmp_path: pathlib.Path) -> None:
    """LEARNINGS 1: under a fixed W, q falls with the utility scale (here exp(-2 sigma_p)) without
    end, while the scale-free CUE objective does not move. The gate fails on q_fixed."""
    prob = scale_toy()
    sps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    rule = NotMonotoneDecreasing("sigma_p", fail_message="Profile check failed: q is still monotone in {along}.")
    res = grid_profile(prob, None, {"sigma_p": sps}, run_dir=tmp_path / "scale", pass_rule=rule)
    q = res.column("q_fixed")
    np.testing.assert_allclose(q[1:] / q[:-1], math.exp(-1.0), rtol=1e-10)
    np.testing.assert_allclose(res.column("sd_xi")[1:] / res.column("sd_xi")[:-1], math.exp(-0.5), rtol=1e-10)
    v = res.classify("sigma_p", "q_fixed")[()]
    assert v.shape is ProfileShape.MONOTONE_DECREASING and v.strict and v.x_min == 3.5
    assert res.classify("sigma_p", "q_cue")[()].shape is ProfileShape.FLAT
    assert res.verdict is not None and not res.verdict[0]
    assert res.verdict[1].startswith("Profile check failed: q is still monotone in sigma_p.")
    text = (tmp_path / "scale" / "profile.txt").read_text()
    assert "FAIL: Profile check failed" in text and "MONOTONE_DECREASING" in text and "is FLAT" in text


# ------------------------------------------------------------------------------------------------
# failures, economics callback, run directory and resume
# ------------------------------------------------------------------------------------------------


def test_failed_points_and_economics_callback() -> None:
    prob = linear_toy()
    prob.model.fail_above = (0, 2.2)

    def econ(problem: Any, ev: Any) -> dict[str, float]:
        if ev.theta[0] == 0.0:
            raise ValueError("no elasticity here")
        return {"median_own_elasticity": -float(ev.theta[0]), "inside_share": 0.1}

    res = grid_profile(prob, THETA_STAR, {"t0": [0.0, 0.5, 1.0, 2.0, 2.5]}, economics=econ)
    assert [p.status for p in res.points] == ["ok"] * 4 + ["inversion_failed"]
    assert np.isnan(res.points[-1].values.get("q_fixed", np.nan))
    assert res.points[1].values["median_own_elasticity"] == -0.5
    assert "economics callback failed: ValueError" in res.points[0].note and "q_fixed" in res.points[0].values
    assert "median_own_elasticity" in res.columns and "inside_share" in res.columns
    v = res.classify("t0")[()]
    assert v.shape is ProfileShape.INTERIOR_MINIMUM and v.evidence["failed_x"] == [2.5]


def test_run_directory_resume_and_mismatch(tmp_path: pathlib.Path) -> None:
    prob = linear_toy()
    rd = tmp_path / "prof"
    grid = {"t0": [0.0, 0.5, 1.0, 1.5]}
    first = grid_profile(prob, THETA_STAR, grid, run_dir=rd)
    assert sorted(p.name for p in (rd / "points").glob("*.json")) == [f"p{i:05d}.json" for i in range(4)]
    header = (rd / "profile.csv").read_text().splitlines()[0].split(",")
    assert header[:4] == ["index", "t0", "status", "q_fixed"] and "theta:t2" in header
    calls = prob.model.calls
    again = grid_profile(prob, THETA_STAR, grid, run_dir=rd)
    assert again.resumed_points == 4 and prob.model.calls == calls  # nothing recomputed
    for a, b in zip(first.points, again.points):
        assert a.values == b.values and np.array_equal(a.theta, b.theta)
    with pytest.raises(ProfileMismatchError):
        grid_profile(prob, THETA_STAR, {"t0": [0.0, 0.5, 1.0, 2.0]}, run_dir=rd)
    fresh = grid_profile(prob, THETA_STAR, {"t0": [0.0, 0.5, 1.0, 2.0]}, run_dir=rd, new_run=True)
    assert fresh.resumed_points == 0
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("prof.old-")]


def test_chained_profile_resumes_after_crash(tmp_path: pathlib.Path) -> None:
    """A chained concentrated profile that dies mid-way resumes from its saved points and warm state
    and ends where the uninterrupted run ends."""
    grid = {"t0": [-1.0, 0.0, 1.0, 2.0, 3.0]}
    kw = dict(concentrate=True, cold=False, modes=("fixed",))
    want = grid_profile(linear_toy(), THETA_STAR + 0.3, grid, run_dir=tmp_path / "ref", **kw)

    prob = linear_toy()
    prob.model.raise_after = 12
    with pytest.raises(RuntimeError, match="killed"):
        grid_profile(prob, THETA_STAR + 0.3, grid, run_dir=tmp_path / "run", **kw)
    saved = sorted((tmp_path / "run" / "points").glob("p*.json"))
    assert 0 < len(saved) < 5 and len(list((tmp_path / "run" / "points").glob("p*.state.npz"))) == len(saved)
    prob = linear_toy()
    got = grid_profile(prob, THETA_STAR + 0.3, grid, run_dir=tmp_path / "run", **kw)
    assert got.resumed_points == len(saved)
    assert prob.model.accepted is not None
    for a, b in zip(want.points, got.points):
        assert b.values["q_fixed"] == pytest.approx(a.values["q_fixed"], rel=1e-9, abs=1e-14)
        np.testing.assert_allclose(b.theta, a.theta, atol=1e-8)


def test_cold_profile_leaves_problem_state_and_weighting() -> None:
    prob = linear_toy()
    prob.evaluate(THETA_STAR)
    prob.commit(THETA_STAR)
    W0 = prob.W.copy()
    W = 2.0 * prob.W
    res = grid_profile(prob, THETA_STAR, {"t0": [0.0, 2.0]}, weighting=W)
    np.testing.assert_array_equal(prob.model.accepted, THETA_STAR)
    np.testing.assert_array_equal(prob.W, W0)
    assert res.points[0].values["q_fixed"] == pytest.approx(2.0 * _direct_q(prob, res.points[0].theta)[0], rel=1e-12)


# ------------------------------------------------------------------------------------------------
# parallel workers
# ------------------------------------------------------------------------------------------------


class _Spare:
    """A problem wrapper whose model reports spare capacity for profile points."""

    def __init__(self, problem: NFPGMMProblem, capacity: int) -> None:
        self._p = problem
        self.capacity = capacity

    def spare_capacity(self) -> int:
        return self.capacity

    def __getattr__(self, name: str) -> Any:
        return getattr(self._p, name)


def test_parallel_workers_match_sequential() -> None:
    grid = {"t0": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], "t1": [-1.0, 0.0]}
    seq = grid_profile(linear_toy(), THETA_STAR, grid)
    assert seq.workers == 1
    made: list[tuple[int, int]] = []

    def factory(worker: int, threads: int) -> NFPGMMProblem:
        made.append((worker, threads))
        return linear_toy(sleep=0.05)

    t0 = time.perf_counter()
    par = grid_profile(linear_toy(), THETA_STAR, grid, workers=4, problem_factory=factory)
    elapsed = time.perf_counter() - t0
    assert par.workers == 4 and sorted(w for w, _ in made) == [0, 1, 2, 3]
    assert all(t == max(1, (os.cpu_count() or 1) // 4) for _, t in made)
    assert elapsed < 12 * 0.05 * 0.75  # 12 points x 0.05 s per solve, spread over 4 workers
    for a, b in zip(seq.points, par.points):
        assert a.index == b.index and a.coords == b.coords
        assert a.values == pytest.approx(b.values, rel=1e-12)
    # spare capacity reported by the problem picks the worker count; without it, points run one by one
    auto = grid_profile(_Spare(linear_toy(), 3), THETA_STAR, grid, problem_factory=factory)
    assert auto.workers == 3
    assert grid_profile(_Spare(linear_toy(), 3), THETA_STAR, grid).workers == 1  # no factory: sequential
    with pytest.raises(ValueError, match="problem_factory"):
        grid_profile(linear_toy(), THETA_STAR, grid, workers=2)
    # a chained profile never runs in parallel
    assert grid_profile(linear_toy(), THETA_STAR, grid, workers=4, problem_factory=factory, cold=False).workers == 1


def test_axis_validation() -> None:
    prob = linear_toy()
    with pytest.raises(KeyError):
        grid_profile(prob, THETA_STAR, {"nope": [1.0]})
    with pytest.raises(ValueError, match="two axes"):
        profile(prob, THETA_STAR, [ParamAxis("t0", [1.0]), RayAxis({"t0": 1.0}, [0.0])])
    with pytest.raises(ValueError):
        RayAxis({"t0": 1.0}, [])


# ------------------------------------------------------------------------------------------------
# kill and resume (subprocess)
# ------------------------------------------------------------------------------------------------

KILL_GRID = {"t0": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]}


def _child(run_dir: str, sleep: float) -> None:
    prob = linear_toy(sleep=sleep)
    res = grid_profile(prob, THETA_STAR, KILL_GRID, run_dir=run_dir)
    print(f"RESUMED {res.resumed_points} SOLVES {prob.model.calls}", flush=True)
    out = {"theta": [[float(v).hex() for v in p.theta] for p in res.points],
           "q": [float(p.values["q_fixed"]).hex() for p in res.points]}
    (pathlib.Path(run_dir) / "result.json").write_text(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        _child(sys.argv[2], float(sys.argv[3]))
    sys.exit(0)


def test_killed_profile_resumes(tmp_path: pathlib.Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    script = str(pathlib.Path(__file__).resolve())
    ref = tmp_path / "ref"
    out = subprocess.run([PY, script, "child", str(ref), "0.0"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    want = json.loads((ref / "result.json").read_text())

    run = tmp_path / "run"
    proc = subprocess.Popen([PY, script, "child", str(run), "0.3"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and len(list((run / "points").glob("p*.json"))) < 3:
        time.sleep(0.02)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL
    n_saved = len(list((run / "points").glob("p*.json")))
    assert 3 <= n_saved < len(KILL_GRID["t0"]) and not (run / "result.json").exists()
    assert not list((run / "points").glob("*.tmp"))

    again = subprocess.run([PY, script, "child", str(run), "0.0"], env=env, capture_output=True, text=True,
                           timeout=120)
    assert again.returncode == 0, again.stderr
    left = len(KILL_GRID["t0"]) - n_saved
    assert f"RESUMED {n_saved} SOLVES {left}" in again.stdout  # one solve per remaining point
    got = json.loads((run / "result.json").read_text())
    assert got == want  # bit for bit
    rows = (run / "profile.csv").read_text().splitlines()
    assert len(rows) == 1 + len(KILL_GRID["t0"])


# ------------------------------------------------------------------------------------------------
# PCM: sigma_p ray profile on the WP7 synthetic data, with and without micro moments
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pcm_medium() -> dict[str, Any]:
    pytest.importorskip("nfp_harness._native")
    from nfp_harness.tests.test_pcm_model import make_synthetic

    return make_synthetic(3, 44, 128, 8, seed=6)


def _pcm_problem(syn: dict[str, Any], micro: bool, threads: int = 16) -> NFPGMMProblem:
    from nfp_harness.models.pcm import PCMModel, PCMSpec
    from nfp_harness.tests.test_pcm_model import CHARS, THETA_TRUE, build_problem, synthetic_layout

    if micro:
        return build_problem(syn, THETA_TRUE, threads=threads)
    layout = synthetic_layout(THETA_TRUE)
    model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=threads)
    return NFPGMMProblem(model, [IVMoments(syn["Z"])], LinearIV(syn["X1"], syn["Z"]), layout)


@pytest.mark.parametrize("micro", [True, False], ids=["micro", "no_micro"])
def test_pcm_sigma_p_ray_has_interior_minimum_at_truth(pcm_medium: dict[str, Any], micro: bool,
                                                        tmp_path: pathlib.Path) -> None:
    """The data were generated at theta_true, so g(theta_true) = 0 and the sigma_p ray through it has
    its minimum at the true sigma_p = 1, with q rising on both sides."""
    from nfp_harness.tests.test_pcm_model import THETA_TRUE

    prob = _pcm_problem(pcm_medium, micro)
    W1 = prob.initial_weighting(THETA_TRUE)
    lambdas = [-0.4, -0.2, 0.0, 0.25, 0.5, 1.0]  # sigma_p = 0.6 ... 2.0
    rule = NotMonotoneDecreasing("lambda")
    res = ray_profile(prob, THETA_TRUE, {"sigma_p": 1.0}, lambdas, weighting=W1, run_dir=tmp_path / "ray",
                      pass_rule=rule)
    print("\n" + res.table())
    assert all(p.status == "ok" for p in res.points)
    np.testing.assert_allclose(res.column("sigma_p"), 1.0 + np.array(lambdas))
    v = res.classify("lambda", "q_fixed")[()]
    assert v.shape is ProfileShape.INTERIOR_MINIMUM and v.x_min == 0.0 and v.q_min < 1e-12
    assert res.verdict is not None and res.verdict[0]
    assert ("q_fixed:micro" in res.columns) == micro
    assert (tmp_path / "ray" / "profile.csv").exists() and (tmp_path / "ray" / "profile.txt").exists()
    prob.model.close()


def test_pcm_concentrated_sigma_p_profile_small() -> None:
    """Concentrated sigma_p profile on the WP7 2-market fixture: the other four parameters are
    re-optimised at every sigma_p, warm-started along the grid; minimum at the truth."""
    pytest.importorskip("nfp_harness._native")
    from nfp_harness.optimizer import OptimizerOptions
    from nfp_harness.tests.test_pcm_model import THETA_TRUE, build_problem, make_synthetic

    syn = make_synthetic(2, 8, 32, 8, seed=2)
    prob = build_problem(syn, THETA_TRUE, threads=8)
    W1 = prob.initial_weighting(THETA_TRUE)
    res = grid_profile(prob, THETA_TRUE, {"sigma_p": [0.8, 1.0, 1.25]}, weighting=W1, concentrate=True, cold=False,
                       optimizer_options=OptimizerOptions(max_iterations=60, record_cue=False))
    print("\n" + res.table())
    assert all(p.status == "ok" for p in res.points)
    q = res.column("q_fixed")
    assert q[1] < 1e-10 and q[0] > 1e3 * max(q[1], 1e-14) and q[2] > 1e3 * max(q[1], 1e-14)
    assert res.classify("sigma_p")[()].shape is ProfileShape.INTERIOR_MINIMUM
    # concentrating can only lower q relative to holding the other parameters at the truth
    fixed = grid_profile(prob, THETA_TRUE, {"sigma_p": [0.8, 1.0, 1.25]}, weighting=W1, modes=("fixed",))
    assert np.all(q <= fixed.column("q_fixed") * (1 + 1e-9) + 1e-9)  # the optimizer stops near, not at, q = 0
    prob.model.close()


# ------------------------------------------------------------------------------------------------
# DESIGN WP9a acceptance on Petrin: nu_per_agent = 1, a 2 x 2 grid runs and writes the CSV
# ------------------------------------------------------------------------------------------------


def test_petrin_nu1_two_by_two_grid(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("nfp_harness._native")
    from nfp_harness.models.pcm import PCMModel, PCMSpec
    from nfp_harness.tests.test_pcm_model import _petrin_frames, _petrin_layout

    products, agents = _petrin_frames()
    chars = ("const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")
    x1 = ["hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv", "pgnp", "trend", "trend2"]
    X1 = np.column_stack([np.ones(len(products))] + [products[c].to_numpy(float) for c in x1])
    Z = np.column_stack([X1] + [products[f"demand_instruments{k}"].to_numpy(float) for k in range(22)])
    clusters = products.clustering_ids.to_numpy()

    def factory(worker: int, threads: int) -> NFPGMMProblem:
        layout = _petrin_layout()
        model = PCMModel(PCMSpec(chars), {"products": products, "agents": agents}, layout, 1, 0,
                         total_threads=threads)
        return NFPGMMProblem(model, [IVMoments(Z, clusters)], LinearIV(X1, Z), layout)

    prob = factory(0, 1)
    assert prob.identification["effective_moments"] == 35 and prob.identification["unknowns"] == 31
    sig_ref = {"sigma_const": 2.0, "sigma_hpwt": 2.0, "sigma_space": 2.0, "sigma_air": 1.0, "sigma_mpd": 1.0,
               "sigma_fwd": 2.0, "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0}
    res = ray_profile(prob, None, sig_ref, [0.5, 1.0], {"sigma_p": [1.0, 1.5]}, origin=0.0,
                      run_dir=tmp_path / "petrin", workers=4, problem_factory=factory)
    print("\n" + res.table())
    prob.model.close()
    assert res.workers == 4 and len(res.points) == 4
    assert all(p.status == "ok" for p in res.points), [(p.coords, p.status, p.note) for p in res.points]
    assert all(np.isfinite(p.values["q_fixed"]) and np.isfinite(p.values["q_cue"]) for p in res.points)
    rows = (tmp_path / "petrin" / "profile.csv").read_text().splitlines()
    assert len(rows) == 5 and rows[0].startswith("index,lambda,sigma_p,status,q_fixed,q_cue")

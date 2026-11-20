"""Tests for the optimizer (WP5; DESIGN 4.8 tests 1-7, 9-10). Test 8 (kill and resume) is in test_checkpoint.py.

Every problem here is a pure-numpy toy that implements ``types.Problem``; nothing depends on the kernel,
``params.py`` or ``problem.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pytest

from nfp_harness.diagnostics.convergence import classify, convergence_block
from nfp_harness.optimizer import OptimizerOptions, cue_gradient, cue_objective, minimize, projected_gradient
from nfp_harness.types import ConvergenceClass, EvalStatus, Evaluation, HistoryRecord, ParameterSpec, StepLimit

INF = float("inf")


# ------------------------------------------------------------------------------------------------
# toy problems
# ------------------------------------------------------------------------------------------------


class ToyLayout:
    """Duck-typed ParameterLayout: specs, names, P, units(data)."""

    def __init__(self, specs: list[ParameterSpec]) -> None:
        self.specs = list(specs)
        self.names = [s.name for s in specs]
        self.P = len(specs)

    def units(self, data: Any) -> np.ndarray:
        return np.array([1.0 if s.unit is None else s.unit for s in self.specs])


def spec(name: str, start: float, lower: float = -INF, upper: float = INF, **kw: Any) -> ParameterSpec:
    return ParameterSpec(name=name, role="toy", column=None, basis=None, start=start, lower=lower, upper=upper, **kw)


class ToyProblem:
    """g(theta), G(theta) from ``fn``; optional failing region, max_step oracle, covariance and CUE terms.

    Keeps an accepted state and a trial state to check the commit/restore protocol.
    """

    def __init__(self, fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], specs: list[ParameterSpec],
                 n: float = 1.0, sleep: float = 0.0, fails: Callable[[np.ndarray], bool] | None = None,
                 oracle: Callable[[np.ndarray, np.ndarray], StepLimit] | None = None,
                 cov: Callable[[np.ndarray], np.ndarray] | None = None,
                 cue: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None) -> None:
        self.fn, self.layout, self.n, self.sleep, self.fails = fn, ToyLayout(specs), n, sleep, fails
        self._oracle, self._cov, self._cue = oracle, cov, cue
        self.accepted: np.ndarray | None = None
        self.trial: np.ndarray | None = None
        self.commits = self.restores = self.evals = 0
        self.W: np.ndarray | None = None

    def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation:
        self.evals += 1
        if self.sleep:
            time.sleep(self.sleep)
        theta = np.array(theta, dtype=float)
        self.trial = theta.copy()
        if self.fails is not None and self.fails(theta):
            return Evaluation(theta=theta, g=np.full(1, np.nan), G=None, status=EvalStatus.INVERSION_FAILED,
                              blocks={}, aux={}, seconds=0.0)
        g, G = self.fn(theta)
        return Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK, blocks={"all": slice(0, len(g))}, aux={},
                          seconds=0.0)

    def covariance(self, ev: Evaluation) -> np.ndarray:
        if self._cov is None:
            raise NotImplementedError
        return self._cov(ev.theta)

    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]:
        if self._cue is None:
            raise NotImplementedError
        return self._cue(ev.theta)

    def set_weighting(self, W: np.ndarray) -> None:
        self.W = np.array(W)

    def max_step(self, ev: Evaluation, dtheta: np.ndarray) -> StepLimit | None:
        return None if self._oracle is None else self._oracle(ev.theta, dtheta)

    def commit(self, theta: np.ndarray) -> None:
        assert self.trial is not None and np.array_equal(self.trial, theta), "commit of a theta that was not the trial"
        self.accepted, self.trial = self.trial, None
        self.commits += 1

    def restore(self, theta: np.ndarray) -> None:
        self.trial = None
        self.restores += 1

    def state_dict(self) -> dict[str, Any]:
        return {"accepted": None if self.accepted is None else self.accepted.copy()}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.accepted = d["accepted"]


def rosenbrock(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = theta
    return np.array([10.0 * (y - x * x), 1.0 - x]), np.array([[-20.0 * x, 10.0], [-1.0, 0.0]])


def rosenbrock_problem(**kw: Any) -> ToyProblem:
    return ToyProblem(rosenbrock, [spec("x", -1.2), spec("y", 1.0)], **kw)


EYE2 = np.eye(2)


# ------------------------------------------------------------------------------------------------
# 1. linear least squares
# ------------------------------------------------------------------------------------------------


def test_linear_least_squares_one_step() -> None:
    rng = np.random.default_rng(0)
    A, _ = np.linalg.qr(rng.standard_normal((6, 3)))
    b = rng.standard_normal(6)
    theta_star = np.linalg.lstsq(A, b, rcond=None)[0]
    p = ToyProblem(lambda t: (A @ t - b, A), [spec(f"t{i}", 0.0) for i in range(3)])
    res = minimize(p, np.eye(6), OptimizerOptions(Delta_0=10.0, Delta_max=10.0))
    assert res.convergence is ConvergenceClass.CONVERGED
    assert res.accepted_steps == 1
    np.testing.assert_allclose(res.theta, theta_star, rtol=0, atol=1e-10)
    assert res.q == pytest.approx(float(np.sum((A @ theta_star - b) ** 2)), rel=1e-12)


# ------------------------------------------------------------------------------------------------
# 2. Rosenbrock
# ------------------------------------------------------------------------------------------------


def test_rosenbrock_converges() -> None:
    res = minimize(rosenbrock_problem(), EYE2, OptimizerOptions())
    assert res.convergence is ConvergenceClass.CONVERGED
    np.testing.assert_allclose(res.theta, [1.0, 1.0], atol=1e-5)  # stop rule: dq_gn <= 1e-6 (1 + q)
    assert res.projected_gradient <= res.pg_tol and res.gn_decrease <= res.gn_tol
    # q never rises along accepted steps
    qs = [r.q for r in res.history if r.accepted]
    assert all(b <= a for a, b in zip(qs, qs[1:]))


# ------------------------------------------------------------------------------------------------
# 3. optimum outside the box
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [False, True])
def test_optimum_outside_box(cap: bool) -> None:
    c = np.array([2.0, -1.0])
    p = ToyProblem(lambda t: (t - c, np.eye(2)), [spec("a", 0.0, -INF, 1.0, cap=cap), spec("b", 0.0)])
    res = minimize(p, EYE2, OptimizerOptions())
    want = ConvergenceClass.CONVERGED_AT_CAP if cap else ConvergenceClass.CONVERGED
    assert res.convergence is want
    np.testing.assert_allclose(res.theta, [1.0, -1.0], atol=1e-10)
    assert ("a at its cap 1" if cap else "a at its bound 1") in res.message
    assert res.projected_gradient == pytest.approx(0.0, abs=1e-9)  # outward gradient is projected away


# ------------------------------------------------------------------------------------------------
# 4. objective falling slowly forever
# ------------------------------------------------------------------------------------------------


def test_exp_decay_under_iteration_limit_not_converged() -> None:
    p = ToyProblem(lambda t: (np.exp(-t), np.diag(-np.exp(-t))), [spec("t", 0.0)])
    res = minimize(p, np.eye(1), OptimizerOptions(max_iterations=5))
    assert not res.convergence.converged
    assert res.convergence in (ConvergenceClass.ITERATION_LIMIT, ConvergenceClass.FLAT_PROGRESS)


def slow_problem() -> ToyProblem:
    # q = 1 + 1/(1+t)^2: falls forever, by less and less, with t running to infinity
    return ToyProblem(lambda t: (np.array([1.0, 1.0 / (1.0 + t[0])]), np.array([[0.0], [-1.0 / (1.0 + t[0]) ** 2]])),
                      [spec("t", 0.0, 0.0, INF)])


def test_slow_descent_never_converged() -> None:
    res = minimize(slow_problem(), np.eye(2), OptimizerOptions(max_iterations=60))
    assert not res.convergence.converged
    assert res.convergence in (ConvergenceClass.FLAT_PROGRESS, ConvergenceClass.ITERATION_LIMIT)
    assert res.gn_decrease > res.gn_tol  # it stopped because the decrease test still fails
    acc = [r.theta[0] for r in res.history if r.accepted]
    assert acc[-1] > 30 and all(b > a for a, b in zip(acc, acc[1:]))


def test_slow_descent_flags_run_to_infinity() -> None:
    from nfp_harness.diagnostics import degeneracies

    res = minimize(slow_problem(), np.eye(2), OptimizerOptions(max_iterations=60))
    try:
        out = degeneracies.run_to_infinity(res.history, slow_problem().layout)
    except NotImplementedError:
        pytest.skip("diagnostics.degeneracies.run_to_infinity is WP6 and not implemented yet")
    assert out["detected"], out


# ------------------------------------------------------------------------------------------------
# 5. classifier: flat stop with a large gradient is never CONVERGED
# ------------------------------------------------------------------------------------------------


def _rec(i: int, q: float, accepted: bool = True) -> HistoryRecord:
    return HistoryRecord(iteration=i, accepted=accepted, q=q, q_fixed=q, q_cue=q, sd_xi=1.0, rho=0.5, Delta=0.3,
                         tau_cap=1.0, status=EvalStatus.OK, seconds=0.1, theta=np.zeros(2))


def test_classifier_flat_large_gradient() -> None:
    opts = OptimizerOptions()
    hist = [_rec(i, 0.803 - 1e-7 * i) for i in range(40)]  # q change below 1e-6 per step
    q = hist[-1].q
    cls = classify(hist, projected_gradient=0.38, gn_decrease=1e-9, q=q, options=opts, Delta=0.3)
    assert cls is ConvergenceClass.FLAT_PROGRESS
    for limit in ("iterations", "time"):
        assert not classify(hist[:5], 0.38, 1e-9, q, opts, limit=limit, Delta=0.3).converged
    assert classify(hist[:5], 0.38, 1e-9, q, opts, Delta=0.3) is None  # no rule fires: continue
    assert classify(hist[:5], 0.38, 1e-9, q, opts, Delta=1e-9) is ConvergenceClass.STALLED
    rej = hist[:5] + [_rec(5 + i, np.nan, accepted=False) for i in range(10)]
    assert classify(rej, 0.38, 1e-9, q, opts, Delta=0.3) is ConvergenceClass.STALLED
    # the same history with a small gradient is converged
    assert classify(hist, 1e-4, 1e-9, q, opts, Delta=0.3) is ConvergenceClass.CONVERGED
    assert classify(hist, 1e-4, 1e-9, q, opts, at_cap=True, Delta=0.3) is ConvergenceClass.CONVERGED_AT_CAP
    assert classify(hist, 1e-4, 1e-9, q, opts, limit="inversion_failed") is ConvergenceClass.INVERSION_FAILED


def test_convergence_block_message() -> None:
    res = minimize(rosenbrock_problem(), EYE2, OptimizerOptions())
    blk = convergence_block(res)
    assert blk.status == "ok"
    assert blk.messages[0].startswith("Convergence: CONVERGED. Projected gradient ")
    assert "accepted steps" in blk.messages[0] and blk.values["class"] == "CONVERGED"


def test_projected_gradient_zeroes_outward_components() -> None:
    G, g, W = np.eye(3), np.array([1.0, -1.0, 1.0]), np.eye(3)
    theta, lo, hi = np.array([0.0, 1.0, 0.5]), np.zeros(3), np.ones(3)
    pg = projected_gradient(G, W, g, 2, np.array([1.0, 1.0, 2.0]), theta, lo, hi)
    np.testing.assert_allclose(pg, [0.0, 0.0, 8.0])  # at lower with grad > 0, at upper with grad < 0


# ------------------------------------------------------------------------------------------------
# 6. kink with a max_step oracle
# ------------------------------------------------------------------------------------------------

KINK = 1.0


def kinked(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = theta
    slope = 3.0 if a > KINK else 1.0
    return np.array([a - 3.0 + 2.0 * max(a - KINK, 0.0), b - 0.5]), np.array([[slope, 0.0], [0.0, 1.0]])


def kink_oracle(theta: np.ndarray, dtheta: np.ndarray) -> StepLimit:
    a, da = theta[0], dtheta[0]
    tau_k = (KINK - a) / da if da != 0 else -1.0
    if tau_k > 0:
        prof = np.array([[0.0, 0.0], [tau_k, 0.0], [tau_k * (1 + 1e-3), 1.0]])
        return StepLimit(tau_cap=INF, tau_first=tau_k, tau_vanish_lb=INF, mass_profile=prof, note="kink ahead")
    return StepLimit(tau_cap=INF, tau_first=INF, tau_vanish_lb=INF, mass_profile=np.array([[0.0, 0.0], [1.0, 0.0]]),
                     note="no kink")


def test_kink_limited_steps() -> None:
    opts = OptimizerOptions(Delta_0=10.0, Delta_max=10.0)
    specs = [spec("a", -2.0), spec("b", 0.0)]
    res = minimize(ToyProblem(kinked, specs, oracle=kink_oracle), EYE2, opts)
    assert res.convergence is ConvergenceClass.CONVERGED
    np.testing.assert_allclose(res.theta, [5.0 / 3.0, 0.5], atol=1e-10)
    assert res.kink_limited_steps >= 1
    pts = [r.theta[0] for r in res.history if r.accepted]
    for a0, a1 in zip(pts, pts[1:]):
        if (a0 - KINK) * (a1 - KINK) < 0:  # crossed the kink: only just past it
            assert abs(a1 - KINK) <= 1.01e-3 * abs(a1 - a0)
    # without the oracle the first trial lands far past the kink (at the linear model's root a = 3)
    free = minimize(ToyProblem(kinked, specs), EYE2, opts)
    assert free.history[1].theta[0] == pytest.approx(3.0)
    assert free.kink_limited_steps == 0


# ------------------------------------------------------------------------------------------------
# 7. failing evaluations in a region
# ------------------------------------------------------------------------------------------------


def test_failing_region_rejected_and_restored() -> None:
    p = ToyProblem(lambda t: (np.arctan(t - 1.0), np.diag(1.0 / (1.0 + (t - 1.0) ** 2))), [spec("t", -3.0)],
                   fails=lambda t: t[0] > 1.2)
    res = minimize(p, np.eye(1), OptimizerOptions(Delta_0=5.0))
    failed = [i for i, r in enumerate(res.history) if r.status is EvalStatus.INVERSION_FAILED]
    assert failed, "the test needs at least one trial in the failing region"
    assert p.restores >= len(failed)
    for i in failed:
        assert not res.history[i].accepted
        nxt = res.history[i + 1]
        assert nxt.Delta < res.history[i].Delta  # radius shrank after the failure
    assert all(r.theta[0] <= 1.2 for r in res.history if r.accepted)
    assert res.convergence is ConvergenceClass.CONVERGED
    assert res.theta[0] == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(p.accepted, res.theta)


def test_failing_start_is_inversion_failed() -> None:
    p = ToyProblem(lambda t: (t, np.eye(1)), [spec("t", 2.0)], fails=lambda t: True)
    res = minimize(p, np.eye(1), OptimizerOptions())
    assert res.convergence is ConvergenceClass.INVERSION_FAILED and res.evaluation is None


def test_failing_start_names_the_failed_markets() -> None:
    """A start point whose inversion fails in some markets says which in the result message (read by the stability
    report)."""
    class MarketToy(ToyProblem):
        def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation:
            ev = super().evaluate(theta, need_jacobian)
            ev.aux["per_market"] = [{"market_id": 1980, "status": "ok", "kernel_status": "ok", "stages": "warm ok"},
                                    {"market_id": 1991, "status": "inversion_failed", "kernel_status": "not_converged",
                                     "stages": "warm failed; path failed; cold failed"}]
            return ev

    p = MarketToy(lambda t: (t, np.eye(1)), [spec("t", 2.0)], fails=lambda t: True)
    res = minimize(p, np.eye(1), OptimizerOptions())
    assert res.convergence is ConvergenceClass.INVERSION_FAILED and res.accepted_steps == 0
    assert res.message.startswith("start point could not be evaluated (status inversion_failed; markets 1991: "
                                  "not_converged (warm failed; path failed; cold failed))")
    assert "1980" not in res.message


# ------------------------------------------------------------------------------------------------
# 9. two-step on a linear IV toy
# ------------------------------------------------------------------------------------------------


@dataclass
class TwoStep:  # duck type of weighting.TwoStep
    W1: np.ndarray
    mode: str = "two_step"


def iv_toy(seed: int = 3) -> tuple[ToyProblem, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 500
    Z = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    v = rng.standard_normal(n)
    x = Z[:, 1] + 0.5 * Z[:, 2] - 0.3 * Z[:, 3] + v
    X = np.column_stack([np.ones(n), x])
    e = (0.5 * v + rng.standard_normal(n)) * (1.0 + np.abs(Z[:, 1]))  # endogenous, heteroskedastic
    y = X @ np.array([1.0, 2.0]) + e

    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return Z.T @ (y - X @ t) / n, -Z.T @ X / n

    def cov(t: np.ndarray) -> np.ndarray:
        ze = Z * (y - X @ t)[:, None]
        return ze.T @ ze / n

    p = ToyProblem(fn, [spec("b0", 0.0), spec("b1", 0.0)], n=n, cov=cov)
    return p, X, Z, y


def test_two_step_matches_efficient_gmm() -> None:
    p, X, Z, y = iv_toy()
    n = len(y)
    W1 = np.linalg.inv(Z.T @ Z / n)

    def gmm(W: np.ndarray) -> np.ndarray:
        ZX, Zy = Z.T @ X, Z.T @ y
        return np.linalg.solve(ZX.T @ W @ ZX, ZX.T @ W @ Zy)

    t1 = gmm(W1)
    ze = Z * (y - X @ t1)[:, None]
    t2 = gmm(np.linalg.inv(ze.T @ ze / n))
    res = minimize(p, TwoStep(W1), OptimizerOptions(Delta_0=5.0, Delta_max=50.0))
    assert res.stage == "two_step.step2" and res.weighting == "two_step"
    assert res.convergence is ConvergenceClass.CONVERGED
    np.testing.assert_allclose(res.theta, t2, rtol=1e-9, atol=1e-10)
    step1 = [r for r in res.history if r.stage == "two_step.step1" and r.accepted]
    np.testing.assert_allclose(step1[-1].theta, t1, rtol=1e-9, atol=1e-10)
    S1 = p.covariance(p.evaluate(step1[-1].theta))
    np.testing.assert_array_equal(p.W, res.W)  # set_weighting(W2) was called with the stage's W
    np.testing.assert_allclose(res.W, np.linalg.inv(S1), rtol=1e-12)


def test_two_step_gate_blocks_step2() -> None:
    p, X, Z, y = iv_toy()
    W1 = np.linalg.inv(Z.T @ Z / len(y))
    opts = OptimizerOptions(Delta_0=5.0, Delta_max=50.0, two_step_gate=lambda r: "SCALE_ARTIFACT detected")
    res = minimize(p, TwoStep(W1), opts)
    assert res.stage == "two_step.step1" and "Step 2 not started: SCALE_ARTIFACT detected" in res.message


# ------------------------------------------------------------------------------------------------
# 10. CUE gradient against finite differences, clustered S
# ------------------------------------------------------------------------------------------------


def cue_toy(seed: int = 5) -> ToyProblem:
    rng = np.random.default_rng(seed)
    n, C = 120, 30
    clusters = np.repeat(np.arange(C), n // C)
    Z = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    X = rng.standard_normal((n, 2))
    y = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1]) + 0.3 * rng.standard_normal(n) + 0.2 * Z[:, 1]
    onehot = np.eye(C)[clusters]  # (n, C)

    def resid(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.exp(X @ t)
        return y - m, -(m[:, None] * X)  # e (n,), de/dtheta (n, P)

    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e, de = resid(t)
        return Z.T @ e / n, Z.T @ de / n

    def cue(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e, de = resid(t)
        V = onehot.T @ (Z * e[:, None])  # (C, k) cluster sums of z_j e_j
        dV = np.einsum("jc,jk,jp->ckp", onehot, Z, de)
        return V, dV

    def cov(t: np.ndarray) -> np.ndarray:
        V, _ = cue(t)
        return V.T @ V / n

    return ToyProblem(fn, [spec("t0", 0.1), spec("t1", 0.1)], n=n, cov=cov, cue=cue)


def test_cue_gradient_matches_finite_differences() -> None:
    p = cue_toy()

    def q(t: np.ndarray) -> float:
        ev = p.evaluate(t)
        return cue_objective(ev, p.covariance(ev), p.n)

    for t in (np.array([0.1, 0.1]), np.array([0.4, -0.3])):
        ev = p.evaluate(t)
        V, dV = p.cue_terms(ev)
        grad = cue_gradient(ev, p.covariance(ev), V, dV, p.n)
        h = 1e-6
        fd = np.array([(q(t + h * e) - q(t - h * e)) / (2 * h) for e in np.eye(2)])
        np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-6 * max(1.0, np.abs(fd).max()))


def test_cue_run_converges_to_stationary_point() -> None:
    p = cue_toy()
    res = minimize(p, "cue", OptimizerOptions())
    assert res.convergence is ConvergenceClass.CONVERGED and res.weighting == "cue"
    ev = p.evaluate(res.theta)
    V, dV = p.cue_terms(ev)
    grad = cue_gradient(ev, p.covariance(ev), V, dV, p.n)
    assert np.max(np.abs(grad)) <= res.pg_tol


# ------------------------------------------------------------------------------------------------
# round 3: noise floor (CONVERGED_NOISE) and Levenberg-Marquardt damping
# ------------------------------------------------------------------------------------------------


def noisy(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g = x - 1 + 0.003 sin(1e7 x): a smooth residual plus a wiggle far finer than any trust-region step, which
    the analytic G (= 1) does not see. Like the supply block's frontier jumps, it puts a floor under q that no
    step can resolve."""
    x = float(theta[0])
    return np.array([x - 1.0 + 0.003 * np.sin(1e7 * x)]), np.array([[1.0]])


def test_noise_floor_stops_as_converged_noise() -> None:
    opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200)
    res = minimize(ToyProblem(noisy, [spec("x", 0.0)]), np.eye(1),
                   replace(opts, noise_probe=1e-5, noise_Delta=1e-4, noise_rejections=3))
    assert res.convergence is ConvergenceClass.CONVERGED_NOISE and res.convergence.converged
    assert 1e-4 < res.noise_floor < 0.1   # probes at x = 0 move q by about 2 |g| 0.003 sin(100)
    tail = res.history[-3:]
    assert all(not r.accepted and r.Delta < 1e-4 and abs(r.q - res.q) < res.noise_floor for r in tail)
    assert abs(res.theta[0] - 1.0) < 0.01 and "noise floor" in res.message
    # without the probes the same run can only stall
    res0 = minimize(ToyProblem(noisy, [spec("x", 0.0)]), np.eye(1), opts)
    assert res0.convergence is ConvergenceClass.STALLED and math.isnan(res0.noise_floor)


def test_noise_stop_needs_small_radius_and_small_change() -> None:
    from nfp_harness.diagnostics.convergence import noise_stop

    def rej(q: float, Delta: float) -> HistoryRecord:
        return HistoryRecord(iteration=0, accepted=False, q=q, q_fixed=q, q_cue=float("nan"), sd_xi=float("nan"),
                             rho=-1.0, Delta=Delta, tau_cap=1.0, status=EvalStatus.OK, seconds=0.0,
                             theta=np.zeros(1))

    opts = OptimizerOptions(noise_probe=1e-5)
    small = [rej(1.001, 5e-5)] * 3
    assert noise_stop(small, 1.0, 0.01, opts)
    assert not noise_stop(small[:2], 1.0, 0.01, opts)                       # only two rejections
    assert not noise_stop([rej(1.001, 5e-4)] * 3, 1.0, 0.01, opts)          # radius not small
    assert not noise_stop([rej(1.1, 5e-5)] * 3, 1.0, 0.01, opts)            # change above the floor
    assert not noise_stop(small, 1.0, float("nan"), opts)                   # no floor measured


def test_lm_damping_grows_on_negative_rho_and_resets_on_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    import nfp_harness.optimizer as opt

    seen: list[float] = []
    real = opt._box_lsq

    def spy(A: np.ndarray, r: np.ndarray, dk: np.ndarray, mu: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        if np.all(np.isfinite(lo)) and np.max(np.abs(lo)) < 1e6:   # trust-region steps (the stopping test is unboxed)
            seen.append(mu)
        return real(A, r, dk, mu, lo, hi)

    monkeypatch.setattr(opt, "_box_lsq", spy)
    # g = x^3 - 8 from x = 0.1: the Gauss-Newton step overshoots far past 2 and q rises (rho < 0)
    p = ToyProblem(lambda t: (np.array([t[0] ** 3 - 8.0]), np.array([[3.0 * t[0] ** 2]])), [spec("x", 0.1)])
    res = minimize(p, np.eye(1), OptimizerOptions(Delta_0=5.0, Delta_max=5.0, lm_damping=True, lm_start=1e-3))
    assert res.convergence is ConvergenceClass.CONVERGED
    rhos = [r.rho for r in res.history[1:]]
    acc = [r.accepted for r in res.history[1:]]
    assert seen[0] == 0.0
    for i in range(len(rhos) - 1):    # the damping of trial i + 1 follows trial i's outcome
        if acc[i]:
            assert seen[i + 1] == 0.0
        elif rhos[i] < 0:
            assert seen[i + 1] == (1e-3 if seen[i] == 0.0 else pytest.approx(10.0 * seen[i]))
    assert any(r < 0 for r in rhos) and any(m > 0 for m in seen)


# ------------------------------------------------------------------------------------------------
# round 4 (SPEC_ANALYSIS v4 change 9): More scaling, Cauchy stop, sub-noise rejections, terminal probe
# ------------------------------------------------------------------------------------------------


def test_more_scaling_shapes_the_box() -> None:
    """A stiff and a flat linear residual: with More scaling the first step moves the flat coordinate past the
    unscaled radius and the stiff one by less; both runs converge to the same point."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array([100.0 * (t[0] - 1.0), 0.1 * (t[1] - 30.0)]), np.diag([100.0, 0.1])

    specs = [spec("x", 0.0), spec("y", 0.0)]
    base = OptimizerOptions(Delta_0=0.3, Delta_max=5.0)
    r0 = minimize(ToyProblem(fn, specs), np.eye(2), base)
    r1 = minimize(ToyProblem(fn, specs), np.eye(2), replace(base, more_scaling=True))
    for r in (r0, r1):
        assert r.convergence is ConvergenceClass.CONVERGED
        np.testing.assert_allclose(r.theta, [1.0, 30.0], atol=1e-6)
    step0, step1 = r0.history[1].theta, r1.history[1].theta
    assert abs(step0[0]) <= 0.3 + 1e-12 and abs(step0[1]) <= 0.3 + 1e-12
    w = np.array([100.0, 0.1]) / math.sqrt(100.0 * 0.1)          # dk / geomean(dk)
    assert abs(step1[0]) <= 0.3 / w[0] + 1e-9 and abs(step1[1]) > 0.3


def test_cauchy_prediction_matches_line_search() -> None:
    from nfp_harness.optimizer import _cauchy_pred, _Model

    rng = np.random.default_rng(0)
    A, r = rng.normal(size=(6, 3)), rng.normal(size=6)
    m = _Model(q=float(r @ r), r=r, A=A, grad=np.zeros(3), pg=np.zeros(3), W=np.eye(6))
    w = np.array([1.0, 2.0, 0.5])
    lo, hi = -0.2 / w, 0.2 / w
    p = -(A.T @ r) / w ** 2
    t_box = float(np.min(np.where(p > 0, hi / p, lo / p)))
    ts = np.linspace(0.0, t_box, 200001)
    best = max(float(r @ r - np.sum((r + A @ (t * p)) ** 2)) for t in ts)
    assert _cauchy_pred(m, 1.0, w, lo, hi) == pytest.approx(best, rel=1e-9)
    # a coordinate at its bound whose descent points out of the box does not move
    k = int(np.argmax(np.abs(p)))
    lo0, hi0 = lo.copy(), hi.copy()
    if p[k] < 0:
        lo0[k] = 0.0
    else:
        hi0[k] = 0.0
    assert _cauchy_pred(m, 1.0, w, lo0, hi0) < _cauchy_pred(m, 1.0, w, lo, hi)
    # on the Rosenbrock toy the Cauchy stopping test still reaches CONVERGED
    res = minimize(rosenbrock_problem(), EYE2, OptimizerOptions(cauchy_stop=True))
    assert res.convergence is ConvergenceClass.CONVERGED and "Cauchy" in res.message


def test_sub_noise_rejection_keeps_radius_and_probe_converges() -> None:
    """The noisy toy under noise_converge: the stage ends CONVERGED_NOISE through the terminal probe, before the
    round-3 radius collapse, and the radius at the stop is the radius of the last trial."""
    opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                            noise_converge=True)

    def faint(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:   # the noisy toy with a 100 times smaller wiggle
        x = float(theta[0])
        return np.array([x - 1.0 + 3e-5 * np.sin(1e7 * x)]), np.array([[1.0]])

    res = minimize(ToyProblem(faint, [spec("x", 0.0)]), np.eye(1), opts)
    assert res.convergence is ConvergenceClass.CONVERGED_NOISE
    # the stop is where the model's predicted decrease (x - 1)^2 falls below 10 x the floor
    assert "terminal probe" in res.message
    assert (res.theta[0] - 1.0) ** 2 < 10 * res.noise_floor
    rejected = [r for r in res.history if not r.accepted]
    assert all(r.Delta >= 1e-4 for r in rejected)          # never reached the round-3 radius
    if rejected and abs(rejected[-1].q - res.q) < res.noise_floor:
        assert res.Delta == rejected[-1].Delta             # a sub-noise rejection does not shrink the radius


def test_terminal_probe_restarts_from_a_better_point() -> None:
    """G misses the y direction (its column is zero), so the Gauss-Newton model cannot move y; the probe along y
    finds q lower by far more than the floor, the stage restarts there and ends at y = -0.1."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = float(t[0]), float(t[1])
        g = np.array([x - 1.0 + 0.003 * np.sin(1e7 * x), 10.0 * (y + 0.1)])
        return g, np.array([[1.0, 0.0], [0.0, 0.0]])

    opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                            noise_converge=True, probe_steps=(0.01, 0.03, 0.1))
    res = minimize(ToyProblem(fn, [spec("x", 0.0), spec("y", 0.0)]), np.eye(2), opts)
    assert res.convergence is ConvergenceClass.CONVERGED_NOISE
    assert res.theta[1] == pytest.approx(-0.1, abs=1e-12)
    assert "restarts from a probe" in res.message
    assert any(math.isnan(r.rho) and r.accepted for r in res.history[1:])   # the restart record


# ------------------------------------------------------------------------------------------------
# round 5 (LEARNINGS 32): probe restarts reset the radius, the radius is floored in noise mode, compass search
# ------------------------------------------------------------------------------------------------


def sawtooth(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: a residual with a sawtooth of period 0.01 that G (= 1) does not see, like the supply block's frontier jumps;
    y: a smooth residual 10 (y + 0.5) that the model gets right."""
    x, y = float(theta[0]), float(theta[1])
    g = np.array([(x - 1.0) + 0.05 * (((x / 0.01) % 1.0) - 0.5), 10.0 * (y + 0.5)])
    return g, np.array([[1.0, 0.0], [0.0, 10.0]])


NOISE_OPTS = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                              noise_converge=True, lm_damping=True, Delta_max=0.5)


def test_probe_restart_resets_the_radius_and_does_not_loop() -> None:
    """The step-2 crawl of LEARNINGS 32: the stage sits at a collapsed radius (1.22e-5), where the predicted
    decrease is below 10 x the floor, so the terminal probe runs and wins at 0.1 along y. Keeping the collapsed radius
    on the restart re-probed at once, five times over (118 evaluations, no trust-region step). The restart now sets
    the radius to the winning probe step, so the Gauss-Newton steps take y to its optimum: one restart."""
    p = ToyProblem(sawtooth, [spec("x", 0.5037), spec("y", 0.0)])
    res = minimize(p, np.eye(2), replace(NOISE_OPTS, Delta_0=1.22e-5))
    assert res.convergence is ConvergenceClass.CONVERGED_NOISE
    restarts = [h for h in res.history[1:] if math.isnan(h.rho) and h.accepted]
    trials = [h for h in res.history[1:] if not math.isnan(h.rho)]
    assert len(restarts) == 1
    assert restarts[0].Delta == pytest.approx(0.1)            # the winning probe step, not the collapsed radius
    assert trials and trials[0].Delta == pytest.approx(0.1)   # the next trust-region step uses it
    assert res.theta[1] == pytest.approx(-0.5, abs=1e-9)
    assert p.evals < 60


def test_noise_mode_floors_the_radius_at_the_secant_step() -> None:
    """In noise mode the radius never falls below noise_Delta_floor (1e-3), and a rejection at the floor goes to the
    probe, which ends the stage CONVERGED_NOISE on the sawtooth (no probe step beats 10 x the floor near x = 1)."""
    def fn(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = float(theta[0])
        return np.array([(x - 1.0) + 0.05 * (((x / 0.01) % 1.0) - 0.5)]), np.array([[1.0]])

    res = minimize(ToyProblem(fn, [spec("x", 0.5037)]), np.eye(1), replace(NOISE_OPTS, Delta_0=0.1))
    assert res.convergence is ConvergenceClass.CONVERGED_NOISE
    assert all(h.Delta >= 1e-3 * (1 - 1e-12) for h in res.history)
    assert res.Delta >= 1e-3 * (1 - 1e-12)
    # outside noise mode the radius may still collapse below the floor
    res0 = minimize(ToyProblem(fn, [spec("x", 0.5037)]), np.eye(1),
                    replace(NOISE_OPTS, Delta_0=0.1, noise_converge=False, noise_probe=None))
    assert min(h.Delta for h in res0.history) < 1e-3


def test_compass_probe_stops_at_the_first_winning_step() -> None:
    """The probe polls the largest step first and stops there when it wins: on the restart toy the probe that wins at
    0.1 costs one poll (2 x 3 directions), not three."""
    from nfp_harness.optimizer import _layout_arrays, _Objective, _terminal_probe

    p = ToyProblem(sawtooth, [spec("x", 1.0037), spec("y", 0.0)])
    arr = _layout_arrays(p)
    obj = _Objective(p, "fixed", np.eye(2), None, NOISE_OPTS, arr)
    theta = np.array([1.0037, 0.0])
    q = obj.q(p.evaluate(theta, False))
    pt, pq, nev, summary, win = _terminal_probe(p, obj, theta, q, np.array([0.0, 1.0]), arr, (0.01, 0.03, 0.1),
                                                gain_needed=1e-3)
    assert win == 0.1 and nev == 6 and pt[1] == pytest.approx(-0.1) and pq < q
    assert "+-0.1 units" in summary
    # nothing beats the threshold: every step is tried, largest first, and win is 0
    *_, nev0, _, win0 = _terminal_probe(p, obj, theta, q, np.array([0.0, 1.0]), arr, (0.01, 0.03, 0.1),
                                        gain_needed=1e9)
    assert win0 == 0.0 and nev0 == 18


# ------------------------------------------------------------------------------------------------
# GMY round 1: the trust-region collapse on a kinked objective (radius floor relative to the parameter scale, restart
# radius at the probe step, probe restarts while the compass still finds real decreases)
# ------------------------------------------------------------------------------------------------


def kinked_bowl(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five residuals, each a slope 1 plus a sawtooth of period 0.01 and amplitude 0.1; G is the active-piece slope
    11 (the analytic Jacobian of a kinked law, like the second choices'), exact inside a piece and 11 times too steep
    over steps that cross many pieces."""
    t = np.asarray(theta, dtype=float)
    saw = 0.1 * (((t / 0.01) % 1.0) - 0.5)
    return (t - 1.0) + saw, 11.0 * np.eye(t.size)


def test_radius_floor_relative_to_scale_avoids_the_crawl() -> None:
    """From 2 units away, a floor of 1e-3 lets the radius collapse onto the sawtooth and the stage ends far from the
    optimum (40 trials end near t = 0.16, accepted steps of 1e-3); a floor of 0.05 natural units keeps the steps at
    the scale where they cross many pieces and reaches the optimum."""
    specs = [spec(f"t{k}", -0.9973, unit=1.0) for k in range(5)]
    base = replace(NOISE_OPTS, Delta_0=0.1, max_iterations=40, more_scaling=True)
    lo = minimize(ToyProblem(kinked_bowl, specs), np.eye(5), base)
    hi = minimize(ToyProblem(kinked_bowl, specs), np.eye(5),
                  replace(base, noise_Delta_floor=0.05, probe_restart_at_step=True))
    assert all(h.Delta >= 0.05 * (1 - 1e-12) for h in hi.history)
    assert hi.q < 0.1 * lo.q
    assert np.max(np.abs(hi.theta - 1.0)) < 0.1


def test_probe_restarts_continue_while_the_gain_is_real() -> None:
    """G misses y entirely (zero column), so only the compass moves it, at most 0.1 per restart; y must travel 0.5.
    With max_probe_restarts = 2 the stage ends CONVERGED_NOISE at y = -0.2 although a 0.1 step still lowers q by 5;
    with probe_restart_gain = 0.5 it keeps restarting while the gain exceeds 0.5 and ends at the optimum."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = float(t[0]), float(t[1])
        g = np.array([x - 1.0 + 0.003 * np.sin(1e7 * x), 10.0 * (y + 0.5)])
        return g, np.array([[1.0, 0.0], [0.0, 0.0]])

    opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                            noise_converge=True, probe_steps=(0.01, 0.03, 0.1), max_probe_restarts=2)
    capped = minimize(ToyProblem(fn, [spec("x", 0.0), spec("y", 0.0)]), np.eye(2), opts)
    assert capped.convergence is ConvergenceClass.CONVERGED_NOISE
    assert capped.theta[1] == pytest.approx(-0.2, abs=1e-12)
    free = minimize(ToyProblem(fn, [spec("x", 0.0), spec("y", 0.0)]), np.eye(2),
                    replace(opts, probe_restart_gain=0.5))
    assert free.convergence is ConvergenceClass.CONVERGED_NOISE
    assert free.theta[1] == pytest.approx(-0.5, abs=1e-12)
    assert sum(1 for h in free.history[1:] if math.isnan(h.rho) and h.accepted) == 5


def test_probe_restart_gain_relative_to_q() -> None:
    """The toy of the previous test: y = 0, -0.1, .., -0.5 has q = 25, 16, 9, 4, 1, 0 (x contributes about 1e-5), so
    the restarts gain 9, 7, 5, 3, 1, i.e. 0.36, 0.44, 0.56, 0.75 and 1.0 of q at their origin. probe_restart_gain_rel =
    0.4 refuses the first restart although the cap allows ten; 0.3 restarts past a cap of 2 to the optimum; unset, the
    cap of ten is the only rule, as before."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = float(t[0]), float(t[1])
        g = np.array([x - 1.0 + 0.003 * np.sin(1e7 * x), 10.0 * (y + 0.5)])
        return g, np.array([[1.0, 0.0], [0.0, 0.0]])

    def run(**kw: Any) -> Any:
        opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                                noise_converge=True, probe_steps=(0.01, 0.03, 0.1), **kw)
        return minimize(ToyProblem(fn, [spec("x", 0.0), spec("y", 0.0)]), np.eye(2), opts)

    def restarts(res: Any) -> int:
        return sum(1 for h in res.history[1:] if math.isnan(h.rho) and h.accepted)

    unset = run(max_probe_restarts=10)
    assert unset.theta[1] == pytest.approx(-0.5, abs=1e-12) and restarts(unset) == 5
    refused = run(max_probe_restarts=10, probe_restart_gain_rel=0.4)
    assert refused.convergence is ConvergenceClass.CONVERGED_NOISE
    assert refused.theta[1] == 0.0 and restarts(refused) == 0
    assert "does not restart" in refused.message
    past_cap = run(max_probe_restarts=2, probe_restart_gain_rel=0.3)
    assert past_cap.theta[1] == pytest.approx(-0.5, abs=1e-12) and restarts(past_cap) == 5
    # with an absolute gain as well, past the cap the gain must exceed both: 2 restarts, then gains of 5 > 0.3 x 9 but
    # not > 6
    both = run(max_probe_restarts=2, probe_restart_gain_rel=0.3, probe_restart_gain=6.0)
    assert both.theta[1] == pytest.approx(-0.2, abs=1e-12) and restarts(both) == 2
    assert "probe_restart_gain_rel" not in repr(OptimizerOptions(probe_restart_gain_rel=0.1))


def test_restart_radius_is_at_least_the_probe_step_under_more_scaling() -> None:
    """With More scaling a coordinate whose weight is small would restart at (weight x 0.1); with
    probe_restart_at_step the restart radius is at least the 0.1 the probe measured."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = float(t[0]), float(t[1])
        g = np.array([100.0 * (x - 1.0) + 0.3 * np.sin(1e7 * x), 10.0 * (y + 0.5), 0.0])
        return g, np.array([[100.0, 0.0], [0.0, 0.0], [0.0, 0.01]])

    opts = OptimizerOptions(pg_tol=1e-12, gn_tol=1e-16, max_rejections=40, flat_window=200, noise_probe=1e-5,
                            noise_converge=True, probe_steps=(0.01, 0.03, 0.1), more_scaling=True, Delta_0=0.004,
                            max_probe_restarts=1)
    res = minimize(ToyProblem(fn, [spec("x", 1.0), spec("y", 0.0)]), np.eye(3),
                   replace(opts, probe_restart_at_step=True))
    restarts = [h for h in res.history[1:] if math.isnan(h.rho) and h.accepted]
    assert restarts and restarts[0].Delta >= 0.1 * (1 - 1e-12)


def test_failed_evaluations_lower_the_radius_floor() -> None:
    """A warm-started inversion fails when the step from the accepted point is too long (GMY 2015/2018 at a 0.05 floor:
    every trial failed and the floor reset the radius to the same failing length). Here a trial fails when it moves
    more than 0.02 from the accepted point; the optimum is 1 unit away. With the floor at 0.05 a failure must lower the
    floor below the failing radius, so the stage keeps accepting steps and reaches the optimum, and the terminal probe
    never polls steps longer than the failing length."""
    def fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array([float(t[0]) - 1.0]), np.array([[1.0]])

    p = ToyProblem(fn, [spec("x", 0.0, unit=1.0)])
    p.fails = lambda th: p.accepted is not None and abs(float(th[0]) - float(p.accepted[0])) > 0.02 + 1e-12
    p.accepted = np.array([0.0])
    opts = replace(NOISE_OPTS, Delta_0=0.1, max_iterations=400, noise_Delta_floor=0.05, probe_restart_at_step=True)
    res = minimize(p, np.eye(1), opts)
    accepted = [h for h in res.history[1:] if h.accepted]
    failed = [h for h in res.history[1:] if h.status is EvalStatus.INVERSION_FAILED]
    assert len(accepted) >= 40                  # 1 unit in steps of at most 0.02
    assert len(failed) <= 10                    # the floor drops after a failure instead of repeating it
    assert res.theta[0] == pytest.approx(1.0, abs=0.02)

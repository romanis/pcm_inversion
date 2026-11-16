"""Tests for problem (WP4; DESIGN 2.2, WP4 acceptance).

The fake inner model is a random-coefficients logit solved exactly by Newton's method, written in
pure numpy. Its delta(theta) is known to machine precision and its D = -(ds/ddelta)^{-1} ds/dtheta is
analytic, so the chain rule in ``NFPGMMProblem`` can be checked against central finite differences.
``test_moments.py`` and ``test_weighting.py`` import the fixtures from here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import IdentificationError, LinearIV
from nfp_harness.moments import MICRO_OUTPUTS, IVMoments, MicroDataset, MicroPart, RatioMicroMoment
from nfp_harness.params import ParameterLayout
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.types import (
    EvalStatus,
    InnerSolution,
    KernelRequests,
    MarketIndex,
    ParameterSpec,
    StepLimit,
)

# --------------------------------------------------------------------------------------------
# Fake inner model
# --------------------------------------------------------------------------------------------


class LogitRCModel:
    """u_ij = delta_j + sum_p theta_p B_p[i, j] + logit error, outside utility 0.

    B_0 = nu_i x_j (taste spread on x), B_1 = fs_i mi_j (interaction), B_2 = -age_i p_j (price).
    """

    def __init__(self, products: dict[str, np.ndarray], agents: list[dict[str, np.ndarray]],
                 market_ids: np.ndarray, shares: np.ndarray, with_max_step: bool = False) -> None:
        self.products = products
        self.market_ids = np.asarray(market_ids)
        self.shares = np.asarray(shares, dtype=float)
        self.n_products = self.shares.size
        ids = list(dict.fromkeys(self.market_ids.tolist()))
        self.markets = [MarketIndex(t, np.flatnonzero(self.market_ids == t)) for t in ids]
        self.agents = agents  # one dict per market: w, nu, fs, age, mid (R_t,)
        self.accepted: list[np.ndarray] = [np.log(self.shares[m.rows]) - np.log(1 - self.shares[m.rows].sum())
                                           for m in self.markets]
        self.trial: list[np.ndarray] | None = None
        self.calls: list[str] = []
        self.fail_above: float | None = None  # theta_0 above this: inversion fails
        if with_max_step:
            self.max_step = self._max_step

    def basis(self, t: int, theta: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        rows = self.markets[t].rows
        a = self.agents[t]
        x, mi, p = (self.products[c][rows] for c in ("x", "mi", "prices"))
        B = [np.outer(a["nu"], x), np.outer(a["fs"], mi), -np.outer(a["age"], p)]
        mu = sum(th * b for th, b in zip(theta, B))
        return B, mu

    @staticmethod
    def choice_probs(delta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        v = delta[None, :] + mu
        vmax = np.maximum(v.max(axis=1, keepdims=True), 0.0)
        e = np.exp(v - vmax)
        return e / (np.exp(-vmax) + e.sum(axis=1, keepdims=True))

    def market_shares(self, t: int, delta_t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        _, mu = self.basis(t, theta)
        return self.agents[t]["w"] @ self.choice_probs(delta_t, mu)

    def _invert(self, t: int, theta: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, float]:
        rows = self.markets[t].rows
        w = self.agents[t]["w"]
        _, mu = self.basis(t, theta)
        target = self.shares[rows]
        d = start.copy()
        for _ in range(100):
            s_ij = self.choice_probs(d, mu)
            s = w @ s_ij
            r = s - target
            if np.max(np.abs(r)) < 1e-15:
                break
            J = np.diag(s) - (s_ij * w[:, None]).T @ s_ij
            d = d - np.linalg.solve(J, r)
        resid = float(np.max(np.abs(w @ self.choice_probs(d, mu) - target)))
        return d, resid

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        self.calls.append("solve")
        theta = np.asarray(theta, dtype=float)
        delta = np.zeros(self.n_products)
        D: list[np.ndarray] = []
        per_market: list[dict[str, Any]] = []
        parts: dict[str, list[dict[str, Any]]] = {p.name: [] for p in requests.micro_parts}
        trial = []
        if self.fail_above is not None and theta[0] > self.fail_above:
            return InnerSolution(delta=delta, D=None, status=EvalStatus.INVERSION_FAILED,
                                 per_market=[{"converged": False}], outputs={})
        for t, mkt in enumerate(self.markets):
            d, resid = self._invert(t, theta, self.accepted[t])
            trial.append(d)
            delta[mkt.rows] = d
            per_market.append({"converged": resid < 1e-12, "residual": resid})
            B, mu = self.basis(t, theta)
            s_ij = self.choice_probs(d, mu)
            w = self.agents[t]["w"]
            if need_jacobian:
                s = w @ s_ij
                ds_dd = np.diag(s) - (s_ij * w[:, None]).T @ s_ij
                ds_dth = np.column_stack([w @ (s_ij * (b - (s_ij * b).sum(1, keepdims=True))) for b in B])
                D.append(-np.linalg.solve(ds_dd, ds_dth))
            for p in requests.micro_parts:
                parts[p.name].append(self.micro_part(t, p, s_ij, B, need_jacobian))
        self.trial = trial
        outputs: dict[str, object] = {MICRO_OUTPUTS: parts} if requests.micro_parts else {}
        return InnerSolution(delta=delta, D=D if need_jacobian else None, status=EvalStatus.OK,
                             per_market=per_market, outputs=outputs)

    def micro_part(self, t: int, part: MicroPart, s_ij: np.ndarray, B: list[np.ndarray],
                   need_jacobian: bool) -> dict[str, Any]:
        """Kernel output contract of nfp_harness.moments for one market and one part."""
        rows = self.markets[t].rows
        a = self.agents[t]
        h = np.ones(s_ij.shape[0])
        for c in part.agent_columns:
            h = h * a[c]
        v = np.ones(rows.size)
        for c in part.product_columns:
            v = v * self.products[c][rows]
        u = v - part.outside_value
        wh = a["w"] * h
        out: dict[str, Any] = {"sum": float(wh @ (s_ij @ u)), "h_sum": float(wh.sum()), "w_sum": float(a["w"].sum()),
                               "d_delta": None, "d_theta": None}
        if need_jacobian:
            ubar = s_ij @ u  # (R,)
            out["d_delta"] = wh @ (s_ij * (u[None, :] - ubar[:, None]))
            out["d_theta"] = np.array([wh @ ((s_ij * u) * (b - (s_ij * b).sum(1, keepdims=True))).sum(1) for b in B])
        return out

    def commit(self, theta: np.ndarray) -> None:
        self.calls.append("commit")
        if self.trial is not None:
            self.accepted = [d.copy() for d in self.trial]

    def restore(self, theta: np.ndarray) -> None:
        self.calls.append("restore")
        self.trial = None

    def state_dict(self) -> dict[str, Any]:
        return {"accepted": [d.copy() for d in self.accepted]}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.accepted = [np.asarray(a, dtype=float).copy() for a in d["accepted"]]

    def _max_step(self, sol: InnerSolution, dtheta: np.ndarray) -> StepLimit:
        return StepLimit(tau_cap=0.5, tau_first=0.25, tau_vanish_lb=1.0, mass_profile=np.zeros((1, 2)), note="fake")


THETA_TRUE = np.array([0.8, 0.3, 0.6])


def make_data(seed: int = 0, sizes: tuple[int, ...] = (4, 5, 6), R: int = 40) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    market_ids = np.repeat(np.arange(len(sizes)), sizes)
    n = market_ids.size
    x = rng.normal(size=n)
    mi = (rng.random(n) < 0.4).astype(float)
    mi[:2] = (1.0, 0.0)
    prices = rng.uniform(1.0, 3.0, size=n)
    products = {"x": x, "mi": mi, "prices": prices}
    agents = []
    for _ in sizes:
        w = rng.uniform(0.5, 1.5, size=R)
        agents.append({
            "w": w / w.sum(),
            "nu": rng.normal(size=R),
            "fs": rng.integers(1, 6, size=R).astype(float),
            "age": (rng.random(R) < 0.5).astype(float),
            "mid": (rng.random(R) < 0.3).astype(float),
        })
    X1 = np.column_stack([np.ones(n), x, mi])
    Z_excl = np.column_stack([rng.normal(size=n) + 0.5 * prices, rng.normal(size=n) + x**2])
    beta_true = np.array([-1.0, 0.5, 0.3])
    xi = 0.3 * rng.normal(size=n)
    delta_true = X1 @ beta_true + xi
    # shares at the truth
    probe = LogitRCModel(products, agents, market_ids, np.full(n, 1e-3))
    shares = np.concatenate([probe.market_shares(t, delta_true[m.rows], THETA_TRUE)
                             for t, m in enumerate(probe.markets)])
    clusters = np.arange(n) // 2
    return {"products": products, "agents": agents, "market_ids": market_ids, "shares": shares, "X1": X1,
            "Z_excl": Z_excl, "Z": np.column_stack([X1, Z_excl]), "delta_true": delta_true, "clusters": clusters}


def make_layout(free: tuple[bool, bool, bool] = (True, True, True)) -> ParameterLayout:
    return ParameterLayout([
        ParameterSpec("sigma_x", "taste", "x", "nu0", 0.5, 0.0, 10.0, unit=1.0, free=free[0]),
        ParameterSpec("pi_mi_fs", "taste", "mi", "fs", 0.1, -10.0, 10.0, unit=1.0, free=free[1]),
        ParameterSpec("alpha_age", "scale", None, "age", 0.4, -10.0, 10.0, unit=1.0, free=free[2]),
    ])


def make_micro(dataset: MicroDataset | None = None) -> RatioMicroMoment:
    mi = MicroPart("E[mi]", (), ("mi",), 0.0)
    age_mi = MicroPart("E[age mi]", ("age",), ("mi",), 0.0)
    fs_mi = MicroPart("E[fs mi]", ("fs",), ("mi",), 0.0)
    mid_in = MicroPart("E[mid inside]", ("mid",), (), 0.0)
    mid = MicroPart("E[mid]", ("mid",), (), 1.0)
    return RatioMicroMoment(
        "micro", dataset or MicroDataset("survey", observations=500),
        numerators=[age_mi, fs_mi, mid_in], denominators=[mi, mi, mid], values=np.array([0.55, 3.1, 0.6]),
        labels=["E[age|mi]", "E[fs|mi]", "E[inside|mid]"])


def make_problem(seed: int = 0, micro: bool = True, with_max_step: bool = False,
                 layout: ParameterLayout | None = None) -> tuple[NFPGMMProblem, LogitRCModel, dict[str, Any]]:
    d = make_data(seed)
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"], with_max_step=with_max_step)
    linear = LinearIV(d["X1"], d["Z"], names=["const", "x", "mi"])
    moments: list[Any] = [IVMoments(d["Z"], d["clusters"])]
    if micro:
        moments.append(make_micro())
    problem = NFPGMMProblem(model, moments, linear, layout or make_layout())
    return problem, model, d


def central_fd(f, theta: np.ndarray, h: float = 1e-5) -> np.ndarray:
    cols = []
    for p in range(theta.size):
        e = np.zeros_like(theta)
        e[p] = h
        cols.append((f(theta + e) - f(theta - e)) / (2 * h))
    return np.column_stack(cols)


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def test_fake_model_inverts_exactly() -> None:
    problem, model, d = make_problem()
    ev = problem.evaluate(THETA_TRUE)
    assert ev.ok
    np.testing.assert_allclose(ev.aux["delta"], d["delta_true"], atol=1e-12)
    assert all(pm["converged"] for pm in ev.aux["per_market"])


def test_D_matches_finite_differences_of_delta() -> None:
    problem, model, _ = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    ev = problem.evaluate(theta)
    fd = central_fd(lambda th: problem.evaluate(th, need_jacobian=False).aux["delta"], theta)
    np.testing.assert_allclose(ev.aux["D"], fd, atol=1e-8)


def test_chain_rule_G_matches_central_finite_differences() -> None:
    """WP4 acceptance: total G (IV + micro, chain rule through delta) against central FD to 1e-7."""
    problem, _, _ = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    ev = problem.evaluate(theta)
    assert ev.ok and ev.G is not None and ev.G.shape == (problem.k, 3)
    fd = central_fd(lambda th: problem.evaluate(th, need_jacobian=False).g, theta, h=1e-5)
    err = np.max(np.abs(ev.G - fd)) / max(1.0, np.max(np.abs(ev.G)))
    print(f"chain-rule G vs central FD: max scaled error {err:.2e} (k={problem.k}, P=3)")
    assert err < 1e-7
    assert np.max(np.abs(ev.G[problem.blocks["micro"]])) > 1e-3  # micro rows carry real derivatives


def test_blocks_and_aux() -> None:
    problem, _, _ = make_problem()
    ev = problem.evaluate(np.array([0.6, 0.2, 0.5]))
    assert ev.blocks == {"aggregate": slice(0, 5), "micro": slice(5, 8)}
    assert ev.g.shape == (8,)
    q = problem.n * ev.g @ problem.W @ ev.g
    assert ev.aux["q"] == pytest.approx(q)
    assert sum(ev.aux["q_blocks"].values()) == pytest.approx(q)
    np.testing.assert_allclose(ev.aux["micro_predicted"]["micro"] - ev.aux["micro_data"]["micro"],
                               ev.g[ev.blocks["micro"]])
    assert ev.aux["beta_names"] == ("const", "x", "mi")


def test_identification_count_and_values() -> None:
    problem, _, _ = make_problem()
    info = problem.identification
    assert info["rank_aggregate"] == 5 and info["k_micro"] == 3
    assert info["unknowns"] == 3 + 3 and info["overidentifying"] == 2


def test_refuses_when_Z_lacks_X1() -> None:
    """LEARNINGS 16 / F1: Z = excluded instruments only, so rank(Z'X1) < K and beta is not identified."""
    d = make_data()
    with pytest.raises(IdentificationError, match="rank\\(Z'X1\\)"):
        LinearIV(d["X1"], d["Z_excl"])


def test_refuses_when_moments_fewer_than_unknowns() -> None:
    """Z = [X1, Z_excl] has rank 5; without micro moments 5 < 3 nonlinear + 3 linear unknowns."""
    d = make_data()
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
    linear = LinearIV(d["X1"], d["Z"])
    with pytest.raises(IdentificationError, match="cannot identify") as exc:
        NFPGMMProblem(model, [IVMoments(d["Z"])], linear, make_layout())
    print(exc.value)
    # fixing two nonlinear parameters leaves 5 >= 1 + 3 unknowns: accepted
    NFPGMMProblem(model, [IVMoments(d["Z"])], linear, make_layout(free=(True, False, False)))


def test_refuses_rank_deficient_instruments() -> None:
    """A duplicated instrument column adds a column but no rank: 6 columns, rank 5, 6 unknowns."""
    d = make_data()
    Z = np.column_stack([d["Z"], d["Z_excl"][:, 0]])
    with pytest.raises(IdentificationError, match="rank"):
        LinearIV(d["X1"], Z)
    # the problem-level count uses rank(Z), not the column count (Absorb has no Z of its own)
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
    from nfp_harness.linear import Absorb
    layout6 = ParameterLayout(make_layout().specs + [
        ParameterSpec(f"extra{i}", "taste", "x", None, 0.0, -1.0, 1.0, unit=1.0) for i in range(3)])
    with pytest.raises(IdentificationError, match=r"rank\(Z\) 5 \+ 0 micro moments = 5 < 6"):
        NFPGMMProblem(model, [IVMoments(Z)], Absorb(d["market_ids"]), layout6)


def test_out_of_domain_and_inversion_failure_do_not_raise() -> None:
    problem, model, _ = make_problem()
    ev = problem.evaluate(np.array([-0.1, 0.2, 0.5]))
    assert ev.status is EvalStatus.OUT_OF_DOMAIN and ev.G is None and np.all(np.isnan(ev.g))
    assert "sigma_x" in ev.aux["note"]
    model.fail_above = 1.0
    ev = problem.evaluate(np.array([1.5, 0.2, 0.5]))
    assert ev.status is EvalStatus.INVERSION_FAILED and not ev.ok


def test_commit_restore_and_state_dict() -> None:
    problem, model, _ = make_problem()
    th1 = np.array([0.6, 0.2, 0.5])
    problem.evaluate(th1)
    problem.commit(th1)
    np.testing.assert_allclose(problem.accepted_theta, th1)
    accepted = [a.copy() for a in model.accepted]
    problem.evaluate(np.array([0.9, 0.1, 0.4]))
    problem.restore(th1)
    for a, b in zip(model.accepted, accepted):
        np.testing.assert_array_equal(a, b)
    assert model.calls[-1] == "restore" and "commit" in model.calls
    state = problem.state_dict()
    problem2, model2, _ = make_problem()
    problem2.load_state_dict(state)
    np.testing.assert_array_equal(problem2.W, problem.W)
    for a, b in zip(model2.accepted, accepted):
        np.testing.assert_array_equal(a, b)


def test_max_step_hook_follows_model() -> None:
    problem, _, _ = make_problem()
    assert not hasattr(problem, "max_step")
    problem, _, _ = make_problem(with_max_step=True)
    ev = problem.evaluate(np.array([0.6, 0.2, 0.5]))
    assert problem.max_step(ev, np.ones(3)).tau_cap == 0.5


def test_set_weighting_reconcentrates_and_refuses_coupled_W() -> None:
    problem, _, d = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    beta1 = problem.evaluate(theta).aux["beta"]
    W = np.eye(problem.k)
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 5))
    W[:5, :5] = A @ A.T + np.eye(5)
    problem.set_weighting(W)
    ev = problem.evaluate(theta)
    Z, X1 = d["Z"], d["X1"]
    beta_expected = np.linalg.solve(X1.T @ Z @ W[:5, :5] @ Z.T @ X1, X1.T @ Z @ W[:5, :5] @ Z.T @ ev.aux["delta"])
    np.testing.assert_allclose(ev.aux["beta"], beta_expected, rtol=1e-10)
    assert not np.allclose(beta1, ev.aux["beta"])
    W_bad = W.copy()
    W_bad[0, 6] = W_bad[6, 0] = 0.1
    with pytest.raises(ValueError, match="block-diagonal"):
        problem.set_weighting(W_bad)


def test_covariance_resolves_product_parts_and_scales_micro_block() -> None:
    problem, model, _ = make_problem()
    theta = np.array([0.6, 0.2, 0.5])
    ev = problem.evaluate(theta)
    n_solves = model.calls.count("solve")
    S = problem.covariance(ev)
    assert model.calls.count("solve") == n_solves + 1  # product parts were not in the evaluation
    assert S.shape == (8, 8)
    np.testing.assert_allclose(S[:5, 5:], 0.0)
    V = problem.micro_covariance(ev)
    np.testing.assert_allclose(S[5:, 5:], problem.n * V, rtol=1e-12)
    np.testing.assert_allclose(S[:5, :5], problem.agg.covariance_block(ev.aux["solution"], ev.aux["linear_fit"]))
    W1 = problem.initial_weighting(theta)
    np.testing.assert_allclose(np.linalg.inv(W1[5:, 5:]), problem.n * V, rtol=1e-8)


def test_full_jacobian_layout() -> None:
    problem, _, d = make_problem()
    ev = problem.evaluate(np.array([0.6, 0.2, 0.5]))
    J, names = problem.full_jacobian(ev)
    assert names == ["const", "x", "mi", "sigma_x", "pi_mi_fs", "alpha_age"]
    np.testing.assert_allclose(J[:5, :3], -d["Z"].T @ d["X1"] / problem.n)
    np.testing.assert_allclose(J[5:, :3], 0.0)
    np.testing.assert_array_equal(J[:, 3:], ev.G)

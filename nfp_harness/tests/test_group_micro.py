"""Tests for the GMY household micro moments (GMY spec WP-G3): values, Jacobian, never-buyers, fixed covariance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.models.pcm import PCMModel, PCMSpec
from nfp_harness.moments import GROUP_OUTPUTS, GroupMicroMoments, GroupMomentRow, MicroDataset
from nfp_harness.params import ParameterLayout
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.types import EvalStatus, InnerSolution, MarketIndex, ParameterSpec

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel

J, A, NU, T = 12, 16, 4, 2   # 64 draws per market (a power of two)
CHARS = ("const", "x1")
ROWS = [GroupMomentRow("diff", "prices", "A", "B"), GroupMomentRow("diff", "x1", "A", "notA"),
        GroupMomentRow("ratio", None, "A", "B"), GroupMomentRow("ratio", None, "notA", "A"),
        GroupMomentRow("diff", "prices", "B", "notA")]


#: a characteristic x demographic taste (as GMY's round-4 pi_age x z log footprint), added by ``layout(interaction=True)``
INTERACTION = ParameterSpec("pi_x1_demo", "taste", "x1", "demo", 0.25, -5.0, 5.0, unit=0.1)


def layout(interaction: bool = False) -> ParameterLayout:
    return ParameterLayout([
        ParameterSpec("sigma_x1", "taste", "x1", "nu0", 0.3, 0.0, 5.0, unit=0.1),
        ParameterSpec("pi_const_demo", "taste", "const", "demo", 0.2, -5.0, 5.0, unit=0.1),
    ] + ([INTERACTION] if interaction else []) + [
        ParameterSpec("phi_sc", "scale", None, "sc", 0.5, -4.0, 4.0, unit=0.1),
        ParameterSpec("sigma_p", "sigma_p", None, None, 1.2, 0.5, 3.0, unit=0.1),
        ParameterSpec("a_lo", "bound", None, None, 0.2, 0.2, 0.2, unit=0.1, free=False),
    ])


def make_data(stingy: bool = False, seed: int = 5) -> tuple[dict[str, Any], np.ndarray]:
    """Two markets of 12 products and 16 households x 4 draws; delta concave in price, so most products are bought.
    With ``stingy`` household 0 of each market has a price scale e^10: under the bound a_i >= 0.2 it never buys."""
    rng = np.random.default_rng(seed)
    mk = np.repeat(np.arange(1, T + 1), J)
    p = np.tile(1.0 + 0.35 * np.arange(J), T) + 0.05 * rng.random(T * J)
    x1 = rng.normal(size=T * J)
    delta = 1.6 * p - 0.08 * p ** 2 - 0.4 + 0.01 * rng.normal(size=T * J)
    am = np.repeat(np.arange(1, T + 1), A)
    w = rng.uniform(0.5, 1.5, T * A)
    w = w / np.repeat([w[am == t].sum() for t in range(1, T + 1)], A)
    sc = 0.3 * rng.normal(size=T * A)
    if stingy:
        sc[::A] = 20.0
    Aset = (rng.random(T * A) < 0.5).astype(float)
    Aset[:3] = (1.0, 0.0, 1.0)
    Aset[::A] = 1.0   # household 0 of each market (the stingy one) is in set A
    agents = {"market_ids": am, "weights": w, "demo": rng.normal(size=T * A), "sc": sc, "A": Aset,
              "notA": 1.0 - Aset, "B": (rng.random(T * A) < 0.4).astype(float)}
    products = {"market_ids": mk, "prices": p, "x1": x1, "shares": np.full(T * J, 0.01)}
    return {"products": products, "agents": agents}, delta


def model_at(stingy: bool = False, interaction: bool = False) -> tuple[PCMModel, np.ndarray, dict[str, Any]]:
    data, delta = make_data(stingy)
    lay = layout(interaction)
    model = PCMModel(PCMSpec(CHARS, draws_per_market=A * NU, tol=1e-12), data, lay, NU, 11, total_threads=4)
    model.group_sums_backend = "oracle"
    th, ph, sp = model.kernel_params(lay.starts())
    s = np.empty(T * J)
    for t, m in enumerate(model.markets):
        s[m.rows] = pcm_kernel.evaluate_market(delta[m.rows], model.kernel_market(t), th, ph, sp, derivatives=False,
                                               **model.kernel_kwargs(lay.starts()))["s"]
    data["products"]["shares"] = s
    return model, delta, data


def moment(model: PCMModel, data: dict[str, Any], never_buyers: str = "drop") -> GroupMicroMoments:
    g = GroupMicroMoments("micro_g", MicroDataset("survey", 1), ROWS, np.zeros(len(ROWS)),
                          np.linspace(1e-3, 5e-3, len(ROWS)), never_buyers=never_buyers)
    g.bind(model.markets, model.n_products, model.P)
    g.bind_agents(data["agents"], [model.market_draws(t)["agent_rows"] for t in range(model.T)])
    return g


def solution(model: PCMModel, delta: np.ndarray, theta: np.ndarray, derivatives: bool) -> InnerSolution:
    out = [model.group_sums(t, delta[m.rows], theta, ("prices", "x1"), derivatives) for t, m in enumerate(model.markets)]
    return InnerSolution(delta=delta, D=None, status=EvalStatus.OK, per_market=[], outputs={GROUP_OUTPUTS: out})


def hand_values(model: PCMModel, delta: np.ndarray, theta: np.ndarray, data: dict[str, Any],
                never_buyers: str = "drop") -> np.ndarray:
    """Explicit loops over households and draws, per-draw shares from one-draw markets."""
    th, ph, sp = model.kernel_params(theta)
    kw = model.kernel_kwargs(theta)
    hh: list[tuple[int, float, float, dict[str, float]]] = []   # (agent row, w_g, P_g, {char: S_g})
    for t, m in enumerate(model.markets):
        km = model.kernel_market(t)
        md = model.market_draws(t)
        x, p = np.asarray(km.x), np.asarray(km.p)
        for g, arow in enumerate(md["agent_rows"]):
            wg = Pg = 0.0
            Sg = {"prices": 0.0, "x1": 0.0}
            for i in np.flatnonzero(md["agent_of_draw"] == g):
                one = pcm_kernel.Market(x, p, np.ones(1), np.asarray(km.taste_basis)[i:i + 1], list(km.taste_col),
                                        np.asarray(km.scale_basis)[i:i + 1])
                s_i = np.asarray(pcm_kernel.evaluate_market(delta[m.rows], one, th, ph, sp, derivatives=False,
                                                            **kw)["s"])
                wi = float(md["weights"][i])
                wg += wi
                Pg += wi * s_i.sum()
                Sg["prices"] += wi * float(p @ s_i)
                Sg["x1"] += wi * float(x[:, 1] @ s_i)
            hh.append((int(arow), wg, Pg, Sg))
    ag = data["agents"]

    def M(s: str, c: str) -> float:
        num = den = 0.0
        for arow, wg, Pg, Sg in hh:
            if ag[s][arow] != 1:
                continue
            if Pg > 0:
                num += wg * Sg[c] / Pg
                den += wg
            elif never_buyers == "keep":
                den += wg
        return num / den

    def Arate(s: str) -> float:
        return sum(Pg for arow, wg, Pg, _ in hh if ag[s][arow] == 1) / sum(wg for arow, wg, _, _ in hh
                                                                            if ag[s][arow] == 1)

    return np.array([M(r.set_a, r.char) - M(r.set_b, r.char) if r.kind == "diff" else Arate(r.set_a) / Arate(r.set_b)
                     for r in ROWS])


def test_values_match_hand_computation() -> None:
    model, delta, data = model_at()
    theta = layout().starts()
    g = moment(model, data)
    np.testing.assert_allclose(g.predicted(solution(model, delta, theta, False)), hand_values(model, delta, theta, data),
                               rtol=1e-12, atol=1e-13)
    model.close()


@pytest.mark.parametrize("interaction", [False, True], ids=["base", "x1_x_demo"])
def test_jacobian_matches_finite_differences(interaction: bool) -> None:
    """With ``interaction`` the layout adds a characteristic x demographic taste (pi_x1_demo, GMY round 4's age x
    footprint form); every free column, that one too, matches central differences and the bound's column is zero."""
    model, delta, data = model_at(interaction=interaction)
    lay = layout(interaction)
    theta = lay.starts()
    g = moment(model, data)
    _, dd, dt = g.moments(solution(model, delta, theta, True), True)
    assert dd is not None and dt is not None
    h = 1e-6
    rows = model.markets[0].rows
    for l in rows[[1, 4, 7]].tolist() + model.markets[1].rows[[2, 9]].tolist():
        e = np.zeros_like(delta)
        e[l] = h
        fd = (g.predicted(solution(model, delta + e, theta, False))
              - g.predicted(solution(model, delta - e, theta, False))) / (2 * h)
        np.testing.assert_allclose(dd[:, l], fd, rtol=1e-6, atol=1e-8)
    nb = lay.names.index("a_lo")
    for k in range(lay.P):   # free parameters; the bound's column is zero
        if k == nb:
            continue
        e = np.zeros_like(theta)
        e[k] = h
        fd = (g.predicted(solution(model, delta, theta + e, False))
              - g.predicted(solution(model, delta, theta - e, False))) / (2 * h)
        np.testing.assert_allclose(dt[:, k], fd, rtol=1e-6, atol=1e-8)
    assert np.all(dt[:, nb] == 0)
    model.close()


def test_never_buyers_drop_and_keep() -> None:
    model, delta, data = model_at(stingy=True)
    theta = layout().starts()
    sol = solution(model, delta, theta, False)
    zero = [int(np.sum(e["S"][:, 0] == 0)) for e in sol.outputs[GROUP_OUTPUTS]]
    assert zero == [1, 1]
    drop, code = moment(model, data, "drop"), moment(model, data, "keep")
    v_drop, v_code = drop.predicted(sol), code.predicted(sol)
    np.testing.assert_allclose(v_drop, hand_values(model, delta, theta, data, "drop"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(v_code, hand_values(model, delta, theta, data, "keep"), rtol=1e-12, atol=1e-13)
    assert drop.last_info["micro_never_buyers"]["A"] == 2   # household 0 of both markets is in set A
    # row 1 is M_A[x1] - M_notA[x1] and notA holds no never-buyer, so the two rows differ by
    # M_A,drop - M_A,code = M_A,drop (W_A - W_A+) / W_A = M_A,drop w_never / W_A
    num = W_plus = W_all = w_never = 0.0
    for t, e in enumerate(sol.outputs[GROUP_OUTPUTS]):
        inA = data["agents"]["A"][model.market_draws(t)["agent_rows"]] == 1
        buy = inA & (e["S"][:, 0] > 0)
        num += float(np.sum(e["w"][buy] * e["S"][buy, 2] / e["S"][buy, 0]))
        W_plus += float(e["w"][buy].sum())
        W_all += float(e["w"][inA].sum())
        w_never += float(e["w"][inA & ~(e["S"][:, 0] > 0)].sum())
    M_A = num / W_plus
    np.testing.assert_allclose(v_drop[1] - v_code[1], M_A * w_never / W_all, rtol=1e-10, atol=1e-15)
    model.close()


def test_fixed_covariance_weighting() -> None:
    model, delta, data = model_at()
    X1 = np.column_stack([np.ones(T * J), data["products"]["x1"]])
    Z = np.column_stack([X1, data["products"]["prices"] ** 2])
    g = GroupMicroMoments("micro_g", MicroDataset("survey", 1), ROWS, np.full(len(ROWS), 0.1),
                          np.linspace(1e-3, 5e-3, len(ROWS)))
    prob = NFPGMMProblem(model, [g], LinearIV(X1, Z), layout())
    n = prob.n
    W1 = prob.initial_weighting(layout().starts())
    np.testing.assert_allclose(W1, np.diag(1.0 / g.variances) / n, rtol=1e-14)
    prob.set_weighting(W1)
    ev = prob.evaluate(layout().starts(), need_jacobian=False)
    assert ev.ok, ev.aux.get("note")
    np.testing.assert_allclose(prob.covariance(ev), n * np.diag(g.variances), rtol=1e-14)
    np.testing.assert_allclose(ev.aux["q_blocks"]["micro_g"], np.sum((ev.g / np.sqrt(g.variances)) ** 2), rtol=1e-12)
    assert "micro_never_buyers" in ev.aux["micro_g"]
    np.testing.assert_allclose(ev.aux["micro_predicted"]["micro_g"],
                               hand_values(model, ev.aux["delta"], layout().starts(), data), rtol=1e-10)
    model.close()


def test_gmy_rows_reproduce_the_logit_evaluator() -> None:
    from nfp_harness.micro_sets import MicroMomentEvaluator
    from nfp_harness.tests.test_gmy_instruments import gmy_module
    gmy = gmy_module()
    try:
        data = gmy.load_data(n_households=200, years=(1985, 1995, 2003))
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    p, a = data.products, gmy.add_set_columns(data.agents)
    rng = np.random.default_rng(0)
    years = np.unique(p.market_ids)
    P_list, entries, markets, agent_rows = [], [], [], []
    for y in years:
        rows = np.flatnonzero(p.market_ids.to_numpy() == y)
        arow = np.flatnonzero(a.market_ids.to_numpy() == y)
        u = rng.normal(size=(rows.size, arow.size)) - 4.0
        P = np.exp(u) / (1.0 + np.exp(u).sum(axis=0, keepdims=True))
        P_list.append(P)
        w = a.weights.to_numpy(float)[arow]
        cols = ("prices", "van", "log_footprint", "truck")
        S = np.column_stack([P.sum(axis=0)] + [p[c].to_numpy(float)[rows] @ P for c in cols]) * w[:, None]
        entries.append({"columns": cols, "S": S, "w": w, "d_delta": None, "d_theta": None})
        markets.append(MarketIndex(int(y), rows))
        agent_rows.append(arow)
    want = MicroMomentEvaluator(p, a, data.micro, gmy.MICRO_CHAR)(P_list)
    rows = [GroupMomentRow(*r) for r in gmy.group_moment_rows(data.micro)]
    g = GroupMicroMoments("micro_gmy", MicroDataset("GMY", 1), rows, data.micro.values, data.micro.variances)
    g.bind(markets, len(p), 3)
    g.bind_agents(a, agent_rows)
    got = g.predicted(InnerSolution(delta=np.zeros(len(p)), D=None, status=EvalStatus.OK, per_market=[],
                                    outputs={GROUP_OUTPUTS: entries}))
    assert got.shape == (33,)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)

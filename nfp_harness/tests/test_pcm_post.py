"""Tests for pcm_post and diagnostics.economics (WP8; DESIGN 7 WP8 acceptance).

A small synthetic PCM (2 markets, 256 draws per market, 3 firms, one body-type-like dummy) is
evaluated at a fixed delta; its data shares are the kernel's shares there, so no inversion is needed.
The agent table carries a constant column ``one`` with a scale parameter ``phi_one`` so that the joint
utility rescaling (delta, t, c) -> kappa (delta, t, c) is a move in theta (LEARNINGS 11).
"""

from __future__ import annotations

import math
import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nfp_harness.data import break_price_ties
from nfp_harness.params import ParameterLayout
from nfp_harness.types import ConvergenceClass, EvalStatus, Evaluation, OptimizeResult, ParameterSpec

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel
from nfp_harness.diagnostics import DiagnosticInputs, build_report, validate  # noqa: E402
from nfp_harness.diagnostics.economics import economics_block, plausible_flags  # noqa: E402
from nfp_harness.diagnostics.report import render_text, to_jsonable  # noqa: E402
from nfp_harness.models import pcm_post as pp  # noqa: E402
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHARS = ("const", "x1", "x2", "g1")
THETA = np.array([2.0, 1.5, 1.0, 0.3, 0.6, 0.0, 0.8])  # sigma_x1, sigma_x2, sigma_g1, pi_x1_demo, phi_inc, phi_one, sigma_p
I_TASTE, I_ONE = [0, 1, 2, 3], 5


def layout() -> ParameterLayout:
    return ParameterLayout([
        ParameterSpec("sigma_x1", "taste", "x1", "nu0", 2.0, 0.0, 50.0, unit=1.0),
        ParameterSpec("sigma_x2", "taste", "x2", "nu1", 1.5, 0.0, 50.0, unit=1.0),
        ParameterSpec("sigma_g1", "taste", "g1", "nu2", 1.0, 0.0, 50.0, unit=1.0),
        ParameterSpec("pi_x1_demo", "taste", "x1", "demo", 0.3, -math.inf, math.inf, unit=1.0),
        ParameterSpec("phi_inc", "scale", None, "inc", 0.6, -math.inf, math.inf, unit=1.0),
        ParameterSpec("phi_one", "scale", None, "one", 0.0, -math.inf, math.inf, unit=1.0),
        ParameterSpec("sigma_p", "sigma_p", None, None, 0.8, 0.05, 3.0, unit=0.1),
    ])


def _shares(model: PCMModel, theta: np.ndarray, delta: np.ndarray) -> np.ndarray:
    th, ph, sp = model.kernel_params(theta)
    s = np.empty(model.n_products)
    for t, m in enumerate(model.markets):
        s[m.rows] = pcm_kernel.evaluate_market(delta[m.rows], model.kernel_market(t), th, ph, sp,
                                               derivatives=False)["s"]
    return s


def make_market(seed: int = 5, J0: int = 12, A: int = 64, dominated: bool = False) -> dict[str, Any]:
    """Products and agents with every share above 1e-4 at (THETA, delta); optionally one extra product
    per market that copies a cheaper product's characteristics with delta lower by 1 and price higher
    by 0.5, so it is strictly dominated for every draw."""
    rng = np.random.default_rng(seed)
    n = 2 * J0
    mk = np.repeat([1984, 1985], J0)
    a = rng.uniform(0, 2 * np.pi, n)
    x1, x2 = np.cos(a), np.sin(a)
    g1 = (rng.uniform(size=n) < 0.35).astype(float)
    p = 2.0 + rng.uniform(0, 4, n) + 0.8 * (1 + x1)
    delta = 1.0 + 0.8 * x1 + 0.4 * x2 + 0.5 * g1 + rng.normal(0, 0.3, n)
    firm = rng.integers(1, 4, n)
    am = np.repeat([1984, 1985], A)
    agents = {"market_ids": am, "weights": np.full(am.size, 1.0 / A), "demo": rng.integers(1, 6, am.size).astype(float),
              "inc": rng.normal(0, 0.5, am.size), "one": np.ones(am.size)}
    keep = np.arange(n)
    for _ in range(30):
        products = {"market_ids": mk[keep], "prices": break_price_ties(p[keep], mk[keep], 0.01), "x1": x1[keep],
                    "x2": x2[keep], "g1": g1[keep], "firm_ids": firm[keep], "shares": np.full(keep.size, 0.01)}
        model = PCMModel(PCMSpec(CHARS), {"products": products, "agents": agents}, layout(), 4, seed, total_threads=4)
        s = _shares(model, THETA, delta[keep])
        model.close()
        if s.min() > 1e-4:
            break
        keep = keep[s > 1e-4]
    else:
        raise AssertionError("pruning did not settle")
    d = delta[keep]
    if dominated:
        add = {k: [] for k in products}
        dd = []
        for m in (1984, 1985):
            rows = np.flatnonzero(products["market_ids"] == m)
            j = rows[np.argsort(products["prices"][rows])[len(rows) // 2]]
            for k in products:
                add[k].append(products[k][j])
            add["prices"][-1] = products["prices"][j] + 0.5 + 1e-3 * rng.uniform()
            dd.append(d[j] - 1.0)
        products = {k: np.r_[products[k], np.asarray(add[k], dtype=products[k].dtype)] for k in products}
        d = np.r_[d, dd]
        if np.any(np.diff(np.sort(products["prices"][products["market_ids"] == 1984])) < 1e-3):
            raise AssertionError("dominated product sits on a price tie")
    data = {"products": products, "agents": agents}
    model = PCMModel(PCMSpec(CHARS), data, layout(), 4, seed, total_threads=4)
    products["shares"] = _shares(model, THETA, d)
    model.close()
    return {"data": data, "delta": d}


@pytest.fixture(scope="module")
def syn() -> dict[str, Any]:
    return make_market()


@pytest.fixture
def model(syn: dict[str, Any]) -> Any:
    m = PCMModel(PCMSpec(CHARS), syn["data"], layout(), 4, 5, total_threads=4)
    yield m
    m.close()


def kappa_point(theta: np.ndarray, delta: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    th = theta.copy()
    th[I_TASTE] *= kappa
    th[I_ONE] += math.log(kappa)
    return th, kappa * delta


# --------------------------------------------------------------------------------------------
# Array functions
# --------------------------------------------------------------------------------------------


def test_array_definitions_on_a_hand_example() -> None:
    s = np.array([0.2, 0.1, 0.05])
    J = np.array([[-0.5, 0.1, 0.02], [0.2, -0.4, 0.0], [0.05, 0.0, -0.3]])  # J[j, k] = ds_j / dp_k
    p = np.array([2.0, 3.0, 4.0])
    e = pp.elasticities(s, J, p)
    assert e[0, 1] == pytest.approx(0.1 * 3 / 0.2) and e[1, 1] == pytest.approx(-0.4 * 3 / 0.1)
    assert pp.nonzero_cross(e).tolist() == [2, 1, 1]
    D = pp.diversion(J)  # D[j, k] = -(ds_k/dp_j) / (ds_j/dp_j)
    assert D[0, 1] == pytest.approx(0.2 / 0.5) and D[1, 0] == pytest.approx(0.1 / 0.4) and D[1, 2] == 0.0
    gd = pp.group_diversion(D, np.array(["a", "a", "b"]), ["a", "b"])
    assert gd["within"][0] == pytest.approx(0.4 / (0.4 + 0.1))
    np.testing.assert_allclose(gd["to_group"].sum(1) + gd["outside"], 1.0)
    nm = pp.nash_markups(s, J, np.array([1, 1, 2]))
    assert nm["status"] == pp.NASH_OK and nm["foc_residual"] < 1e-15
    A = np.array([[-0.5, 0.2, 0], [0.1, -0.4, 0], [0, 0, -0.3]])  # Omega * J'
    np.testing.assert_allclose(nm["markup"], -np.linalg.solve(A, s))
    assert pp.inside_share_elasticity(s, J, p) == pytest.approx(float((J @ p).sum() / s.sum()))


def test_singular_nash_is_a_named_status_not_an_exception() -> None:
    s = np.array([0.1, 0.1, 0.0])
    J = np.array([[-0.5, 0.1, 0.0], [0.1, -0.5, 0.0], [0.0, 0.0, 0.0]])  # third product has no buyers
    nm = pp.nash_markups(s, J, np.array([1, 2, 3]))
    assert nm["status"] == "NASH_JACOBIAN_SINGULAR" and np.all(np.isnan(nm["markup"]))
    J2 = np.array([[-0.5, 0.5], [0.5, -0.5]])  # exactly singular after ownership weighting
    assert pp.nash_markups(np.array([0.1, 0.1]), J2, np.array([1, 1]))["status"] == "NASH_JACOBIAN_SINGULAR"


# --------------------------------------------------------------------------------------------
# Synthetic market
# --------------------------------------------------------------------------------------------


def _signatures(km: Any, delta: np.ndarray, th: np.ndarray, ph: np.ndarray, sp: float) -> np.ndarray:
    return np.asarray(pcm_kernel.frontier_signature(delta, km, th, ph, sp, 1))


def test_elasticities_match_finite_differences_of_shares(model: Any, syn: dict[str, Any]) -> None:
    th, ph, sp = model.kernel_params(THETA)
    worst = 0.0
    for t, m in enumerate(model.markets):
        post = pp.market_post(model, THETA, syn["delta"], t)
        km = model.kernel_market(t)
        f = {k: getattr(km, k) for k in ("x", "weights", "taste_basis", "taste_col", "scale_basis")}
        d = syn["delta"][m.rows]
        p = np.asarray(km.p)
        base = _signatures(km, d, th, ph, sp)
        for k in range(p.size):
            h = 1e-6
            while True:
                mk = [pcm_kernel.Market(f["x"], p + sg * h * np.eye(p.size)[k], f["weights"], f["taste_basis"],
                                        f["taste_col"], f["scale_basis"]) for sg in (1, -1)]
                if all(np.array_equal(_signatures(q, d, th, ph, sp), base) for q in mk) or h < 1e-10:
                    break
                h /= 10
            sp_, sm_ = (pcm_kernel.evaluate_market(d, q, th, ph, sp, derivatives=False)["s"] for q in mk)
            fd = (sp_ - sm_) / (2 * h)
            e_fd = fd * p[k] / post["s"]
            err = np.max(np.abs(e_fd - post["elasticity"][:, k])) / max(1.0, np.max(np.abs(post["elasticity"][:, k])))
            worst = max(worst, err)
    print(f"elasticity vs finite differences, worst relative error {worst:.2e}")
    assert worst < 1e-6


def test_nash_costs_satisfy_the_foc(model: Any, syn: dict[str, Any]) -> None:
    post = pp.postestimation(model, THETA, syn["delta"], keep_matrices=True)
    worst = 0.0
    for m in post["markets"]:
        r = m["post"]
        assert r["nash"]["status"] == "ok"
        res = pp.foc_residual(r["s"], r["ds_dp"], np.asarray(syn["data"]["products"]["firm_ids"])[r["rows"]],
                              r["p"], r["nash"]["cost"])
        worst = max(worst, float(np.max(np.abs(res))))
    print(f"Nash FOC residual at computed costs {worst:.2e}")
    assert worst < 1e-8
    pr = post["products"]
    assert np.all(np.isfinite(pr["margin"])) and np.all(pr["own_elasticity"] < 0)
    np.testing.assert_allclose(pr["shares"], syn["data"]["products"]["shares"], atol=1e-14)


def test_group_diversion_adds_up(model: Any, syn: dict[str, Any]) -> None:
    post = pp.postestimation(model, THETA, syn["delta"], groups={"g1": "g1"}, group_other="base")
    assert post["group_names"] == ["g1", "base"]
    pr = post["products"]
    tot = pr["diversion_to_g1"] + pr["diversion_to_base"] + pr["outside_diversion"]
    np.testing.assert_allclose(tot, 1.0, atol=1e-12)
    np.testing.assert_allclose(post["group_matrix"].sum(1), 1.0, atol=1e-12)
    ok = np.isfinite(pr["diversion_within_group"])
    np.testing.assert_allclose(pr["diversion_within_group"][ok] + pr["diversion_leaving_group"][ok], 1.0)
    # the pooled cross-group diversion is the group matrix's inside flow leaving its group, weighted by group sales
    from nfp_harness.diagnostics.economics import pooled_cross_group_diversion

    gm, lab, s = post["group_matrix"], np.asarray(pr["group"]).astype(str), pr["shares"]
    fin = np.isfinite(pr["diversion_to_g1"]) & np.isfinite(pr["diversion_to_base"])
    sales = np.array([s[fin & (lab == g)].sum() for g in post["group_names"]])
    inside = gm[:, :-1].sum(1)
    expect = 1.0 - float(sales @ np.diag(gm)) / float(sales @ inside)
    assert pooled_cross_group_diversion(post) == pytest.approx(expect, rel=1e-12)
    assert 0.0 < expect < 1.0
    # the product median is a separate statistic: here it differs from the pooled flow
    assert np.nanmedian(pr["diversion_leaving_group"]) != pytest.approx(expect, rel=1e-6)
    assert pooled_cross_group_diversion(pp.postestimation(model, THETA, syn["delta"])) is None


def test_window_anatomy_frontier_size_histogram(model: Any, syn: dict[str, Any]) -> None:
    """The buyer-mass histogram by frontier size is the kernel's per-draw frontier_size weighted by w_i x the draw's
    inside share, pooled and per market; it sums to 1 and its mean over the bins below 4+ matches the mean frontier
    size when no draw has 4 or more frontier products."""
    wa = pp.window_anatomy(model, THETA, syn["delta"])
    th, ph, sp = model.kernel_params(THETA)
    tot = {b: 0.0 for b in pp.FRONTIER_SIZE_BINS}
    mass = 0.0
    sizes = []
    for t, m in enumerate(model.markets):
        km = model.kernel_market(t)
        r = pcm_kernel.frontier_anatomy(syn["delta"][np.asarray(m.rows)], km, th, ph, sp, 2, **pp._bound(model, THETA))
        wm = np.asarray(km.weights) * np.asarray(r["mass"])
        size = np.asarray(r["frontier_size"])
        sizes.append(size[wm > 0])
        for k, b in enumerate(pp.FRONTIER_SIZE_BINS, start=1):
            tot[b] += float(wm[size >= k].sum() if b == "4+" else wm[size == k].sum())
        mass += float(wm.sum())
        hm = wa["by_market"][str(m.market_id)]["frontier_size_mass"]
        assert sum(hm.values()) == pytest.approx(1.0, abs=1e-12)
    h = wa["frontier_size_mass"]
    assert list(h) == list(pp.FRONTIER_SIZE_BINS)
    for b in pp.FRONTIER_SIZE_BINS:
        assert h[b] == pytest.approx(tot[b] / mass, rel=1e-12, abs=1e-15)
    assert wa["single_frontier_buyer_share"] == h["1"]
    assert sum(h.values()) == pytest.approx(1.0, abs=1e-12)
    assert len(set(np.concatenate(sizes).tolist())) > 1   # the toy has buyers on frontiers of different sizes
    if h["4+"] == 0.0:
        assert sum(int(b) * v for b, v in h.items() if b != "4+") == pytest.approx(
            wa["frontier_products_per_buying_draw"], rel=1e-12)


def test_pooled_cross_group_diversion_when_the_product_median_is_zero() -> None:
    """Three products of group a divert only within a (leaving share 0), one big product of group b sends half its
    inside diversion to a: the median over products is 0, the pooled flow is 1 - (0.1 * 3 * 0.8 + 0.6 * 0.4) / (0.1 * 3
    * 0.8 + 0.6 * 0.8) = 1/3. The pre-round-2 statistic (the median) would read as a segmented corner."""
    from nfp_harness.diagnostics.economics import pooled_cross_group_diversion

    pr = {"shares": np.array([0.1, 0.1, 0.1, 0.6]), "group": np.array(["a", "a", "a", "b"], dtype=object),
          "diversion_to_a": np.array([0.8, 0.8, 0.8, 0.4]), "diversion_to_b": np.array([0.0, 0.0, 0.0, 0.4]),
          "diversion_leaving_group": np.array([0.0, 0.0, 0.0, 0.5])}
    post = {"products": pr, "group_names": ["a", "b"]}
    assert float(np.median(pr["diversion_leaving_group"])) == 0.0
    assert pooled_cross_group_diversion(post) == pytest.approx(1.0 - (0.24 + 0.24) / (0.24 + 0.48), rel=1e-12)
    # SEGMENTED_CORNER reads the pooled flow: 33% is not a corner, the product median of 0 was
    from nfp_harness.diagnostics.degeneracies import segmented_corner
    from nfp_harness.types import ParameterSpec

    specs = [ParameterSpec("sigma_x", "taste", "x", "nu0", 1.0, 0.0, 10.0)]
    assert not segmented_corner(np.array([1.0]), specs, 10.0, pooled_cross_group_diversion(post))["detected"]
    assert segmented_corner(np.array([1.0]), specs, 10.0, 0.0)["detected"]
    pr["diversion_to_a"][3] = np.nan     # a row with non-finite diversion is left out, as in the group matrix
    assert pooled_cross_group_diversion(post) == pytest.approx(0.0, abs=1e-15)


def test_removing_a_dominated_product_leaves_cs_unchanged() -> None:
    syn = make_market(dominated=True)
    with PCMModel(PCMSpec(CHARS), syn["data"], layout(), 4, 5, total_threads=4) as model:
        pr = syn["data"]["products"]
        dom = np.zeros(pr["prices"].size, dtype=bool)
        dom[-2:] = True  # the two appended products
        assert np.all(pr["shares"][dom] == 0.0)
        for mkt in (1984, 1985):
            cf = pp.drop_products_counterfactual(model, THETA, syn["delta"], mkt, dom)
            assert cf["dropped_rows"].size == 1
            print(f"market {mkt}: CS change from dropping a dominated product {cf['cs_change_fixed']:.2e}")
            assert abs(cf["cs_change_fixed"]) < 1e-10
            np.testing.assert_allclose(cf["fixed_prices"]["per_draw"], cf["observed"]["per_draw"], rtol=0, atol=1e-12)
            assert cf["fixed_prices"]["inside_share"] == pytest.approx(cf["observed"]["inside_share"], abs=1e-14)
            # the Nash counterfactual stops with the named status: a zero-share product makes Omega * ds/dp singular
            cfn = pp.drop_products_counterfactual(model, THETA, syn["delta"], mkt, dom, resolve_prices=True)
            assert cfn["nash"]["status"] == "NASH_JACOBIAN_SINGULAR"


def test_cs_is_invariant_to_a_joint_rescaling_of_utility(model: Any, syn: dict[str, Any]) -> None:
    drop = np.asarray(syn["data"]["products"]["g1"]) == 1
    base = pp.drop_products_counterfactual(model, THETA, syn["delta"], 1984, drop, resolve_prices=True)
    post0 = pp.postestimation(model, THETA, syn["delta"])["products"]
    for kappa in (0.37, 3.7):
        th, d = kappa_point(THETA, syn["delta"], kappa)
        cs0 = pp.consumer_surplus(model, THETA, syn["delta"], 1984)
        cs1 = pp.consumer_surplus(model, th, d, 1984)
        cf = pp.drop_products_counterfactual(model, th, d, 1984, drop, resolve_prices=True)
        post1 = pp.postestimation(model, th, d)["products"]
        rel = abs(cs1["total"] - cs0["total"]) / abs(cs0["total"])
        print(f"kappa {kappa}: CS {cs0['total']:.12f} -> {cs1['total']:.12f} (relative change {rel:.1e}); "
              f"fixed-price change {base['cs_change_fixed']:.10f} vs {cf['cs_change_fixed']:.10f}; "
              f"Nash change {base['cs_change_nash']:.10f} vs {cf['cs_change_nash']:.10f}")
        assert rel < 1e-12
        np.testing.assert_allclose(cs1["per_draw"], cs0["per_draw"], rtol=1e-11, atol=1e-13)
        assert cf["cs_change_fixed"] == pytest.approx(base["cs_change_fixed"], rel=1e-10, abs=1e-13)
        assert cf["cs_change_nash"] == pytest.approx(base["cs_change_nash"], rel=1e-6)
        np.testing.assert_allclose(post1["own_elasticity"], post0["own_elasticity"], rtol=1e-9)
        np.testing.assert_allclose(post1["markup"], post0["markup"], rtol=1e-9)


def test_bounded_nash_after_removal(model: Any, syn: dict[str, Any]) -> None:
    pr = syn["data"]["products"]
    drop = np.asarray(pr["g1"]) == 1
    cf = pp.drop_products_counterfactual(model, THETA, syn["delta"], 1984, drop, resolve_prices=True,
                                         tabulate=("demo", ("inc", [0.0])))
    nash = cf["nash"]
    print(f"bounded Nash: {nash['status']} after {nash['sweeps']} sweeps, {nash['evaluations']} evaluations, "
          f"{nash['box_binding']['count']} at a box end, largest +-2% deviation gain {nash['deviation']['max_gain']:.2e}; "
          f"CS observed {cf['observed']['cs']:.6f}, fixed prices {cf['fixed_prices']['cs']:.6f}, bounded Nash "
          f"{nash['cs']:.6f}")
    assert nash["method"] == "bounded_nash" and nash["status"] in ("converged", "cycle")
    # off the box ends no +-2% deviation pays; at a binding box end it may (the box is the bound)
    assert nash["sweeps"] <= 20 and nash["deviation"]["max_gain_off_box"] < 1e-3
    p_obs = cf["observed"]["prices"]
    kept = np.isin(cf["observed"]["rows"], cf["nash"]["rows"])
    assert np.all(np.abs(np.sort(nash["prices"]) / np.sort(p_obs[kept]) - 1.0) <= 0.10 + 1e-3)
    assert np.all(nash["shares"] > 0)
    # removing options never raises any draw's surplus at fixed prices
    assert np.all(cf["fixed_prices"]["per_draw"] <= cf["observed"]["per_draw"] + 1e-12)
    assert cf["cs_change_fixed"] < 0
    # tables add up to the totals and the quintiles split the mass into fifths
    for key in ("alpha_quintile", "demo", "inc"):
        rows = cf["tables"][key]
        assert sum(r["mass"] for r in rows) == pytest.approx(1.0)
        for sc, tot in (("observed", cf["observed"]["cs"]), ("fixed_prices", cf["fixed_prices"]["cs"]),
                        ("nash", nash["cs"])):
            assert sum(r[f"cs_total_{sc}"] for r in rows) == pytest.approx(tot, rel=1e-12)
    q = cf["tables"]["alpha_quintile"]
    assert [r["group"] for r in q] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([r["mass"] for r in q], 0.2, atol=0.02)


def test_buyer_alpha_reproduces_the_inside_share(model: Any, syn: dict[str, Any]) -> None:
    sb = pp.spread_in_buyer_dollars(model, THETA, syn["delta"])
    s = syn["data"]["products"]["shares"]
    inside = np.mean([s[m.rows].sum() for m in model.markets])
    assert sb["inside_share"] == pytest.approx(inside, rel=1e-10)
    assert set(sb["spreads"]) == {"sigma_x1", "sigma_x2", "sigma_g1"}
    assert sb["spreads"]["sigma_x1"] == pytest.approx(2.0 / sb["mean_alpha_buyers"])
    th, d = kappa_point(THETA, syn["delta"], 3.7)  # spreads in buyer dollars carry no utility units
    sb2 = pp.spread_in_buyer_dollars(model, th, d)
    assert sb2["spreads"]["sigma_x1"] == pytest.approx(sb["spreads"]["sigma_x1"], rel=1e-10)


def test_relative_quality_dispersion_by_hand(model: Any, syn: dict[str, Any]) -> None:
    rq = pp.relative_quality_dispersion(model, THETA, syn["delta"], exclude=("x2",))
    mab = rq["mean_alpha_buyers"]
    t = 0
    km = model.kernel_market(t)
    x, basis, w = np.asarray(km.x), np.asarray(km.taste_basis), np.asarray(km.weights)
    # taste columns: sigma_x1 (nu0), sigma_x2 (nu1, excluded), sigma_g1 (nu2), pi_x1_demo (demo)
    T = np.zeros((w.size, x.shape[1]))
    T[:, CHARS.index("x1")] = THETA[0] * basis[:, 0] + THETA[3] * basis[:, 3]
    T[:, CHARS.index("g1")] = THETA[2] * basis[:, 2]
    U = T @ x.T
    J = x.shape[0]
    j = 3
    d = U[:, j] - U[:, j + 1]
    w = w / w.sum()
    sd = math.sqrt(w @ (d - w @ d) ** 2)
    assert rq["per_product"][model.markets[t].rows[j]] == pytest.approx(sd / mab, rel=1e-10)
    last = U[:, J - 1] - U[:, J - 2]
    assert rq["per_product"][model.markets[t].rows[J - 1]] == pytest.approx(math.sqrt(w @ (last - w @ last) ** 2) / mab,
                                                                            rel=1e-10)
    assert np.isfinite(rq["median"])


def test_economics_block_flags_and_report(model: Any, syn: dict[str, Any]) -> None:
    n = model.n_products
    xi = np.random.default_rng(0).normal(0, 0.3, n)
    ev = Evaluation(theta=THETA, g=np.zeros(3), G=np.zeros((3, THETA.size)), status=EvalStatus.OK,
                    blocks={"aggregate": slice(0, 3)}, aux={"delta": syn["delta"], "xi": xi}, seconds=0.0)
    res = OptimizeResult(theta=THETA, q=0.0, convergence=ConvergenceClass.CONVERGED, evaluation=ev, history=[],
                         accepted_steps=1, evaluations=1, wall_seconds=1.0, projected_gradient=0.0, pg_tol=1.0,
                         gn_decrease=0.0, gn_tol=1.0, Delta=1.0, q_mass=0.0, W=np.eye(3))
    spec = SimpleNamespace(groups={"g1": "g1"}, group_other="base", plausible={
        "own_elasticity_median": (-12, -2), "share_own_elasticity_below_-50": (None, 0.01),
        "nonzero_cross_median": (100, None),  # violated on purpose
        "diversion_leaving_group_median": (0.05, None), "margin_median": (0.05, 0.95),
        "spread_buyer_dollars": {"sigma_g1": (None, "price_range")}, "sd_xi_over_sd_delta": (None, 1.0)})
    problem = SimpleNamespace(model=model, layout=model.layout)
    blk = economics_block(problem, res, spec)
    print("\n".join(blk.messages))
    assert blk.status == "flag"
    med = blk.values["nonzero_cross"]["median"]
    assert f"FLAG: nonzero_cross_median = {med:.4g} is outside the declared plausible range [100, inf]." in blk.messages
    assert sum(m.startswith("FLAG:") for m in blk.messages) == 1
    assert "spread_buyer_dollars:sigma_g1" in blk.thresholds["plausible"]
    assert blk.values["price_range"] == pytest.approx(np.ptp(syn["data"]["products"]["prices"]))
    from nfp_harness.diagnostics.economics import pooled_cross_group_diversion
    post = pp.postestimation(model, THETA, syn["delta"], groups={"g1": "g1"}, group_other="base")
    assert blk.values["cross_group_diversion"] == pytest.approx(pooled_cross_group_diversion(post), rel=1e-12)
    assert blk.values["cross_group_diversion_product_median"] == blk.values["diversion_leaving_group"]["median"]
    rep = build_report(DiagnosticInputs(result=res, specs=model.layout.specs, units=np.ones(THETA.size), n=n,
                                        economics=blk, xi=xi))
    d = to_jsonable(rep.to_dict())
    validate(d)
    assert "nonzero_cross_median" in render_text(rep)
    assert blk.values["own_elasticity"]["median"] == pytest.approx(
        float(np.median(pp.postestimation(model, THETA, syn["delta"])["products"]["own_elasticity"])))


def test_economics_block_not_run_without_an_evaluation() -> None:
    res = OptimizeResult(theta=THETA, q=0.0, convergence=ConvergenceClass.INVERSION_FAILED, evaluation=None,
                         history=[], accepted_steps=0, evaluations=1, wall_seconds=0.0, projected_gradient=0.0,
                         pg_tol=1.0, gn_decrease=0.0, gn_tol=1.0, Delta=1.0, q_mass=0.0)
    blk = economics_block(SimpleNamespace(model=None), res, None)
    assert blk.status == "not_run" and blk.messages


def test_plausible_flag_template() -> None:
    flags, notes, _ = plausible_flags({"margin_median": 0.9, "own_elasticity_median": None},
                                      {"margin_median": (0.05, 0.6), "own_elasticity_median": (-12, -2)})
    assert flags == ["FLAG: margin_median = 0.9 is outside the declared plausible range [0.05, 0.6]."]
    assert notes == ["Plausible range for own_elasticity_median not checked: statistic not computed."]


def test_compare_blp_table() -> None:
    txt = pp.compare_blp({"own_elasticity_median": -4.0, "minivan_cs_gain_1984_musd": 300.0},
                         {"BLP micro": {"own_elasticity_median": -4.48, "minivan_cs_gain_1984_musd": 425.9}},
                         note="PCM uses demand moments only.")
    assert "own-price elasticity, median" in txt and "-4.480" in txt and "425.900" in txt and "-" in txt
    assert txt.endswith("PCM uses demand moments only.")


# --------------------------------------------------------------------------------------------
# DESIGN 7 WP8: the old body-sigma nu=16 estimate as a numeric fixture
# --------------------------------------------------------------------------------------------

FIXTURE = ROOT / "petrin_replication" / "results"
PETRIN = ROOT / "nevo_replication" / ".vendor" / "pyblp" / "data"


@pytest.mark.skipif(not ((FIXTURE / "pcm_estimates_body_nu16.npz").exists()
                         and (FIXTURE / "pcm_product_level_body_nu16.csv").exists()
                         and (PETRIN / "petrin_products.csv").exists()), reason="local Petrin fixture not present")
def test_old_estimate_elasticities_and_margins_match_fixture() -> None:
    """Rebuilds the markets of the old ``pcm_petrin.build_markets`` (PCM_BODY_SIGMA=1, nu_per_agent=16,
    seed 0) and checks own elasticities, margins and nonzero cross counts against the local
    ``pcm_product_level_body_nu16.csv`` written by the old ``pcm_postestimation.py``."""
    import warnings

    import pandas as pd
    from scipy.stats import norm, qmc

    P = pd.read_csv(PETRIN / "petrin_products.csv")
    A = pd.read_csv(PETRIN / "petrin_agents.csv")
    est = np.load(FIXTURE / "pcm_estimates_body_nu16.npz")
    ref = pd.read_csv(FIXTURE / "pcm_product_level_body_nu16.csv")
    X = ["const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv"]
    taste = [(X.index(c), f"nu{i}") for i, c in enumerate(X)] + [(X.index(c), "logfs_fv") for c in X[6:]]
    th, ph, sp, delta = est["theta"][:14], est["theta"][14:17], float(est["theta"][17]), est["delta"]
    own = np.full(len(P), np.nan)
    margin, ncross = own.copy(), own.copy()
    for mkt, prod in P.groupby("market_ids", sort=True):
        prod = prod.assign(ps=break_price_ties(prod.prices.to_numpy(float), np.zeros(len(prod)), 0.01)
                           ).sort_values("ps")
        ag = A[A.market_ids == mkt]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nu = norm.ppf(qmc.Sobol(10, scramble=True, seed=int(mkt)).random(len(ag) * 16))
        rep = np.repeat(np.arange(len(ag)), 16)
        cols = {f"nu{i}": nu[:, i] for i in range(10)}
        cols["logfs_fv"] = (np.log(ag.fs.to_numpy(float)) * ag.fv.to_numpy(float))[rep]
        inc = ag.income.to_numpy(float)[rep]
        sb = np.column_stack([ag[lv].to_numpy(float)[rep] / inc for lv in ("low", "mid", "high")])
        x = np.column_stack([np.ones(len(prod))] + [prod[c].to_numpy(float) for c in X[1:]])
        p = prod.ps.to_numpy(float)
        km = pcm_kernel.Market(x, p, np.repeat(ag.weights.to_numpy(float), 16) / 16,
                               np.column_stack([cols[b] for _, b in taste]), [c for c, _ in taste], sb)
        rows = prod.index.to_numpy()
        r = pcm_kernel.evaluate_market(delta[rows], km, th, ph, sp, derivatives=False, price_derivatives=True,
                                       n_threads=16)
        e = pp.elasticities(r["s"], r["ds_dp"], p)
        nm = pp.nash_markups(r["s"], r["ds_dp"], prod.firm_ids.to_numpy(), rcond_min=0.0)
        assert nm["status"] == "ok"
        own[rows], ncross[rows], margin[rows] = np.diag(e), pp.nonzero_cross(e), nm["markup"] / p
    e_own = np.max(np.abs(own - ref.own_elasticity) / np.maximum(1, np.abs(ref.own_elasticity)))
    e_mar = np.max(np.abs(margin - ref.margin) / np.maximum(1, np.abs(ref.margin)))
    print(f"old nu=16 fixture: own elasticity max relative error {e_own:.1e}, margin {e_mar:.1e}")
    assert e_own < 1e-8 and e_mar < 1e-8
    np.testing.assert_array_equal(ncross, ref.n_cross_nonzero)


def _logit(a: np.ndarray, b: float):
    def shares(p: np.ndarray) -> np.ndarray:
        e = np.exp(a - b * p)
        return e / (1.0 + e.sum())
    return shares


def test_best_response_box_finds_the_logit_nash_equilibrium() -> None:
    """Single- and multi-product logit firms: the markup of firm f solves p - c = 1 / (b (1 - S_f)); best
    responses in a box around a point 5% off the equilibrium reach it to the golden-section tolerance."""
    a, b, c = np.array([1.0, 1.5, 0.8, 1.2]), 0.9, np.array([1.0, 1.2, 0.9, 1.1])
    S = _logit(a, b)
    for firms in (np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])):
        p = c + 1.0
        for _ in range(500):   # the equilibrium by its fixed point (a contraction for logit)
            s = S(p)
            Sf = np.array([s[firms == f].sum() for f in firms])
            p = c + 1.0 / (b * (1.0 - Sf))
        br = pp.best_response_box(S, 1.05 * p, c, firms, box=0.10, golden_tol=1e-4, price_tol=1e-7, max_sweeps=60)
        assert br["status"] == "converged", br["max_change"]
        np.testing.assert_allclose(br["prices"], p, rtol=2e-4)
        assert br["box_binding"]["count"] == 0 and br["deviation"]["max_gain"] <= 0.0


def test_best_response_box_rejects_zero_shares_and_reports_the_box() -> None:
    """Profit rises in p_0 (constant shares) until product 0 loses every buyer above 1.05 p0: the zero-share trials
    are rejected, so p_0 stops just below the cliff; product 1's profit rises to the box end."""
    p0 = np.array([10.0, 20.0])

    def shares(p: np.ndarray) -> np.ndarray:
        return np.array([0.0 if p[0] > 10.5 else 0.1, 0.2])

    br = pp.best_response_box(shares, p0, np.array([5.0, 5.0]), np.array([0, 1]), box=0.10,
                              surplus=lambda p: -float(p.sum()))
    assert br["status"] == "converged"
    assert 10.49 <= br["prices"][0] <= 10.5 and br["prices"][1] == pytest.approx(22.0)
    assert br["box_binding"]["count"] == 1 and br["box_binding"]["positions"] == [1]
    assert np.all(br["shares"] > 0) and "cs_range" not in br

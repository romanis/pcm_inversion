"""Tests for the GMY PCM spec on the pipeline (GMY spec WP-G7): problems, identification, run keys, the table."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pytest

from nfp_harness.comparison import near_twins, trimmed_mean
from nfp_harness.tests.test_gmy_instruments import gmy_module
from nfp_harness.weighting import efficient_on_excluded


@pytest.fixture(scope="module")
def gmy() -> Any:
    try:
        gmy_module().package_root()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    pytest.importorskip("nfp_harness._native")
    return gmy_module()


@pytest.fixture(scope="module")
def spec(gmy: Any) -> Any:
    return gmy.GMYSpec((1991, 1999))


def test_problems_and_identification(gmy: Any, spec: Any) -> None:
    prob = spec.build_problem(1, 0, 4, "all")
    names = [m.name for m in prob.moments]
    assert names == ["aggregate", "micro_gmy", "second_choice"]
    assert prob.blocks["micro_gmy"].stop - prob.blocks["micro_gmy"].start == 33
    assert prob.blocks["second_choice"].stop - prob.blocks["second_choice"].start == 20   # electric NaN in both
    info = prob.identification
    # round 2: GMY's 22 (pi_const_inc free again, D8 undone) plus the 3 extra spreads (D7); round 4: 2 age tastes
    assert info["p_free"] == 27 and info["overidentifying"] > 0 and not info.get("refused")
    assert prob.model.group_sums_backend == "kernel"
    W1 = prob.initial_weighting(prob.layout.starts())
    sc = prob.blocks["second_choice"]
    np.testing.assert_allclose(np.diag(W1)[sc] * prob.n, 900.0 / prob.n * prob.n)
    prob.model.close()
    pre = spec.build_problem(1, 0, 4, "prefit")
    assert [m.name for m in pre.moments] == ["micro_gmy"] and pre.k == 20
    pre.model.close()
    spec.set_moments("gmy_nolinear")
    nol = spec.build_problem(1, 0, 4, "all")
    assert [m.name for m in nol.moments] == ["micro_gmy", "second_choice"] and nol.agg is None
    nol.model.close()
    spec.set_moments("demand_supply")
    ds = spec.build_problem(1, 0, 4, "all")
    assert [m.name for m in ds.moments][-1] == "supply"
    # step 1's W1 exists with the supply block: its excluded instruments are standardised (steel_weight is ~6e5)
    assert np.allclose(spec.Z_supply[:, spec.cost_basis.shape[1]:].std(axis=0), 1.0)
    W1 = ds.initial_weighting(ds.layout.starts())
    assert np.all(np.isfinite(W1)) and np.linalg.eigvalsh(W1[ds.supply_rows, ds.supply_rows])[0] > 0
    ds.model.close()
    spec.set_moments("gmy")


def test_normalisations_and_run_key(gmy: Any, spec: Any) -> None:
    text = " ".join(spec.normalisations)
    for tag in ("D1", "D2", "D3", "D4", "D5", "D6"):
        assert tag in text
    base = spec.run_key()
    assert base["years"] == [1991, 1999] and base["a_lo"] == 0.2
    spec.set_moments("gmy_nolinear")
    assert spec.run_key() != base
    spec.set_moments("gmy")
    ap = argparse.ArgumentParser()
    spec.add_arguments(ap)
    other = gmy.GMYSpec((1991, 1999))
    other.configure(ap.parse_args(["--years", "1991", "1999", "--instruments", "gh"]))
    assert other.run_key()["instruments"] == "gh" and other.Z.shape[1] > spec.X1.shape[1]
    assert other.run_key() != base
    back = gmy.GMYSpec((1991, 1999))
    back.configure(ap.parse_args(["--years", "1991", "1999"]))
    assert back.run_key() == base
    # D9: products below --min-share are dropped and the threshold is part of the run key
    assert "D9" in text and base["min_share"] == 1e-5 and spec.dropped_small["products"] > 0
    full = gmy.GMYSpec((1991, 1999))
    full.configure(ap.parse_args(["--years", "1991", "1999", "--min-share", "0"]))
    assert full.run_key() != base and full.dropped_small["products"] == 0
    assert len(full.products) == len(spec.products) + spec.dropped_small["products"]
    assert full.products.shares.min() < 1e-5 <= spec.products.shares.min()


def test_warm_from_delta(gmy: Any, spec: Any) -> None:
    prob = spec.build_problem(1, 0, 4, "prefit")
    m = prob.model
    delta = np.zeros(m.n_products)
    assert m.warm_from_delta(delta, prob.layout.starts()) == 2
    np.testing.assert_array_equal(m.accepted_delta(), delta)
    delta[m.markets[0].rows[0]] = np.nan
    assert m.warm_from_delta(delta, prob.layout.starts()) == 1
    m.close()


def test_comparison_table_renders_missing_entries() -> None:
    pcm = {"PCM a_lo 0.2": {
        "table5": {1980: {"mean_over_products": [-20.0, -19.0, -18.0, -17.0, -16.0]}},
        "lerner_median_by_year": {1980: 0.05, 1991: 0.06},
        "own_elasticity_mean_by_year": {1980: -22.0, 1991: -21.0},
        "own_elasticity_median_by_year": {1980: -20.0, 1991: -19.5},
        "second_choice": {1991: {"data": [0.7] * 11, "pcm": [0.9] * 11}}, "second_choice_rmse": 0.2,
        "micro": [{"name": "m1", "data": 0.1, "pcm": 0.2}], "diversion_to_own_type": {"car": 0.6},
        "cs_per_household_usd_by_year": {1980: 1234.0}, "convergence": "CONVERGED",
        "window_anatomy": {"frontier_size_mass": {"1": 0.501, "2": 0.376, "3": 0.1, "4+": 0.023}},
        "margin_median": 0.369, "margin_median_sales_weighted": 0.512, "products_own_elasticity_above_-1": 114,
        "q": 12.0, "q_blocks": {"micro_gmy": 5.0}, "at_bound": []}}
    blp = {"BLP fit": {"table5": {"1980": {"mean_over_products": [-6.0, -5.8, -5.5, -5.1, -4.3]}},
                       "lerner_median_by_year": {"1980": 0.3}, "micro": [{"name": "m1", "model": 0.15}],
                       "second_choice": {"1991": {"model": [0.71] * 11}}}}
    paper = {"table5": {1980: (-5.96, -5.78, -5.49, -5.13, -4.30)}, "lerner_median": {"intro": {1980: 0.311}},
             "table6_data": {f"c{k}": 0.5 for k in range(11)}, "table6_model": {f"c{k}": 0.5 for k in range(11)}}
    text = gmy_module().comparison_table(pcm, blp, paper)
    assert "1980 Q1" in text and "-20.00" in text and "-6.00" in text and "-5.96" in text
    assert "2018 Q1" in text and " -" in text   # 2018 missing everywhere: rendered as '-'
    assert "c0 model (data)" in text and "0.710" in text
    assert "m1" in text and "0.150" in text
    assert "all years: sales-weighted median" in text and "0.512" in text and "0.369" in text
    assert "|own elasticity| < 1" in text and " 114" in text
    assert "buyer mass on frontiers of 1 product " in text and "0.501" in text and "frontiers of 4+ products" in text


def test_step2_weighting_keeps_ols_and_handles_singular_x1_rows(gmy: Any, spec: Any) -> None:
    from types import SimpleNamespace

    from nfp_harness.linear import LinearIV
    from nfp_harness.weighting import clustered_covariance

    rng = np.random.default_rng(0)
    n, K = spec.X1.shape
    Z = spec.Z
    k = Z.shape[1]
    xi = rng.normal(size=n)
    xi -= spec.X1 @ np.linalg.lstsq(spec.X1, xi, rcond=None)[0]       # OLS residual: X1 rows of Z'xi are zero
    S_a = clustered_covariance(Z, xi, spec.clusters)
    assert np.linalg.eigvalsh(S_a)[0] / np.linalg.eigvalsh(S_a)[-1] < 1e-12   # singular by construction
    S = np.zeros((k + 3, k + 3))
    S[:k, :k] = S_a
    S[k:, k:] = np.diag([1.0, 2.0, 3.0])
    blocks = {"aggregate": slice(0, k), "micro_gmy": slice(k, k + 3)}
    prob = SimpleNamespace(covariance=lambda ev: S, agg=SimpleNamespace(name="aggregate", Z=Z), agg_rows=blocks["aggregate"],
                           linear=SimpleNamespace(X1=spec.X1))
    W = efficient_on_excluded(prob, SimpleNamespace(blocks=blocks))
    assert np.linalg.eigvalsh(W)[0] > 0
    np.testing.assert_allclose(W[k:, k:], np.diag([1.0, 1 / 2, 1 / 3]), rtol=1e-12)
    delta = rng.normal(size=n)
    beta_w2 = LinearIV(spec.X1, Z, W_aa=W[:k, :k]).fit(delta).beta
    beta_ols = np.linalg.lstsq(spec.X1, delta, rcond=None)[0]
    np.testing.assert_allclose(beta_w2, beta_ols, rtol=1e-8, atol=1e-8)


def test_round2_layout_and_the_round1_layout_by_robustness(gmy: Any) -> None:
    new = gmy.GMYSpec((1991, 1999))
    lay = new.layout()
    names = lay.names
    # D8 undone: pi_const_inc free at the BLP value / median |alpha| (about 0.17), and free in the prefit
    pic = lay.specs[names.index("pi_const_inc")]
    assert pic.free and pic.start == new.blp["theta"]["const x inc"] / new.alpha_median
    assert 0.1 < pic.start < 0.3
    assert new.prefit_free == ("phi_inc", "phi_age", "pi_const_inc", "pi_z_log_footprint_age", "pi_lux_age")
    assert [n for n in names if n[len("sigma_"):] in gmy.EXTRA_SPREADS] == [f"sigma_{c}" for c in gmy.EXTRA_SPREADS]
    assert all(lay.specs[names.index(f"sigma_{c}")].start == 0.5 for c in gmy.EXTRA_SPREADS)
    assert new.pcm_spec().characteristics == gmy.TASTE + gmy.EXTRA_SPREADS
    assert new.income_basis == "level" and new.prefit_variants() == ()
    assert new.optimizer_options["probe_restart_gain_rel"] == 1e-3
    assert any(t.startswith("D8 undone") for t in new.normalisations)
    r1 = gmy.GMYSpec((1991, 1999), robustness=("no_pi_const_inc",))
    assert "pi_const_inc" not in r1.layout().names and len(r1.layout().names) == len(names) - 1
    assert r1.prefit_free == ("phi_inc", "phi_age", "pi_z_log_footprint_age", "pi_lux_age")
    assert r1.run_key() != new.run_key()
    with pytest.raises(ValueError, match="no_pi_const_inc"):
        gmy.GMYSpec((1991, 1999), robustness=("pi_const_inc",))
    old = gmy.GMYSpec((1991, 1999), robustness=("gmy_spreads_only", "no_age_taste"))
    assert len(old.layout().names) == 24 and "pi_const_inc" in old.layout().names   # GMY's 22 free + phi_gmy, a_lo
    assert old.pcm_spec().characteristics == gmy.TASTE
    assert old.n_merged >= new.n_merged   # twins must agree on the extra columns too
    assert old.run_key() != new.run_key()


def test_round4_age_taste_and_the_round2_layout_by_robustness(gmy: Any) -> None:
    """Round 4: pi_age on z log footprint and lux (agent column age), start 0, free in the prefit, in the layout by
    default; robustness no_age_taste is the round-2 layout in the same order, with the round-2 run key."""
    new = gmy.GMYSpec((1991, 1999))
    lay = new.layout()
    for c in ("z_log_footprint", "lux"):
        sp = lay.specs[lay.names.index(f"pi_{c}_age")]
        assert sp.free and sp.start == 0.0 and sp.role == "taste" and sp.column == c and sp.basis == "age"
    r2 = gmy.GMYSpec((1991, 1999), robustness=("no_age_taste",))
    assert [n for n in lay.names if not n.endswith("_age") or n == "phi_age"] == r2.layout().names
    assert r2.run_key()["layout_round"] == 2 and new.run_key()["layout_round"] == 4
    assert not any("pi_" in n and n.endswith("_age") for n in r2.prefit_free)
    assert any(t.startswith("round 4") for t in new.normalisations)
    # the 8 age-price rows alone, phi_age and the age tastes free
    ap = argparse.ArgumentParser()
    new.add_arguments(ap)
    age = gmy.GMYSpec((1991, 1999))
    age.configure(ap.parse_args(["--years", "1991", "1999", "--prefit-rows", "age"]))
    assert age.prefit_free == ("phi_age", "pi_z_log_footprint_age", "pi_lux_age")
    assert age.run_key()["prefit_rows"] == "age" and "prefit_rows" not in new.run_key()
    labels = age.micro_moment("prefit")[0].labels
    assert len(labels) == 8 and all(lb.startswith("Age") and " Price " in lb for lb in labels)
    pre = age.build_problem(1, 0, 4, "prefit")
    assert pre.k == 8 and {n for n, s in zip(pre.layout.names, pre.layout.specs) if s.free} == set(age.prefit_free)
    pre.model.close()
    # the taste basis of a draw is its household's age
    prob = new.build_problem(1, 0, 4, "prefit")
    m = prob.model
    md = m.market_draws(0)
    col = [n for n, s in zip(lay.names, lay.specs) if s.role == "taste"].index("pi_lux_age")
    np.testing.assert_array_equal(np.asarray(m.kernel_market(0).taste_basis)[:, col],
                                  new.agents.age.to_numpy()[md["agent_rows"]][md["agent_of_draw"]])
    m.close()


def _fake_stage(tmp: Any, names: list[str], theta: np.ndarray, state: dict[str, Any] | None = None) -> Any:
    """A run directory with stages/step1 holding ``theta`` (and ``state``) and a history.csv naming the layout."""
    from nfp_harness.checkpoint import HISTORY_FIELDS, RunDirectory

    rd = RunDirectory(tmp, "gmy_pcm", "hash", "key0123456789")
    rd.save_stage("step1", {"theta": theta, **(state or {})})
    (rd.path / "history.csv").write_text(",".join(list(HISTORY_FIELDS) + [f"theta_{n}" for n in names]) + "\n")
    return rd.path / "stages" / "step1.npz"


def test_start_theta_and_warm_theta_by_name(gmy: Any, tmp_path: Any) -> None:
    """A round-1 stage (no pi_const_inc) starts the round-2 layout by name: pi_const_inc keeps its start, the
    --start-theta-except parameters keep theirs; the warm theta has pi_const_inc at 0, its value in that run."""
    r1 = gmy.GMYSpec((1991, 1999), robustness=("no_pi_const_inc",))
    src_names = r1.layout().names
    th = r1.layout().starts() + 0.01 * np.arange(len(src_names))
    rows = r1.market_rows(1991)
    path = _fake_stage(tmp_path, src_names, th, {"problem_state": {"model": {
        "market_ids": ["1991"], "delta": [np.zeros(rows.size)]}}})
    ap = argparse.ArgumentParser()
    r1.add_arguments(ap)
    sp = gmy.GMYSpec((1991, 1999))
    d7 = [f"sigma_{c}" for c in gmy.EXTRA_SPREADS]
    sp.configure(ap.parse_args(["--years", "1991", "1999", "--start-theta", str(path), "--warm-delta", str(path),
                                "--start-theta-except", *d7]))
    lay, ref = sp.layout(), gmy.GMYSpec((1991, 1999)).layout()
    src = dict(zip(src_names, th))
    for n, s in zip(lay.names, lay.specs):
        if n == "pi_const_inc" or n in d7 or not s.free:
            assert s.start == ref.specs[ref.names.index(n)].start, n
        else:
            assert s.start == pytest.approx(float(np.clip(src[n], s.lower, s.upper)), abs=1e-15), n
    assert sp.run_key()["start_theta_except"] == sorted(d7)
    delta, theta_w = sp.read_warm_delta()
    assert np.all(delta[rows] == 0.0) and np.isnan(delta[sp.market_rows(1999)]).all()
    assert theta_w.size == lay.P and theta_w[lay.names.index("pi_const_inc")] == 0.0
    for n in src_names:
        assert theta_w[lay.names.index(n)] == src[n]
    bad = gmy.GMYSpec((1991, 1999))
    bad.configure(ap.parse_args(["--years", "1991", "1999", "--start-theta", str(path), "--start-theta-except", "x"]))
    with pytest.raises(ValueError, match="not parameters"):
        bad.layout()


def test_income_basis_variants(gmy: Any) -> None:
    sp = gmy.GMYSpec((1991, 1999), income_basis="auto")
    assert sp.income_basis == "auto" and sp.prefit_variants() == ("level", "log")
    lay = sp.layout()
    assert lay.specs[lay.names.index("phi_inc")].basis == "inc"
    sp.set_prefit_variant("log")
    assert sp.layout().specs[lay.names.index("phi_inc")].basis == "log_inc_ratio"
    assert sp.hash_layout().specs[lay.names.index("phi_inc")].basis == "inc"   # the hash ignores the choice
    a = sp.agents
    np.testing.assert_allclose(np.exp(a.log_inc_ratio) * (a.income - a.inc * 1e4), a.income, rtol=1e-12)
    fixed = gmy.GMYSpec((1991, 1999), income_basis="log")
    assert fixed.prefit_variants() == () and fixed.layout().specs[lay.names.index("phi_inc")].basis == "log_inc_ratio"


def test_income_law_petrin_bounded(gmy: Any) -> None:
    """--income-law petrin_bounded: log alpha on Petrin's log(ybar / clip(y, 1, 150)) (y in $1000, ybar the pooled
    weighted median) through phi_y fixed at 1, phi_gmy fixed at 0; phi_inc, phi_age and pi_const_inc stay free. The
    default layout and run key are unchanged."""
    base = gmy.GMYSpec((1991, 1999))
    ap = argparse.ArgumentParser()
    base.add_arguments(ap)
    sp = gmy.GMYSpec((1991, 1999))
    sp.configure(ap.parse_args(["--years", "1991", "1999", "--income-law", "petrin_bounded"]))
    lay, ref = sp.layout(), base.layout()
    assert "phi_y" not in ref.names and "income_law" not in base.run_key() and "income_law" not in base.hash_payload()
    assert sp.run_key()["income_law"] == "petrin_bounded" and sp.run_key() != base.run_key()
    y = lay.specs[lay.names.index("phi_y")]
    g = lay.specs[lay.names.index("phi_gmy")]
    assert (y.role, y.basis, y.start, y.free) == ("scale", "log_ybar_inc_w", 1.0, False)
    assert (g.start, g.free) == (0.0, False) and ref.specs[ref.names.index("phi_gmy")].start == 1.0
    assert all(lay.specs[lay.names.index(n)].free for n in ("phi_inc", "phi_age", "pi_const_inc"))
    assert [n for n in lay.names if n != "phi_y"] == ref.names
    a = sp.agents
    inc = a.income.to_numpy(float) / 1e3
    np.testing.assert_array_equal(a.log_ybar_inc_w, np.log(sp.income_law_ybar / np.clip(inc, 1.0, 150.0)))
    w = a.weights.to_numpy(float)
    assert abs(w[inc < sp.income_law_ybar].sum() / w.sum() - 0.5) < 0.01   # ybar is the weighted median
    assert any("petrin_bounded" in t for t in sp.normalisations)
    prob = sp.build_problem(1, 0, 4, "prefit")
    m = prob.model
    md = m.market_draws(0)
    col = [n for n, s in zip(lay.names, lay.specs) if s.role == "scale"].index("phi_y")
    np.testing.assert_array_equal(np.asarray(m.kernel_market(0).scale_basis)[:, col],
                                  a.log_ybar_inc_w.to_numpy()[md["agent_rows"]][md["agent_of_draw"]])
    m.close()


def test_antithetic_taste(gmy: Any) -> None:
    """--antithetic-taste: every taste draw dimension in antithetic pairs (zero sum per market), in the run key and
    the hash payload only when on."""
    import math

    base = gmy.GMYSpec((1991, 1999))
    ap = argparse.ArgumentParser()
    base.add_arguments(ap)
    sp = gmy.GMYSpec((1991, 1999))
    sp.configure(ap.parse_args(["--years", "1991", "1999", "--antithetic-taste"]))
    assert sp.run_key()["antithetic_taste"] is True and "antithetic_taste" not in base.run_key()
    assert "draws" in sp.hash_payload() and "draws" not in base.hash_payload()
    prob = sp.build_problem(1, 0, 4, "prefit")
    m = prob.model
    lay = prob.layout
    taste = [s for s in lay.specs if s.role == "taste"]
    nu_cols = [k for k, s in enumerate(taste) if s.basis is not None and s.basis.startswith("nu")]
    assert len(m.antithetic) == len(nu_cols) == 14
    for t in range(m.T):
        b = np.asarray(m.kernel_market(t).taste_basis)
        assert all(math.fsum(b[:, k]) == 0.0 for k in nu_cols)
    m.close()


def test_fix_sigma_p(gmy: Any) -> None:
    """--fix sigma_p=3.0: sigma_p fixed at 3 in every problem (also above its bound 4: the bound widens), in the run key;
    a fixed parameter stays fixed in the prefit; an unknown name is refused."""
    base = gmy.GMYSpec((1991, 1999))
    ap = argparse.ArgumentParser()
    base.add_arguments(ap)
    sp = gmy.GMYSpec((1991, 1999))
    sp.configure(ap.parse_args(["--years", "1991", "1999", "--fix", "sigma_p=3.0"]))
    lay = sp.layout()
    s = lay.specs[lay.names.index("sigma_p")]
    assert (s.start, s.free, s.lower, s.upper) == (3.0, False, gmy.SIGMA_P["lower"], gmy.SIGMA_P["upper"])
    assert sp.run_key()["fix"] == {"sigma_p": 3.0} and "fix" not in base.run_key() and "fix" not in base.hash_payload()
    assert sum(sp_.free for sp_ in lay.specs) == sum(sp_.free for sp_ in base.layout().specs) - 1
    assert any(t.startswith("fixed by --fix") for t in sp.normalisations)
    hi = gmy.GMYSpec((1991, 1999))
    hi.configure(ap.parse_args(["--years", "1991", "1999", "--fix", "sigma_p=4.5", "--fix", "phi_age=-1"]))
    lh = hi.layout()
    assert lh.specs[lh.names.index("sigma_p")].upper == 4.5
    pre = hi.build_problem(1, 0, 4, "prefit")
    free = {n for n, s_ in zip(pre.layout.names, pre.layout.specs) if s_.free}
    assert "phi_age" not in free and "phi_inc" in free and pre.layout.specs[pre.layout.names.index("phi_age")].start == -1
    pre.model.close()
    bad = gmy.GMYSpec((1991, 1999))
    bad.configure(ap.parse_args(["--years", "1991", "1999", "--fix", "nope=1"]))
    with pytest.raises(ValueError, match="not parameters"):
        bad.layout()


def test_trimmed_mean_and_near_twins(gmy: Any) -> None:
    v = np.r_[np.arange(1.0, 101.0), -1e4, np.nan]
    assert trimmed_mean(v) == pytest.approx(np.mean(np.arange(5.0, 97.0)), rel=0.02)
    assert abs(trimmed_mean(v)) < 60
    import pandas as pd
    p = pd.DataFrame({"market_ids": [1, 1, 1, 1, 2], "veh_type": ["car", "car", "van", "car", "car"],
                      "prices_data": [1.0, 1.005, 1.002, 2.0, 1.0]})
    assert near_twins(p, gmy.NEAR_TWIN_GAP, "veh_type", "prices_data") == {1: 2, 2: 0}

"""Tests for nfp_harness.gmy_data on the Grieco, Murry & Yurukoglu (2024) package. Skipped when the package is not
unpacked under gmy_replication/data/GMY_Replication (the raw data are not committed)."""

from __future__ import annotations

import numpy as np
import pytest

from nfp_harness import gmy_data as g
from nfp_harness.second_choice import first_second_correlations

try:
    ROOT = g.package_root()
except FileNotFoundError:   # pragma: no cover - depends on the local data
    pytest.skip("GMY replication package not unpacked", allow_module_level=True)


@pytest.fixture(scope="module")
def products():
    return g.load_products(ROOT)


@pytest.fixture(scope="module")
def agents():
    return g.load_agents(ROOT, n_households=200, years=(1980, 1991, 1999, 2005, 2015, 2018))


def test_row_counts(products):
    n = products.groupby("market_ids").size()
    assert len(products) == 9694
    assert n.index.tolist() == list(range(1980, 2019))
    assert n.min() == 165 and n.max() == 323
    assert (products.position.to_numpy() == products.groupby("market_ids").cumcount().to_numpy()).all()


def test_prices_shares_and_ownership(products):
    assert np.allclose(products.prices, products.msrp / 10)
    inside = products.groupby("market_ids").shares.sum()
    assert (inside > 0).all() and (inside < 1).all()
    assert inside.min() > 0.2 and inside.max() < 0.45     # sales / (households / 2.5)
    assert (products.shares > 0).all()
    assert products.firm_ids.notna().all() and products.firm_ids.nunique() > 10
    for c in g.Z_XTERMS:
        assert abs(products[f"z_{c}"].mean()) < 1e-12 and abs(products[f"z_{c}"].std(ddof=1) - 1) < 1e-12
    for c in g.STAGE2_X + ("demand_instruments0", "IV_N", "IV_N_weight"):
        assert products[c].notna().all(), c


def test_second_choices_aligned_to_products(products):
    sc = g.load_second_choices(ROOT, products)
    assert sorted(sc) == list(g.SECOND_CHOICE_YEARS)
    corr = []
    for y, v in sc.items():
        pr = products[products.market_ids == y]
        J = len(pr)
        P = v["prob"]
        assert P.shape == (J, J) and v["count"].shape == (J, J)
        assert np.all(np.diag(P) == 0) and np.all(np.diag(v["count"]) == 0)   # never one's own second choice
        rs = P.sum(axis=1)
        assert np.all(np.isclose(rs, 0) | np.isclose(rs, 1))
        assert (rs > 0).sum() > 0.75 * J
        sh = pr.shares.to_numpy()
        assert sh[rs == 0].mean() < 0.5 * sh[rs > 0].mean()   # first choices without respondents are small sellers
        assert np.allclose(P[rs > 0], v["count"][rs > 0] / v["count"][rs > 0].sum(axis=1, keepdims=True))
        corr.append(first_second_correlations(pr.shares, P, pr[list(g.SECOND_CHOICE_CHARS)].to_numpy(float)))
    # Table 6, "Data" column (working paper): a misaligned matrix would not reproduce these
    table6 = (0.71, 0.64, 0.84, 0.71, 0.60, 0.65, 0.48, 0.28, 0.37, 0.34, 0.48)
    assert np.allclose(np.nanmean(corr, axis=0), table6, atol=0.011)


def test_micro_moment_shapes_and_values():
    t = g.load_micro_tables(ROOT)
    assert t["cex"]["value"].shape == (44, 75) and t["cex"]["variance"].shape == (44, 75)
    assert t["mri"]["value"].shape == (44, 76) and "PurchaseProb" in t["mri"]["value"].columns
    m = g.load_micro_moments(ROOT)
    assert len(m.table) == 33 and m.weight.shape == (33, 33)
    assert (m.variances > 0).all() and np.isfinite(m.values).all()
    v = dict(zip(m.table.name, m.values))
    # Table 8 "Data" values; the paper prints the CEX and MRI columns swapped relative to the files the code reads
    assert v["Income5MINUSIncome1 Price mri"] == pytest.approx(0.60, abs=0.005)
    assert v["Income5MINUSIncome1 Price cex"] == pytest.approx(0.22, abs=0.005)
    assert v["Age60+MINUSAge20-29 Price cex"] == pytest.approx(0.52, abs=0.005)
    assert v["FamilySize5+MINUSFamilySize1 Van mri"] == pytest.approx(0.12, abs=0.005)
    assert v["UrbanMINUSRural Truck cex"] == pytest.approx(-0.10, abs=0.005)
    assert v["Income5OVERIncome1 PurchaseProb mri"] == pytest.approx(7.81, abs=0.005)


def test_agents(agents):
    assert agents.groupby("market_ids").size().eq(200).all()
    assert np.allclose(agents.groupby("market_ids").weights.sum(), 1)
    assert agents.income.between(*g.INCOME_WINSOR).all()
    assert set(agents.inc_quintile) <= {1, 2, 3, 4, 5} and set(agents.age_group) <= {1, 2, 3, 4, 5}
    assert np.allclose(agents.inc2, agents.inc ** 2 / 100)
    assert (agents.fs2 + agents.fs34 + agents.fs5 <= 1).all()
    q = agents.groupby("inc_quintile").weights.sum() / agents.weights.sum()
    assert q.between(0.08, 0.35).all()
    assert (agents.cex == agents.market_ids.between(1983, 2004)).all()
    assert (agents.mri == agents.market_ids.between(1992, 2014)).all()


def test_micro_evaluator_on_flat_probabilities(products, agents):
    m = g.load_micro_moments(ROOT)
    # one year in both survey windows: with every household alike, differences are 0 and ratios 1 (pooling
    # several years would not give 0, since the sets differ in their mix of years)
    pr = products[products.market_ids == 1999].reset_index(drop=True)
    ag = agents[agents.market_ids == 1999].reset_index(drop=True)
    ev = g.MicroMomentEvaluator(pr, ag, m)
    out = ev([np.full((len(pr), len(ag)), 0.001)])
    diff = (m.table.kind == "diff").to_numpy()
    assert np.allclose(out[diff], 0, atol=1e-12) and np.allclose(out[~diff], 1)

"""Petrin's minivan counterfactual and elasticities under the PCM, next to the two BLP fits.

Consumer surplus in the PCM. Consumer draw i (tastes t_i, price scale c_i) with price sensitivity
a ~ LogNormal(0, sigma_p) gets u_ij = q_ij - c_i a p_j with q_ij = delta_j + x_j t_i. In dollars
(dividing by c_i a) the surplus is max_j (q_ij / (c_i a) - p_j)^+. For alpha = c_i a the frontier
products k with endpoints e_k > e_{k+1} (e from the upper convex hull of (p, q/c_i)) give
    CS_i = sum_k [ (q_k / c_i) E(1/alpha; e_{k+1} < alpha < e_k) - p_k P(e_{k+1} < alpha < e_k) ]
with E(1/alpha; a < alpha < b) = exp(s^2/2) [Phi((ln b + s^2)/s) - Phi((ln a + s^2)/s)] for LN(0, s).
Prices and delta are in $1000; the market CS is households * sum_i w_i CS_i, reported in $ million.

Counterfactual (Petrin): 1984 without the minivans, marginal costs from the multi-product Nash
first-order condition at observed prices, new prices from the same condition without minivans.

Run: PYTHONPATH=nevo_replication/.vendor:build-rel2/python_bindings:petrin_replication \
     PCM_TAG=_nu4 python petrin_replication/pcm_postestimation.py
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd
import pyblp
import python_pcm_demand as pd_cpp
from scipy.stats import norm

import pcm_petrin as pp

OUT = pathlib.Path(__file__).resolve().parent / "results"


def nash_markup(s, Jp, firms):
    om = (firms[:, None] == firms[None, :]).astype(float)
    return -np.linalg.solve(om * Jp.T, s)


def market_eval(x, p, w, tb, tc, sb, delta, th, ph, sp):
    mk = pd_cpp.Market(x, p, w, tb, tc, sb)
    d = pd_cpp.shares(delta, mk, th, ph, sp, True)
    return d["s"], d["d_price"]


def consumer_surplus(x, p, w, tb, tc, sb, delta, th, ph, sp):
    """Sum_i w_i CS_i in $1000 per household, by the frontier formula above."""
    R = len(w)
    M = len(th)
    t = np.zeros((R, x.shape[1]))
    for m in range(M):
        t[:, tc[m]] += th[m] * tb[:, m]
    c = np.exp(sb @ ph) if sb.shape[1] else np.ones(R)
    q = (delta[None, :] + t @ x.T) / c[:, None]                  # R x N conditional qualities
    s2 = sp ** 2
    E_inv = lambda e: np.exp(s2 / 2) * norm.cdf((np.log(e) + s2) / sp)   # E[1/a 1{a < e}]
    F = lambda e: norm.cdf(np.log(e) / sp)
    total = 0.0
    N = len(p)
    for i in range(R):
        qi = q[i]
        # upper convex hull of (0,0), (p_k, q_k): monotone chain over price-sorted products
        hull = [-1]
        px = lambda k: 0.0 if k < 0 else p[k]
        qy = lambda k: 0.0 if k < 0 else qi[k]
        for k in range(N):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (px(b) - px(a)) * (qy(k) - qy(a)) - (qy(b) - qy(a)) * (px(k) - px(a)) >= 0:
                    hull.pop()
                else:
                    break
            hull.append(k)
        # increasing part only, and slopes (endpoints) decreasing
        cs = 0.0
        prev_e = np.inf
        for h in range(1, len(hull)):
            a, b = hull[h - 1], hull[h]
            if qy(b) <= qy(a):
                break
            e_hi = prev_e
            e_lo = (qy(b) - qy(a)) / (px(b) - px(a))      # alpha below this prefers b over a
            if e_lo >= e_hi:
                continue
            # product b is bought for alpha in (e_lo, e_hi)
            p_mass = (F(e_hi) if np.isfinite(e_hi) else 1.0) - F(e_lo)
            inv_mass = (E_inv(e_hi) if np.isfinite(e_hi) else np.exp(s2 / 2)) - E_inv(e_lo)
            cs += qy(b) * inv_mass - px(b) * p_mass
            prev_e = e_lo
        total += w[i] * cs
    return total


def main():
    tag = os.environ.get("PCM_TAG", "")
    products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    est = np.load(OUT / f"pcm_estimates{tag}.npz")
    cfg = json.load(open(OUT / f"pcm_estimates{tag}.json"))
    theta, delta_all = est["theta"], est["delta"]
    th, ph, sp = pp.split(theta)
    markets = pp.build_markets(products, agents, int(cfg["nu_per_agent"]))

    n = len(products)
    own = np.empty(n)
    margins = np.empty(n)
    n_cross = np.empty(n)
    for m in markets:
        x, p, w = m.cpp.x, m.cpp.p, None
        d = pd_cpp.shares(delta_all[m.rows], m.cpp, th, ph, sp, True)
        s, Jp = d["s"], d["d_price"]
        assert np.max(np.abs(s - m.shares)) < 1e-9, m.market_id
        el = Jp * p[None, :] / s[:, None]
        own[m.rows] = np.diag(el)
        n_cross[m.rows] = (np.abs(el) > 1e-10).sum(1) - 1
        firms = products.firm_ids.to_numpy()[m.rows]
        margins[m.rows] = nash_markup(s, Jp, firms) / p

    # ---- 1984 minivan counterfactual ---------------------------------------------------
    m = next(mm for mm in markets if mm.market_id == 1984)
    rows = m.rows
    prod84 = products.loc[rows]
    firms = prod84.firm_ids.to_numpy()
    x = m.cpp.x
    p0 = m.cpp.p
    delta = delta_all[rows]
    # pull the consumer arrays back out of the Market: rebuild them the same way build_markets did
    ag = agents[agents.market_ids == 1984]
    nu_per = int(cfg["nu_per_agent"])
    from scipy.stats import qmc
    A = len(ag)
    nu = norm.ppf(qmc.Sobol(len(pp.SIGMA_ON), scramble=True, seed=1984).random(A * nu_per))
    rep = np.repeat(np.arange(A), nu_per)
    w = np.repeat(ag.weights.to_numpy(float), nu_per) / nu_per
    cols = {f"nu{k}": nu[:, k] for k in range(len(pp.SIGMA_ON))}
    cols["logfs_fv"] = (np.log(ag.fs.to_numpy(float)) * ag.fv.to_numpy(float))[rep]
    inc = ag.income.to_numpy(float)[rep]
    for lvl in ("low", "mid", "high"):
        cols[f"{lvl}_inc"] = ag[lvl].to_numpy(float)[rep] / inc
    tb = np.column_stack([cols[b] for _, _, b in pp.TASTE])
    sb = np.column_stack([cols[b] for _, b in pp.SCALE])
    tc = [c for _, c, _ in pp.TASTE]

    s0, Jp0 = market_eval(x, p0, w, tb, tc, sb, delta, th, ph, sp)
    assert np.max(np.abs(s0 - m.shares)) < 1e-9
    costs = p0 - nash_markup(s0, Jp0, firms)
    cs0 = consumer_surplus(x, p0, w, tb, tc, sb, delta, th, ph, sp)

    keep = prod84.mi.to_numpy() == 0
    xk, dk, ck, fk = x[keep], delta[keep], costs[keep], firms[keep]
    p = p0[keep].copy()
    for it in range(200):
        order = np.argsort(p)
        inv = np.empty_like(order); inv[order] = np.arange(len(p))
        ps = pp.break_price_ties(p[order])
        s, Jp = market_eval(xk[order], ps, w, tb, tc, sb, dk[order], th, ph, sp)
        target = ck[order] + nash_markup(s, Jp, fk[order])
        step = target - ps
        p_new = ps + 0.5 * step
        p = p_new[inv]
        if np.max(np.abs(step)) < 1e-7:
            break
    order = np.argsort(p)
    inv = np.empty_like(order); inv[order] = np.arange(len(p))
    ps = pp.break_price_ties(p[order])
    cs1 = consumer_surplus(xk[order], ps, w, tb, tc, sb, dk[order], th, ph, sp)
    s1, _ = market_eval(xk[order], ps, w, tb, tc, sb, dk[order], th, ph, sp)
    households = prod84.households.iloc[0]
    gain = households * (cs0 - cs1) / 1000.0        # $1000 per household * households -> $ million

    blp = {}
    for v in ("micro", "nomicro"):
        f = OUT / f"blp_{v}_summary.json"
        if f.exists():
            blp[v] = json.load(open(f))
    rows_out = [("minivan CS gain 1984 ($M)", gain, blp.get("micro", {}).get("minivan_cs_gain_1984_musd"), blp.get("nomicro", {}).get("minivan_cs_gain_1984_musd")),
                ("own-price elasticity, median", np.median(own), blp.get("micro", {}).get("own_elasticity_median"), blp.get("nomicro", {}).get("own_elasticity_median")),
                ("own-price elasticity, mean", np.mean(own), blp.get("micro", {}).get("own_elasticity_mean"), blp.get("nomicro", {}).get("own_elasticity_mean")),
                ("margin (p-c)/p, median", np.median(margins), blp.get("micro", {}).get("margin_median"), blp.get("nomicro", {}).get("margin_median")),
                ("margin (p-c)/p, mean", np.mean(margins), blp.get("micro", {}).get("margin_mean"), blp.get("nomicro", {}).get("margin_mean"))]
    table = pd.DataFrame(rows_out, columns=["quantity", "PCM", "BLP micro", "BLP no micro"]).set_index("quantity")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    print(f"\nPCM: products with a nonzero cross-price elasticity, median per product {np.median(n_cross):.0f} "
          f"of ~{np.median(products.groupby('market_ids').size()) - 1:.0f}; minivan share 1984 {s0[~keep].sum():.4f}, "
          f"inside share with/without minivans {s0.sum():.4f}/{s1.sum():.4f}, price iterations {it + 1}")
    print(f"1984 minivan margins (p-c)/p: {np.round(margins[rows][~keep], 3).tolist()}")
    table.to_csv(OUT / f"pcm_vs_blp{tag}.csv")
    pd.DataFrame({"market_ids": products.market_ids, "firm_ids": products.firm_ids, "prices": products.prices,
                  "shares": products.shares, "mi": products.mi, "own_elasticity": own, "margin": margins,
                  "n_cross_nonzero": n_cross}).to_csv(OUT / f"pcm_product_level{tag}.csv", index=False)


if __name__ == "__main__":
    main()

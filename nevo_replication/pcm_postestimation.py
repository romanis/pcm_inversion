"""Elasticities and Nevo's margin decomposition from the PCM estimates, side by side with BLP.

At observed prices, Bertrand-Nash first-order conditions give the margin each conduct implies:
    s + (Omega o (ds/dp)') (p - c) = 0   =>   p - c = -(Omega o (ds/dp)')^-1 s
with Omega(j,k) = 1 when j and k are priced jointly:
    single_product : Omega = I           (each brand priced alone)
    multi_product  : Omega from firm_ids (the actual portfolios)
    collusion      : Omega = all ones    (one firm sets every price)

Run after pcm_nevo.py and blp_baseline.py:
    PYTHONPATH=nevo_replication/.vendor:build-rel/python_bindings python nevo_replication/pcm_postestimation.py
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pyblp
import python_pcm_demand as pd_cpp

import pcm_nevo as pn

OUT = pathlib.Path(__file__).resolve().parent / "results"


def main() -> None:
    raw = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
    tag = os.environ.get("PCM_TAG", "")
    est = np.load(OUT / f"pcm_estimates{tag}.npz")
    theta, delta = est["theta"], est["delta"]
    cfg = pd.read_json(OUT / f"pcm_estimates{tag}.json", typ="series")
    # rebuild exactly the sample and consumers the estimation used
    pn.configure(bool(cfg.get("firm_taste", False)))
    agents = pn.expand_demographics(agents, int(cfg.get("demo_draws", 0)))
    products = raw
    if cfg.get("drop"):
        ds, dp, ratio = (float(v) for v in cfg["drop"].split(","))
        products = pn.drop_near_duplicates(raw, ds, dp, ratio)[0]
    # rows of the BLP arrays (estimated on all 2,256 products) that match the kept products
    key = raw.reset_index()[["index", "market_ids", "product_ids"]]
    blp_rows = products.merge(key, on=["market_ids", "product_ids"], how="left")["index"].to_numpy()
    th, ph, sp = pn.split(theta)

    n = len(products)
    own = np.empty(n)
    n_cross_nonzero = np.empty(n)
    margins = {k: np.empty(n) for k in ("single_product", "multi_product", "collusion")}
    for m in pn.build_markets(products, agents, int(cfg["gh_nodes"]), int(cfg.get("nu_draws", 0))):
        d = pd_cpp.shares(delta[m.rows], m.cpp, th, ph, sp, True)
        s, Jp, p = d["s"], d["d_price"], m.cpp.p
        assert np.max(np.abs(s - m.shares)) < 1e-10, m.market_id
        el = Jp * p[None, :] / s[:, None]           # el[j,k] = d log s_j / d log p_k
        own[m.rows] = np.diag(el)
        n_cross_nonzero[m.rows] = (np.abs(el) > 1e-12).sum(1) - 1
        firms = products.firm_ids.to_numpy()[m.rows]
        omegas = {
            "single_product": np.eye(len(s)),
            "multi_product": (firms[:, None] == firms[None, :]).astype(float),
            "collusion": np.ones((len(s), len(s))),
        }
        for k, om in omegas.items():
            markup = -np.linalg.solve(om * Jp.T, s)
            margins[k][m.rows] = markup / p

    blp = np.load(OUT / "blp_baseline.npz")
    p_all = products.prices.to_numpy()
    blp_margin = {k: blp[f"markup_{k}"].ravel()[blp_rows] / p_all for k in margins}
    blp_own = blp["own_elasticities"].ravel()[blp_rows]
    rows = []
    for k in margins:
        b = blp_margin[k]
        rows.append((f"margin (p-c)/p, {k}", np.median(margins[k]), np.mean(margins[k]), np.median(b), np.mean(b)))
    rows.insert(0, ("own-price elasticity", np.median(own), np.mean(own), np.median(blp_own), np.mean(blp_own)))
    table = pd.DataFrame(rows, columns=["quantity", "PCM median", "PCM mean", "BLP median", "BLP mean"]).set_index("quantity")
    # Nevo's decomposition: share of the collusive margin explained by each source
    for model, mm in (("PCM", margins), ("BLP", blp_margin)):
        med = {k: np.median(v) for k, v in mm.items()}
        table.loc[f"portfolio effect (multi - single), {model}", f"{model} median"] = med["multi_product"] - med["single_product"]
        table.loc[f"collusion effect (collusion - multi), {model}", f"{model} median"] = med["collusion"] - med["multi_product"]
    pd.set_option("display.width", 200)
    print(f"sample: {n} brand-markets (BLP columns use the same rows of the full-sample BLP fit)")
    print(table.round(4).to_string())
    print(f"\nPCM brands with a nonzero cross-price elasticity, per brand: median {np.median(n_cross_nonzero):.0f} "
          f"of the other brands in the market (BLP: all of them by construction)")
    table.to_csv(OUT / f"pcm_vs_blp{tag}.csv")
    pd.DataFrame({"market_ids": products.market_ids, "product_ids": products.product_ids,
                  "own_elasticity": own, "n_cross_nonzero": n_cross_nonzero,
                  **{f"margin_{k}": v for k, v in margins.items()}}).to_csv(OUT / f"pcm_product_level{tag}.csv", index=False)


if __name__ == "__main__":
    main()

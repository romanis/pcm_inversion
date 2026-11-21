"""BLP baselines on Petrin's (2002) minivan data, following the pyblp micro-moments tutorial.

Three fits on the same demand specification:
  micro        : demand and log-cost supply, with Petrin's CEX micro moments (the tutorial)
  nomicro      : the same problem without the micro moments
  demand_micro : demand only, aggregate + micro moments, no supply side: the same moments as the PCM
                 (DESIGN 9, owner decision 4), so the PCM comparison is like for like
Each fit reports the comparison quantities of the PCM post stage under the same keys
(nfp_harness.comparison): own-price elasticities, nonzero cross-elasticities, Nash margins, diversion
by body type, inside share, the 1984 no-minivan consumer-surplus change at fixed prices and with
prices re-solved to a Nash equilibrium (Petrin's headline), and that change by income group and
family size. Households are in thousands and prices in $1000, so households x CS is in $ million.

Run: python petrin_replication/estimate_blp.py [micro|nomicro|demand_micro|all]
Outputs: petrin_replication/results/blp_<variant>_{summary.json,results.txt,results.pkl,product_level.csv}, blp_<variant>.npz
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")   # anaconda's MKL otherwise runs a spinning 192-thread pool
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyblp  # noqa: E402

from nfp_harness import comparison  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
pyblp.options.digits = 3
pyblp.options.verbose = True

VARIANTS = ("micro", "nomicro", "demand_micro")
YEAR = 1984
BODY = ["mi", "sw", "su", "pv"]
GROUP_NAMES = BODY + ["car"]


def compute_ratio(v):
    return v[0] / v[1]


def compute_ratio_gradient(v):
    return [1 / v[1], -v[0] / v[1] ** 2]


def build_micro_moments():
    ds = pyblp.MicroDataset(name="CEX", observations=29125,
                            compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)))
    # X2 columns: 0 const, 1 -prices, 2 hpwt, 3 space, 4 air, 5 mpd, 6 fwd, 7 mi, 8 sw, 9 su, 10 pv
    # demographics: 0 const, 1 low/inc, 2 mid/inc, 3 high/inc, 4 log(fs)*fv, 5 age, 6 fs, 7 mid, 8 high
    body = {"mi": 7, "sw": 8, "su": 9, "pv": 10}

    def part(name, d, x, outside=0):
        return pyblp.MicroPart(name=name, dataset=ds,
                               compute_values=lambda t, p, a, d=d, x=x: np.outer(a.demographics[:, d], np.r_[outside, p.X2[:, x]]))

    values = dict(zip(*pd.read_csv(pyblp.data.PETRIN_VALUES_LOCATION).iloc[:, :2].to_numpy().T))
    moments = []
    body_parts = {b: part(f"E[{b}_j]", 0, x) for b, x in body.items()}
    for demo_name, d in (("age", 5), ("fs", 6)):
        for b, x in body.items():
            moments.append(pyblp.MicroMoment(
                name=f"E[{demo_name}_i | {b}_j]", value=values[f"E[{demo_name} | {b}]"],
                parts=[part(f"E[{demo_name}_i * {b}_j]", d, x), body_parts[b]],
                compute_value=compute_ratio, compute_gradient=compute_ratio_gradient))
    for inc_name, d in (("mid", 7), ("high", 8)):
        moments.append(pyblp.MicroMoment(
            name=f"E[1{{j>0}} | {inc_name}_i]", value=values[f"E[new | {inc_name}]"],
            parts=[part(f"E[1{{j>0}} * {inc_name}_i]", d, 0), part(f"E[{inc_name}_i]", d, 0, outside=1)],
            compute_value=compute_ratio, compute_gradient=compute_ratio_gradient))
    return moments


def body_labels(products: pd.DataFrame) -> np.ndarray:
    """Body type of each product: the first of mi, sw, su, pv whose dummy is 1, else car. Two products
    (1988 and 1989, firm 2) carry both mi and pv; they count as minivans, as in estimate_pcm.py."""
    lab = np.full(len(products), "car", dtype=object)
    for b in reversed(BODY):
        lab[products[b].to_numpy() == 1] = b
    return lab.astype(str)


def income_labels(agents: pd.DataFrame) -> np.ndarray:
    return np.select([agents["low"] == 1, agents["mid"] == 1, agents["high"] == 1], ["low", "mid", "high"], "none")


def fs_labels(agents: pd.DataFrame) -> np.ndarray:
    fs = agents["fs"].to_numpy(int)
    return np.where(fs >= 5, "5+", fs.astype(str))


def comparison_quantities(results, product_data: pd.DataFrame, costs: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Per-product elasticities, diversion by body type and margins, summarised as in the PCM post stage."""
    labels = body_labels(product_data)
    markets, frames = [], []
    for t in np.unique(product_data.market_ids):
        sel = (product_data.market_ids == t).to_numpy()
        s = product_data.shares.to_numpy(float)[sel]
        p = product_data.prices.to_numpy(float)[sel]
        e = np.asarray(results.compute_elasticities(market_id=t))   # e[j, k] = (ds_j/dp_k) p_k / s_j
        ds_dp = e * s[:, None] / p[None, :]
        m = comparison.market_quantities(s, ds_dp, p, labels[sel], GROUP_NAMES, costs[sel])
        markets.append(m)
        frames.append(pd.DataFrame({k: v for k, v in m.items() if k != "to_group"}
                                   | {"market_ids": t, "row": np.flatnonzero(sel)}
                                   | {f"diversion_to_{g}": m["to_group"][:, i] for i, g in enumerate(GROUP_NAMES)}))
    prod, gm = comparison.products_from_markets(markets, GROUP_NAMES)
    return comparison.summarise(prod, gm, GROUP_NAMES), pd.concat(frames, ignore_index=True)


def counterfactual_1984(results, product_data, agent_data, product_formulations, agent_formulation, costs):
    """Petrin's counterfactual: 1984 without minivans at fixed prices and with prices re-solved (Nash)."""
    d84 = product_data[product_data.market_ids == YEAR]
    a84 = agent_data[agent_data.market_ids == YEAR]
    keep = d84.mi.to_numpy() == 0
    costs84 = costs[(product_data.market_ids == YEAR).to_numpy()]
    sim = pyblp.Simulation(
        product_formulations=product_formulations[:2],
        product_data=d84[keep],
        xi=results.xi[(product_data.market_ids == YEAR).to_numpy()][keep],
        agent_formulation=agent_formulation, agent_data=a84,
        beta=results.beta, sigma=results.sigma, pi=results.pi,
    )
    nash = sim.replace_endogenous(costs=costs84[keep])
    fixed = sim.replace_endogenous(costs=costs84[keep], prices=d84.prices.to_numpy()[keep],
                                   iteration=pyblp.Iteration("return"))
    households = float(d84.households.iloc[0])
    w = a84.weights.to_numpy(float)
    per = {"observed": np.asarray(results.compute_consumer_surpluses(market_id=YEAR, keep_all=True)).ravel(),
           "fixed_prices": np.asarray(fixed.compute_consumer_surpluses(keep_all=True)).ravel(),
           "nash": np.asarray(nash.compute_consumer_surpluses(keep_all=True)).ravel()}
    # agent-level surpluses integrate to pyblp's own market surplus
    agg = float(np.asarray(results.compute_consumer_surpluses(market_id=YEAR)).ravel()[0])
    assert abs(w @ per["observed"] - agg) <= 1e-10 * max(1.0, abs(agg)), (w @ per["observed"], agg)
    out = {
        "minivan_cs_gain_1984_musd": households * (w @ per["observed"] - w @ per["nash"]),
        "minivan_cs_gain_1984_fixed_prices_musd": households * (w @ per["observed"] - w @ per["fixed_prices"]),
        "nash_status_1984": "converged" if bool(np.all(np.asarray(nash.fp_converged))) else "not_converged",
        "inside_share_1984_observed": float(d84.shares.sum()),
        "inside_share_1984_fixed_prices": float(np.asarray(fixed.product_data.shares).sum()),
        "inside_share_1984_nash": float(np.asarray(nash.product_data.shares).sum()),
    }
    tables = {}
    for name, labels in (("income", income_labels(a84)), ("fs", fs_labels(a84))):
        rows = comparison.cs_gain_by_group(per, w, labels, households)
        tables[name] = rows
        for r in rows:
            out[f"cs_gain_1984_{name}_{r['group']}_musd"] = r["gain_nash_musd"]
            out[f"cs_gain_1984_fixed_prices_{name}_{r['group']}_musd"] = r["gain_fixed_prices_musd"]
    return out, tables


def main(variant: str) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {VARIANTS} or 'all', got {variant!r}")
    product_data = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agent_data = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    product_formulations = (
        pyblp.Formulation("1 + hpwt + space + air + mpd + fwd + mi + sw + su + pv + pgnp + trend + trend2"),
        pyblp.Formulation("1 + I(-prices) + hpwt + space + air + mpd + fwd + mi + sw + su + pv"),
        pyblp.Formulation("1 + log(hpwt) + log(wt) + log(mpg) + air + fwd + trend * (jp + eu) + log(q)"),
    )
    agent_formulation = pyblp.Formulation("1 + I(low / income) + I(mid / income) + I(high / income) + I(log(fs) * fv) + age + fs + mid + high")
    supply = variant != "demand_micro"
    if supply:
        problem = pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data, costs_type="log")
    else:
        problem = pyblp.Problem(product_formulations[:2], product_data, agent_formulation, agent_data)
    print(problem)

    initial_sigma = np.diag([3.23, 0, 4.43, 0.46, 0.01, 2.58, 4.42, 0, 0, 0, 0])
    initial_pi = np.zeros((11, 9))
    initial_pi[1, 1:4] = [7.52, 31.13, 34.49]
    initial_pi[7:11, 4] = [0.57, 0.28, 0.31, 0.42]
    results = problem.solve(
        sigma=initial_sigma, pi=initial_pi,
        optimization=pyblp.Optimization("bfgs", {"gtol": 1e-4}),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-13}),
        se_type="clustered", W_type="clustered",
        micro_moments=build_micro_moments() if variant != "nomicro" else [],
    )
    print(results)

    costs = np.asarray(results.compute_costs()).ravel()   # Nash FOC costs (the supply fits also estimate them)
    summary, products = comparison_quantities(results, product_data, costs)
    cf, tables = counterfactual_1984(results, product_data, agent_data, product_formulations, agent_formulation, costs)
    summary = {"variant": variant, "supply_side": supply, "objective": float(np.asarray(results.objective).ravel()[0]),
               **cf, **summary}
    print(json.dumps(comparison.scalar_items(summary), indent=1))
    own = np.asarray(products.own_elasticity)
    np.savez(OUT / f"blp_{variant}.npz", delta=results.delta, xi=results.xi, beta=results.beta, sigma=results.sigma,
             pi=results.pi, gamma=results.gamma, own_elasticities=own, costs=costs, margins=products.margin.to_numpy())
    with open(OUT / f"blp_{variant}_summary.json", "w") as fh:
        json.dump(summary | {"cs_tables_1984": tables}, fh, indent=2, default=float)
    products.to_csv(OUT / f"blp_{variant}_product_level.csv", index=False)
    with open(OUT / f"blp_{variant}_results.txt", "w") as fh:
        fh.write(str(results) + "\n\n" + json.dumps(summary | {"cs_tables_1984": tables}, indent=1, default=float) + "\n")
    with open(OUT / f"blp_{variant}_results.pkl", "wb") as fh:
        pickle.dump(results.to_dict(), fh)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "micro"
    for v in (VARIANTS if arg == "all" else (arg,)):
        main(v)

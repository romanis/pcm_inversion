"""BLP baselines on Petrin's (2002) minivan data, following the pyblp micro-moments tutorial.

Two fits with identical demand and supply specifications:
  micro   : with Petrin's CEX micro moments (the tutorial)
  nomicro : the same problem without them
Both report the 1984 consumer-surplus change from removing minivans (Petrin's $367M headline),
so the PCM fit without any micro data can be placed against both.

Run: PYTHONPATH=nevo_replication/.vendor python petrin_replication/blp_baseline.py [micro|nomicro]
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pyblp

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
pyblp.options.digits = 3
pyblp.options.verbose = True


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


def main(variant: str) -> None:
    product_data = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agent_data = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    product_formulations = (
        pyblp.Formulation("1 + hpwt + space + air + mpd + fwd + mi + sw + su + pv + pgnp + trend + trend2"),
        pyblp.Formulation("1 + I(-prices) + hpwt + space + air + mpd + fwd + mi + sw + su + pv"),
        pyblp.Formulation("1 + log(hpwt) + log(wt) + log(mpg) + air + fwd + trend * (jp + eu) + log(q)"),
    )
    agent_formulation = pyblp.Formulation("1 + I(low / income) + I(mid / income) + I(high / income) + I(log(fs) * fv) + age + fs + mid + high")
    problem = pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data, costs_type="log")
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
        micro_moments=build_micro_moments() if variant == "micro" else [],
    )
    print(results)

    # Petrin's counterfactual: 1984 without minivans, prices re-equilibrated at estimated costs
    year = 1984
    d84 = product_data[product_data.market_ids == year]
    costs84 = results.compute_costs(market_id=year)
    sim = pyblp.Simulation(
        product_formulations=product_formulations[:2],
        product_data=d84[d84.mi == 0],
        xi=results.xi[product_data.market_ids == year][d84.mi.to_numpy() == 0],
        agent_formulation=agent_formulation, agent_data=agent_data[agent_data.market_ids == year],
        beta=results.beta, sigma=results.sigma, pi=results.pi,
    )
    sim_results = sim.replace_endogenous(costs=costs84[d84.mi.to_numpy() == 0])
    households = d84.households.iloc[0]
    cs = households * results.compute_consumer_surpluses(market_id=year)
    cs_cf = households * sim_results.compute_consumer_surpluses()
    gain = float((cs - cs_cf).ravel()[0])
    el = results.compute_elasticities()
    own = results.extract_diagonals(el)
    costs = results.compute_costs()
    margins = (product_data.prices.to_numpy() - costs.ravel()) / product_data.prices.to_numpy()
    summary = {
        "variant": variant, "objective": float(np.asarray(results.objective).ravel()[0]),
        "minivan_cs_gain_1984_musd": gain,
        "own_elasticity_median": float(np.median(own)), "own_elasticity_mean": float(np.mean(own)),
        "margin_median": float(np.median(margins)), "margin_mean": float(np.mean(margins)),
    }
    print(summary)
    np.savez(OUT / f"blp_{variant}.npz", delta=results.delta, xi=results.xi, beta=results.beta, sigma=results.sigma,
             pi=results.pi, gamma=results.gamma, own_elasticities=own, costs=costs, margins=margins)
    pd.Series(summary).to_json(OUT / f"blp_{variant}_summary.json", indent=2)
    with open(OUT / f"blp_{variant}_results.txt", "w") as fh:
        fh.write(str(results) + "\n\n" + str(summary) + "\n")
    with open(OUT / f"blp_{variant}_results.pkl", "wb") as fh:
        pickle.dump(results.to_dict(), fh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "micro")

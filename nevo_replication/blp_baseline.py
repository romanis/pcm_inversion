"""BLP baseline on Nevo's fake cereal data, following the pyblp tutorial.

Specification is the pyblp "Random Coefficients Logit Tutorial with the Fake Cereal Data":
https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/nevo.html
plus the post-estimation tutorial for elasticities and markups under alternative
ownership structures (single-product, actual multi-brand firms, full collusion),
which is the decomposition of Nevo (1998/2001).

Run:  PYTHONPATH=nevo_replication/.vendor python nevo_replication/blp_baseline.py
"""
from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
import pyblp

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

pyblp.options.digits = 4
pyblp.options.verbose = True


def main() -> None:
    products = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)

    product_formulations = (
        pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
        pyblp.Formulation("1 + prices + sugar + mushy"),
    )
    agent_formulation = pyblp.Formulation("0 + income + income_squared + age + child")
    problem = pyblp.Problem(product_formulations, products, agent_formulation, agents)

    initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
    initial_pi = np.array([
        [5.4819, 0, 0.2037, 0],
        [15.8935, -1.2000, 0, 2.6342],
        [-0.2506, 0, 0.0511, 0],
        [1.2650, 0, -0.8091, 0],
    ])
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=pyblp.Optimization("bfgs", {"gtol": 1e-5}),
        method="1s",
    )
    print(results)

    # --- post-estimation: elasticities and markups ---------------------------
    elasticities = results.compute_elasticities()
    own = results.extract_diagonals(elasticities)
    means = results.extract_diagonal_means(elasticities)

    # Nevo's decomposition: at observed prices, the margin implied by each conduct.
    n = len(products)
    conduct = {
        "single_product": np.arange(n).reshape(-1, 1),   # every brand its own firm
        "multi_product": None,                           # actual firm_ids
        "collusion": np.zeros((n, 1), dtype=int),        # one firm per market
    }
    p = products["prices"].to_numpy().reshape(-1, 1)
    markups = {k: p - results.compute_costs(firm_ids=f) for k, f in conduct.items()}
    costs = p - markups["multi_product"]
    summary = pd.DataFrame({
        k: [float(np.median(v / p)), float(np.mean(v / p))] for k, v in markups.items()
    }, index=["median (p-c)/p", "mean (p-c)/p"])
    print("\nOwn-price elasticity: median", np.median(own), "mean", np.mean(own))
    print("Mean own elasticity per market, median over markets:", np.median(means))
    print("\nMargins implied by marginal costs from the multi-product Nash model:\n", summary)

    np.savez(
        OUT / "blp_baseline.npz",
        delta=results.delta, xi=results.xi, beta=results.beta, sigma=results.sigma,
        pi=results.pi, own_elasticities=own, costs=costs,
        **{f"markup_{k}": v for k, v in markups.items()},
    )
    summary.to_csv(OUT / "blp_margins.csv")
    with open(OUT / "blp_results.txt", "w") as fh:
        fh.write(str(results))
        fh.write("\n\n" + summary.to_string() + "\n")
    with open(OUT / "blp_results.pkl", "wb") as fh:
        pickle.dump(results.to_dict(), fh)


if __name__ == "__main__":
    main()

"""Nevo (2000) cereal demand estimated with the pure characteristics model (PCM).

Same data, instruments, fixed effects and demographic interactions as the pyblp
tutorial specification run in blp_baseline.py, with the logit error removed:

    u_ijt = delta_jt + x_j . t_i - c_i * a_i * p_jt,      u_i0t = 0
    x_j   = [1, sugar_j, mushy_j]
    t_i   = sigma * nu_i + Pi D_i                 (taste for cereal, sugar, mushy)
    c_i   = exp(phi . [income, income^2, child])  (demographic shifters of price sensitivity)
    a_i   ~ LogNormal(0, sigma_p)                  (integrated in closed form)
    delta_jt = d_j (brand fixed effect) + xi_jt

PCM utility has no scale of its own, so the median price sensitivity at mean
demographics is normalised to 1 and delta is in dollars. That is why there is no
linear price coefficient: BLP's beta_price becomes this normalisation.

Integration: each market's 20 demographic draws (Nevo's CPS draws) are crossed with
a Gauss-Hermite product grid over nu, so every market uses 20 * n^3 consumers.

GMM: one-step, W = (Z'Z/N)^-1, Z = 20 excluded instruments demeaned by brand,
as pyblp method="1s". Gradient is analytic: d delta/d theta = -J_delta^-1 J_theta.

Run: PYTHONPATH=nevo_replication/.vendor:build-rel/python_bindings \
     python nevo_replication/pcm_nevo.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyblp
import python_pcm_demand as pd_cpp
from scipy.optimize import minimize

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

DEMOS = ["income", "income_squared", "age", "child"]
CHARS = ["const", "sugar", "mushy"]
# taste parameters: (name, characteristic column, basis)
TASTE = [
    ("sigma_const", 0, "nu0"), ("sigma_sugar", 1, "nu1"), ("sigma_mushy", 2, "nu2"),
    ("pi_const_income", 0, "income"), ("pi_const_age", 0, "age"),
    ("pi_sugar_income", 1, "income"), ("pi_sugar_age", 1, "age"),
    ("pi_mushy_income", 2, "income"), ("pi_mushy_age", 2, "age"),
]
# demographic shifters of log price sensitivity, mirroring BLP's price x {income, income^2, child}
SCALE = [("phi_price_income", "income"), ("phi_price_income_sq", "income_squared"), ("phi_price_child", "child")]
NAMES = [t[0] for t in TASTE] + [s[0] for s in SCALE] + ["sigma_p"]
N_TASTE, N_SCALE = len(TASTE), len(SCALE)


def gauss_hermite(dim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    nodes = np.stack(np.meshgrid(*[z] * dim, indexing="ij"), -1).reshape(-1, dim)
    wts = np.prod(np.stack(np.meshgrid(*[w] * dim, indexing="ij"), -1).reshape(-1, dim), axis=1)
    return nodes, wts


@dataclass
class MarketData:
    market_id: str
    rows: np.ndarray            # row indices into the product table, in ascending-price order
    shares: np.ndarray
    cpp: object                 # pd_cpp.Market
    delta: np.ndarray = field(default=None)  # warm start


def build_markets(products: pd.DataFrame, agents: pd.DataFrame, gh_nodes: int) -> list[MarketData]:
    nu, nu_w = gauss_hermite(3, gh_nodes)
    markets = []
    for mkt, prod in products.groupby("market_ids", sort=True):
        prod = prod.sort_values("prices")
        if prod.prices.duplicated().any():
            raise ValueError(f"{mkt}: tied prices")
        ag = agents[agents.market_ids == mkt]
        D = ag[DEMOS].to_numpy(float)
        # cross product: each demographic draw with every nu node
        Dx = np.repeat(D, len(nu), axis=0)
        nux = np.tile(nu, (len(D), 1))
        w = np.repeat(ag.weights.to_numpy(float), len(nu)) * np.tile(nu_w, len(D))
        cols = {"nu0": nux[:, 0], "nu1": nux[:, 1], "nu2": nux[:, 2]}
        cols.update({d: Dx[:, k] for k, d in enumerate(DEMOS)})
        taste_basis = np.column_stack([cols[b] for _, _, b in TASTE])
        scale_basis = np.column_stack([cols[b] for _, b in SCALE])
        x = np.column_stack([np.ones(len(prod)), prod.sugar, prod.mushy]).astype(float)
        m = pd_cpp.Market(x, prod.prices.to_numpy(float), w, taste_basis,
                          [c for _, c, _ in TASTE], scale_basis)
        markets.append(MarketData(mkt, prod.index.to_numpy(), prod.shares.to_numpy(float), m))
    return markets


def split(theta: np.ndarray):
    return theta[:N_TASTE], theta[N_TASTE:N_TASTE + N_SCALE], float(theta[-1])


class PCMProblem:
    def __init__(self, products: pd.DataFrame, agents: pd.DataFrame, gh_nodes: int = 5, workers: int = 94):
        self.products = products
        self.markets = build_markets(products, agents, gh_nodes)
        self.n = len(products)
        brand = products.product_ids.to_numpy()
        self.brand_codes = pd.factorize(brand)[0]
        Z = products[[f"demand_instruments{k}" for k in range(20)]].to_numpy(float)
        self.Z = Z - self._brand_mean(Z)
        self.W = np.linalg.inv(self.Z.T @ self.Z / self.n)
        self.pool = ThreadPoolExecutor(workers)
        self.evals = 0
        self.last = None

    def _brand_mean(self, a: np.ndarray) -> np.ndarray:
        a2 = a.reshape(self.n, -1)
        sums = np.zeros((self.brand_codes.max() + 1, a2.shape[1]))
        np.add.at(sums, self.brand_codes, a2)
        counts = np.bincount(self.brand_codes)[:, None]
        return (sums / counts)[self.brand_codes].reshape(a.shape)

    def _solve_market(self, m: MarketData, theta: np.ndarray, grad: bool):
        th, ph, sp = split(theta)
        warm = m.delta if m.delta is not None else np.empty(0)
        r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, warm, 1e-13, 20000)
        if not r["converged"]:
            # retry cold: a warm start from a distant theta can sit in a bad basin
            r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, np.empty(0), 1e-13, 20000)
        out = {"delta": r["delta"], "converged": r["converged"], "resid": r["max_abs_residual"]}
        if grad:
            d = pd_cpp.shares(r["delta"], m.cpp, th, ph, sp, True)
            out["ddelta"] = -np.linalg.solve(d["d_delta"], d["d_theta"])
        return out

    def delta(self, theta: np.ndarray, grad: bool):
        res = list(self.pool.map(lambda m: self._solve_market(m, theta, grad), self.markets))
        delta = np.empty(self.n)
        ddelta = np.empty((self.n, len(theta))) if grad else None
        ok = True
        worst = 0.0
        for m, r in zip(self.markets, res):
            delta[m.rows] = r["delta"]
            if grad:
                ddelta[m.rows] = r["ddelta"]
            ok &= bool(r["converged"])
            worst = max(worst, float(r["resid"]))
            if r["converged"]:
                m.delta = r["delta"]
        return delta, ddelta, ok, worst

    def objective(self, theta: np.ndarray):
        t0 = time.time()
        self.evals += 1
        delta, ddelta, ok, worst = self.delta(theta, grad=True)
        xi = delta - self._brand_mean(delta)
        g = self.Z.T @ xi / self.n
        q = float(self.n * g @ self.W @ g)
        G = self.Z.T @ ddelta / self.n
        dq = 2 * self.n * (G.T @ self.W @ g)
        self.last = dict(theta=theta.copy(), delta=delta, xi=xi, q=q, G=G, ok=ok)
        print(f"eval {self.evals:4d}  q={q:.6e}  |grad|={np.abs(dq).max():.2e}  "
              f"inversions_ok={ok} worst_resid={worst:.1e}  {time.time() - t0:.1f}s", flush=True)
        if not ok:
            return 1e10, np.zeros_like(theta)
        return q, dq

    def standard_errors(self, theta: np.ndarray) -> np.ndarray:
        self.objective(theta)
        G, xi = self.last["G"], self.last["xi"]
        S = (self.Z * xi[:, None] ** 2).T @ self.Z / self.n
        bread = np.linalg.inv(G.T @ self.W @ G)
        V = bread @ G.T @ self.W @ S @ self.W @ G @ bread / self.n
        return np.sqrt(np.diag(V))


def start_values() -> np.ndarray:
    th = np.zeros(len(NAMES))
    th[NAMES.index("sigma_const")] = 0.05
    th[NAMES.index("sigma_sugar")] = 0.003
    th[NAMES.index("sigma_mushy")] = 0.03
    th[NAMES.index("sigma_p")] = 0.5
    return th


def main() -> None:
    products = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
    gh = int(os.environ.get("PCM_GH_NODES", "5"))
    prob = PCMProblem(products, agents, gh_nodes=gh)
    print(f"{len(prob.markets)} markets, {len(products)} products, "
          f"{prob.markets[0].cpp.p.size} brands, consumers per market {20 * gh ** 3}")

    bounds = [(0.0, None) if n.startswith("sigma") else (None, None) for n in NAMES]
    bounds[NAMES.index("sigma_p")] = (1e-3, None)
    t0 = time.time()
    res = minimize(prob.objective, start_values(), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 500})
    print(res.message, f"{time.time() - t0:.0f}s")
    se = prob.standard_errors(res.x)
    table = pd.DataFrame({"estimate": res.x, "robust_se": se}, index=NAMES)
    print(table)

    delta = prob.last["delta"]
    np.savez(OUT / "pcm_estimates.npz", theta=res.x, se=se, delta=delta, xi=prob.last["xi"])
    table.to_csv(OUT / "pcm_estimates.csv")
    with open(OUT / "pcm_estimates.json", "w") as fh:
        json.dump({"objective": float(res.fun), "converged": bool(res.success), "message": str(res.message),
                   "evaluations": prob.evals, "gh_nodes": gh}, fh, indent=2)


if __name__ == "__main__":
    main()

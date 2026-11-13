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
    theta_ok: np.ndarray = field(default=None)  # theta at which delta solves the inversion


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


def expand_demographics(agents: pd.DataFrame, per_market: int, seed: int = 0) -> pd.DataFrame:
    """Smooth each market's demographic draws to `per_market` draws.

    The fake-data agent file has 20 draws per market, so demographic tastes take only ~20 values and
    shares jump when a brand wins or loses one whole type. Each new draw copies a random original
    draw from the same market, adds Gaussian noise to income and age with Silverman's bandwidth
    (computed per market), keeps the child flag, and recomputes income_squared with the identity
    the file satisfies exactly: income_squared = income^2 + 18.1139 income - 0.8824.
    """
    if per_market <= 0:
        return agents
    rng = np.random.default_rng(seed)
    a_sq = np.linalg.lstsq(np.column_stack([np.ones(len(agents)), agents.income, agents.income ** 2]),
                           agents.income_squared, rcond=None)[0]
    out = []
    for mkt, g in agents.groupby("market_ids", sort=True):
        n = len(g)
        pick = rng.integers(0, n, per_market)
        new = g.iloc[pick].copy()
        for col in ("income", "age"):
            h = 1.06 * g[col].std() * n ** (-0.2)
            new[col] = new[col].to_numpy() + h * rng.standard_normal(per_market)
        new["income_squared"] = a_sq[0] + a_sq[1] * new.income + a_sq[2] * new.income ** 2
        new["weights"] = 1.0 / per_market
        out.append(new)
    return pd.concat(out, ignore_index=True)


def drop_near_duplicates(products: pd.DataFrame, max_dsugar: float, max_rel_dprice: float,
                         min_share_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within each market, for every pair of brands with the same mushy value, sugar within
    max_dsugar grams and prices within max_rel_dprice of each other, drop the smaller brand when
    the larger outsells it by more than min_share_ratio. Dropped shares go to the outside good.
    Returns (kept products, dropped rows with the brand that dominated each one)."""
    drop = []
    for mkt, d in products.groupby("market_ids"):
        idx = d.index.to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ra, rb = products.loc[idx[a]], products.loc[idx[b]]
                if ra.mushy != rb.mushy or abs(ra.sugar - rb.sugar) > max_dsugar:
                    continue
                if abs(ra.prices - rb.prices) / min(ra.prices, rb.prices) >= max_rel_dprice:
                    continue
                big, small = (ra, rb) if ra.shares >= rb.shares else (rb, ra)
                if big.shares / small.shares > min_share_ratio:
                    drop.append((small.name, big.product_ids))
    dropped_idx = sorted({i for i, _ in drop})
    dropped = products.loc[dropped_idx, ["market_ids", "product_ids", "prices", "sugar", "mushy", "shares"]].copy()
    dropped["dominated_by"] = [next(b for i, b in drop if i == k) for k in dropped_idx]
    return products.drop(index=dropped_idx).reset_index(drop=True), dropped


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
        r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, warm, 1e-13, 3000)
        if not r["converged"] and m.delta is not None:
            # Path in theta: from the last theta where this market inverted, move to the new
            # theta in short steps, each warm-started from the previous solution.
            d, tau, step = m.delta, 0.0, 0.25
            while tau < 1.0 and step > 1e-4:
                t = min(1.0, tau + step)
                tt = m.theta_ok + t * (theta - m.theta_ok)
                rr = pd_cpp.invert(m.shares, m.cpp, *split(tt), d, 1e-13, 3000)
                if rr["converged"]:
                    d, tau, step = rr["delta"], t, min(0.5, 2 * step)
                else:
                    step /= 2
            if tau >= 1.0:
                r = rr
        if not r["converged"]:
            rw = r
            r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, np.empty(0), 1e-13, 6000)
            print(f"  {m.market_id}: warm start and theta path failed (resid {rw['max_abs_residual']:.1e}); "
                  f"cold: converged={r['converged']} resid {r['max_abs_residual']:.1e}", flush=True)
        out = {"delta": r["delta"], "converged": r["converged"], "resid": r["max_abs_residual"]}
        if grad:
            d = pd_cpp.shares(r["delta"], m.cpp, th, ph, sp, True)
            try:
                out["ddelta"] = -np.linalg.solve(d["d_delta"], d["d_theta"])
            except np.linalg.LinAlgError:
                zero = np.flatnonzero(np.abs(d["d_delta"]).sum(1) == 0)
                print(f"  {m.market_id}: singular ds/ddelta, rows with no derivative {zero.tolist()}, "
                      f"shares {d['s'][zero].tolist()}", flush=True)
                out["ddelta"] = np.full_like(d["d_theta"], np.nan)
                out["converged"] = False
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
                m.theta_ok = theta.copy()
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
              f"inversions_ok={ok} worst_resid={worst:.1e}  {time.time() - t0:.1f}s\n"
              f"      theta={np.array2string(theta, precision=4, max_line_width=250)}", flush=True)
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


def parameter_scales() -> np.ndarray:
    """Typical size of each parameter. Delta is in dollars per serving (prices 0.05 to 0.25), so
    a taste spread of 0.05 on the constant, 0.003 per gram of sugar (sugar 0 to 20) and 0.03 on
    mushy each move utility by a fraction of the price level. The optimizer works in u = theta / scale,
    so its unit first step stays within that range instead of jumping sigma_sugar to ~1."""
    sc = {"const": 0.05, "sugar": 0.003, "mushy": 0.03}
    out = []
    for n in NAMES:
        if n.startswith(("sigma_const", "sigma_sugar", "sigma_mushy", "pi_")):
            out.append(sc[n.split("_")[1]])
        elif n == "phi_price_income_sq":
            out.append(0.02)      # income^2 has sd 16
        elif n.startswith("phi_"):
            out.append(0.3)
        else:
            out.append(0.5)       # sigma_p
    return np.array(out)


def main() -> None:
    products = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
    gh = int(os.environ.get("PCM_GH_NODES", "5"))
    tag = os.environ.get("PCM_TAG", "")
    demo = int(os.environ.get("PCM_DEMO_DRAWS", "0"))
    agents = expand_demographics(agents, demo)
    if os.environ.get("PCM_DROP"):          # e.g. "2,0.02,10" = dsugar, rel dprice, share ratio
        ds, dp, ratio = (float(v) for v in os.environ["PCM_DROP"].split(","))
        products, dropped = drop_near_duplicates(products, ds, dp, ratio)
        dropped.to_csv(OUT / f"pcm_dropped{tag}.csv", index=False)
        print(f"dropped {len(dropped)} near-duplicate brand-markets in {dropped.market_ids.nunique()} markets:\n"
              f"{dropped.round(4).to_string(index=False)}")
    prob = PCMProblem(products, agents, gh_nodes=gh)
    print(f"{len(prob.markets)} markets, {len(products)} products, "
          f"consumers per market {agents.groupby("market_ids").size().iloc[0] * gh ** 3}")

    scale = parameter_scales()
    # sigmas are optimised in logs: sigma = scale * exp(u). At sigma = 0 the model can lose the
    # continuous taste heterogeneity it needs to rationalise every brand's share (the inversion
    # then has no solution), so the boundary must not be reachable by a single step.
    is_log = np.array([n.startswith("sigma") for n in NAMES])

    def to_theta(u):
        return np.where(is_log, scale * np.exp(np.where(is_log, u, 0.0)), scale * u)

    def scaled(u):
        th = to_theta(u)
        q, dq = prob.objective(th)
        return q, dq * np.where(is_log, th, scale)

    u0 = np.where(is_log, np.log(np.maximum(start_values(), 1e-12) / scale), start_values() / scale)
    t0 = time.time()
    # Restart L-BFGS from its own answer: with finite draws the objective has kinks where a brand
    # enters or leaves a consumer's frontier, and a kink can end a line search early.
    best = None
    for restart in range(6):
        res = minimize(scaled, u0 if best is None else best.x, jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-6, "ftol": 1e-12, "maxiter": 500})
        print(f"restart {restart}: q={res.fun:.6e} {res.message}", flush=True)
        if best is not None and res.fun > best.fun - 1e-7 * abs(best.fun):
            best = res if res.fun < best.fun else best
            break
        best = res
    res = best
    res.x = to_theta(res.x)
    print(res.message, f"{time.time() - t0:.0f}s")
    se = prob.standard_errors(res.x)
    table = pd.DataFrame({"estimate": res.x, "robust_se": se}, index=NAMES)
    print(table)

    delta = prob.last["delta"]
    np.savez(OUT / f"pcm_estimates{tag}.npz", theta=res.x, se=se, delta=delta, xi=prob.last["xi"])
    table.to_csv(OUT / f"pcm_estimates{tag}.csv")
    with open(OUT / f"pcm_estimates{tag}.json", "w") as fh:
        json.dump({"objective": float(res.fun), "converged": bool(res.success), "message": str(res.message),
                   "evaluations": prob.evals, "gh_nodes": gh, "demo_draws": demo, "drop": os.environ.get("PCM_DROP", "")}, fh, indent=2)


if __name__ == "__main__":
    main()

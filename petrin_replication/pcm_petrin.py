"""Petrin (2002) minivan demand estimated with the pure characteristics model (PCM).

Same data, instruments and heterogeneity as the pyblp Petrin tutorial, with the logit error
removed and no micro moments:

    u_ijt = delta_jt + x_jt . t_i - c_i * a_i * p_jt,         u_i0t = 0
    x_j   = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv]
    t_i   = sigma * nu_i                  on [1, hpwt, space, air, mpd, fwd]   (Petrin's sigmas)
          + pi * log(fs_i) * fv_i         on [mi, sw, su, pv]                   (Petrin's pis)
    c_i   = exp(phi . [low_i, mid_i, high_i] / income_i)                        (Petrin's price x income)
    a_i   ~ LogNormal(0, sigma_p)
    delta_jt = x1_jt beta + xi_jt,  x1 = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv, pgnp, trend, trend2]

PCM has no utility scale, so the median price sensitivity at c_i = 1 is normalised to one and
delta is in $1000 (the price unit). beta is concentrated out by IV with the 22 demand instruments.
Consumers: each of the 1000 CEX agent rows per market is crossed with PCM_NU_PER_AGENT scrambled
Sobol normal draws over the 6 taste dimensions.

Run: PYTHONPATH=nevo_replication/.vendor:build-rel2/python_bindings python petrin_replication/pcm_petrin.py
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
from scipy.stats import norm, qmc

OUT = pathlib.Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

X_COLS = ["const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv"]
X1_COLS = X_COLS + ["pgnp", "trend", "trend2"]
SIGMA_ON = ["const", "hpwt", "space", "air", "mpd", "fwd"]
PI_ON = ["mi", "sw", "su", "pv"]
TASTE = [(f"sigma_{c}", X_COLS.index(c), f"nu{k}") for k, c in enumerate(SIGMA_ON)] + \
        [(f"pi_{c}_logfs_fv", X_COLS.index(c), "logfs_fv") for c in PI_ON]
SCALE = [("phi_low_inc", "low_inc"), ("phi_mid_inc", "mid_inc"), ("phi_high_inc", "high_inc")]
NAMES = [t[0] for t in TASTE] + [s[0] for s in SCALE] + ["sigma_p"]
N_TASTE, N_SCALE = len(TASTE), len(SCALE)
N_ZD = 22
TOL = 1e-10          # max |s_model - s_data|; shares are 1e-5 to 5e-3, so this is 1e-5 relative at worst


def split(theta):
    return theta[:N_TASTE], theta[N_TASTE:N_TASTE + N_SCALE], float(theta[-1])


@dataclass
class MarketData:
    market_id: int
    rows: np.ndarray
    shares: np.ndarray
    cpp: object
    delta: np.ndarray = field(default=None)
    theta_ok: np.ndarray = field(default=None)


def break_price_ties(p: np.ndarray, min_gap: float = 0.01) -> np.ndarray:
    """Strictly increasing prices with a minimum gap (default $10 on prices in $1000).

    Petrin's prices are list prices deflated by the CPI, so 95 pairs are exact ties and 1% of
    neighbouring gaps are under $1. In the PCM the conditional share derivative between two
    products is f(e)/(price gap), so a $0.001 gap makes the Jacobian singular to working
    precision. Spreading near-ties to $10 changes nothing economically at $5,000 to $50,000.
    Order among tied products follows their row order."""
    p = p.copy()
    order = np.argsort(p, kind="stable")
    for k in range(1, len(p)):
        i, j = order[k - 1], order[k]
        if p[j] < p[i] + min_gap:
            p[j] = p[i] + min_gap
    return p


def build_markets(products: pd.DataFrame, agents: pd.DataFrame, nu_per_agent: int, seed: int = 0) -> list[MarketData]:
    dim = len(SIGMA_ON)
    markets = []
    for mkt, prod in products.groupby("market_ids", sort=True):
        prod = prod.assign(p_sort=break_price_ties(prod.prices.to_numpy(float))).sort_values("p_sort")
        ag = agents[agents.market_ids == mkt]
        A = len(ag)
        nu = norm.ppf(qmc.Sobol(dim, scramble=True, seed=seed + int(mkt)).random(A * nu_per_agent))
        rep = np.repeat(np.arange(A), nu_per_agent)
        w = np.repeat(ag.weights.to_numpy(float), nu_per_agent) / nu_per_agent
        cols = {f"nu{k}": nu[:, k] for k in range(dim)}
        cols["logfs_fv"] = (np.log(ag.fs.to_numpy(float)) * ag.fv.to_numpy(float))[rep]
        inc = ag.income.to_numpy(float)[rep]
        for lvl in ("low", "mid", "high"):
            cols[f"{lvl}_inc"] = ag[lvl].to_numpy(float)[rep] / inc
        taste_basis = np.column_stack([cols[b] for _, _, b in TASTE])
        scale_basis = np.column_stack([cols[b] for _, b in SCALE])
        x = np.column_stack([np.ones(len(prod))] + [prod[c].to_numpy(float) for c in X_COLS[1:]])
        m = pd_cpp.Market(x, prod.p_sort.to_numpy(float), w, taste_basis, [c for _, c, _ in TASTE], scale_basis)
        markets.append(MarketData(int(mkt), prod.index.to_numpy(), prod.shares.to_numpy(float), m))
    return markets


class PCMProblem:
    def __init__(self, products, agents, nu_per_agent=4, workers=13, path_seconds=60.0):
        self.products = products
        self.path_seconds = path_seconds
        self.markets = build_markets(products, agents, nu_per_agent)
        self.n = len(products)
        self.X1 = np.column_stack([np.ones(self.n)] + [products[c].to_numpy(float) for c in X1_COLS[1:]])
        self.Z = products[[f"demand_instruments{k}" for k in range(N_ZD)]].to_numpy(float)
        self.set_weighting(np.linalg.inv(self.Z.T @ self.Z / self.n))
        self.pool = ThreadPoolExecutor(workers)
        self.evals = 0
        self.last = None

    def set_weighting(self, W):
        self.W = W
        ZW = self.Z @ W @ self.Z.T / self.n
        XZWZ = self.X1.T @ ZW
        self.beta_of = np.linalg.solve(XZWZ @ self.X1, XZWZ)         # beta = beta_of @ delta
        self.M = np.eye(self.n) - self.X1 @ self.beta_of              # xi = M delta

    def _solve_market(self, m: MarketData, theta, grad):
        th, ph, sp = split(theta)
        t_start, stages = time.time(), []
        warm = m.delta if m.delta is not None else np.empty(0)
        # a trial point gets a short first try; most of the run's time was going into trial
        # points the optimizer then rejected
        r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, warm, TOL, 8000 if m.delta is not None else 20000)
        stages.append(f"warm {time.time() - t_start:.0f}s")
        if not r["converged"] and m.delta is not None:
            t0 = time.time()
            d, tau, step = m.delta, 0.0, 0.5
            while tau < 1.0 and step > 1e-3 and time.time() - t0 < self.path_seconds:
                t = min(1.0, tau + step)
                rr = pd_cpp.invert(m.shares, m.cpp, *split(m.theta_ok + t * (theta - m.theta_ok)), d, TOL, 2000)
                if rr["converged"]:
                    d, tau, step = rr["delta"], t, min(0.5, 2 * step)
                else:
                    step /= 2
            if tau >= 1.0:
                r = rr
            stages.append(f"theta path to tau={tau:.3f} {time.time() - t0:.0f}s")
        if not r["converged"] and m.delta is None:
            t0 = time.time()
            r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, np.empty(0), TOL, 40000)
            stages.append(f"cold {time.time() - t0:.0f}s")
        if not r["converged"]:
            # no cold restart at a trial point: a far-away theta the optimizer is probing is
            # rejected (objective 1e10) and L-BFGS shortens its step
            print(f"  {m.market_id}: inversion failed (resid {r['max_abs_residual']:.1e}): {', '.join(stages)}", flush=True)
        if time.time() - t_start > 60:
            print(f"  {m.market_id}: slow inversion {time.time() - t_start:.0f}s: {', '.join(stages)}", flush=True)
        out = {"delta": r["delta"], "converged": r["converged"], "resid": r["max_abs_residual"]}
        if grad:
            d = pd_cpp.shares(r["delta"], m.cpp, th, ph, sp, True)
            try:
                out["ddelta"] = -np.linalg.solve(d["d_delta"], d["d_theta"])
            except np.linalg.LinAlgError:
                zero = np.flatnonzero(np.abs(d["d_delta"]).sum(1) == 0)
                print(f"  {m.market_id}: singular ds/ddelta at rows {zero.tolist()}", flush=True)
                out["ddelta"] = np.full_like(d["d_theta"], np.nan)
                out["converged"] = False
        return out

    def delta(self, theta, grad):
        res = list(self.pool.map(lambda m: self._solve_market(m, theta, grad), self.markets))
        delta = np.empty(self.n)
        ddelta = np.empty((self.n, len(theta))) if grad else None
        ok, worst = True, 0.0
        for m, r in zip(self.markets, res):
            delta[m.rows] = r["delta"]
            if grad:
                ddelta[m.rows] = r["ddelta"]
            ok &= bool(r["converged"])
            worst = max(worst, float(r["resid"]))
            if r["converged"]:
                m.delta, m.theta_ok = r["delta"], theta.copy()
        return delta, ddelta, ok, worst

    def objective(self, theta):
        t0 = time.time()
        self.evals += 1
        delta, ddelta, ok, worst = self.delta(theta, grad=True)
        xi = self.M @ delta
        g = self.Z.T @ xi / self.n
        q = float(self.n * g @ self.W @ g)
        G = self.Z.T @ (self.M @ ddelta) / self.n
        dq = 2 * self.n * (G.T @ self.W @ g)
        self.last = dict(theta=theta.copy(), delta=delta, xi=xi, beta=self.beta_of @ delta, q=q, G=G, ok=ok)
        print(f"eval {self.evals:4d}  q={q:.6e}  |grad|={np.abs(dq).max():.2e}  inversions_ok={ok} "
              f"worst_resid={worst:.1e}  {time.time() - t0:.1f}s\n"
              f"      theta={np.array2string(theta, precision=4, max_line_width=250)}", flush=True)
        if not ok:
            return 1e10, np.zeros_like(theta)
        return q, dq

    def standard_errors(self, theta, cluster):
        self.objective(theta)
        G, xi = self.last["G"], self.last["xi"]
        codes = pd.factorize(cluster)[0]
        zxi = self.Z * xi[:, None]
        S = np.zeros((N_ZD, N_ZD))
        for c in np.unique(codes):
            v = zxi[codes == c].sum(0)
            S += np.outer(v, v)
        S /= self.n
        bread = np.linalg.inv(G.T @ self.W @ G)
        V = bread @ G.T @ self.W @ S @ self.W @ G @ bread / self.n
        return np.sqrt(np.diag(V)), S


def start_values():
    th = np.zeros(len(NAMES))
    st = {"sigma_const": 3.0, "sigma_hpwt": 2.0, "sigma_space": 1.0, "sigma_air": 0.5, "sigma_mpd": 1.0, "sigma_fwd": 1.0,
          "pi_mi_logfs_fv": 0.3, "pi_sw_logfs_fv": 0.3, "pi_su_logfs_fv": 0.3, "pi_pv_logfs_fv": 0.3,
          "phi_low_inc": 0.0, "phi_mid_inc": 0.0, "phi_high_inc": 0.0, "sigma_p": 0.7}
    for n, v in st.items():
        th[NAMES.index(n)] = v
    if os.environ.get("PCM_START"):
        prev = pd.read_csv(os.environ["PCM_START"], index_col=0)["estimate"]
        for n, v in prev.items():
            if n in NAMES:
                th[NAMES.index(n)] = v
    return th


def parameter_scales():
    """delta is in $1000; a taste spread of a few $1000 on a characteristic is the natural unit."""
    sc = {"sigma_const": 3.0, "sigma_hpwt": 2.0, "sigma_space": 1.0, "sigma_air": 0.5, "sigma_mpd": 1.0, "sigma_fwd": 1.0,
          "pi_mi_logfs_fv": 0.3, "pi_sw_logfs_fv": 0.3, "pi_su_logfs_fv": 0.3, "pi_pv_logfs_fv": 0.3,
          "phi_low_inc": 1.0, "phi_mid_inc": 1.0, "phi_high_inc": 1.0, "sigma_p": 0.5}
    return np.array([sc[n] for n in NAMES])


def main():
    products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    nu_per_agent = int(os.environ.get("PCM_NU_PER_AGENT", "4"))
    tag = os.environ.get("PCM_TAG", "")
    prob = PCMProblem(products, agents, nu_per_agent=nu_per_agent)
    print(f"{len(prob.markets)} markets, {len(products)} products, consumers per market {1000 * nu_per_agent}", flush=True)

    scale = parameter_scales()
    is_log = np.array([n.startswith("sigma") for n in NAMES])

    def to_theta(u):
        return np.where(is_log, scale * np.exp(np.where(is_log, u, 0.0)), scale * u)

    def scaled(u):
        th = to_theta(u)
        q, dq = prob.objective(th)
        return q, dq * np.where(is_log, th, scale)

    th0 = start_values()
    u0 = np.where(is_log, np.log(np.maximum(th0, 1e-12) / scale), th0 / scale)
    # sigma_p in [0.05, 2]: a log price-sensitivity sd above 2 (95% range of 3000x) makes the
    # inversion fail in every market; the other parameters are free
    sp_lo, sp_hi = np.log(0.05 / scale[-1]), np.log(2.0 / scale[-1])
    radius = float(os.environ.get("PCM_TRUST", "0.3"))

    def trust_bounds(center):
        """Box of +-radius (in scaled units) around the last accepted point: L-BFGS-B has no
        step limit, and its long trial steps toward small sigmas landed where the inversion
        stalls (residual 1e-4 to 1e-3 after 20,000 evaluations) at 250-450 s each."""
        b = [(c - radius, c + radius) for c in center]
        k = NAMES.index("sigma_p")
        b[k] = (max(b[k][0], sp_lo), min(b[k][1], sp_hi))
        return b

    t0 = time.time()
    best = None
    for step in range(2):                       # two-step GMM
        for restart in range(40):
            x_start = u0 if best is None else best.x
            res = minimize(scaled, x_start, jac=True, method="L-BFGS-B", bounds=trust_bounds(x_start),
                           options={"gtol": 1e-6, "ftol": 1e-12, "maxiter": 300})
            at_wall = np.any(np.abs(res.x - x_start) > radius - 1e-9)
            print(f"step {step} restart {restart}: q={res.fun:.6e} at_trust_wall={at_wall} {res.message}", flush=True)
            improved = best is None or res.fun < best.fun - 1e-6 * abs(best.fun)
            if res.fun < (best.fun if best is not None else np.inf):
                best = res
            if not improved and not at_wall:
                break
        if step == 0:
            _, S = prob.standard_errors(to_theta(best.x), products.clustering_ids.to_numpy())
            prob.set_weighting(np.linalg.inv(S))
            best.fun = prob.objective(to_theta(best.x))[0]
    theta = to_theta(best.x)
    print(best.message, f"{time.time() - t0:.0f}s")
    se, _ = prob.standard_errors(theta, products.clustering_ids.to_numpy())
    table = pd.DataFrame({"estimate": theta, "clustered_se": se}, index=NAMES)
    print(table)
    beta = pd.Series(prob.last["beta"], index=X1_COLS)
    print("beta (delta on x1, $1000):\n", beta.round(4))

    np.savez(OUT / f"pcm_estimates{tag}.npz", theta=theta, se=se, delta=prob.last["delta"], xi=prob.last["xi"], beta=beta.to_numpy())
    table.to_csv(OUT / f"pcm_estimates{tag}.csv")
    beta.to_csv(OUT / f"pcm_beta{tag}.csv")
    with open(OUT / f"pcm_estimates{tag}.json", "w") as fh:
        json.dump({"objective": float(best.fun), "message": str(best.message), "evaluations": prob.evals,
                   "nu_per_agent": nu_per_agent}, fh, indent=2)


if __name__ == "__main__":
    main()

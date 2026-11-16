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
# PCM_BODY_SIGMA=1 adds a random coefficient on each body type. Petrin's BLP has none: the logit
# error gives a minivan its own following. Without it the PCM must fit the Caravan's 0.5% share
# by climbing the vertical ladder past a dense cluster of same-price sedans, which fails.
BODY_SIGMA = os.environ.get("PCM_BODY_SIGMA", "0") == "1"
if BODY_SIGMA:
    SIGMA_ON = SIGMA_ON + PI_ON
TASTE = [(f"sigma_{c}", X_COLS.index(c), f"nu{k}") for k, c in enumerate(SIGMA_ON)] + \
        [(f"pi_{c}_logfs_fv", X_COLS.index(c), "logfs_fv") for c in PI_ON]
SCALE = [("phi_low_inc", "low_inc"), ("phi_mid_inc", "mid_inc"), ("phi_high_inc", "high_inc")]
NAMES = [t[0] for t in TASTE] + [s[0] for s in SCALE] + ["sigma_p"]
N_TASTE, N_SCALE = len(TASTE), len(SCALE)
N_ZD = 22
TOL = 1e-9           # max |s_model - s_data|; shares are 1e-5 to 5e-3. Newton on 4000-consumer shares stalls near 2e-10


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
    def __init__(self, products, agents, nu_per_agent=4, workers=13, path_seconds=90.0):
        self.products = products
        self.path_seconds = path_seconds
        self.markets = build_markets(products, agents, nu_per_agent)
        self.n = len(products)
        self.X1 = np.column_stack([np.ones(self.n)] + [products[c].to_numpy(float) for c in X1_COLS[1:]])
        self.Z = products[[f"demand_instruments{k}" for k in range(N_ZD)]].to_numpy(float)
        self.set_weighting(np.linalg.inv(self.Z.T @ self.Z / self.n))
        self.cue = os.environ.get("PCM_CUE", "0") == "1"
        self.cluster = pd.factorize(products.clustering_ids.to_numpy())[0]
        self.n_clusters = int(self.cluster.max()) + 1
        self.pool = ThreadPoolExecutor(workers)
        self.evals = 0
        self.last = None
        # Per-market OpenMP thread budget: each Python thread gets a slice of the available CPUs.
        # Inversion work per market scales roughly as J * I (products x consumers), so we allocate
        # threads proportionally to empirical timing (which varies 10x across markets at same J),
        # falling back to product count. This equalises wall-clock time across parallel market calls,
        # cutting the bottleneck from ~430s (1 OMP thread/market at nu16) toward ~30s.
        try:
            import ctypes
            self._gomp = ctypes.CDLL("libgomp.so.1")
            omp_total = int(os.environ.get("PCM_OMP_TOTAL", str(os.cpu_count() or 13)))
            # Allocation weight: J^2 per market (Newton work per consumer scales as J^2 for the
            # Jacobian linear solve). This is nu-agnostic; empirical timing from one nu is a
            # poor proxy for another because the per-market rank order changes substantially.
            sizes = np.array([len(m.shares) ** 2 for m in self.markets], dtype=float)
            raw = omp_total * sizes / sizes.sum()
            # floor to int >= 1, then give leftover to the slowest market
            alloc = np.maximum(raw.astype(int), 1)
            alloc[np.argmax(sizes)] += omp_total - int(alloc.sum())
            self._omp_threads = {m.market_id: int(t) for m, t in zip(self.markets, alloc)}
            print(f"OMP thread budget (total={omp_total}): "
                  + ", ".join(f"{mid}={t}" for mid, t in sorted(self._omp_threads.items())),
                  flush=True)
        except OSError:
            self._gomp = None
            self._omp_threads = {}

    def set_weighting(self, W):
        self.W = W
        ZW = self.Z @ W @ self.Z.T / self.n
        XZWZ = self.X1.T @ ZW
        self.beta_of = np.linalg.solve(XZWZ @ self.X1, XZWZ)         # beta = beta_of @ delta
        self.M = np.eye(self.n) - self.X1 @ self.beta_of              # xi = M delta

    def _solve_market(self, m: MarketData, theta, grad):
        th, ph, sp = split(theta)
        t_start, stages = time.time(), []
        # Set per-market OMP thread count so the slowest market gets proportionally more cores.
        if self._gomp is not None and m.market_id in self._omp_threads:
            self._gomp.omp_set_num_threads(self._omp_threads[m.market_id])
        warm = m.delta if m.delta is not None else np.empty(0)
        # a trial point gets a short first try; most of the run's time was going into trial
        # points the optimizer then rejected
        r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, warm, TOL, 2500 if m.delta is not None else 20000)
        stages.append(f"warm {time.time() - t_start:.0f}s")
        if not r["converged"] and m.delta is not None and time.time() - t_start < self.path_seconds:
            t0 = time.time()
            d, tau, step = m.delta, 0.0, 0.5
            while tau < 1.0 and step > 1e-2 and time.time() - t_start < self.path_seconds:
                t = min(1.0, tau + step)
                rr = pd_cpp.invert(m.shares, m.cpp, *split(m.theta_ok + t * (theta - m.theta_ok)), d, TOL, 1500)
                if rr["converged"]:
                    d, tau, step = rr["delta"], t, min(0.5, 2 * step)
                else:
                    step /= 2
            if tau >= 1.0:
                r = rr
            stages.append(f"theta path to tau={tau:.3f} {time.time() - t0:.0f}s")
        if not r["converged"]:
            # cold start as last resort, also at trial points: the sigma-ray profile inverted every
            # grid point cold, while warm starts across a 35% move in all sigmas did not
            t0 = time.time()
            r = pd_cpp.invert(m.shares, m.cpp, th, ph, sp, np.empty(0), TOL, 40000)
            stages.append(f"cold {time.time() - t0:.0f}s")
        if not r["converged"]:
            print(f"  {m.market_id}: inversion failed (resid {r['max_abs_residual']:.1e}): {', '.join(stages)}", flush=True)
        if time.time() - t_start > 60:
            print(f"  {m.market_id}: slow inversion {time.time() - t_start:.0f}s: {', '.join(stages)}", flush=True)
        out = {"delta": r["delta"], "converged": r["converged"], "resid": r["max_abs_residual"]}
        if grad:
            d = pd_cpp.shares(r["delta"], m.cpp, th, ph, sp, True)
            # Regularise: products with zero share have an all-zero row in d_delta, making
            # the system singular.  Add a small diagonal floor (1/n_consumers) to those rows
            # so the solve is defined.  The gradient contribution from a zero-share product
            # is negligible and the floor is well below the residual tolerance.
            dD = d["d_delta"].copy()
            zero_rows = np.flatnonzero(np.abs(dD).sum(1) == 0)
            if len(zero_rows):
                n_consumers = m.cpp.shape[0]        # rows = consumers
                floor_val = 1.0 / max(n_consumers, 1)
                dD[zero_rows, zero_rows] = floor_val
                print(f"  {m.market_id}: {len(zero_rows)} zero-share rows regularised (floor={floor_val:.2e})", flush=True)
            try:
                out["ddelta"] = -np.linalg.solve(dD, d["d_theta"])
            except np.linalg.LinAlgError:
                zero = np.flatnonzero(np.abs(dD).sum(1) == 0)
                print(f"  {m.market_id}: singular ds/ddelta after regularisation at rows {zero.tolist()}", flush=True)
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
        G = self.Z.T @ (self.M @ ddelta) / self.n
        if self.cue:
            # Continuously updated GMM: q = n g' S(theta)^-1 g with S the clustered covariance of
            # z_j xi_j at the same theta. The PCM's utility scale is free (median alpha = 1 pins it
            # to a non-buyer), so a fixed-W objective is quadratic in a nuisance scale and is
            # minimised by sigma_p -> inf, sigma_x -> 0. CUE is invariant to xi -> kappa xi.
            # beta is still concentrated out with the fixed 2SLS weighting (self.M), which keeps
            # the gradient simple; that is a valid, if not fully efficient, GMM estimator.
            dxi = self.M @ ddelta                                    # n x P
            V = self._cluster_sum(self.Z * xi[:, None])              # C x K
            dV = self._cluster_sum3(self.Z, dxi)                     # C x K x P
            S = V.T @ V / self.n
            a = np.linalg.solve(S, g)
            q = float(self.n * g @ a)
            Va = V @ a                                               # C
            dVa = np.einsum("ckp,k->cp", dV, a)                      # C x P
            dq = 2 * self.n * (G.T @ a) - 2 * (dVa.T @ Va)
            self.W = np.linalg.inv(S)
        else:
            q = float(self.n * g @ self.W @ g)
            dq = 2 * self.n * (G.T @ self.W @ g)
        self.last = dict(theta=theta.copy(), delta=delta, xi=xi, beta=self.beta_of @ delta, q=q, G=G, ok=ok)
        print(f"eval {self.evals:4d}  q={q:.6e}  |grad|={np.abs(dq).max():.2e}  inversions_ok={ok} "
              f"worst_resid={worst:.1e}  sd_xi={xi.std():.4f}  {time.time() - t0:.1f}s\n"
              f"      theta={np.array2string(theta, precision=4, max_line_width=250)}", flush=True)
        if not ok:
            return 1e10, np.zeros_like(theta)
        return q, dq

    def _cluster_sum(self, a):
        out = np.zeros((self.n_clusters, a.shape[1]))
        np.add.at(out, self.cluster, a)
        return out

    def _cluster_sum3(self, Z, dxi):
        out = np.zeros((self.n_clusters, Z.shape[1], dxi.shape[1]))
        np.add.at(out, self.cluster, Z[:, :, None] * dxi[:, None, :])
        return out

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
    st = {"sigma_const": 3.0, "sigma_hpwt": 2.0, "sigma_space": 1.0, "sigma_air": 0.5, "sigma_mpd": 1.0, "sigma_fwd": 1.0, "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0,
          "pi_mi_logfs_fv": 0.3, "pi_sw_logfs_fv": 0.3, "pi_su_logfs_fv": 0.3, "pi_pv_logfs_fv": 0.3,
          "phi_low_inc": 0.0, "phi_mid_inc": 0.0, "phi_high_inc": 0.0, "sigma_p": 0.7}
    for n, v in st.items():
        if n in NAMES:                        # skip names absent in the current spec (e.g. body sigmas)
            th[NAMES.index(n)] = v
    if os.environ.get("PCM_START"):
        prev = pd.read_csv(os.environ["PCM_START"], index_col=0)["estimate"]
        for n, v in prev.items():
            if n in NAMES:
                th[NAMES.index(n)] = v
    return th


def parameter_scales():
    """delta is in $1000; a taste spread of a few $1000 on a characteristic is the natural unit."""
    sc = {"sigma_const": 3.0, "sigma_hpwt": 2.0, "sigma_space": 1.0, "sigma_air": 0.5, "sigma_mpd": 1.0, "sigma_fwd": 1.0, "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0,
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
    radius = float(os.environ.get("PCM_TRUST", "0.30"))

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
                           options={"gtol": 1e-5, "ftol": 1e-15, "maxiter": 500})
            at_wall = np.any(np.abs(res.x - x_start) > radius - 1e-9)
            print(f"step {step} restart {restart}: q={res.fun:.6e} at_trust_wall={at_wall} {res.message}", flush=True)
            improved = best is None or res.fun < best.fun - 1e-6 * abs(best.fun)
            if res.fun < (best.fun if best is not None else np.inf):
                best = res
            if not improved and not at_wall:
                break
        if step == 0 and not prob.cue:
            _, S = prob.standard_errors(to_theta(best.x), products.clustering_ids.to_numpy())
            prob.set_weighting(np.linalg.inv(S))
            best.fun = prob.objective(to_theta(best.x))[0]
        elif step == 0:
            break                                   # CUE is one step
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
                   "nu_per_agent": nu_per_agent, "cue": prob.cue, "body_sigma": BODY_SIGMA}, fh, indent=2)


if __name__ == "__main__":
    main()

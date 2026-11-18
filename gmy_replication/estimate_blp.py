"""BLP benchmark on Grieco, Murry & Yurukoglu (2024, QJE): their main specification (spec 6 of codes/estimation.m).

    u_ijt = delta_jt + mu_ijt + eps_ijt,  eps logit,  u_i0t = 0
    mu_ij = p_j (th_inc inc_i + th_inc2 inc2_i + th_age age_i) + const inc_i th_ci
            + van (fs2, fs34, fs5) + log footprint (fs2, fs34, fs5) + truck rural            (11 demographic terms)
            + sum_c sigma_c nu_ic x_jc on van, suv, truck, log footprint, log hp, log mpg, lux, sport, electric,
              EuroBrand, USBrand                                                              (11 random coefficients)
    (log footprint, log hp, log mpg z-scored inside mu; nu_i standard normal Latin hypercube draws per year)
    delta_jt = alpha p_jt + x_jt beta + year FE + make2 FE + xi_jt,   alpha_i = alpha + p-interactions

Estimation follows the authors' three steps (Appendix A, codes/estimation.m, mom_fnc.m, linearDemand.do):
  step 1  theta (22) by nonlinear least squares on [W^1/2 (micro data - model); 30 (second-choice corr data - model)],
          33 CEX/MRI micro moments with W = diag(1 / variance) and 41 first/second-choice correlations (11
          characteristics x 1991, 1999, 2005, 2015, less the 3 electric ones that are undefined without EVs), delta(theta)
          by the BLP contraction, no aggregate IV moment (delta is a free product effect in this step);
  step 2  alpha and beta by 2SLS of delta(theta_hat) on price (instrument lag_pl_con) and the 13 characteristics with
          year and make2 fixed effects, clustered by make2;
  post    own elasticities by income quintile (Table 5), Nash margins with ownership by parent and the Lerner index.

pyblp cannot express step 1: GMY's micro moments average each household's expected x given a purchase with household
weights (ExjCondH.m), and their second-choice probabilities renormalise over inside goods per household, neither of
which a pyblp MicroPart can weight; pyblp's own second-choice parts would also need (I, J, J) arrays per part. So step
1 runs in numpy here (nfp_harness.gmy_data, nfp_harness.second_choice), and step 2 and the post-estimation run in
pyblp (vendored 1.3.0) at theta_hat: a pyblp.Problem with GMY's X1 / X2 / demographics, the same agents and nodes,
absorbed year and make2 effects and Optimization("return"), so pyblp's one-step GMM is the authors' 2SLS and its
elasticities and Nash costs are those of the fitted model. pyblp's delta is checked against the numpy contraction.

Stages: ``published`` evaluates everything at Table 4's theta (no step-1 search); ``fit`` runs step 1 from Table 4's
theta. ``--years`` fits a subset of markets (the micro moments then pool over the years kept; the second-choice
moments need 1991, 1999, 2005, 2015 among them).

Run: python gmy_replication/estimate_blp.py {published,fit} [--households 1000]
     [--threads 32] [--max-nfev 200] [--years 1980 1991 ...]
Outputs (untracked): gmy_replication/results/blp_<stage>_h<H>[_years].{json,npz}, blp_<stage>_h<H>.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

THREADS_DEFAULT = 32
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # parallelism is across markets, one BLAS thread each
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import norm, qmc  # noqa: E402

from nfp_harness import gmy_data  # noqa: E402
from nfp_harness.second_choice import first_second_correlations, logit_second_choice_matrix  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
SEED = 8673320
log = logging.getLogger("gmy_blp")

# X2 of the utility deviations: GMY xTerms used by spec 6 (Car and AsianBrand carry no coefficient there)
X2 = ("prices", "const", "van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport",
      "electric", "regionEU", "regionUS")
DEMOG = ("inc", "inc2", "age", "fs2", "fs34", "fs5", "rural")
# NEWPATTERN of loadWardsSalesData.m: (X2 column, demographic)
PATTERN = (("prices", "inc"), ("prices", "inc2"), ("prices", "age"), ("van", "fs2"), ("van", "fs34"), ("van", "fs5"),
           ("z_log_footprint", "fs2"), ("z_log_footprint", "fs34"), ("z_log_footprint", "fs5"), ("truck", "rural"),
           ("const", "inc"))
UNOBS = ("van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport", "electric", "regionEU",
         "regionUS")
NAMES = tuple(f"{c} x {d}" for c, d in PATTERN) + tuple(f"sigma {c}" for c in UNOBS)
# Table 4 of the working paper (NBER w29013), in PATTERN then UNOBS order
TABLE4_THETA = np.array([0.094, -0.464, 2.068, 1.668, 3.563, 5.653, 0.483, 0.463, 0.645, 3.009, 0.362,
                         5.348, 3.646, 6.309, 1.884, 1.249, 1.636, 2.627, 2.62, 3.835, 1.923, 2.14])
TABLE4_ALPHA = -3.200
LOWER = np.r_[-25, -25, -25, np.full(8, -np.inf), np.zeros(11)]   # estimation.m bounds
UPPER = np.r_[25, 25, 25, np.full(8, np.inf), np.full(11, 15.0)]
SC_WEIGHT = 30.0   # mom_fnc.m: second-choice residuals x 30
SC_CHARS_PRODUCT = ("van", "suv", "truck", "log_footprint", "log_hp", "log_mpg", "lux", "sport", "electric",
                    "regionEU", "regionUS")   # raw columns for the correlations (xTerms, not z-scored)
# Headline numbers of the paper (working paper text; the published QJE version could not be fetched here)
TABLE5 = {1980: (-5.96, -5.78, -5.49, -5.13, -4.30), 2000: (-8.24, -7.83, -7.40, -6.88, -6.21),
          2018: (-9.37, -8.56, -7.69, -6.90, -6.46)}
LERNER = {"introduction": {1980: 0.311, 2018: 0.196}, "section 5.3": {1980: 0.248, 2018: 0.1373}}
TABLE6_DATA = dict(zip(UNOBS, (0.71, 0.64, 0.84, 0.71, 0.60, 0.65, 0.48, 0.28, 0.37, 0.34, 0.48)))
TABLE6_MODEL = dict(zip(UNOBS, (0.71, 0.64, 0.80, 0.69, 0.59, 0.65, 0.49, 0.28, 0.19, 0.34, 0.47)))
TABLE8_MODEL = {  # Table 8 "Model" values, keyed by the code's survey label (Table 8 prints CEX and MRI swapped)
    "Income5MINUSIncome1 Price mri": 0.40, "Income4MINUSIncome1 Price mri": 0.27, "Income3MINUSIncome1 Price mri": 0.15,
    "Income2MINUSIncome1 Price mri": 0.08, "Income5MINUSIncome1 Price cex": 0.38, "Income4MINUSIncome1 Price cex": 0.23,
    "Income3MINUSIncome1 Price cex": 0.13, "Income2MINUSIncome1 Price cex": 0.07,
    "Age60+MINUSAge20-29 Price mri": 0.34, "Age50-59MINUSAge20-29 Price mri": 0.29,
    "Age40-49MINUSAge20-29 Price mri": 0.24, "Age30-39MINUSAge20-29 Price mri": 0.14,
    "Age60+MINUSAge20-29 Price cex": 0.37, "Age50-59MINUSAge20-29 Price cex": 0.34,
    "Age40-49MINUSAge20-29 Price cex": 0.29, "Age30-39MINUSAge20-29 Price cex": 0.15,
    "UrbanMINUSRural Truck cex": -0.10, "Income2OVERIncome1 PurchaseProb mri": 1.66,
    "Income3OVERIncome1 PurchaseProb mri": 2.71, "Income4OVERIncome1 PurchaseProb mri": 5.08,
    "Income5OVERIncome1 PurchaseProb mri": 9.24}


class GMYLogit:
    """Mixed logit of GMY spec 6 on one GMYData: contraction, micro and second-choice moments, in numpy."""

    def __init__(self, data: gmy_data.GMYData, threads: int, seed: int = SEED, tol: float = 1e-13) -> None:
        self.data, self.tol = data, tol
        p, a = data.products, data.agents
        self.years = np.unique(p.market_ids.to_numpy())
        self.rows = [np.flatnonzero(p.market_ids.to_numpy() == y) for y in self.years]
        arows = [np.flatnonzero(a.market_ids.to_numpy() == y) for y in self.years]
        self.X = [p.loc[r, list(X2)].to_numpy(float) for r in self.rows]
        self.D = [a.loc[r, list(DEMOG)].to_numpy(float) for r in arows]
        self.w = [a.weights.to_numpy(float)[r] for r in arows]
        self.H = [r.size for r in arows]
        # estimation.m: lhsnorm draws per market, one column per random coefficient
        self.nu = [norm.ppf(qmc.LatinHypercube(d=len(UNOBS), seed=seed + t).random(h))
                   for t, h in enumerate(self.H)]
        self.log_s = [np.log(p.shares.to_numpy(float)[r]) for r in self.rows]
        s0 = [np.exp(ls).sum() for ls in self.log_s]
        self.delta0 = [ls - np.log(1 - s) for ls, s in zip(self.log_s, s0)]   # logit start, as delta_start
        self.delta = [d.copy() for d in self.delta0]
        self.pattern = [(X2.index(c), DEMOG.index(d)) for c, d in PATTERN]
        self.unobs = [X2.index(c) for c in UNOBS]
        self.micro = gmy_data.MicroMomentEvaluator(p, a, data.micro)
        self.sc_years = [y for y in gmy_data.SECOND_CHOICE_YEARS if y in set(self.years.tolist())]
        self.sc_x, self.sc_s, self.sc_data = {}, {}, {}
        for y in self.sc_years:
            r = self.rows[list(self.years).index(y)]
            self.sc_x[y] = p.loc[r, list(SC_CHARS_PRODUCT)].to_numpy(float)
            self.sc_s[y] = p.shares.to_numpy(float)[r]
            self.sc_data[y] = first_second_correlations(self.sc_s[y], data.second_choices[y]["prob"], self.sc_x[y])
        self.sc_keep = {y: np.isfinite(self.sc_data[y]) for y in self.sc_years}
        self.pool = ThreadPoolExecutor(max_workers=max(1, min(threads, len(self.years))))
        self.evaluations = 0

    def coefficients(self, theta: np.ndarray, t: int) -> np.ndarray:
        """(K2, H_t) coefficient on each X2 column for each household of market t."""
        C = np.zeros((len(X2), self.H[t]))
        for k, (c, d) in enumerate(self.pattern):
            C[c] += theta[k] * self.D[t][:, d]
        for m, c in enumerate(self.unobs):
            C[c] += theta[len(PATTERN) + m] * self.nu[t][:, m]
        return C

    def probabilities_at(self, delta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        u = delta[:, None] + mu
        m = np.maximum(u.max(axis=0), 0.0)
        e = np.exp(u - m)
        return e / (np.exp(-m) + e.sum(axis=0))

    def solve_market(self, theta: np.ndarray, t: int, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """BLP contraction with SQUAREM; returns (delta, P (J, H), iterations)."""
        mu = self.X[t] @ self.coefficients(theta, t)
        w, ls = self.w[t], self.log_s[t]

        def g(d: np.ndarray) -> np.ndarray:
            return d + ls - np.log(self.probabilities_at(d, mu) @ w)

        x = delta.copy()
        for it in range(1, 5001):
            x1 = g(x)
            r = x1 - x
            if np.max(np.abs(r)) < self.tol:
                x = x1
                break
            x2 = g(x1)
            v = x2 - 2 * x1 + x
            nv = float(v @ v)
            step = -np.sqrt(float(r @ r) / nv) if nv > 0 else -1.0
            step = min(step, -1.0)
            xn = x - 2 * step * r + step * step * v
            x = g(xn) if np.all(np.isfinite(xn)) else x2
        else:
            raise RuntimeError(f"contraction did not converge in market {self.years[t]}")
        return x, self.probabilities_at(x, mu), it

    def solve(self, theta: np.ndarray, warm: bool = True) -> list[np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        start = self.delta if warm else self.delta0
        res = list(self.pool.map(lambda t: self.solve_market(theta, t, start[t]), range(len(self.years))))
        self.evaluations += 1
        return [r[0] for r in res], [r[1] for r in res]

    def moments(self, theta: np.ndarray, commit: bool = False) -> dict[str, np.ndarray]:
        delta, P = self.solve(theta)
        if commit:
            self.delta = [d.copy() for d in delta]
        g_micro = self.micro(P)
        sc_model = {}
        for y in self.sc_years:
            t = list(self.years).index(y)
            C = logit_second_choice_matrix(P[t], self.w[t])
            sc_model[y] = first_second_correlations(self.sc_s[y], C, self.sc_x[y])
        return {"delta": np.concatenate(delta), "P": P, "micro": g_micro, "sc": sc_model}

    def residuals_of(self, m: dict[str, np.ndarray]) -> np.ndarray:
        mic = self.data.micro
        r_micro = np.sqrt(1.0 / mic.variances) * (mic.values - m["micro"])
        r_sc = [SC_WEIGHT * (self.sc_data[y] - m["sc"][y])[self.sc_keep[y]] for y in self.sc_years]
        return np.concatenate([r_micro] + r_sc)

    def residuals(self, theta: np.ndarray, commit: bool = False) -> np.ndarray:
        return self.residuals_of(self.moments(theta, commit))

    def commit_delta(self, delta: np.ndarray) -> None:
        self.delta = [delta[r].copy() for r in self.rows]


def fit_step1(model: GMYLogit, theta0: np.ndarray, max_nfev: int) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    best = {"q": np.inf}

    def fun(th: np.ndarray) -> np.ndarray:
        m = model.moments(th)
        r = model.residuals_of(m)
        q = float(r @ r)
        if q < best["q"]:   # warm-start later contractions from the best point so far
            best["q"] = q
            model.commit_delta(m["delta"])
        log.info("eval %d  q %.6f  best %.6f  %.0fs", model.evaluations, q, best["q"], time.time() - t0)
        return r

    # lsqnonlin with forward differences of step 1e-6 (estimation.m options)
    res = least_squares(fun, theta0, bounds=(LOWER, UPPER), method="trf", diff_step=1e-6, jac="2-point",
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-10, gtol=1e-8, verbose=0)
    return res.x, {"status": int(res.status), "message": res.message, "nfev": int(res.nfev),
                   "q": float(2 * res.cost), "optimality": float(res.optimality), "seconds": time.time() - t0}


# ---------------------------------------------------------------- step 2 and post-estimation in pyblp

def pyblp_results(data: gmy_data.GMYData, model: GMYLogit, theta: np.ndarray):
    """pyblp ProblemResults at theta: absorbed year and make2 effects, X1 = [price, STAGE2_X], ZD = [STAGE2_X,
    lag_pl_con] (just identified, so one-step GMM = 2SLS), X2 / demographics / nodes as GMYLogit."""
    import pyblp
    pyblp.options.verbose = False
    p = data.products.copy()
    p["clustering_ids"] = p.make2_do
    # pyblp puts the intercept first: X2 order is (const, prices, rest of X2)
    order = ["const", "prices"] + [c for c in X2 if c not in ("const", "prices")]
    a_frames = []
    for t, y in enumerate(model.years):
        f = pd.DataFrame(model.D[t], columns=list(DEMOG))
        f["market_ids"] = y
        f["weights"] = model.w[t]
        nodes = np.zeros((model.H[t], len(X2)))
        nodes[:, [order.index(c) for c in UNOBS]] = model.nu[t]
        for k in range(len(X2)):
            f[f"nodes{k}"] = nodes[:, k]
        a_frames.append(f)
    agents = pd.concat(a_frames, ignore_index=True)
    x1 = "0 + prices + " + " + ".join(gmy_data.STAGE2_X)
    x2 = "1 + " + " + ".join(order[1:])
    form = (pyblp.Formulation(x1, absorb="C(market_ids) + C(make2_do)"), pyblp.Formulation(x2))
    problem = pyblp.Problem(form, p, pyblp.Formulation("0 + " + " + ".join(DEMOG)), agents)
    sigma = np.zeros((len(X2), len(X2)))
    for m, c in enumerate(UNOBS):
        sigma[order.index(c), order.index(c)] = theta[len(PATTERN) + m]
    pi = np.zeros((len(X2), len(DEMOG)))
    for k, (c, d) in enumerate(PATTERN):
        pi[order.index(c), DEMOG.index(d)] = theta[k]
    log.info("pyblp problem:\n%s", problem)
    res = problem.solve(sigma=sigma, pi=pi, optimization=pyblp.Optimization("return"), method="1s",
                        iteration=pyblp.Iteration("squarem", {"atol": 1e-13, "max_evaluations": 10000}),
                        se_type="clustered")
    return problem, res


def post_estimation(data: gmy_data.GMYData, model: GMYLogit, theta: np.ndarray, delta: np.ndarray) -> dict:
    problem, res = pyblp_results(data, model, theta)
    p = data.products
    beta = np.asarray(res.beta).ravel()
    alpha = float(beta[0])
    se = np.asarray(res.beta_se).ravel()
    out: dict = {"alpha": alpha, "alpha_se_clustered": float(se[0]),
                 "beta": dict(zip(gmy_data.STAGE2_X, beta[1:].tolist())),
                 "max_abs_delta_numpy_vs_pyblp": float(np.max(np.abs(np.asarray(res.delta).ravel() - delta)))}
    # alpha_i = alpha + th_inc inc + th_inc2 inc2 + th_age age (getAlphas.m)
    pr = [k for k, (c, _) in enumerate(model.pattern) if c == 0]
    lerner_median, lerner_q, own_mean, table5 = {}, {}, {}, {}
    for t, y in enumerate(model.years):
        rows = model.rows[t]
        s = p.shares.to_numpy(float)[rows]
        prices = p.prices.to_numpy(float)[rows]
        a_i = alpha + sum(theta[k] * model.D[t][:, model.pattern[k][1]] for k in pr)
        P = np.asarray(res.compute_probabilities(market_id=y))       # (J, H), pyblp agent order = ours
        own = (prices / s) * ((P * (1 - P)) @ (model.w[t] * a_i))
        own_mean[int(y)] = float(own.mean())
        costs = np.asarray(res.compute_costs(market_id=y)).ravel()
        lerner = (prices - costs) / prices
        lerner_median[int(y)] = float(np.median(lerner))
        lerner_q[int(y)] = [float(np.percentile(lerner, q)) for q in (10, 25, 50, 75, 90)]
        if int(y) in TABLE5:
            quint = data.agents.inc_quintile.to_numpy()[data.agents.market_ids.to_numpy() == y]
            row, alt = [], []
            for q in range(1, 6):
                sel = quint == q
                wq = model.w[t] * sel
                sq = P @ wq
                eq = (prices / sq) * ((P * (1 - P)) @ (wq * a_i))       # quintile demand's own elasticity
                row.append(float(eq.mean()))
                ind = a_i[None, :] * prices[:, None] * (1 - P)          # individual own elasticity
                alt.append(float(np.sum(P * ind * wq) / np.sum(P * wq)))
            table5[int(y)] = {"mean_over_products": row, "purchase_weighted_individual": alt}
    out.update({"own_elasticity_mean_by_year": own_mean, "lerner_median_by_year": lerner_median,
                "lerner_p10_25_50_75_90_by_year": lerner_q, "table5": table5})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("published", "fit"))
    ap.add_argument("--households", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.threads > 32:
        raise SystemExit("at most 32 threads")
    OUT.mkdir(exist_ok=True)
    tag = f"blp_{args.stage}_h{args.households}" + (f"_{len(args.years)}y" if args.years else "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / f"{tag}.log"), logging.StreamHandler()])
    data = gmy_data.load(n_households=args.households, seed=SEED, years=args.years)
    model = GMYLogit(data, args.threads)
    log.info("%d products, %d markets, %d households per market", len(data.products), len(model.years),
             args.households)
    theta = TABLE4_THETA.copy()
    t0 = time.time()
    r0 = model.residuals(theta, commit=True)
    log.info("q at Table 4 theta %.4f (%.1fs)", float(r0 @ r0), time.time() - t0)
    fit = None
    if args.stage == "fit":
        theta, fit = fit_step1(model, theta, args.max_nfev)
        log.info("step 1: %s", fit)
    m = model.moments(theta, commit=True)
    r = model.residuals(theta)
    summary = {"stage": args.stage, "households": args.households, "years": [int(y) for y in model.years],
               "theta": dict(zip(NAMES, theta.tolist())), "theta_table4": dict(zip(NAMES, TABLE4_THETA.tolist())),
               "q": float(r @ r), "q_table4": float(r0 @ r0), "step1": fit,
               "micro": [{"name": n, "data": float(v), "model": float(g), "table8_model": TABLE8_MODEL.get(n)}
                         for n, v, g in zip(data.micro.table.name, data.micro.values, m["micro"])],
               "second_choice": {int(y): {"data": model.sc_data[y].tolist(), "model": m["sc"][y].tolist()}
                                 for y in model.sc_years}}
    if model.sc_years:
        mod = np.nanmean([m["sc"][y] for y in model.sc_years], axis=0)
        dat = np.nanmean([model.sc_data[y] for y in model.sc_years], axis=0)
        summary["table6"] = {c: {"data": float(d), "model": float(v), "paper_data": TABLE6_DATA[c],
                                 "paper_model": TABLE6_MODEL[c]} for c, d, v in zip(UNOBS, dat, mod)}
    summary.update(post_estimation(data, model, theta, m["delta"]))
    summary["paper"] = {"alpha": TABLE4_ALPHA, "table5": TABLE5, "lerner_median": LERNER}
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez(OUT / f"{tag}.npz", theta=theta, delta=m["delta"])
    log.info("alpha %.3f (Table 4 %.3f)", summary["alpha"], TABLE4_ALPHA)
    for y, v in summary["table5"].items():
        log.info("Table 5 %d: %s  (paper %s)", y, np.round(v["mean_over_products"], 2), TABLE5[y])
    for y in (1980, 2018):
        if y in summary["lerner_median_by_year"]:
            log.info("median Lerner %d: %.3f", y, summary["lerner_median_by_year"][y])
    log.info("wrote %s", OUT / f"{tag}.json")


if __name__ == "__main__":
    main()

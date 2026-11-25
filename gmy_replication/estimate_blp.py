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
1 runs in numpy here: the inner model is the harness's MixedLogitModel (nfp_harness.models.blp, the BLP contraction on
the InnerModel interface the PCM uses), the moments the data layer of estimate_pcm.py, nfp_harness.micro_sets and
nfp_harness.second_choice; step 2 and the post-estimation run in
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
import importlib.util
import json
import logging
import os
import pathlib
import sys
import time
import types
from typing import Any, Mapping, Sequence

THREADS_DEFAULT = 32
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # parallelism is across markets, one BLAS thread each
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import norm, qmc  # noqa: E402

from nfp_harness.diagnostics.report import to_jsonable  # noqa: E402
from nfp_harness.linear import LinearIV  # noqa: E402
from nfp_harness.models.blp import MixedLogitModel  # noqa: E402
from nfp_harness.moments import GroupMicroMoments, GroupMomentRow, MicroDataset, SecondChoiceMoments  # noqa: E402
from nfp_harness.params import ParameterLayout, ParameterSpec  # noqa: E402
from nfp_harness.problem import NFPGMMProblem  # noqa: E402
from nfp_harness.second_choice import first_second_correlations  # noqa: E402
from nfp_harness.types import KernelRequests, MarketIndex  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def _gmy_pcm() -> types.ModuleType:
    """estimate_pcm.py of this folder, which holds the GMY data layer, under its own module name (petrin_replication has
    an estimate_pcm too)."""
    name = "gmy_replication_estimate_pcm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "estimate_pcm.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


gmy_pcm = _gmy_pcm()
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
PARAMETER_UNIT = 0.1   # natural unit of every theta entry on the harness's problem path (optimizer scaling only)
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
    """Mixed logit of GMY spec 6 on one GMYData: the harness's MixedLogitModel (nfp_harness.models.blp, the BLP
    contraction) on GMY's X2, demographics and Latin hypercube draws, and GMY's micro and second-choice moments."""

    def __init__(self, data: gmy_pcm.GMYData, threads: int, seed: int = SEED, tol: float = 1e-13) -> None:
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
        self.pattern = [(X2.index(c), DEMOG.index(d)) for c, d in PATTERN]
        self.unobs = [X2.index(c) for c in UNOBS]
        self.model = MixedLogitModel([MarketIndex(y, r) for y, r in zip(self.years, self.rows)], self.X, self.D,
                                     self.w, self.nu, self.pattern, self.unobs,
                                     [p.shares.to_numpy(float)[r] for r in self.rows], threads=threads, tol=tol)
        self.micro = gmy_pcm.micro_evaluator(p, a, data.micro)
        self.sc_years = [y for y in gmy_pcm.SECOND_CHOICE_YEARS if y in set(self.years.tolist())]
        self.sc_x, self.sc_s, self.sc_data = {}, {}, {}
        for y in self.sc_years:
            r = self.rows[list(self.years).index(y)]
            self.sc_x[y] = p.loc[r, list(SC_CHARS_PRODUCT)].to_numpy(float)
            self.sc_s[y] = p.shares.to_numpy(float)[r]
            self.sc_data[y] = first_second_correlations(self.sc_s[y], data.second_choices[y]["prob"], self.sc_x[y])
        self.sc_keep = {y: np.isfinite(self.sc_data[y]) for y in self.sc_years}

    @property
    def evaluations(self) -> int:
        return self.model.evaluations

    def moments(self, theta: np.ndarray, commit: bool = False) -> dict[str, np.ndarray]:
        """delta, P, GMY's micro moments (ExjCondH.m) and the first/second-choice correlations of the model's logit
        second choices, from one ``MixedLogitModel.solve``."""
        sol = self.model.solve(theta, KernelRequests(second_choice=True, second_choice_markets=tuple(self.sc_years)),
                               need_jacobian=False)
        if commit:
            self.model.commit(theta)
        P = sol.outputs["probabilities"]
        g_micro = self.micro(P)
        sc_model = {}
        for y in self.sc_years:
            t = list(self.years).index(y)
            sc_model[y] = first_second_correlations(self.sc_s[y], sol.outputs["second_choice"][t]["C"], self.sc_x[y])
        return {"delta": np.concatenate([sol.delta[r] for r in self.rows]), "P": P, "micro": g_micro, "sc": sc_model}

    def residuals_of(self, m: dict[str, np.ndarray]) -> np.ndarray:
        mic = self.data.micro
        r_micro = np.sqrt(1.0 / mic.variances) * (mic.values - m["micro"])
        r_sc = [SC_WEIGHT * (self.sc_data[y] - m["sc"][y])[self.sc_keep[y]] for y in self.sc_years]
        return np.concatenate([r_micro] + r_sc)

    def residuals(self, theta: np.ndarray, commit: bool = False) -> np.ndarray:
        return self.residuals_of(self.moments(theta, commit))

    def commit_delta(self, delta: np.ndarray) -> None:
        """Accept ``delta`` (product-table order, as ``moments`` returns it) as the contraction's warm start."""
        self.model.warm_from_delta(delta)

    def problem(self) -> NFPGMMProblem:
        """The same objective on the harness's problem path: an NFPGMMProblem on this MixedLogitModel with the
        harness's GroupMicroMoments (the 33 rows, weight diag(1 / variance)) and SecondChoiceMoments (weight 30), no
        aggregate block (beta is GMY's step 2). Its q under W1 is the sum of squared ``residuals``
        (tests/test_blp_model.py); ``main`` logs both."""
        data, p = self.data, self.data.products
        agents = gmy_pcm.add_set_columns(data.agents)
        tab = data.micro.table
        micro = GroupMicroMoments("micro_gmy", MicroDataset("GMY_CEX_MRI", 1),
                                  [GroupMomentRow(*x) for x in gmy_pcm.group_moment_rows(data.micro)],
                                  tab.value.to_numpy(float), tab.variance.to_numpy(float))
        ms = [micro] + ([SecondChoiceMoments("second_choice", self.sc_years, SC_CHARS_PRODUCT, self.sc_data, self.sc_x,
                                             self.sc_s, weight=SC_WEIGHT)] if self.sc_years else [])
        self.model.product_columns = {c: p[c].to_numpy(float) for c in set(gmy_pcm.MICRO_CHAR.values())}
        self.model.agents = agents
        self.model.agent_rows = [np.flatnonzero(agents.market_ids.to_numpy() == y) for y in self.years]
        layout = ParameterLayout([ParameterSpec(n.replace(" ", "_"), "taste", None, None, float(v), float(lo), float(hi),
                                                unit=PARAMETER_UNIT)
                                  for n, v, lo, hi in zip(NAMES, TABLE4_THETA, LOWER, UPPER)])
        n = self.model.n_products
        return NFPGMMProblem(self.model, ms, LinearIV(np.ones((n, 1)), np.ones((n, 1))), layout, max_step=False)


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

def pyblp_results(data: gmy_pcm.GMYData, model: GMYLogit, theta: np.ndarray):
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
    x1 = "0 + prices + " + " + ".join(gmy_pcm.STAGE2_X)
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


def post_estimation(data: gmy_pcm.GMYData, model: GMYLogit, theta: np.ndarray, delta: np.ndarray) -> dict:
    problem, res = pyblp_results(data, model, theta)
    p = data.products
    beta = np.asarray(res.beta).ravel()
    alpha = float(beta[0])
    se = np.asarray(res.beta_se).ravel()
    out: dict = {"alpha": alpha, "alpha_se_clustered": float(se[0]),
                 "beta": dict(zip(gmy_pcm.STAGE2_X, beta[1:].tolist())),
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


# ------------------------------------------------------------------------------------------------
# same-moments: a mixed logit on the PCM's moment set under the joint CUE weight
# ------------------------------------------------------------------------------------------------

SAME_YEARS = (1980, 1991, 1997, 1999, 2001, 2005, 2015, 2018)   # the 8-year PCM joint_cue runs' markets
SAME_NAMES = ("alpha",) + tuple(n.replace(" ", "_") for n in NAMES)
UTILITIES = ("gmy", "pcm_matched")
#: pcm_matched: bounds of the lognormal price factor's sd and of the log price-coefficient scale
SIGMA_P_LOGIT = (0.0, 3.0)
LOG_ALPHA0 = (-5.0, 5.0)


def _pcm_matched_design(spec: Any, start_theta: Mapping[str, float] | None) -> dict[str, Any]:
    """The PCM's utility as a mixed logit, read from the GMY PCM spec's own layout (``GMYSpec.layout()``, the
    starts of ``start_theta`` applied): every taste of the PCM (a sigma on a draw, a pi on a demographic) on the same
    characteristic, and the PCM's price sensitivity alpha_h = exp(log_alpha0 + log c_h + phi_inc inc_h + phi_age
    age_h + sigma_p nu_ph) (``MixedLogitModel`` exponential law; log c_h is GMY's BLP income-age law fixed at
    coefficient 1 as in the PCM, phi_gmy). log_alpha0 is the scale the PCM fixes by its median normalisation and the
    logit's error does not; price leaves delta, so the linear part is the PCM's X1. The PCM's bound a_lo has no
    counterpart (the lognormal is untruncated). Starts: the PCM's tastes times exp(log_alpha0) at log_alpha0 =
    log(median |alpha| of GMY's BLP fit), its phi, sigma_p at min(PCM sigma_p, 1)."""
    lay = spec.layout()
    if start_theta:
        lay = lay.replace({n: {"start": float(np.clip(v, s.lower, s.upper))} for n, v in start_theta.items()
                           if n in lay.names for s in [lay.specs[lay.names.index(n)]] if s.free})
    chars = ["prices"] + [c for c in spec.kernel_chars]
    pattern, unobs, demog, names, starts, lower, upper = [], [], [], [], [], [], []
    scale = float(spec.alpha_median)
    for s in lay.specs:
        if s.role != "taste":
            continue
        k = chars.index(s.column)
        if s.basis.startswith("nu"):
            unobs.append((k, s))
        else:
            if s.basis not in demog:
                demog.append(s.basis)
            pattern.append((k, demog.index(s.basis), s))
    th_pi = [(s.name, s.start * scale, -25.0, 25.0) for _, _, s in pattern]
    th_sig = [(s.name, s.start * scale, 0.0, 25.0) for _, s in unobs]
    exp_names = [("log_alpha0", "one")]
    for s in lay.specs:
        if s.role == "scale" and s.free:
            exp_names.append((s.name, s.basis))
    for d in [b for _, b in exp_names]:
        if d not in demog:
            demog.append(d)
    sp = lay.specs[lay.names.index("sigma_p")]
    th_exp = ([("log_alpha0", float(np.log(scale)), *LOG_ALPHA0)]
              + [(n, lay.specs[lay.names.index(n)].start, -4.0, 4.0) for n, _ in exp_names[1:]]
              + [("sigma_p", float(min(sp.start, 1.0)), *SIGMA_P_LOGIT)])
    for n, v, lo, hi in th_pi + th_sig + th_exp:
        names.append(n)
        starts.append(v)
        lower.append(lo)
        upper.append(hi)
    phi_gmy = lay.specs[lay.names.index("phi_gmy")]
    offset_col = phi_gmy.basis if phi_gmy.start != 0 else None
    return {"chars": chars, "pattern": [(k, d) for k, d, _ in pattern], "unobs": [k for k, _ in unobs],
            "demog": demog, "exp_terms": [("D", demog.index(b)) for _, b in exp_names] + [("nu", len(unobs))],
            "offset": (offset_col, float(phi_gmy.start)), "names": names, "start": np.array(starts),
            "lower": np.array(lower), "upper": np.array(upper)}


def same_moments_problem(threads: int, households: int = 2000, years: tuple[int, ...] = SAME_YEARS,
                         fd_max_gap: int = 3, seed: int = SEED, utility: str = "gmy",
                         moments: str = "demand_supply_msfd_own", drop_blocks: Sequence[str] = (),
                         start_theta: Mapping[str, float] | None = None, cost_shifters: str = "excluded",
                         alo_moments: str = "off") -> tuple[Any, Any, dict[str, Any]]:
    """(NFPGMMProblem, GMYSpec, design): a mixed logit (``MixedLogitModel``, the harness's BLP contraction) on the
    moment blocks of the PCM's ``--moments <moments> --weighting joint_cue [--drop-block ...]`` problem
    (``GMYSpec.moment_blocks``): the same products (twins merged, shares below 1e-5 dropped, price ties spread), the
    same households and ownership coding, the same X1 and Z, micro weight diag(1 / variance), second choices x 30, and
    ``set_joint("cue")`` (a set with no product-level block, e.g. ``gmy_nolinear``, keeps the fixed weight W1 of its
    micro and second-choice blocks, GMY's step-1 objective). Taste draws are GMY's Latin hypercube normals, one per household and market
    (estimation.m), seed + market position.

    ``utility``: ``gmy`` is GMY's spec 6 with alpha the first nonlinear parameter (a pattern term on price with a
    demographic of ones, so ds/dp, which the supply blocks read, moves with it); ``pcm_matched`` is the PCM's utility
    (``_pcm_matched_design``). ``cost_shifters`` and ``alo_moments`` are the PCM's options of the same names."""
    from nfp_harness.problem import NFPGMMProblem

    if utility not in UTILITIES:
        raise ValueError(f"utility must be one of {UTILITIES}, got {utility!r}")
    spec = gmy_pcm.GMYSpec(years=years, households=households)
    spec.fd_max_gap = int(fd_max_gap)
    spec.set_cost_shifters(cost_shifters)
    spec.set_alo_moments(alo_moments)
    spec.set_moments(moments)
    spec.set_weighting_mode("joint_cue", list(drop_blocks))
    ms, _ = spec.moment_blocks(moments, True)
    p, a = spec.products, spec.agents.copy()
    a["one"] = 1.0
    yrs = [int(y) for y in spec.years]
    rows = [spec.market_rows(y) for y in yrs]   # price order, as the PCM model's markets and the second-choice data
    arows = [np.flatnonzero(a.market_ids.to_numpy() == y) for y in yrs]
    w = [a.weights.to_numpy(float)[r] for r in arows]
    chars = sorted({c for m in ms for c in getattr(m, "chars", ()) if isinstance(c, str) and c in p})
    common = dict(threads=threads, tol=1e-13, product_columns={c: p[c].to_numpy(float) for c in chars}, agents=a,
                  agent_rows=arows)
    shares = [p.shares.to_numpy(float)[r] for r in rows]
    markets = [MarketIndex(y, r) for y, r in zip(yrs, rows)]

    def col(c: str, r: np.ndarray) -> np.ndarray:
        return np.ones(r.size) if c == "const" and c not in p else p[c].to_numpy(float)[r]

    if utility == "gmy":
        X = [np.column_stack([col(c, r) for c in X2]) for r in rows]
        demog = DEMOG + ("one",)
        D = [np.column_stack([a[d].to_numpy(float)[r] for d in demog]) for r in arows]
        nu = [norm.ppf(qmc.LatinHypercube(d=len(UNOBS), seed=seed + t).random(r.size)) for t, r in enumerate(arows)]
        pattern = [(X2.index("prices"), demog.index("one"))] + [(X2.index(c), DEMOG.index(d)) for c, d in PATTERN]
        model = MixedLogitModel(markets, X, D, w, nu, pattern, [X2.index(c) for c in UNOBS], shares,
                                price_column=X2.index("prices"), **common)
        names, start = SAME_NAMES, np.r_[TABLE4_ALPHA, TABLE4_THETA]
        lo, hi = np.r_[-25.0, LOWER], np.r_[0.0, UPPER]
        design: dict[str, Any] = {"utility": "gmy", "names": list(names)}
    else:
        dz = _pcm_matched_design(spec, start_theta)
        X = [np.column_stack([col(c, r) for c in dz["chars"]]) for r in rows]
        D = [np.column_stack([a[d].to_numpy(float)[r] for d in dz["demog"]]) for r in arows]
        n_nu = len(dz["unobs"]) + 1
        nu = [norm.ppf(qmc.LatinHypercube(d=n_nu, seed=seed + t).random(r.size)) for t, r in enumerate(arows)]
        oc, ov = dz["offset"]
        offset = [ov * a[oc].to_numpy(float)[r] if oc else np.zeros(r.size) for r in arows]
        model = MixedLogitModel(markets, X, D, w, nu, dz["pattern"], dz["unobs"], shares, price_column=0,
                                exp_column=0, exp_terms=dz["exp_terms"], exp_offset=offset, **common)
        names, start, lo, hi = tuple(dz["names"]), dz["start"], dz["lower"], dz["upper"]
        design = {"utility": "pcm_matched", "names": list(names), "chars": dz["chars"], "demog": dz["demog"],
                  "price_law": "alpha_h = exp(log_alpha0 + log c_h + phi' d_h + sigma_p nu_ph), untruncated"}
    layout = ParameterLayout([ParameterSpec(n, "taste", None, None, float(np.clip(v, l_, h)), float(l_), float(h),
                                            unit=1.0) for n, v, l_, h in zip(names, start, lo, hi)])
    prob = NFPGMMProblem(model, ms, spec.linear_for("all"), layout, max_step=False)
    if prob.agg is not None:
        prob.set_joint("cue")
    else:   # no product-level block (e.g. gmy_nolinear, GMY's step 1): every block has a fixed covariance, W = W1
        prob.set_weighting(prob.initial_weighting(layout.starts()))
    design.update(moments=moments, drop_blocks=list(spec.drop_blocks), fd_max_gap=int(fd_max_gap),
                  cost_shifters=cost_shifters)
    if alo_moments != "off":   # only when set, so the design of an earlier run is unchanged
        design["alo_moments"] = alo_moments
    return prob, spec, design


def same_moments_evaluate(prob: Any, theta: np.ndarray) -> dict[str, Any]:
    """q_CUE, its blocks, Hansen J and the economics of the mixed logit at theta (no Jacobian): every moment row
    (data and model) of the micro and second-choice blocks, Lerner indices and own elasticities by year and segment.
    A problem with no product-level block (``same_moments_problem`` on e.g. ``gmy_nolinear``) is scored with its fixed
    W, and without a supply block the Lerner and elasticity fields are left out."""
    from scipy.stats import chi2

    from nfp_harness.weighting import weight_from_covariance

    ev = prob.evaluate(np.asarray(theta, dtype=float), need_jacobian=False)
    if not ev.ok:
        return {"status": ev.status.value, "note": ev.aux.get("note")}
    if ev.aux.get("joint") is not None:
        W = weight_from_covariance(prob.covariance(ev))
        q = float(prob.n * ev.g @ W @ ev.g)
    else:   # no product-level block: the problem's fixed W (same_moments_problem)
        q = float(ev.aux["q"])
    ident = prob.identification
    df = int(ident["overidentifying"])
    sol = ev.aux["solution"]
    model = prob.model
    mic = next((m for m in prob.moments if m.name == "micro_gmy"), None)
    sc = next((m for m in prob.moments if m.name == "second_choice"), None)
    out = {"status": "ok", "theta": dict(zip(prob.layout.names, map(float, theta))), "q_cue": q,
           "q_blocks": ev.aux["q_blocks"], "q_blocks_own": ev.aux.get("q_blocks_own", {}),
           "moments": int(ident["effective_moments"]), "free_nonlinear": int(ident["p_free"]),
           "linear": int(ident["k_linear"]), "df": df, "J_p_value": float(chi2.sf(q, df)),
           "price_coefficient_max": max(float(model.price_coefficients(theta, t).max())
                                        for t in range(len(model.markets))),
           "micro": [] if mic is None else [
               {"name": lab, "data": float(d), "model": float(v), "z": float((v - d) / np.sqrt(var))}
               for lab, d, v, var in zip(mic.labels, mic.values, ev.aux["micro_predicted"]["micro_gmy"],
                                         mic.variances)],
           "second_choice": [] if sc is None else [
               {"row": lab, "data": float(d), "model": float(v)}
               for lab, d, v in zip(sc.labels, sc.values, ev.aux["micro_predicted"]["second_choice"])]}
    sup = next((m for m in prob.moments if getattr(m, "is_supply", False)), None)
    if sup is None:
        return out
    _, _, info = sup.log_costs(sol, False)
    lerner, own = np.full(prob.n, np.nan), np.full(prob.n, np.nan)
    mids = np.empty(prob.n, dtype=int)
    for t, m in enumerate(model.markets):
        r = np.asarray(m.rows)
        pr = model.kernel_market(t).p
        s = np.asarray(sol.outputs["shares"])[r]
        lerner[r] = info["markup"][r] / pr
        own[r] = np.diag(sol.outputs["ds_dp"][t]) * pr / s
        mids[r] = int(m.market_id)
    years = [int(m.market_id) for m in model.markets]
    s_all = np.asarray(sol.outputs["shares"])
    out.update({"lerner_median": float(np.nanmedian(lerner)),
                "lerner_sales_weighted_mean": float(np.nansum(lerner * s_all) / np.sum(s_all[np.isfinite(lerner)])),
                "lerner_median_by_year": {y: float(np.nanmedian(lerner[mids == y])) for y in years},
                "own_elasticity_median": float(np.nanmedian(own)),
                "own_elasticity_median_by_year": {y: float(np.nanmedian(own[mids == y])) for y in years},
                "share_mc_clipped": float(info.get("share_mc_clipped", float("nan")))})
    seg = getattr(prob, "segments", None)
    if seg is not None:
        out["lerner_median_by_segment"] = {k: float(np.nanmedian(lerner[seg == k])) for k in np.unique(seg)}
    evb = next((m for m in prob.moments if m.name == "event_cost"), None)
    if evb is not None:
        out["events"] = evb.summary(ev.aux["log_mc"])
    out["markup"], out["lerner"], out["own_elasticity"] = info["markup"], lerner, own
    return out


def same_moments_fit(prob: Any, theta0: np.ndarray, max_nfev: int, max_rounds: int = 6, tag: str = "blp_same_moments"
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """CUE by nonlinear least squares on r(theta) = sqrt(n) L(theta)^-1 g(theta) with S(theta) = L L', so r'r =
    q_CUE, and a central-difference Jacobian (the supply blocks' log mc has no analytic derivative on the logit),
    restarted from its endpoint while q still falls; the contraction warm-starts from the best point so far. A theta
    at which some household's price coefficient is not negative is refused (no Nash markups). Standard errors
    (J'J)^-1 from the final Jacobian of r."""
    import scipy.linalg

    lay = prob.layout
    names = lay.names
    lo, hi = (np.asarray(b, dtype=float) for b in lay.bounds())
    best: dict[str, Any] = {"q": np.inf}
    k = prob.k
    t_start = time.time()

    def resid(th: np.ndarray) -> np.ndarray:
        a_max = max(float(prob.model.price_coefficients(th, t).max()) for t in range(len(prob.model.markets)))
        if not a_max < 0:   # demand slopes up for some household: no Nash markups, outside the model's domain
            return np.full(k, 1e3)
        ev = prob.evaluate(th, need_jacobian=False)
        if not ev.ok:
            log.info("evaluation failed at %s: %s %s", np.round(th, 3).tolist(), ev.status.value, ev.aux.get("note"))
            return np.full(k, 1e3)
        L = np.linalg.cholesky(prob.covariance(ev))
        r = np.sqrt(prob.n) * scipy.linalg.solve_triangular(L, ev.g, lower=True)
        q = float(r @ r)
        if q < best["q"]:
            best["q"] = q
            prob.commit(th)
            log.info("q_CUE %.4f (evaluations %d, %.0f s)", q, prob.model.evaluations, time.time() - t_start)
        return r

    x = np.clip(theta0, lo, hi)
    rounds: list[dict[str, Any]] = []
    for _ in range(max_rounds):   # restart from the endpoint while it still lowers q (trust-region radius reset)
        res = least_squares(resid, x, jac="3-point", bounds=(lo, hi), x_scale="jac", diff_step=1e-5,
                            xtol=1e-12, ftol=1e-12, gtol=1e-8, max_nfev=max_nfev, verbose=0)
        q_new = float(2 * res.cost)
        rounds.append({"q": q_new, "nfev": int(res.nfev), "status": int(res.status), "message": str(res.message),
                       "optimality": float(res.optimality)})
        log.info("least_squares round %d: q %.5f, %s, optimality %.3g", len(rounds), q_new, res.message,
                 res.optimality)
        improved = len(rounds) == 1 or rounds[-2]["q"] - q_new > 1e-4
        x = np.asarray(res.x, dtype=float)
        np.savez(OUT / f"{tag}.checkpoint.npz", theta=x, names=np.array(names))
        if not improved or res.status == 1 or res.nfev >= max_nfev:
            break
    J = res.jac
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)), 0, None))
    info = {"status": int(res.status), "message": str(res.message), "nfev": int(res.nfev), "q": float(2 * res.cost),
            "rounds": rounds, "optimality": float(res.optimality),
            "active_bounds": [names[i] for i in np.flatnonzero(res.active_mask)],
            "se": dict(zip(names, map(float, se))), "seconds": time.time() - t_start,
            "evaluations": int(prob.model.evaluations)}
    return np.asarray(res.x, dtype=float), info


def _start_theta(path: str | None) -> dict[str, float]:
    """Named starting values from a harness stage file (``.npz`` with theta and names) or a same-moments JSON."""
    if not path:
        return {}
    pth = pathlib.Path(path)
    if pth.suffix == ".json":
        d = json.loads(pth.read_text())
        th = d.get("estimate", d).get("theta", d.get("theta"))
        return {k.replace(" ", "_"): float(v) for k, v in th.items()}
    z = np.load(pth, allow_pickle=True)
    if "names" in z:   # a same-moments .npz
        return dict(zip([str(n) for n in z["names"]], map(float, np.asarray(z["theta"], dtype=float).ravel())))
    from nfp_harness.checkpoint import read_stage_file, stage_parameter_names

    names = stage_parameter_names(pth)
    if names is None:
        raise ValueError(f"{pth}: a stage without parameter names")
    return dict(zip(names, map(float, np.asarray(read_stage_file(pth)["theta"], dtype=float).ravel())))


def est_delta(prob: Any) -> np.ndarray:
    """(n,) row-order accepted delta of a same-moments problem's model."""
    out = np.full(prob.n, np.nan)
    for m, d in zip(prob.model.markets, prob.model.delta):
        out[np.asarray(m.rows)] = d
    return out


def _logit_counterfactuals(model: MixedLogitModel, theta: np.ndarray, delta: np.ndarray, products: Any,
                           price_coefficient: float | None = None) -> dict[str, Any]:
    """``estimate_pcm.gmy_counterfactuals`` on a mixed logit: ``LogitMarketDemand`` per market (Nash by the FOC
    fixed point), ``price_coefficient`` the linear one inside delta (GMY's alpha; None: the model's)."""
    from nfp_harness.counterfactuals import LogitMarketDemand

    firms = products.firm_ids.astype(str).to_numpy()
    return gmy_pcm.gmy_counterfactuals(
        lambda t: LogitMarketDemand(model, theta, delta, t, firms, price_coefficient), model.markets, products)


def same_moments_main(args: Any) -> None:
    """Fit a mixed logit on the PCM's moments and write results/<tag>.{json,npz,log}."""
    years = tuple(args.years) if args.years else gmy_pcm.ALL_YEARS
    tag = (f"blp_same_moments_{args.utility}_{len(years)}y" + (f"_{args.tag}" if args.tag else ""))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / f"{tag}.log"), logging.StreamHandler()])
    t0 = time.time()
    st = _start_theta(args.start_theta)
    # a pcm_matched same-moments fit of its own (it names log_alpha0) is a warm start of the logit's layout as it
    # stands; a PCM stage file is read in the PCM's units by _pcm_matched_design
    own_fit = args.utility == "pcm_matched" and "log_alpha0" in st
    prob, spec, design = same_moments_problem(args.threads, households=args.households, years=years,
                                              fd_max_gap=args.fd_max_gap, utility=args.utility,
                                              moments=args.moments, drop_blocks=args.drop_block or (),
                                              start_theta=st if args.utility == "pcm_matched" and not own_fit
                                              else None,
                                              cost_shifters=args.cost_shifters, alo_moments=args.alo_moments)
    prob.segments = spec.products[spec.groups].astype(str).to_numpy()   # type: ignore[attr-defined]
    log.info("%s: %d products, %d moments, identification %s", design, prob.n, prob.k, prob.identification)
    theta0 = prob.layout.starts()
    if (args.utility == "gmy" or own_fit) and st:
        theta0 = np.array([st.get(n, v) for n, v in zip(prob.layout.names, theta0)])
    if args.max_nfev <= 0:   # evaluate only
        out = {"design": design, "theta0": dict(zip(prob.layout.names, map(float, theta0)))}
        e = same_moments_evaluate(prob, theta0)
        prob.commit(theta0)
        out["estimate"] = {k: v for k, v in e.items() if k not in ("markup", "lerner", "own_elasticity")}
        if args.counterfactuals:
            out["counterfactuals"] = _logit_counterfactuals(prob.model, theta0, est_delta(prob), spec.products)
        (OUT / f"{tag}.json").write_text(json.dumps(to_jsonable(out), indent=1))
        log.info("evaluated at the start: q_CUE %s (%.0f s)", out["estimate"].get("q_cue"), time.time() - t0)
        return
    theta, fit = same_moments_fit(prob, theta0, args.max_nfev, tag=tag)
    est = same_moments_evaluate(prob, theta)
    prob.commit(theta)
    try:
        cfs = _logit_counterfactuals(prob.model, theta, est_delta(prob), spec.products)
    except Exception as exc:   # the estimate is written whatever happens to the counterfactuals
        log.exception("counterfactuals failed")
        cfs = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    out = {"design": design, "theta0": dict(zip(prob.layout.names, map(float, theta0))), "years": list(spec.years),
           "households": args.households, "recoded_ownership": spec.recoded_ownership, "fit": fit,
           "estimate": {k: v for k, v in est.items() if k not in ("markup", "lerner", "own_elasticity")},
           "counterfactuals": cfs, "seconds": time.time() - t0}
    (OUT / f"{tag}.json").write_text(json.dumps(to_jsonable(out), indent=1))
    np.savez(OUT / f"{tag}.npz", theta=theta, names=np.array(prob.layout.names),
             delta=np.concatenate([d for d in prob.model.delta]),
             rows=np.concatenate([np.asarray(m.rows) for m in prob.model.markets]),
             **{k: np.asarray(est[k]) for k in ("markup", "lerner", "own_elasticity") if k in est})
    log.info("wrote %s: q_CUE %.3f", OUT / f"{tag}.json", out["estimate"].get("q_cue", np.nan))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("published", "fit", "same-moments"))
    ap.add_argument("--households", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    ap.add_argument("--utility", choices=UTILITIES, default="gmy",
                    help="same-moments: gmy (spec 6, alpha in theta) or pcm_matched (the PCM's utility and lognormal "
                         "price sensitivity)")
    ap.add_argument("--moments", default="demand_supply_msfd_own",
                    help="same-moments: the PCM moment set whose blocks the logit is fitted on")
    ap.add_argument("--drop-block", action="append", default=None,
                    help="same-moments: product-level blocks dropped, as the PCM's --drop-block (own = event_cost)")
    ap.add_argument("--fd-max-gap", type=int, default=1, help="same-moments: as the PCM's --fd-max-gap")
    ap.add_argument("--start-theta", default=None,
                    help="same-moments: named starts (a harness stage .npz, e.g. the PCM's, for pcm_matched; a "
                         "same-moments JSON)")
    ap.add_argument("--cost-shifters", choices=gmy_pcm.COST_SHIFTER_MODES, default="excluded",
                    help="same-moments: as the PCM's --cost-shifters")
    ap.add_argument("--alo-moments", choices=gmy_pcm.ALO_MOMENT_MODES, default="off",
                    help="same-moments: as the PCM's --alo-moments (c4c6c needs --cost-shifters in_cost)")
    ap.add_argument("--tag", default="", help="suffix of the output names (default none)")
    ap.add_argument("--counterfactuals", action="store_true",
                    help="published / fit: also run the shared counterfactuals at the estimate (off by default, so "
                         "the outputs of a plain run are unchanged); same-moments with --max-nfev 0: run them at the "
                         "start (a smoke test). A same-moments fit always runs them")
    args = ap.parse_args()
    if args.threads > 64:
        raise SystemExit("at most 64 threads")
    OUT.mkdir(exist_ok=True)
    if args.stage == "same-moments":
        args.drop_block = [gmy_pcm.DROP_ALIASES.get(b, b) for b in (args.drop_block or ())]
        same_moments_main(args)
        return
    tag = (f"blp_{args.stage}_h{args.households}" + (f"_{len(args.years)}y" if args.years else "")
           + (f"_{args.tag}" if args.tag else ""))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / f"{tag}.log"), logging.StreamHandler()])
    data = gmy_pcm.load_data(n_households=args.households, seed=SEED, years=args.years)
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
    if args.counterfactuals:   # the shared counterfactual engine, Nash by the zeta map, alpha the 2SLS coefficient
        try:
            model.model.price_column = X2.index("prices")
            delta_rows = np.full(model.model.n_products, np.nan)
            delta_rows[np.concatenate(model.rows)] = m["delta"]
            summary["counterfactuals"] = to_jsonable(_logit_counterfactuals(
                model.model, theta, delta_rows, data.products, price_coefficient=float(summary["alpha"])))
        except Exception as exc:   # the estimate is written whatever happens to the counterfactuals
            log.exception("counterfactuals failed")
            summary["counterfactuals"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez(OUT / f"{tag}.npz", theta=theta, delta=m["delta"])
    log.info("alpha %.3f (Table 4 %.3f)", summary["alpha"], TABLE4_ALPHA)
    for y, v in summary["table5"].items():
        log.info("Table 5 %d: %s  (paper %s)", y, np.round(v["mean_over_products"], 2), TABLE5[y])
    for y in (1980, 2018):
        if y in summary["lerner_median_by_year"]:
            log.info("median Lerner %d: %.3f", y, summary["lerner_median_by_year"][y])
    log.info("wrote %s", OUT / f"{tag}.json")
    prob = model.problem()   # the same objective on the harness's problem path (a check, not an output)
    ev = prob.evaluate(theta, need_jacobian=False)
    W1 = prob.initial_weighting(theta)
    log.info("q on the NFPGMMProblem path %.10g, of the residuals %.10g", float(prob.n * ev.g @ W1 @ ev.g),
             summary["q"])


if __name__ == "__main__":
    main()

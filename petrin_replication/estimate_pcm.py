"""Petrin (2002) minivans, exact pure characteristics model (PCM), as a spec on nfp_harness.

    u_ijt = delta_jt + x_jt . t_i - alpha_i p_jt,   u_i0t = 0
    x     = [1, hpwt, space, air, mpd, fwd, fam],   fam = max(mi, sw, su, pv)   (the taste columns, TASTE)
    t_ik  = sigma_k nu_ik on the 7 taste columns  +  pi_b log(fs_i) fv_i on b in {mi, sw, su, pv}
    log alpha_i = log(ybar / clip(y_i, 1, 150)) + phi_mid mid_i + phi_high high_i + log a_i   (Petrin's income law)
    a_i ~ LogNormal(0, sigma_p) truncated below at a_lo = 0.2, so alpha_lo,i = a_lo c_i   (LEARNINGS 32)
    delta_jt = X1_jt beta + xi_jt,  X1 = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv, pgnp, trend, trend2]
    Z     = [X1, demand_instruments0..21]  (35 columns; LEARNINGS 16)

Moments ("all"): the 35 aggregate IV moments, Petrin's 10 CEX micro moments (pyblp's Petrin data), with the
micro block of W1 and S fixed at Petrin's sample covariances (PETRIN_COVARIANCES_LOCATION), and Petrin's
log-cost supply side, 28 moments (round-2 analysis, LEARNINGS 21): Nash markups from the PCM's own ds/dp with
the ownership of firm_ids, log mc = w gamma + omega on w = [1, log hpwt, log wt, log mpg, air, fwd, trend, jp,
eu, jp x trend, eu x trend, log q], Z_s = [w, supply_instruments0..15]; an evaluation with any mc <= 0 fails.
73 moments for 13 + 12 linear and 14 nonlinear parameters. "demand" drops the supply block. The aggregate block
is in Sargan form (``SarganIVMoments``, scale-free; LEARNINGS 24, 25) unless ``--robustness fixed_w``. With
``--margin m`` one calibration row binds the markup level: the 25th-75th percentile trimmed mean of the Nash
margins (p - mc)/p equals m (``MarginMoment``, weight 1e4 / (n 0.01^2); LEARNINGS 25). The supply Jacobian is
the central secant over 1e-3 natural units. Taste spreads
start at $3k on const, $2k on fam and $1k per sd on the other five; sigma_p starts at 2.5 in [1, 4], phi_mid
and phi_high at 1.1 and 1.8 (Petrin's pi ratios) in [-4, 4]. Normalisations: median a_i = 1 (mu = 0, LEARNINGS 11) on the untruncated
parent, alpha_i = exp(phi_g) (ybar / y_i) a_i with income winsorised to [1, 150] and ybar the pooled low-group median
income (5.78); the unobserved factor bounded below, a_i >= a_lo = 0.2 (a fixed parameter of role bound, a
calibration of the welfare tail); sigma_fam capped at $25k; taste spreads non-negative; u_i0 = 0 in every year with pgnp and
trends in X1. Data: the 17 same-firm exact twins (identical price and characteristics) are merged into one
row with summed shares, then the remaining price ties are spread to a $10 gap. Consumers: the 1,000 CEX
agents per market crossed with scrambled Sobol standard normal taste draws, 2^ceil(log2(1000 nu)) draws per
market. Exact PCM shares, no smoothing (LEARNINGS 15). Specification: LEARNINGS 19. Stages and their
files: nfp_harness.pipeline (DESIGN 6.4); the gate is a sigma_p profile concentrated in (phi_mid, phi_high),
read on q_fixed and on its supply block. step1 starts from the gate minimum (lambda = 1 slice, lowest q_fixed)
with pi_b re-solved on the eight body micro moments; ``margin_band`` estimates everything at m* in
{0.10, 0.15, 0.20, 0.24, 0.30} (headline 0.24, Petrin's BLP median margin). ``a_lo_band`` fixes a_lo at 0.1, 0.2
and 0.4 and estimates everything else (sigma_p free, start 2.5, bounds [1, 4]) on Sargan + micro + supply from the
prefit point; its headline is a_lo = 0.2, a calibration of the welfare tail (LEARNINGS 32). ``sigma_p_band`` (sigma_p
fixed at 2.5 and 3 at a_lo = 0.2) is kept as a robustness stage. ``--start-theta <run>/stages/<stage>.npz`` starts
the prefit from a stored estimate's tastes, pis and sigma_p (phi at its starts; part of the run key).

Robustness switches (each changes the run directory): ``--robustness body_sigma`` frees taste spreads on
mi, sw, su, pv (cap $25k); ``--robustness income_winsor`` is a no-op alias (the income law is the main
spec); ``--robustness fixed_w`` uses the
fixed-W aggregate block (IVMoments) instead of the Sargan form; ``--robustness sargan`` is accepted as a no-op
alias (Sargan is the default) so commands and run directories written with it keep their run key;
``--robustness a_lo_01`` and ``a_lo_04`` set the bound a_lo to 0.1 and 0.4.
``--outer mpec`` (own run directory) runs the optional demand-only MPEC outer of ``nfp_harness.mpec`` with the
``.envs/mpec`` python; a stage whose problem has the supply block or a margin row refuses it (LEARNINGS 31).
The options every paper shares (``--a-lo``, ``--fix``, ``--start-theta-except``, ``--warm-delta``, ``--cold-draws``,
``--max-iterations``, beside those above) come from nfp_harness.paper.PaperSpec; each is in the run key only when set.

Run (results under petrin_replication/results/harness/, untracked):
  python petrin_replication/estimate_pcm.py run --stage prefit --nu 4
  python petrin_replication/estimate_pcm.py run --stage gate --nu 4 [--detach]
  python petrin_replication/estimate_pcm.py run --stage all --nu 16 --detach
  python petrin_replication/estimate_pcm.py run --stage margin_band --nu 16 \
      --start-gate <gate dir> --detach
  python petrin_replication/estimate_pcm.py run --stage a_lo_band --nu 16 \
      --start-theta <run dir>/stages/band.2.5.step2.npz --detach
  python petrin_replication/estimate_pcm.py status --nu 16
Resume a killed run by relaunching the identical command.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # idle OpenMP teams must not spin on 192 cores
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")    # numpy work is small; keep cores for the kernel
os.environ.setdefault("MKL_NUM_THREADS", "8")         # anaconda's MKL (loaded via scipy/pyblp) spins a 192-thread pool otherwise
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]

import dataclasses  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyblp  # noqa: E402

from nfp_harness import comparison, pipeline, twins  # noqa: E402
from nfp_harness.data import break_price_ties, income_law_basis  # noqa: E402
from nfp_harness.models import pcm_post as pp  # noqa: E402
from nfp_harness.models.pcm import PCMSpec  # noqa: E402
from nfp_harness.moments import (IVMoments, MarginMoment, SarganIVMoments, MicroDataset, MicroPart,  # noqa: E402
                                 RatioMicroMoment, SupplyMoments)
from nfp_harness.paper import PaperSpec  # noqa: E402
from nfp_harness.params import ParameterLayout, ParameterSpec  # noqa: E402
from nfp_harness.profile import NotMonotoneDecreasing, RisesAtBothEnds  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results" / "harness"
CHARS = ("const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")   # the X1 basis
X1_COLS = CHARS + ("pgnp", "trend", "trend2")
BODY = ("mi", "sw", "su", "pv")
TASTE = ("const", "hpwt", "space", "air", "mpd", "fwd", "fam")   # columns with a taste spread
KERNEL_CHARS = TASTE + BODY   # columns of x in the kernel: the body dummies carry pi_b (and body sigmas)
YEAR = 1984
# Petrin's log-cost basis (estimate_blp.py, pyblp X3 "1 + log(hpwt) + log(wt) + log(mpg) + air + fwd + trend * (jp
# + eu) + log(q)", all exogenous) and supply instruments Z_s = [w, supply_instruments0..15] (pyblp's ZS)
COST_BASIS = ("1", "log(hpwt)", "log(wt)", "log(mpg)", "air", "fwd", "trend", "jp", "eu", "jp*trend", "eu*trend",
              "log(q)")
N_SUPPLY_INSTRUMENTS = 16
NASH_RCOND_MIN = 1e-12
# Bounded Nash (LEARNINGS 29): best responses in a +-10% price box, at most 20 Gauss-Seidel sweeps over firms,
# deviation check at +-2%
NASH_BOX = 0.10
NASH_SWEEPS = 20
NASH_DEVIATION = 0.02
# Reference taste spreads in $1000 per sd of each characteristic at alpha = 1 (round-2 analysis, change 1):
# $3k on const, $2k on fam, $1k per sd on the five continuous or binary characteristics, so sigma_c =
# 1 / sd(c) for those (hpwt 10.2, space 6.7, air 2.05, mpd 1.50, fwd 2.00 on the merged products). The
# ray gate (lambda x SIG_REF) and the starts use the same direction.
SIG_FIXED_DOLLARS = {"const": 3.0, "fam": 2.0}
PER_SD_COLUMNS = ("hpwt", "space", "air", "mpd", "fwd")
PER_SD_DOLLARS = 1.0
# Under the bound sigma_p no longer moves the buyers' log-alpha range (LEARNINGS 32): estimated, start 2.5 in [1, 4]
SIGMA_P = {"start": 2.5, "lower": 1.0, "upper": 4.0, "unit": 0.25}
# Lower bound on the unobserved factor a_i of alpha_i = exp(phi_g) (ybar / y_i) a_i (LEARNINGS 32): alpha_lo,i =
# a_lo c_i, so the WTP cap is proportional to income. Nothing in the data pins a_lo; it is a calibration of the
# welfare tail. Headline 0.2 (welfare at BLP's size), robustness 0.1 and 0.4.
A_LO = 0.2
A_LO_ROBUSTNESS = {"a_lo_01": 0.1, "a_lo_04": 0.4}
A_LO_UNIT = 0.1
A_LO_BAND = (0.1, 0.2, 0.4)
A_LO_HEADLINE = 0.2
# Income shifts on log alpha: starts at Petrin's pi ratios, log(12.2 / 3.9) and log(23.7 / 3.9); bounds +-4 so a bound
# is never the estimate (an estimate within 1e-3 of one is flagged AT_BOUND)
PHI_STARTS = {"phi_mid": 1.1, "phi_high": 1.8}
PHI_BOUND = 4.0
GATE_LAMBDAS = (0.5, 1.0, 2.0)
GATE_SIGMA_P = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GATE_SLICE = {"lambda": 1.0}
HAULERS = ("mi", "sw", "su", "pv")
# sigma_p band, kept as a robustness stage at a_lo = 0.2: past 3.0 margins fall with sigma_p (LEARNINGS 32)
BAND_SIGMA_P = (2.5, 3.0)
BAND_MARGIN_TARGET = 0.24   # headline: the band point whose median margin is nearest Petrin's BLP median margin
ANATOMY_COLUMNS = ("frontier_products_per_buying_draw", "log_alpha_range_buyers", "window_width_log_alpha")
BAND_COLUMNS = ("own_elasticity_median", "own_elasticity_times_sigma_p", "own_elasticity_times_window_width",
                "share_own_elasticity_below_-20",
                "margin_median", "inside_share_semi_elasticity", "minivan_cs_gain_1984_fixed_prices_musd",
                "minivan_cs_gain_1984_bounded_nash_musd", "bounded_nash_status_1984",
                "minivan_diversion_haulers_share_of_inside", "car_to_hauler_diversion_over_hauler_inside_share",
                "nonzero_cross_median")
PI_STARTS = {"mi": 0.75, "sw": 0.28, "su": 0.20, "pv": 0.41}   # pyblp BLP-micro pi / 0.65 (SPEC 3.2)
FAM_CAP = 25.0       # $1000: 2 (1 - Phi(66.1 / 25)) = 0.8% of consumers value fam beyond the price range
BODY_SIGMA_CAP = 25.0
INCOME_CLIP = (1.0, 150.0)
TWIN_KEYS = ("market_ids", "firm_ids", "prices", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")
# FOC costs are clipped at $500 (pyblp's costs_bounds convention): a product with |own elasticity| < 1 has a
# Nash markup above its price. The clipped share is reported and flagged; at sigma_p <= 2 it is 1-2 products,
# at sigma_p 3-4 it is 94-336, and the log-cost residual of a clipped product (log 0.5 against about 2.5)
# is the supply block's penalty on that region (LEARNINGS 22).
MC_FLOOR = 0.5
ROBUSTNESS = ("body_sigma", "income_winsor", "sargan", "fixed_w") + tuple(A_LO_ROBUSTNESS)
# Markup-level calibration (round-3 analysis 5.2-5.3, LEARNINGS 25)
MARGIN_TRIM = (0.25, 0.75)
MARGIN_SD = 0.01          # declared sd of the margin row: W1 entry 1 / (n 0.01^2)
MARGIN_BINDING = 1e4      # binding W entry = 1e4 x that, so one margin point off costs 1e4
MARGIN_BAND = (0.10, 0.15, 0.20, 0.24, 0.30)
MARGIN_HEADLINE = 0.24    # median margin of Petrin's BLP with micro and supply (post/comparison.txt)
SUPPLY_FD_STEP = 1e-3     # natural units, central secant, no shrink (round-3 analysis 4 (i))
BODY_MICRO_ROWS = 8       # E[age | b], E[fs | b] for b in mi, sw, su, pv: the first 8 micro rows
MICRO_COVARIANCE_SOURCE = "pyblp.data.PETRIN_COVARIANCES_LOCATION, fixed"
BLP_SUMMARIES = {"BLP micro": "blp_micro_summary.json", "BLP no micro": "blp_nomicro_summary.json",
                 "BLP demand micro": "blp_demand_micro_summary.json"}


def merge_twins(products: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows equal on TWIN_KEYS (same market, firm, price and characteristics) are one product to every
    consumer in the PCM: keep the first row with the summed shares (and quantities). Returns (products,
    number of rows merged away). The cost basis stays consistent: the twins share wt, mpg, jp, eu and trend
    (checked), and q is summed with the shares, so log(q) is the merged product's quantity."""
    return twins.merge_twins(products, TWIN_KEYS, ("shares", "q"), constant_columns=("wt", "mpg", "jp", "eu", "trend"))


def per_sd_spreads(products: pd.DataFrame) -> dict[str, float]:
    """Taste spreads (by characteristic) of the reference direction: SIG_FIXED_DOLLARS on const and fam,
    PER_SD_DOLLARS / sd(c) on PER_SD_COLUMNS (pooled sd over the merged products)."""
    out = dict(SIG_FIXED_DOLLARS)
    for c in PER_SD_COLUMNS:
        out[c] = PER_SD_DOLLARS / float(np.std(products[c].to_numpy(float)))
    return {c: out[c] for c in TASTE}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, float, int]:
    products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    price_range = float(products.prices.max() - products.prices.min())   # pooled, before tie breaking: 66.1
    products, n_merged = merge_twins(products)
    products["fam"] = products[list(BODY)].max(axis=1)
    products["prices"] = break_price_ties(products.prices.to_numpy(float), products.market_ids.to_numpy(), 0.01)
    # body type label for diversion by group: the first of mi, sw, su, pv whose dummy is 1, else car (two
    # products, 1988 and 1989, carry both mi and pv and count as minivans, as in estimate_blp.py)
    body = np.full(len(products), "car", dtype=object)
    for b in reversed(BODY):
        body[products[b].to_numpy() == 1] = b
    products["body"] = body.astype(str)
    if not (products[["hpwt", "wt", "mpg", "q"]].to_numpy() > 0).all():
        raise ValueError("the log-cost basis needs positive hpwt, wt, mpg and q")
    agents["logfs_fv"] = np.log(agents.fs) * agents.fv
    # Price scale c_i = alpha_i / a_i: log c_i = log(ybar / clip(y_i, 1, 150)) + phi_mid mid_i + phi_high high_i,
    # Petrin's income law (LEARNINGS 32) with ybar the pooled low-group median income, so the normalising consumer
    # is the median low-income household; c_i spans at most a factor 150 within a group.
    ybar = float(agents.income[agents.low == 1].median())
    agents["log_ybar_inc_w"] = income_law_basis(agents.income.to_numpy(float), ybar, INCOME_CLIP)
    agents["income_group"] = np.select([agents.low == 1, agents.mid == 1, agents.high == 1], ["low", "mid", "high"], "none")
    agents["fs_group"] = np.where(agents.fs >= 5, "5+", agents.fs.astype(int).astype(str))
    return products, agents, price_range, n_merged


class PetrinSpec(PaperSpec):
    """The Petrin PCM spec read by nfp_harness.pipeline (interface in its module docstring; shared options, run key,
    starts and problem assembly in nfp_harness.paper.PaperSpec)."""

    name = "petrin_pcm"
    firm_column = "firm_ids"
    groups = "body"
    group_other = "car"
    binary_columns = ("air", "fwd", "fam") + BODY
    prefit_free = ("phi_mid", "phi_high")
    prefit_max_iterations = 10
    extensive_free = ("phi_mid", "phi_high", "sigma_p", "sigma_const")
    gate_concentrate_iterations = 20
    # step1 and every margin_band point start at the gate minimum with pi_b re-solved on the body micro moments
    start_from_gate = True
    start_concentrate = tuple(f"pi_{b}" for b in BODY)
    start_slice = GATE_SLICE
    start_column = "q_fixed"
    margin_band_values = MARGIN_BAND
    a_lo_default = A_LO
    robustness_choices = ROBUSTNESS
    robustness_help = ("robustness switch (repeatable): body_sigma frees the 4 body taste spreads (cap 25); "
                       "income_winsor is a no-op alias (the income law is the main spec); fixed_w uses the fixed-W "
                       "aggregate block; sargan is a no-op alias of the default; a_lo_01 / a_lo_04 set the bound on a_i "
                       "to 0.1 / 0.4 (main 0.2)")
    a_lo_robustness = A_LO_ROBUSTNESS
    # --start-theta takes tastes, pis and sigma_p: phi keeps the spec's starts, since the income law changes what phi
    # means, and the bound is a calibration; the values taken are part of the run key
    start_theta_keep = ("phi_mid", "phi_high", "phi_y", "alpha_lo", "a_lo")
    start_theta_key = "values"
    # prefit and gate concentration of (phi_mid, phi_high): the two income purchase-rate rows (the last two micro rows)
    # up-weighted by 1e4, so phi solves those two moments, which identify it exactly, with the rest of q only breaking
    # ties (SPEC 3 steps 3-4, LEARNINGS 19); reported q values keep W1. The gate-minimum start re-solves pi_b on the
    # eight body micro rows (E[age | b], E[fs | b]) up-weighted alike (round-3 analysis 4 (iv)), calibration rows at 0.
    concentrate_rows = {"micro": (-2, None)}
    start_concentrate_rows = {"micro": (0, BODY_MICRO_ROWS)}
    # the body-level direction: sigma_fam, any body sigmas and the family-size pis up in proportion (LEARNINGS 14)
    flat_directions = {"body_level_direction": tuple(f"sigma_{b}" for b in BODY + ("fam",))
                       + tuple(f"pi_{b}" for b in BODY)}

    def __init__(self, robustness: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.band_nash_all = False
        self.margin: float | None = None
        self.products, self.agents, self.price_range, self.n_merged_twins = load_data()
        p = self.products
        self.sigma_starts = per_sd_spreads(p)
        self.sig_ref = {f"sigma_{c}": v for c, v in self.sigma_starts.items()}
        inside = p.shares.to_numpy(float)
        self.hauler_inside_share = float(inside[p.body.isin(HAULERS).to_numpy()].sum() / inside.sum())
        self.X1 = np.column_stack([np.ones(len(p))] + [p[c].to_numpy(float) for c in X1_COLS[1:]])
        self.Z = np.column_stack([self.X1] + [p[f"demand_instruments{k}"].to_numpy(float) for k in range(22)])
        self.cost_basis = cost_basis(p)
        self.Z_supply = np.column_stack([self.cost_basis] + [p[f"supply_instruments{k}"].to_numpy(float)
                                                             for k in range(N_SUPPLY_INSTRUMENTS)])
        self.linear_names = list(X1_COLS)
        self.clusters = p.clustering_ids.to_numpy()
        self.values = dict(pd.read_csv(pyblp.data.PETRIN_VALUES_LOCATION).iloc[:, :2].to_numpy())
        self.micro_covariances = pd.read_csv(pyblp.data.PETRIN_COVARIANCES_LOCATION, index_col=0)
        self.set_robustness(robustness)
        self.plausible = {   # ranges for the exact PCM (round-2 analysis, change 7); BLP's ranges do not apply
            "own_elasticity_median": (-15.0, -5.0), "share_own_elasticity_below_-20": (None, 0.25),
            "margin_median": (0.07, 0.25), "inside_share_elasticity_median": (-1.5, -0.4),
            "minivan_diversion_haulers_share_of_inside": (0.15, None),
            "car_to_hauler_diversion_over_hauler_inside_share": (0.05, None),
            "relative_quality_dispersion_buyer_dollars_median": (1.0, 6.0),
            "sd_xi_over_sd_delta": (None, 1.0), "xi_kurtosis": (None, 10.0),
            "spread_buyer_dollars": {"sigma_fam": (None, FAM_CAP), "sigma_const": (None, 15.0)},
            # the bounded elasticity law: -median own elasticity x window width in log alpha, 20 to 25 (LEARNINGS 32)
            "own_elasticity_times_window_width": (15.0, 30.0),
        }

    # ------------------------------------------------------------------ options and normalisations

    def check_robustness(self, robustness: tuple[str, ...]) -> None:
        if "sargan" in robustness and "fixed_w" in robustness:
            raise ValueError("robustness sargan (the default aggregate block) and fixed_w exclude each other")

    @property
    def base_normalisations(self) -> list[str]:
        """The report's normalisation lines without the margin calibration."""
        return [
            "median a_i = 1 (mu = 0); alpha_i = exp(phi_g) (ybar / y_i) a_i with income winsorised to [1, 150]: "
            "log alpha_i = log(ybar / clip(y_i, 1, 150)) + phi_mid mid_i + phi_high high_i + log a_i, a_i ~ "
            "LogNormal(0, sigma_p) on the untruncated parent, ybar the pooled low-group median income (Petrin's "
            "income law, LEARNINGS 11, 32)",
            self.a_lo_normalisation(self.a_lo) + (f" (robustness {self.a_lo_switch})" if self.a_lo_switch else ""),
            f"taste spread on the family-hauler factor fam = max(mi, sw, su, pv) capped at {FAM_CAP:g} ($1000), "
            f"so under 1% of consumers value it beyond the pooled price range {self.price_range:.1f}",
            "taste spreads on the body dummies mi, sw, su, pv fixed at 0 (only second-choice data would pin "
            "them, LEARNINGS 14)" if "body_sigma" not in self.robustness else
            f"taste spreads on the body dummies mi, sw, su, pv free and capped at {BODY_SIGMA_CAP:g} ($1000) "
            f"(robustness body_sigma)",
            "taste spreads non-negative (draws are symmetric)",
            "u_i0 = 0 in every year, with pgnp, trend and trend2 in X1 (harmless for the 1984 counterfactual)",
            "PCM taste draws are scrambled Sobol standard normals crossed with pyblp's CEX agents; pyblp's nodes "
            "are not used (owner decision 3)",
        ]

    @property
    def normalisations(self) -> list[str]:
        return self.base_normalisations + ([self.margin_normalisation(self.margin)] if self.margin is not None else [])

    def problem_normalisations(self, moments: str, margin: float | None = None) -> list[str]:
        m_star = self.margin if margin is None else float(margin)
        return self.base_normalisations + ([self.margin_normalisation(m_star)]
                                           if m_star is not None and moments in ("all", "supply") else [])

    @staticmethod
    def a_lo_normalisation(a: float) -> str:
        return (f"a_i >= a_lo = {a:g}: the unobserved factor is truncated below at a_lo for every household, so "
                f"alpha_lo,i = a_lo c_i is proportional to 1 / income and the WTP cap to income; a fixed parameter of "
                f"role bound, a calibration of the welfare tail that no moment pins (LEARNINGS 32)")

    def add_paper_arguments(self, parser: Any) -> None:
        parser.add_argument("--band-nash-all", action="store_true",
                            help="sigma_p_band: run the bounded Nash counterfactual at every band point, not only "
                                 "at the headline point (does not change estimates)")
        parser.add_argument("--margin", type=float, default=None,
                            help="calibrated markup level m*: one binding row, trimmed mean (25th-75th percentile) "
                                 "of the Nash margins = m*; absent: no margin row")

    def configure_paper(self, args: Any) -> None:
        self.set_margin(getattr(args, "margin", None))
        self.band_nash_all = bool(getattr(args, "band_nash_all", False))

    def set_margin(self, margin: float | None) -> None:
        """Set the calibrated markup level m* (None: no margin row)."""
        if margin is not None and not 0.0 < float(margin) < 1.0:
            raise ValueError(f"margin must lie in (0, 1), got {margin}")
        self.margin = None if margin is None else float(margin)

    @staticmethod
    def margin_normalisation(m: float) -> str:
        return (f"margin calibration m* = {m:g}: the {100 * MARGIN_TRIM[0]:g}th-{100 * MARGIN_TRIM[1]:g}th percentile "
                f"trimmed mean of the Nash margins (p - mc)/p over all products equals m*, one moment row with W "
                f"entry {MARGIN_BINDING:g} / (n {MARGIN_SD:g}^2) so it binds (LEARNINGS 25)")

    def paper_run_key(self) -> dict[str, Any]:
        """The margin calibration (when set); the robustness switches, the bound a_lo and its kind ``bound: "a"``, the
        start theta and the outer solver come from PaperSpec.run_key."""
        return {"margin": self.margin} if self.margin is not None else {}

    # ------------------------------------------------------------------ parameters and moments

    def base_layout(self) -> ParameterLayout:
        """Nonlinear parameters (SPEC 2, change list 3-6). Body sigmas exist only under ``body_sigma``, so the
        Sobol dimension of the main run is the 7 taste columns."""
        r, p = self.price_range, self.products
        specs = []
        for k, c in enumerate(TASTE):
            name, start = f"sigma_{c}", self.sigma_starts[c]
            if c == "const":
                specs.append(ParameterSpec(name, "taste", c, f"nu{k}", start, 0.0, r, unit=1.0,
                                           note="upper = pooled price range, domain guard"))
            elif c == "fam":
                specs.append(ParameterSpec(name, "taste", c, f"nu{k}", start, 0.0, FAM_CAP, cap=True, unit=1.0,
                                           note="cap = price range / 2.58, under 1% of consumers beyond the "
                                                "price range"))
            else:
                specs.append(ParameterSpec(name, "taste", c, f"nu{k}", start, 0.0, 3 * r / float(np.std(p[c])),
                                           note="upper = 3 x price range per sd, domain guard"))
        if "body_sigma" in self.robustness:
            specs += [ParameterSpec(f"sigma_{b}", "taste", b, f"nu{len(TASTE) + i}", 0.0, 0.0, BODY_SIGMA_CAP,
                                    cap=True, unit=1.0, note="robustness body_sigma; cap normalisation")
                      for i, b in enumerate(BODY)]
        specs += [ParameterSpec(f"pi_{b}", "taste", b, "logfs_fv", PI_STARTS[b], -5.0, 5.0, note="x log(fs) fv")
                  for b in BODY]
        specs += [ParameterSpec("phi_mid", "scale", None, "mid", PHI_STARTS["phi_mid"], -PHI_BOUND, PHI_BOUND,
                                note="log alpha shift for mid income (low = 0); start Petrin's log(pi_mid / pi_low)"),
                  ParameterSpec("phi_high", "scale", None, "high", PHI_STARTS["phi_high"], -PHI_BOUND, PHI_BOUND,
                                note="log alpha shift for high income (low = 0); start Petrin's log(pi_high / pi_low)")]
        specs.append(ParameterSpec("phi_y", "scale", None, "log_ybar_inc_w", 1.0, -1.0, 3.0, free=False,
                                   note="log alpha on log(ybar / clip(income, 1, 150)); fixed at 1, Petrin's income law"))
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, SIGMA_P["start"], SIGMA_P["lower"], SIGMA_P["upper"],
                                   unit=SIGMA_P["unit"], note="sd of log a (untruncated parent); estimated, a_lo is the "
                                                              "calibration axis (LEARNINGS 32)"))
        a = self.a_lo
        specs.append(ParameterSpec("a_lo", "bound", None, None, a, a, a, unit=A_LO_UNIT, free=False,
                                   note="a_i >= a_lo, alpha_lo,i = a_lo c_i; fixed (LEARNINGS 32)"))
        return ParameterLayout(specs)

    def micro_moment(self, which: str) -> RatioMicroMoment:
        """Petrin's CEX micro moments as in estimate_blp.build_micro_moments; ``which`` = "all" or "income".
        The covariance of the moments is Petrin's sample covariance, fixed (SPEC 2, change list 8)."""
        nums, dens, vals, labels, keys = [], [], [], [], []
        if which == "all":
            for demo in ("age", "fs"):
                for b in BODY:
                    nums.append(MicroPart(f"E[{demo}*{b}]", (demo,), (b,)))
                    dens.append(MicroPart(f"E[{b}]", (), (b,)))
                    vals.append(self.values[f"E[{demo} | {b}]"])
                    keys.append(f"E[{demo} | {b}]")
                    labels.append(f"E[{demo}_i | {b}_j]")
        for inc in ("mid", "high"):
            nums.append(MicroPart(f"E[inside*{inc}]", (inc,), ()))
            dens.append(MicroPart(f"E[{inc}]", (inc,), (), outside_value=1.0))
            vals.append(self.values[f"E[new | {inc}]"])
            keys.append(f"E[new | {inc}]")
            labels.append(f"E[1{{j>0}} | {inc}_i]")
        V = self.micro_covariances.loc[keys, keys].to_numpy(float)
        return RatioMicroMoment("micro", MicroDataset("CEX", 29125), nums, dens, np.array(vals), labels, covariance=V)

    # The frontier-event step guard (DESIGN 4.2, 4.4) is off. Products with shares of 1e-5 lie on the
    # frontiers of only one or two draws, so the first event in those draws sends their affected mass to 1
    # and their vanishing bound to the draws' first event time. At the start point the cap was 0.003 of a
    # phi step and 0.014 of the Gauss-Newton step at 1,024 draws, and 5e-13 at 4,096 draws, where 30
    # accepted pre-fit steps left q at 560.673486. Steps are controlled by the trust region and the gain
    # ratio instead; every trial point is re-inverted exactly.
    use_max_step = False
    # Trust radius in natural units (DESIGN 4.3 defaults 0.3 and 5). A 0.6-unit phi step at 4,096 draws sent
    # one market's warm start into the cold fallback for over 30 minutes; smaller boxes keep trial points
    # within reach of the warm start and the theta path.
    # Round 3 (analysis 4 (ii)-(iii)): the noise floor from +-1e-5-unit probes at every stage start and the
    # CONVERGED_NOISE stop on it, and Levenberg-Marquardt damping on rejections with rho < 0.
    # Round 4 (SPEC_ANALYSIS v4 section 4, change 9): More scaling of the trust region, the Cauchy decrease in the
    # stopping test, no radius shrink on sub-noise rejections, CONVERGED_NOISE when the predicted decrease at the
    # current radius is below 10 x the floor, after a terminal probe at +-0.01, 0.03, 0.1 units.
    optimizer_options = {"Delta_0": 0.1, "Delta_max": 0.5, "noise_probe": 1e-5, "noise_Delta": 1e-4,
                         "noise_rejections": 3, "lm_damping": True, "more_scaling": True, "cauchy_stop": True,
                         "noise_converge": True}

    def pcm_spec(self) -> PCMSpec:
        # At a trial phi step from the pre-fit's second point, markets 1990-1993 failed the warm stage (2,500
        # evaluations, 60-73 s), the theta path (stopped at tau 0.25-0.5) and then a 40,000-evaluation cold
        # stage (860-1,380 s): 26 minutes to learn that the step must be rejected. Trial points therefore get
        # no cold stage; the first evaluation and the cold profile points keep the full budget.
        return PCMSpec(KERNEL_CHARS, track_realised_mass=self.use_max_step, trial_cold_evals=0, bound_kind="a")

    def payload(self) -> dict[str, Any]:
        return {"data": "pyblp 1.3.0 Petrin, prices spread to a $10 gap", "chars": CHARS, "taste": TASTE,
                "kernel_chars": KERNEL_CHARS, "x1": X1_COLS,
                "merge": {"rule": "same TWIN_KEYS: keep first row, sum shares and q", "keys": TWIN_KEYS},
                "instruments": "petrin", "z": "X1 + demand_instruments0..21",
                "supply": {"cost_basis": COST_BASIS, "z_s": f"cost basis + supply_instruments0..{N_SUPPLY_INSTRUMENTS - 1}",
                           "markups": "Nash, ownership firm_ids, model ds/dp, data shares",
                           "mc_guard": f"NUMERICAL if rcond < {NASH_RCOND_MIN:g} or any mc <= 0",
                           "gamma": "concentrated, IV with the supply block of W", "jacobian": "central FD at fixed "
                           f"inversions, h = {SUPPLY_FD_STEP:g} natural units, no shrink, frontier-change count kept "
                           "as a diagnostic (SupplyMoments fd_scale='unit')",
                           "moments": "all = aggregate + micro + supply (+ margin)"},
                "aggregate": "IVMoments (fixed W)" if "fixed_w" in self.robustness else
                             "SarganIVMoments (scale-free); W2 block diagonal across aggregate and supply",
                "margin": {"moment": f"trimmed mean over products of (p - mc)/p between percentiles {MARGIN_TRIM} "
                                     "minus m*, same Nash markups and FD pass as the supply block",
                           "target": self.margin, "sd": MARGIN_SD, "binding": MARGIN_BINDING,
                           "band": list(MARGIN_BAND), "headline": MARGIN_HEADLINE},
                "start": {"from_gate": self.start_from_gate, "slice": self.start_slice, "column": self.start_column,
                          "concentrate": list(self.start_concentrate),
                          "weighting": f"first {BODY_MICRO_ROWS} micro rows x 1e4, calibration rows 0"},
                "convergence": "CONVERGED_NOISE: 3 rejections with Delta < 1e-4 units and |q_trial - q| below the "
                               "max block change over +-1e-5-unit probes at the stage start, or a terminal probe "
                               "(+-0.01, 0.03, 0.1 units along Gauss-Newton and each coordinate) finding no decrease "
                               "above 10 x the floor once the predicted decrease at the current radius is below 10 x "
                               "the floor or a rejection changed q by less than the floor; More-scaled trust region, "
                               "Cauchy decrease in the stopping test; counts as converged",
                "phi_by_income_moments": "income rows x 1e4 in prefit and gate concentration",
                "mc_floor": MC_FLOOR,
                "layout": [dataclasses.asdict(s) for s in self.layout().specs],
                "micro": {"values": self.values, "observations": 29125, "covariance": MICRO_COVARIANCE_SOURCE},
                "caps": {"sigma_fam": FAM_CAP, "body_sigma": BODY_SIGMA_CAP, "income_clip": INCOME_CLIP},
                "robustness": list(self.robustness), "pcm": dataclasses.asdict(self.pcm_spec()),
                "a_lo": {"value": self.a_lo, "role": "bound", "free": False, "kind": "a",
                         "distribution": "a_i lognormal(0, sigma_p) truncated below at a_lo for every household"},
                "income_law": {"phi_y": 1.0, "ybar": "pooled low-group median income", "clip": INCOME_CLIP},
                "start_theta": self.start_theta_source,
                "a_lo_band": {"values": list(self.a_lo_band_values), "headline": A_LO_HEADLINE},
                "draws": "Sobol normals, seed + market position, crossed with agents", "max_step": self.use_max_step,
                "optimizer": self.optimizer_options,
                "prefit": {"free": self.prefit_free, "max_iterations": self.prefit_max_iterations},
                "extensive_free": self.extensive_free, "gate_concentrate_iterations": self.gate_concentrate_iterations,
                "sig_ref": {k: float(v).hex() for k, v in self.sig_ref.items()},
                "gate": {"lambdas": list(self.gate_lambdas), "sigma_p": list(self.gate_sigma_p), "slice": GATE_SLICE,
                         "required": list(self.gate_required), "rules": "q_fixed NotMonotoneDecreasing, "
                         "q_fixed:supply RisesAtBothEnds, rise 3.84"},
                "band": {"parameter": self.band_parameter, "values": list(self.band_values),
                         "moments": self.band_moments,
                         "headline": f"median margin nearest {BAND_MARGIN_TARGET:g} (calibration)"},
                "plausible": self.plausible}

    def moment_list(self, moments: str, margin: float | None = None) -> list[Any]:
        """``moments``: all = aggregate + micro + supply (+ the margin row of m* = ``margin`` or --margin), demand =
        aggregate + micro, prefit = aggregate + the two income rows, supply = aggregate + supply (+ margin), aggregate."""
        iv = IVMoments if "fixed_w" in self.robustness else SarganIVMoments   # LEARNINGS 24, 25
        ms: list[Any] = [iv(self.Z, self.clusters)]
        if moments in ("all", "demand", "prefit"):
            ms.append(self.micro_moment("income" if moments == "prefit" else "all"))
        m_star = self.margin if margin is None else float(margin)
        if moments in ("all", "supply"):
            sup = self.supply_moment()
            ms.append(sup)
            if m_star is not None:
                ms.append(MarginMoment(sup, m_star, trim=MARGIN_TRIM, sd=MARGIN_SD, binding_weight=MARGIN_BINDING))
        return ms

    def supply_moment(self) -> SupplyMoments:
        """Petrin's log-cost supply block: Nash markups from the PCM's own ds/dp with the ownership of firm_ids,
        log mc = w gamma + omega, Z_s = [w, supply_instruments0..15], clustered like the demand moments (round-2
        analysis, change 2)."""
        return SupplyMoments(self.Z_supply, self.cost_basis, self.products[self.firm_column].to_numpy(),
                             self.clusters, name="supply", w_names=[f"gamma[{c}]" for c in COST_BASIS],
                             rcond_min=NASH_RCOND_MIN, mc_floor=MC_FLOOR, fd_step=SUPPLY_FD_STEP, fd_scale="unit")

    # ------------------------------------------------------------------ gate (DESIGN 6.4)

    gate_lambdas: tuple[float, ...] = GATE_LAMBDAS
    gate_sigma_p: tuple[float, ...] = GATE_SIGMA_P
    gate_name = "sigma_ray"

    def gate_profiles(self) -> list[tuple[str, str, dict[str, Any]]]:
        """The sigma ray profile on all moments (aggregate, micro, supply), concentrated in (phi_mid, phi_high):
        at every grid point the two income shifts are re-solved with everything else held, so the income
        purchase rates do not drift with sigma_p (LEARNINGS 19). Two verdicts on the lambda = 1 slice (round-2
        analysis, change 5): ``q_fixed`` must not be monotone decreasing in sigma_p, and its supply block
        ``q_fixed:supply`` must rise by at least 3.84 from its minimum to both ends of the grid. step1 needs one of
        them (``gate_required``); with both failed, the pipeline runs the ``sigma_p_band`` fallback (change 8)."""
        econ = ["sd_xi", "median_own_elasticity", "own_elasticity_times_sigma_p", "margin_median",
                "share_mc_nonpositive", "nash_rcond_min", "nonzero_cross_median",
                "minivan_diversion_haulers_share_of_inside", "car_to_hauler_diversion_over_hauler_inside_share",
                "relative_quality_dispersion_buyer_dollars_median", "inside_share_mean", "inside_share_semi_elasticity",
                "phi_mid", "phi_high", "mean_alpha_buyers", "sd_delta"]
        q_cols = ["q_fixed", "q_cue", "q_fixed:aggregate", "q_fixed:micro", "q_fixed:supply", "q_cue:aggregate"]
        rules = {
            "q_fixed": {"rule": NotMonotoneDecreasing(
                "sigma_p", by=("lambda",), column="q_fixed", rise=3.84, only=GATE_SLICE,
                fail_message="Profile check failed: q_fixed (aggregate + micro + supply) is still monotone "
                             "decreasing in sigma_p on the lambda = 1 slice."),
                "columns": q_cols + econ + ["E[1{j>0}|mid_i]", "E[1{j>0}|high_i]"]},
            "supply": {"rule": RisesAtBothEnds(
                "sigma_p", by=("lambda",), column="q_fixed:supply", rise=3.84, only=GATE_SLICE,
                fail_message="Profile check failed: the supply block does not rise by 3.84 at both ends of sigma_p "
                             "on the lambda = 1 slice, so it does not pin sigma_p."),
                "columns": ["q_fixed:supply", "q_fixed"] + econ},
        }
        kw = {"direction": self.sig_ref, "lambdas": list(self.gate_lambdas),
              "grid": {"sigma_p": list(self.gate_sigma_p)}, "origin": 0.0,
              "concentrate": ("phi_mid", "phi_high"), "economics": self.gate_economics, "rules": rules}
        return [(self.gate_name, "all", kw)]

    gate_required = ("q_fixed", "supply")   # any of them; both failed means the margin_band fallback
    fallback_stage = "margin_band"
    band_parameter = "sigma_p"
    band_values = BAND_SIGMA_P
    band_moments = "all"   # Sargan + micro + supply (LEARNINGS 27, SPEC_ANALYSIS v4 change 8)
    a_lo_band_parameter = "a_lo"
    a_lo_band_values = A_LO_BAND
    a_lo_band_moments = "all"

    def band_row(self, pipe: Any, problem: Any, result: Any, value: float) -> dict[str, Any]:
        """One row of band.csv at the band estimate with sigma_p fixed at ``value``: elasticities, margins, the
        market semi-elasticity (inside-share elasticity to a uniform 1% price rise, mean over markets), the 1984
        minivan gain at fixed and Nash prices, diversion to haulers, q by block and the estimate."""
        model, theta = problem.model, np.asarray(result.theta, dtype=float)
        ev = result.evaluation
        delta = np.asarray(ev.aux["delta"])
        post = self.postestimation(model, theta, delta)
        summ = comparison.summarise(post["products"], post["group_matrix"], post["group_names"])
        own = np.asarray(post["products"]["own_elasticity"], dtype=float)
        sigma_p = float(theta[problem.layout.names.index("sigma_p")])
        row: dict[str, Any] = {
            "label": "",   # the headline is chosen once every point is in (band_finish)
            "own_elasticity_median": summ["own_elasticity_median"],
            "own_elasticity_times_sigma_p": -summ["own_elasticity_median"] * sigma_p,
            "share_own_elasticity_below_-20": float(np.mean(own[np.isfinite(own)] < -20.0)),
            "margin_median": summ["margin_median"],
            "inside_share_semi_elasticity": float(np.mean([m["inside_share_elasticity"] for m in post["markets"]])),
            "nonzero_cross_median": summ["nonzero_cross_median"],
        }
        row.update({k: v for k, v in self.diversion_stats(summ["group_diversion"]).items()
                    if k in BAND_COLUMNS})
        cf, _ = self.counterfactual(model, theta, delta, resolve_prices=False)
        row.update({k: cf.get(k) for k in ("minivan_cs_gain_1984_fixed_prices_musd",)})
        names = problem.layout.names
        for n in ("phi_mid", "phi_high", "sigma_p", "a_lo"):
            if n in names:
                row[n] = float(theta[names.index(n)])
        row["mean_alpha_buyers"] = float(pp.spread_in_buyer_dollars(model, theta, delta)["mean_alpha_buyers"])
        wa = pp.window_anatomy(model, theta, delta)
        row.update({k: wa[k] for k in ANATOMY_COLUMNS})
        row.update({f"{k}_1984": wa["by_market"][str(YEAR)][k] for k in ANATOMY_COLUMNS if str(YEAR) in wa["by_market"]})
        row["own_elasticity_times_window_width"] = -summ["own_elasticity_median"] * wa["window_width_log_alpha"]
        row.update(pipeline.estimate_columns(problem, result))
        return row

    def bounded_nash_at(self, problem: Any) -> Any:
        """The ``Pipeline.at_band_points`` callback adding the bounded Nash counterfactual to a band row."""
        def fn(r: dict[str, Any], res: Any) -> None:
            if res is None:
                r["bounded_nash_status_1984"] = "estimate not re-evaluated"
                return
            cf, _ = self.counterfactual(problem.model, np.asarray(res.theta, dtype=float),
                                        np.asarray(res.evaluation.aux["delta"]), resolve_prices=True)
            r.update({k: v for k, v in cf.items() if "bounded_nash" in k or k == "nash_note_1984"})
        return fn

    def a_lo_band_finish(self, pipe: Any, problem: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """After every a_lo point: label a_lo = A_LO_HEADLINE the headline (a calibration of the welfare tail, LEARNINGS
        32) unless a free parameter ended AT_BOUND there, and run the bounded Nash counterfactual at it (at every point
        with --band-nash-all)."""
        for r in rows:
            r["label"] = ""
        head = next((r for r in rows if abs(float(r.get("a_lo", float("nan"))) - A_LO_HEADLINE) < 1e-12), None)
        if head is not None and head.get("at_bound"):
            head["label"] = f"not a headline: {head['at_bound']} AT_BOUND (a failed estimate)"
            head = None
        elif head is not None:
            head["label"] = f"headline (calibration of the welfare tail: a_lo = {A_LO_HEADLINE:g})"
        targets = [r for r in rows if not r.get("at_bound")] if self.band_nash_all else ([head] if head else [])
        pipe.at_band_points(problem, targets, "a_lo_band", self.bounded_nash_at(problem))
        return rows

    def band_finish(self, pipe: Any, problem: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """After every band point: label the headline row, the point whose median margin is nearest
        BAND_MARGIN_TARGET (a calibration of the markup level, not an estimate of sigma_p), and run the bounded Nash
        counterfactual at it (at every point with --band-nash-all)."""
        ok = [r for r in rows if np.isfinite(float(r.get("margin_median", float("nan")))) and not r.get("at_bound")]
        if not ok:
            return rows
        head = min(ok, key=lambda r: abs(float(r["margin_median"]) - BAND_MARGIN_TARGET))
        for r in rows:
            r["label"] = ""
        head["label"] = (f"headline (calibration: median margin {float(head['margin_median']):.3f} nearest "
                         f"{BAND_MARGIN_TARGET:g}, sigma_p = {float(head['sigma_p']):g})")
        pipe.at_band_points(problem, ok if self.band_nash_all else [head], "sigma_p_band", self.bounded_nash_at(problem))
        return rows

    def margin_band_row(self, pipe: Any, problem: Any, result: Any, value: float) -> dict[str, Any]:
        """One row of margin_band/band.csv at m* = ``value``: the ``band_row`` statistics (elasticities, margins,
        market semi-elasticity, minivan diversion to haulers, 1984 minivan CS gain at fixed and Nash prices, q by
        block, theta), plus sigma_p, the taste spreads as multiples of the reference direction
        (``lambda:<spread>``, 1 = the gate's lambda = 1 ray), mean alpha of buyers, sd(delta) and the margin row."""
        row = self.band_row(pipe, problem, result, float("nan"))
        row["label"] = f"headline (margin calibration m* = {MARGIN_HEADLINE:g})" if value == MARGIN_HEADLINE else ""
        theta = np.asarray(result.theta, dtype=float)
        names = problem.layout.names
        row["sigma_p"] = float(theta[names.index("sigma_p")])
        lam = {n: float(theta[names.index(n)]) / v for n, v in self.sig_ref.items() if n in names and v > 0}
        row.update({f"lambda:{n}": x for n, x in lam.items()})
        row["lambda_median"] = float(np.median(list(lam.values()))) if lam else float("nan")
        ev = result.evaluation
        delta = np.asarray(ev.aux["delta"])
        row["mean_alpha_buyers"] = float(pp.spread_in_buyer_dollars(problem.model, theta, delta)["mean_alpha_buyers"])
        row["sd_delta"] = float(np.std(delta))
        row["sd_xi"] = float(ev.aux.get("sd_xi", float("nan")))
        mg = ev.aux.get("margin") or {}
        row.update({k: mg[k] for k in ("margin_trimmed_mean", "margin_moment") if k in mg})
        sup = ev.aux.get("supply") or {}
        row.update({k: sup[k] for k in ("share_mc_clipped", "fd_kink_columns") if k in sup})
        return row

    def band_reference_rows(self) -> list[dict[str, Any]]:
        """BLP rows for band.csv from petrin_replication/results/blp_*_summary.json (columns they carry)."""
        rows = []
        for label, d in comparison.load_summaries({k: HERE / "results" / f for k, f in BLP_SUMMARIES.items()}).items():
            if d.get("group_diversion"):
                d.update(self.diversion_stats(d["group_diversion"]))
            ref = {k: d.get(k, "") for k in BAND_COLUMNS
                   if k not in ("own_elasticity_times_sigma_p", "own_elasticity_times_window_width",
                                "share_own_elasticity_below_-20", "inside_share_semi_elasticity")}
            # BLP's Nash gain is an equilibrium; the PCM column beside it is the bounded Nash approximation
            ref["minivan_cs_gain_1984_bounded_nash_musd"] = d.get("minivan_cs_gain_1984_musd", "")
            ref["bounded_nash_status_1984"] = f"BLP Nash {d.get('nash_status_1984', '')}".strip()
            rows.append({"label": label, **ref})
        return rows

    def gate_economics(self, problem: Any, ev: Any) -> dict[str, float]:
        """Per profile point: the aggregate-only CUE objective, elasticities and the law monitor
        -median own elasticity x sigma_p (about 25 to 55 in the exact PCM on this product set, LEARNINGS 21),
        Nash margins, the supply guard's mc and rcond readings, diversion to family haulers, the relative
        quality dispersion, inside shares, the inside-share elasticity to a uniform 1% price rise (mean over
        markets) and the re-solved phi."""
        post = self.postestimation(problem.model, ev.theta, ev.aux["delta"])
        out = pp.point_economics(problem, ev, post)
        names = problem.layout.names
        gd = pp.group_diversion_table(post)
        out.update({k: v for k, v in self.diversion_stats(gd).items()
                    if k in ("minivan_diversion_haulers_share_of_inside",
                             "car_to_hauler_diversion_over_hauler_inside_share")})
        for n in ("phi_mid", "phi_high"):
            out[n] = float(np.asarray(ev.theta)[names.index(n)])
        pred = ev.aux.get("micro_predicted", {}).get("micro")
        micro = next((m for m in problem.moments if m.name == "micro"), None)
        if pred is not None and micro is not None:
            for lab, v in zip(micro.labels, pred):
                if "1{j>0}" in lab:
                    out[lab.replace(" ", "")] = float(v)
        return out

    def diversion_stats(self, group_diversion: dict[str, dict[str, float]]) -> dict[str, float]:
        """The minivan row (``minivan_diversion``) and ``car_to_hauler_diversion_over_hauler_inside_share``: the
        share of a car's inside diversion that goes to mi, sw, su and pv, divided by those body types' share of
        inside sales (pooled shares). IIA-like substitution gives about 1 (BLP micro 0.73, no micro 0.74, demand micro
        0.70, from their summaries); the exact PCM's local
        substitution about 0.05 to 0.15 (round-2 analysis, change 6)."""
        out = minivan_diversion(group_diversion)
        car = group_diversion.get(self.group_other)
        if car:
            inside = sum(float(v) for k, v in car.items() if k != "outside")
            to_h = sum(float(car.get(b, 0.0)) for b in HAULERS)
            out["car_to_hauler_diversion_over_hauler_inside_share"] = (
                to_h / inside / self.hauler_inside_share if inside > 0 else float("nan"))
        return out

    def extra_economics(self, post: dict[str, Any], values: dict[str, Any]) -> dict[str, float]:
        """Spec statistics for the economics block (``diagnostics.economics``): the diversion ratios above."""
        gd = values.get("group_diversion") or {}
        return {k: v for k, v in self.diversion_stats(gd).items()
                if k in ("minivan_diversion_haulers_share_of_inside", "car_to_hauler_diversion_over_hauler_inside_share")}

    # ------------------------------------------------------------------ postestimation (DESIGN 6.5)

    def counterfactual(self, model: Any, theta: np.ndarray, delta: np.ndarray, resolve_prices: bool
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
        """1984 without minivans (Petrin's headline): CS change at fixed prices (the headline, LEARNINGS 29) and,
        if asked, the bounded Nash approximation: best responses of every firm in a +-10% price box at the FOC
        costs (``pcm_post.best_response_box``). Gains in $M = households (thousands) x CS ($1000)."""
        mi = self.products.mi.to_numpy() == 1
        cf = pp.drop_products_counterfactual(model, theta, delta, YEAR, mi, firm_ids=self.firm_column,
                                             resolve_prices=resolve_prices, box=NASH_BOX, max_sweeps=NASH_SWEEPS,
                                             deviation=NASH_DEVIATION)
        hh = float(self.products.households[self.products.market_ids == YEAR].iloc[0])
        out: dict[str, Any] = {
            "minivan_cs_gain_1984_fixed_prices_musd": hh * (cf["observed"]["cs"] - cf["fixed_prices"]["cs"]),
            "inside_share_1984_observed": cf["observed"]["inside_share"],
            "inside_share_1984_fixed_prices": cf["fixed_prices"]["inside_share"],
            "cs_1984_observed_per_household": cf["observed"]["cs"],
        }
        per = {"observed": cf["observed"]["per_draw"], "fixed_prices": cf["fixed_prices"]["per_draw"]}
        nash = cf["nash"]
        if nash is not None:
            out["bounded_nash_status_1984"] = nash["status"]
            if nash.get("per_draw") is not None:
                gain = hh * (cf["observed"]["cs"] - nash["cs"])
                dev = nash["deviation"]
                out.update(minivan_cs_gain_1984_bounded_nash_musd=gain, minivan_cs_gain_1984_musd=gain,
                           inside_share_1984_bounded_nash=nash["inside_share"],
                           bounded_nash_sweeps_1984=nash["sweeps"], bounded_nash_evaluations_1984=nash["evaluations"],
                           bounded_nash_box_binding_1984=nash["box_binding"]["count"],
                           bounded_nash_max_deviation_gain_1984=dev["max_gain"],
                           bounded_nash_profitable_deviations_1984=dev["profitable"],
                           bounded_nash_max_deviation_gain_off_box_1984=dev["max_gain_off_box"])
                if "cs_range" in nash:
                    lo, hi = nash["cs_range"]
                    out["minivan_cs_gain_1984_bounded_nash_range_musd"] = [hh * (cf["observed"]["cs"] - hi),
                                                                           hh * (cf["observed"]["cs"] - lo)]
                per["nash"] = nash["per_draw"]
                out["nash_note_1984"] = (
                    f"Bounded Nash: best responses of each firm within +-{100 * NASH_BOX:g}% of observed prices at the "
                    f"FOC costs, ended '{nash['status']}' after {nash['sweeps']} sweeps; {nash['box_binding']['count']} "
                    f"prices at a box end; largest profitable unilateral deviation of +-{100 * NASH_DEVIATION:g}% "
                    f"raises a firm's profit by {100 * dev['max_gain']:.3g}%"
                    + (f"; CS gain over {'the cycle' if nash['status'] == 'cycle' else 'the last 5 sweeps'} "
                       f"{out['minivan_cs_gain_1984_bounded_nash_range_musd'][0]:,.0f} to "
                       f"{out['minivan_cs_gain_1984_bounded_nash_range_musd'][1]:,.0f} $M"
                       if "cs_range" in nash else "") + ". It is an approximation, not an equilibrium (LEARNINGS 29).")
            else:
                out["nash_note_1984"] = (f"Bounded Nash not run: {nash['status']} at the observed prices, so there are "
                                         f"no FOC costs.")
        t = pp.market_position(model, YEAR)
        w = np.asarray(model.kernel_market(t).weights)
        tables = {}
        for name, col in (("income", "income_group"), ("fs", "fs_group")):
            rows = comparison.cs_gain_by_group(per, w, pp.draw_labels(model, t, self.agents[col].to_numpy()), hh)
            tables[name] = rows
            for r in rows:
                if "gain_nash_musd" in r:
                    out[f"cs_gain_1984_{name}_{r['group']}_musd"] = r["gain_nash_musd"]
                out[f"cs_gain_1984_fixed_prices_{name}_{r['group']}_musd"] = r["gain_fixed_prices_musd"]
        return out, {"tables": tables, "raw": cf}

    def post(self, pipe: Any, problem: Any, result: Any, out: pathlib.Path) -> dict[str, Any]:
        model, theta, delta = problem.model, np.asarray(result.theta), np.asarray(result.evaluation.aux["delta"])
        post = self.postestimation(model, theta, delta)
        prod = pd.DataFrame({k: v for k, v in post["products"].items()})
        prod.insert(0, "row", np.arange(len(prod)))
        summary = comparison.summarise(post["products"], post["group_matrix"], post["group_names"])
        cf, extra = self.counterfactual(model, theta, delta, resolve_prices=True)
        summary.update(cf)
        summary["inside_share_by_market"] = {str(m["market_id"]): m["inside_share"] for m in post["markets"]}
        summary["nash_singular_markets"] = [str(m) for m in post["nash_singular"]]
        sb = pp.spread_in_buyer_dollars(model, theta, delta)
        summary["mean_alpha_buyers"] = sb["mean_alpha_buyers"]
        summary["spread_buyer_dollars"] = sb["spreads"]
        qd = pp.quality_dispersion(model, theta, delta)
        summary["quality_dispersion_buyer_dollars_median"] = qd["median"]
        prod["quality_dispersion_buyer_dollars"] = qd["per_product"]
        rq = pp.relative_quality_dispersion(model, theta, delta)
        summary["relative_quality_dispersion_buyer_dollars_median"] = rq["median"]
        prod["relative_quality_dispersion_buyer_dollars"] = rq["per_product"]
        prod.to_csv(out / "product_level.csv", index=False)
        summary["inside_share_semi_elasticity"] = float(np.mean([m["inside_share_elasticity"]
                                                                 for m in post["markets"]]))
        summary.update(self.diversion_stats(summary["group_diversion"]))
        summary["own_elasticity_times_sigma_p"] = -summary["own_elasticity_median"] * float(
            theta[problem.layout.names.index("sigma_p")])
        wa = pp.window_anatomy(model, theta, delta)
        summary.update({k: wa[k] for k in ANATOMY_COLUMNS})
        summary["own_elasticity_times_window_width"] = -summary["own_elasticity_median"] * wa["window_width_log_alpha"]
        summary["cs_tables_1984"] = extra["tables"]
        for name, rows in extra["tables"].items():
            pd.DataFrame(rows).to_csv(out / f"cs_1984_by_{name}.csv", index=False)
        blp = comparison.load_summaries({k: HERE / "results" / f for k, f in BLP_SUMMARIES.items()})
        note = ("BLP micro and BLP no micro estimate a log-cost supply side jointly; BLP demand micro uses demand "
                "moments only. The PCM uses aggregate + micro + supply moments and recovers costs from the Nash FOC. "
                f"PCM estimate: stage {result.stage}, {result.convergence.value}, {pipe.cfg.nu} draws per agent. "
                "The headline counterfactual is the fixed-price CS gain; the PCM 'Nash' row is the bounded Nash "
                f"approximation, status {summary.get('bounded_nash_status_1984')}. {summary.get('nash_note_1984', '')}")
        rows = {k: v for k, v in comparison.scalar_items(summary).items() if k != "nash_note_1984"}
        table = pp.compare_blp(rows, {k: comparison.scalar_items(v) for k, v in blp.items()}, note=note)
        gd = ["", "Diversion by body type (share-weighted mean, rows: from, columns: to):"]
        for src, dst in summary["group_diversion"].items():
            gd.append(f"  PCM {src:>4}: " + "  ".join(f"{k} {v:.3f}" for k, v in dst.items()))
        for v in blp.values():
            if v.get("group_diversion"):
                v.update(self.diversion_stats(v["group_diversion"]))
        for label, v in blp.items():
            for src, dst in (v.get("group_diversion") or {}).items():
                gd.append(f"  {label} {src:>4}: " + "  ".join(f"{k} {x:.3f}" for k, x in dst.items()))
        gd += ["", "Minivan diversion (share-weighted mean over minivans): to cars, to sw+su+pv, to outside, "
                   "and sw+su+pv as a share of inside diversion:"]
        for label, v in [("PCM", summary)] + list(blp.items()):
            if "minivan_diversion_to_haulers" in v:
                gd.append(f"  {label:>16}: car {v['minivan_diversion_to_car']:.3f}  sw+su+pv "
                          f"{v['minivan_diversion_to_haulers']:.3f}  outside {v['minivan_diversion_to_outside']:.3f}  "
                          f"haulers/inside {v['minivan_diversion_haulers_share_of_inside']:.3f}")
        (out / "comparison.txt").write_text(table + "\n" + "\n".join(gd) + "\n")
        pipeline.log.info("comparison with BLP:\n%s", table)
        return summary

    def headlines(self, problem: Any, result: Any) -> dict[str, float]:
        """Median own-price elasticity and the fixed-price 1984 minivan CS gain, for the integration check."""
        model, theta, delta = problem.model, np.asarray(result.theta), np.asarray(result.evaluation.aux["delta"])
        post = self.postestimation(model, theta, delta)
        cf, _ = self.counterfactual(model, theta, delta, resolve_prices=False)
        return {"own_elasticity_median": float(np.nanmedian(post["products"]["own_elasticity"])),
                "minivan_cs_gain_1984_fixed_prices_musd": cf["minivan_cs_gain_1984_fixed_prices_musd"]}


def cost_basis(products: pd.DataFrame) -> np.ndarray:
    """(n, 12) Petrin's log-cost basis in COST_BASIS order."""
    p = products
    tr = p.trend.to_numpy(float)
    cols = {"1": np.ones(len(p)), "log(hpwt)": np.log(p.hpwt.to_numpy(float)), "log(wt)": np.log(p.wt.to_numpy(float)),
            "log(mpg)": np.log(p.mpg.to_numpy(float)), "air": p.air.to_numpy(float), "fwd": p.fwd.to_numpy(float),
            "trend": tr, "jp": p.jp.to_numpy(float), "eu": p.eu.to_numpy(float), "jp*trend": p.jp.to_numpy(float) * tr,
            "eu*trend": p.eu.to_numpy(float) * tr, "log(q)": np.log(p.q.to_numpy(float))}
    return np.column_stack([cols[c] for c in COST_BASIS])


def minivan_diversion(group_diversion: dict[str, dict[str, float]]) -> dict[str, float]:
    """Minivan row of a body-type diversion table: to cars, to sw+su+pv, to the outside good, and sw+su+pv as
    a share of inside diversion (SPEC 4: at least 0.15)."""
    row = group_diversion.get("mi")
    if not row:
        return {}
    haulers = sum(float(row.get(b, 0.0)) for b in ("sw", "su", "pv"))
    inside = sum(float(v) for k, v in row.items() if k != "outside")
    return {"minivan_diversion_to_car": float(row.get("car", float("nan"))),
            "minivan_diversion_to_haulers": haulers,
            "minivan_diversion_to_outside": float(row.get("outside", float("nan"))),
            "minivan_diversion_haulers_share_of_inside": haulers / inside if inside > 0 else float("nan")}


if __name__ == "__main__":
    sys.exit(pipeline.main(PetrinSpec(), default_nu=16, default_base=RESULTS))

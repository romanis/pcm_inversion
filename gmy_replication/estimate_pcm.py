"""Grieco, Murry & Yurukoglu (2024, QJE) US automobiles, exact pure characteristics model (PCM), as a spec on
nfp_harness (GMY exact-PCM spec, docs/estimation_harness; mirrors petrin_replication/estimate_pcm.py).

    u_ijt = delta_jt + x_jt . t_i - alpha_i p_jt,   u_i0t = 0,   p in $10k of 2015
    x     = [1, van, suv, truck, z log footprint, z log hp, z log mpg, lux, sport, electric, EuroBrand, USBrand]
    t_ik  = sigma_k nu_ik on the 11 GMY random-coefficient columns
            + sigma_k nu_ik on z log height, z log curb weight, z log number of trims             (PCM extension, D7)
            + pi (van, z log footprint) x (fs2, fs34, fs5) + pi truck x rural + pi const x inc   (GMY's demographics)
            + pi (z log footprint, lux) x age                                (round 4: intensive-margin age taste)
    log alpha_i = log c_i^GMY + phi_inc b_i + phi_age age_i + log a_i, b_i = inc_i or log(income_i / mean)  (D5)
    c_i^GMY = |alpha_bar + th_inc inc + th_inc2 inc2 + th_age age| / median, from the BLP fit (fixed, coefficient 1)
    --income-law petrin_bounded: log c_i^GMY is replaced by Petrin's bounded income law log(ybar / clip(y_i, 1, 150)),
    y_i in $1000 of 2015, ybar the pooled median (fixed, coefficient 1; LEARNINGS 32)
    a_i ~ LogNormal(0, sigma_p) truncated below at a_lo = 0.2 for every household                (LEARNINGS 32)
    delta_jt = X1_jt beta + xi_jt,  X1 = [GMY's 13 second-stage characteristics, year and make2 dummies]
    Z     = [X1, lag_pl_con, iv_prod, steel_weight, IV_N, IV_N_weight]  (D4; --instruments gh / gmy+gh for Gandhi-Houde)

Objective (``--moments gmy``, the default, D1): GMY's structure. The 33 CEX/MRI micro moments of spec 6 (differences
of household-averaged conditional means and purchase-rate ratios, ExjCondH.m; never-buyers dropped, D2) with GMY's
weight diag(1 / variance); the 41 first/second-choice correlation moments (11 characteristics x 1991, 1999, 2005,
2015, less electric before 2015) from the PCM's exact second choices, weight 30 per row (D3, D6); and one aggregate
block on xi with GMY's cost shifters in Sargan form (scale-free, LEARNINGS 24-26), clustered by model. No Nash supply
block: Nash markups (ownership by parent) and the Lerner index are post-estimation only, as makeMarkups.m.
``--moments gmy_nolinear`` drops the aggregate block (GMY's nonlinear stage exactly; beta from the linear fit),
``--moments demand_supply`` adds the log-cost supply block (robustness only). 27 free nonlinear parameters: 14 sigma,
10 pi (2 of them the round-4 age tastes), phi_inc, phi_age, sigma_p (phi_gmy = 1 and a_lo fixed).

Deviations from GMY beyond D1-D6 (rounds 1 and 2, GMY_ROUND1 and GMY_ROUND2): D7, a PCM extension: taste spreads on
three of GMY's own X1 columns outside the second-choice set (z log height, z log curb weight, z log number of trims),
started at 0.5 ($5k per sd; 1.0 in round 1, where they put half the buyer mass on single-product frontiers and gave
8% of products |own elasticity| < 1). The exact PCM has no per-product logit draw, so the same taste vector picks the
first and the second choice and the second choices sort on the 11 measured characteristics at full strength (van
retention 0.95 against 0.75 in the 1999 data); taste dimensions the moments do not condition on dilute that sorting.
Twins are merged on these columns too. ``--robustness gmy_spreads_only`` drops them. D8 (round 1, undone in round 2):
GMY's constant x income taste pi_const_inc is free again (start: BLP value / median |alpha|). The purchase-rate rows
need a shifter of the buy margin that leaves the price paid alone; alpha (phi_inc) moves both, so without
pi_const_inc phi_inc deepened to reach the purchase-rate ratios and overshot the income-price gradient (prefit q 2,215
against 1,331). The prefit frees pi_const_inc beside phi_inc and phi_age. ``--robustness no_pi_const_inc`` keeps the
round-1 layout (pi_const_inc fixed at 0, not in the layout); the round-1 name ``pi_const_inc`` is retired. D5: phi_inc
loads on inc (``--income-basis level``, the default) or on log(income / mean income) (``log``); ``auto`` runs the
prefit on both and keeps the lower q (it chose level in both round-1 runs, by about 1,050). Own elasticities are
reported as medians and 5-95% trimmed means: the means are dominated by near-twin products whose window is resolved by
a few draws. D9: products with a data share
below ``--min-share`` (1e-5) are dropped and their share goes to the outside good (492 of 9,694 products, at most
1.1e-4 of a market's share). Each is bought by about one simulated consumer, so its inversion is simulation noise,
and in 2015 such products made the zero-share repair cycle (the failures fixed in the kernel). ``--min-share 0``
keeps every product. The second-choice joint law drops their rows and columns. Round 4 (GMY_ROUND4 item 2): an
intensive-margin age taste, pi_age on z log footprint and on lux (age / 100), started at 0 and free in the prefit.
phi_age shifts log alpha, which under the a_lo bound moves who buys more than what is bought, so the age-price
gradient stayed flat while phi_age deepened and thinned the frontiers; a taste interaction moves the car chosen.
``--robustness no_age_taste`` keeps the round-2 layout. ``--prefit-rows age`` fits phi_age and the age tastes on the 8
age-price rows alone (a check of the age channel).

Starting theta: every taste and pi of the BLP fit (results/blp_fit_h1000.json) divided by the median |alpha_i|, phi at 0
(the BLP law sits in c_i), sigma_p 2.5. Data: same-firm exact twins merged (same price and x, shares summed; the
second-choice matrices merged likewise), remaining price ties spread to a $10 gap. Consumers: 1,000 CPS households per
year (``--households``; load_agents) crossed with scrambled Sobol normal taste draws, 2^ceil(log2(H nu)) per
market; nu 16 by default (nu 4 fits the micro block's family-size rows to simulation noise, GMY_ROUND1 section 4).
``--antithetic-taste`` draws them in antithetic pairs (each Sobol point followed by its negation on every taste
dimension, GMY_ROUND4 item 1). a_i has no draws: the kernel integrates its lognormal in closed form.

Stages (nfp_harness.pipeline): ``prefit`` (phi_inc, phi_age, pi_const_inc and the age tastes on the 20 price and
purchase-rate micro rows), ``step1`` (fixed W1 from the prefit point; no gate needed, the gate is a diagnostic),
``step2`` (W2 = S^-1: the aggregate block updated, micro and second-choice blocks fixed), ``post`` (section 4 tables,
comparison.txt against GMY's BLP), ``a_lo_band`` (a_lo in 0.1, 0.2, 0.4; headline 0.2), ``stability``, ``gate``. ``--years`` restricts the
markets (development: 1980 1991 1999 2005 2015 2018); ``--warm-delta <run>/stages/<stage>.npz`` warm-starts the
markets of another run's stage at that stage's theta (also under another a_lo: the stored a_lo is a warm start and
the kernel's theta path moves the bound; ``run --stage warm`` inverts every market once at the starting theta and
stores ``stages/warm.npz`` for this), ``--start-theta`` starts the layout there (by parameter name;
``--start-theta-except`` keeps named parameters at this layout's starts); ``--fix sigma_p=3.0`` fixes a parameter at a
value for the whole chain (a calibration band, GMY_ROUND4 item 3; ``ParameterLayout.fixed``).

Run (results under gmy_replication/results/harness/, untracked):
  python gmy_replication/estimate_pcm.py run --stage prefit --nu 4 \\
      --years 1980 1991 1999 2005 2015 2018 --threads 64
  python gmy_replication/estimate_pcm.py run --stage all --nu 16 --threads 192 --detach
  python gmy_replication/estimate_pcm.py status --nu 16
  python gmy_replication/estimate_pcm.py evaluate [--years 1980 1999 2018] [--nu 1]
Resume a killed run by relaunching the identical command.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfp_harness import comparison, micro_sets, pipeline, twins  # noqa: E402
from nfp_harness import instruments as iv  # noqa: E402
from nfp_harness.checkpoint import read_stage_file, stage_parameter_names  # noqa: E402
from nfp_harness.comparison import format_cell as _fmt, year_entry as _year_get  # noqa: E402
from nfp_harness.data import (INCOME_LAW_CLIP, break_price_ties, drop_small_shares, fetch_dataverse_file,  # noqa: E402
                              income_law_basis, zscore)
from nfp_harness.micro_sets import MicroMoments  # noqa: E402
from nfp_harness.diagnostics import at_bound_parameters  # noqa: E402
from nfp_harness.instruments import drop_collinear, drop_collinear_columns, keep_rank, standardise  # noqa: E402
from nfp_harness.linear import LinearIV  # noqa: E402
from nfp_harness.models import pcm_post as pp  # noqa: E402
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.moments import (GroupMicroMoments, GroupMomentRow, IVMoments, MicroDataset,  # noqa: E402
                                 SarganIVMoments, SecondChoiceMoments, SupplyMoments)
from nfp_harness.params import ParameterLayout, ParameterSpec, parse_fixed  # noqa: E402
from nfp_harness.problem import NFPGMMProblem  # noqa: E402
from nfp_harness.profile import NotMonotoneDecreasing  # noqa: E402
from nfp_harness.second_choice import first_second_correlations  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
HARNESS = RESULTS / "harness"
SEED = 8673320
YEARS = (1980, 1999, 2018)                     # the evaluate subcommand's default
ALL_YEARS = tuple(range(1980, 2019))
DEV_YEARS = (1980, 1991, 1999, 2005, 2015, 2018)   # the second-choice years and the Table 5 endpoints
TABLE5_YEARS = (1980, 2018)
A_LO_BAND = (0.1, 0.2, 0.4)
ROBUSTNESS = ("fixed_w", "never_buyers_gmy_code", "no_pi_const_inc", "gmy_spreads_only", "no_age_taste")
#: robustness names of earlier rounds, refused with the reason
RETIRED_ROBUSTNESS = {"pi_const_inc": "pi_const_inc is free by default since round 2 (D8 undone); --robustness "
                                      "no_pi_const_inc keeps the round-1 layout without it"}
#: phi_inc's basis: inc (level, GMY's inc_i) or log(income / pooled mean); "auto" lets the prefit choose
INCOME_BASES = ("auto", "level", "log")
INCOME_BASIS_COLUMN = {"level": "inc", "log": "log_inc_ratio"}
#: the fixed part of log alpha: gmy_blp (log c_i^GMY, the BLP income-age law, the default) or petrin_bounded (Petrin's
#: income law log(ybar / clip(y_i, 1, 150)) with y_i in $1000, GMY_ROUND4 item 4, LEARNINGS 32)
INCOME_LAWS = ("gmy_blp", "petrin_bounded")
ANATOMY_COLUMNS = ("frontier_products_per_buying_draw", "log_alpha_range_buyers", "window_width_log_alpha",
                   "frontier_size_mass", "single_frontier_buyer_share")
TASTE = ("const", "van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport", "electric",
         "regionEU", "regionUS")
UNOBS = ("van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport", "electric", "regionEU",
         "regionUS")
#: PCM extension (round 1, deviation D7): taste spreads on three of GMY's X1 columns outside the second-choice set
EXTRA_SPREADS = ("z_log_height", "z_log_curbweight", "z_log_number_trims")
EXTRA_SPREAD_START = 0.5   # round 2 (1.0 in round 1)
PIS = (("van", "fs2"), ("van", "fs34"), ("van", "fs5"), ("z_log_footprint", "fs2"), ("z_log_footprint", "fs34"),
       ("z_log_footprint", "fs5"), ("truck", "rural"))
PI_CONST_INC = ("const", "inc")   # free by default; robustness no_pi_const_inc drops it (D8)
#: round 4: intensive-margin age taste (GMY_ROUND4 item 2), start 0; robustness no_age_taste drops it
PIS_AGE = (("z_log_footprint", "age"), ("lux", "age"))
#: --prefit-rows: the micro rows of the prefit, price_purchase (the 20 price and purchase-rate rows) or age (the 8
#: age-price rows, with phi_age and the age tastes free)
PREFIT_ROWS = ("price_purchase", "age")
TWIN_KEYS = ("market_ids", "firm_ids", "prices") + TASTE[1:]   # the GMY-only kernel columns; see GMYSpec.twin_keys
TRIM = (0.05, 0.95)   # own-elasticity trimmed means
NEAR_TWIN_GAP = 0.01   # $100 in $10k: price gap to the nearest product of the same vehicle type
MIN_GAP = 0.001          # $10 in $10k
MIN_SHARE = 1e-5         # products below it are bought by about one simulated consumer; dropped (D9)
SIGMA_P = {"start": 2.5, "lower": 1.0, "upper": 4.0, "unit": 0.25}
A_LO = 0.2
SC_WEIGHT = 30.0         # mom_fnc.m line 109: second-choice residuals x 30
PHI_BOUND = 4.0
TASTE_CAP = 10.0         # $100k per sd: above the 1980-2018 price range ($11k-$100k)
BLP_FIT = RESULTS / "blp_fit_h1000.json"
COST_BASIS = ("const", "log_hp", "log_curbweight", "log_mpg", "log_footprint", "van", "suv", "truck", "electric",
              "regionASIA", "regionEU", "trend")
SUPPLY_EXCLUDED = ("lag_pl_con", "steel_weight", "NAFTA_D", "NAFTA_T", "KOREATRDE_D", "KOREATRDE_T")
#: GMY's cost shifters and differentiation counts (data_blp.csv, linearDemand.do): the default excluded set (D4)
GMY_EXCLUDED = ("lag_pl_con", "iv_prod", "steel_weight", "IV_N", "IV_N_weight")
INSTRUMENT_SETS = ("gmy", "gh", "gmy+gh")
BLP_DUMMIES = ("van", "suv", "truck", "lux", "sport", "electric")
GH_CONTINUOUS = ("z_log_footprint", "z_log_hp", "z_log_mpg")
#: --moments names; the stub's old names are aliases (all -> demand_supply, demand -> gmy)
MOMENT_SETS = ("gmy", "gmy_nolinear", "demand_supply", "aggregate", "prefit")
ESTIMATION_MOMENTS = ("gmy", "gmy_nolinear", "demand_supply")
MOMENT_ALIASES = {"demand": "gmy"}
BLP_SUMMARIES = {"BLP fit": RESULTS / "blp_fit_h1000.json", "BLP Table 4": RESULTS / "blp_published_h1000.json"}
#: numbers quoted in the paper (estimate_blp.py TABLE5, LERNER, TABLE6_*); definitions UNVERIFIED (spec section 0)
PAPER = {"table5": {1980: (-5.96, -5.78, -5.49, -5.13, -4.30), 2018: (-9.37, -8.56, -7.69, -6.90, -6.46)},
         "lerner_median": {"introduction": {1980: 0.311, 2018: 0.196}, "section 5.3": {1980: 0.248, 2018: 0.1373}},
         "table6_data": dict(zip(UNOBS, (0.71, 0.64, 0.84, 0.71, 0.60, 0.65, 0.48, 0.28, 0.37, 0.34, 0.48))),
         "table6_model": dict(zip(UNOBS, (0.71, 0.64, 0.80, 0.69, 0.59, 0.65, 0.49, 0.28, 0.19, 0.34, 0.47)))}

# ------------------------------------------------------------------ data
#
# Grieco, Murry & Yurukoglu (2024, QJE 139(2)) US automobile data: products, CPS households, micro moments
# and second-choice matrices, read from the authors' replication package (Harvard Dataverse doi:10.7910/DVN/CZGOKP,
# ``GMY_Replication.tar.gz``, MIT licence). The raw files are not committed: unpack the package so that
# ``<root>/data/data_blp.csv`` exists (default root ``gmy_replication/data/GMY_Replication`` of this repo).
#
# Every rule below is the authors' code in ``codes/`` of the package, cited by file:
#
# * products (``loadWardsSalesData.m``): one market per year (1980-2018, 165-323 models, 9,694 rows, all with
#   ``FINAL_SAMPLE`` = 1); price ``msrp / 10`` in $10,000 of 2015 (msrp is already real, in $1,000 of 2015); market size
#   ``number_households / 2.5`` (average holding 5 years, 2 cars per household), shares ``sales / market size``; the
#   file order is the authors' product order (sorted by year, then make), which the second-choice matrices index.
# * ownership (``makeOwnership.m``): products with the same ``parent`` are jointly priced.
# * utility interactions (``loadWardsSalesData.m`` xTerms, ``getMu_lowMem.m``): the 15 columns ``XTERMS``; within mu the
#   three continuous ones (log footprint, log hp, log mpg) are z-scored over the pooled products (MATLAB zscore, sd with
#   n - 1), and the micro moments use them raw.
# * second stage (``linearDemand.do``): delta on price (instrument ``lag_pl_con``) and ``STAGE2_X`` with year and
#   ``make2`` fixed effects, clustered by ``make2``; the do-file's differentiation counts ``IV_N`` and ``IV_N_weight`` are
#   built here too (the do-file builds them and does not use them in the baseline).
# * households (``loadCPSMktHholds.m``): per year a uniform draw without replacement of CPS households, weights the CPS
#   weights normalised to sum to 1 in each year; income in 2015 dollars, winsorised to [5,000, 225,000]; ``inc`` =
#   (income - pooled mean) / 10,000, ``inc2`` = inc^2 / 100; family size 1 + married + children; age / 100; income
#   quintiles from the national cutoffs (2018 dollars converted to 2015); CEX sets live in 1983-2004, MRI in 1992-2014.
# * micro moments (``loadMicMomData.m``, ``getMicroMoments.m``, ``ExjCondH.m``): the 33 moments of specification 6
#   (``NEWPATTERN``), weight 1 / variance. The model counterpart of E[x | H] is the household-weighted mean over the
#   households in H (pooled over the set's years, each year's weights summing to 1) of that household's expected x
#   given a purchase: ``MicroMomentEvaluator``. The "CEX" and "MRI" column labels of the working paper's Table 8 are
#   swapped relative to the files the code reads (microMomentsCEX.mat holds the values Table 8 prints under MRI, and
#   the reverse; the urban-rural truck row, which the code takes from the CEX file, sits under MRI in the table).
# * second choices (``getSecondChoiceStructure.m``): ``secondChoiceProb_<year>`` has first choices in rows and second
#   choices in columns, in the year's product order, short of the year's product count by up to two trailing
#   rows/columns, which are padded with zeros.
#
# The MicroMoments class of the table and the evaluator of the model counterparts are the harness's
# nfp_harness.micro_sets (MicroMomentEvaluator, with MICRO_CHAR); the household-set columns are
# micro_sets.add_set_columns with SURVEYS and SET_DEMOGRAPHICS.

DATAVERSE_DOI = "doi:10.7910/DVN/CZGOKP"
DEFAULT_ROOT = HERE / "data" / "GMY_Replication"

#: xTerms of loadWardsSalesData.m (columns 1-15), named as in xTerms_names; make dummies are not used by spec 6
XTERMS = ("prices", "const", "car", "van", "suv", "truck", "log_footprint", "log_hp", "log_mpg", "lux", "sport",
          "electric", "regionASIA", "regionEU", "regionUS")
XTERMS_NAMES = ("Price", "Constant", "Car", "Van", "SUV", "Truck", "logSizeLW", "Horsepower", "MilesPerGallon",
                "Luxury", "Sport", "Electric", "AsianBrand", "EuroBrand", "USBrand")
#: columns z-scored inside mu (getMu_lowMem.m, jChar columns 7-9)
Z_XTERMS = ("log_footprint", "log_hp", "log_mpg")
#: exogenous characteristics of the second stage (linearDemand.do $X, from X(:,1:12) plus sport)
STAGE2_X = ("z_log_height", "z_log_footprint", "z_log_hp", "z_log_mpg", "z_log_curbweight", "z_log_number_trims",
            "releaseYear", "yearsSinceDesign_tc", "sport", "electric", "truck", "suv", "van")
SECOND_CHOICE_YEARS = (1991, 1999, 2005, 2015)
#: unObsPattern of spec 6: the characteristics with a random coefficient and a second-choice correlation moment
SECOND_CHOICE_CHARS = ("van", "suv", "truck", "log_footprint", "log_hp", "log_mpg", "lux", "sport", "electric",
                       "regionEU", "regionUS")
CEX_YEARS = (1983, 2004)
MRI_YEARS = (1992, 2014)
INCOME_WINSOR = (5000.0, 225000.0)
HOLDING = 2.5   # number_households / 2.5: average duration 5 years, 2 cars per household
AGE_BINS = (0.30, 0.40, 0.50, 0.60)   # age / 100 cut points of the five age sets
#: micro-moment characteristic (data column name) -> product column of the model counterpart
MICRO_CHAR = {"Price": "prices", "Van": "van", "logSizeLW": "log_footprint", "Truck": "truck"}


def package_root(root: str | pathlib.Path | None = None) -> pathlib.Path:
    """package_root(root=None) -> the unpacked package directory (holding ``data/data_blp.csv``); raises
    FileNotFoundError naming ``fetch_package`` when it is missing."""
    r = pathlib.Path(root) if root is not None else DEFAULT_ROOT
    if not (r / "data" / "data_blp.csv").is_file():
        raise FileNotFoundError(f"{r}/data/data_blp.csv not found: unpack GMY_Replication.tar.gz there "
                                f"(fetch_package in gmy_replication/estimate_pcm.py downloads it from {DATAVERSE_DOI})")
    return r


def fetch_package(dest: str | pathlib.Path) -> pathlib.Path:
    """fetch_package(dest) -> path of ``GMY_Replication.tar.gz`` downloaded into ``dest`` from the Dataverse API
    (dataset ``DATAVERSE_DOI``, latest version). Unpack it with ``tar xzf`` so ``package_root`` finds it."""
    return fetch_dataverse_file(DATAVERSE_DOI, "GMY_Replication.tar.gz", dest)


def _make2(make: pd.Series) -> pd.Series:
    """make2 of linearDemand.do (small UK and Italian makes pooled, datsun -> nissan, merkur -> mercury). The
    do-file's list spells 'alpha romeo' and 'lansia', so alfa romeo and lancia keep their own make, as there."""
    m2 = make.copy()
    m2[make.isin(["rover", "austin-rover", "triumph", "mg", "delorean", "sterling"])] = "small-uk"
    m2[make.isin(["alpha romeo", "lansia"])] = "small-italian"
    m2[make == "datsun"] = "nissan"
    m2[make == "merkur"] = "mercury"
    return m2


def load_products(root: str | pathlib.Path | None = None) -> pd.DataFrame:
    """load_products(root=None) -> product table, one row per model-year in the authors' order, with pyblp-shaped
    columns: ``market_ids`` (year), ``position`` (0-based index within the year, the second-choice matrix index),
    ``product_ids``, ``firm_ids`` (parent), ``clustering_ids`` (make2), ``prices`` ($10k of 2015), ``shares``,
    ``market_size``, ``quantity``, the ``XTERMS`` columns (raw), their z-scored versions ``z_<col>`` for
    ``Z_XTERMS``, the ``STAGE2_X`` columns, ``demand_instruments0`` (lag_pl_con), ``iv_prod``, ``steel_weight``,
    ``IV_N``, ``IV_N_weight`` and ``trend`` (year - 1980), plus every column of data_blp.csv."""
    r = package_root(root)
    d = pd.read_csv(r / "data" / "data_blp.csv")
    if not (d.FINAL_SAMPLE == 1).all():
        d = d[d.FINAL_SAMPLE == 1].reset_index(drop=True)
    if not d.year.is_monotonic_increasing:
        raise ValueError("data_blp.csv is not sorted by year; the second-choice matrices index the file order")
    out = d.copy()
    out["market_ids"] = d.year.astype(int)
    out["position"] = d.groupby("year").cumcount().astype(int)
    out["product_ids"] = d.model.astype(str)
    out["firm_ids"] = d.parent.astype(str)
    out["make2_do"] = _make2(d.make.astype(str))
    out["clustering_ids"] = out.make2_do
    out["prices"] = d.msrp / 10.0
    out["market_size"] = d.number_households / HOLDING
    out["quantity"] = d.sales.astype(float)
    out["shares"] = d.sales / out.market_size
    out["const"] = 1.0
    out["electric"] = np.maximum(d.PHEV, d.EV).astype(float)
    for c in Z_XTERMS:
        out[f"z_{c}"] = zscore(d[c])
    for c in ("log_height", "log_curbweight", "log_number_trims"):
        out[f"z_{c}"] = zscore(d[c])
    out["yearsSinceDesign_tc"] = d.yearsSinceDesign.clip(3, 12).astype(float)
    out["trend"] = (d.year - d.year.min()).astype(float)
    out["demand_instruments0"] = d.lag_pl_con.astype(float)
    # differentiation counts of linearDemand.do: rivals by year x vehicle type x HQ region, and the same split by
    # curb weight above the pooled median of the vehicle type
    veh = np.select([d.van == 1, d.suv == 1, d.truck == 1], ["van", "suv", "truck"], "car")
    out["veh_type"] = veh
    out["IV_N"] = out.groupby(["year", "veh_type", "region"]).year.transform("size").astype(float)
    big = d.curbweight > out.groupby("veh_type").curbweight.transform("median")
    out["big_for_class"] = big.astype(int)
    out["IV_N_weight"] = out.groupby(["year", "veh_type", "region", "big_for_class"]).year.transform("size").astype(float)
    return out


def _cpi(r: pathlib.Path) -> dict[int, float]:
    c = pd.read_csv(r / "data" / "CPIU_1967base.csv", encoding="utf-8-sig")
    return dict(zip(c.year.astype(int), c.CPIU67.astype(float)))


def load_agents(root: str | pathlib.Path | None = None, n_households: int = 1000, seed: int = SEED,
                years: Sequence[int] | None = None) -> pd.DataFrame:
    """load_agents(root=None, n_households=1000, seed=SEED, years=None) -> household table, ``n_households``
    CPS households per year drawn uniformly without replacement (numpy, so not the authors' MATLAB stream).

    Columns: ``market_ids``, ``agent_ids``, ``weights`` (CPS weight / year total), ``income`` (2015 $, winsorised),
    ``inc``, ``inc2``, ``fs2``, ``fs34``, ``fs5``, ``age`` (/100), ``rural``, ``college``, ``inc_quintile`` (1-5),
    ``age_group`` (1-5: <30, 30s, 40s, 50s, 60+), ``fs_group`` (1-4: 1, 2, 3-4, 5+), ``cex`` and ``mri`` (1 when the
    year is in that survey's window). ``years`` restricts the draw; the income mean behind ``inc`` is always the
    pooled mean over all 39 years, as in loadCPSMktHholds.m (each year weighted equally).
    """
    r = package_root(root)
    cps = pd.read_csv(r / "data" / "CPS_households.csv")
    cpi = _cpi(r)
    rng = np.random.default_rng(seed)
    all_years = sorted(int(y) for y in pd.read_csv(r / "data" / "data_blp.csv", usecols=["year"]).year.unique())
    frames = []
    for y in all_years:
        g = cps[cps.year == y]
        if len(g) < n_households:
            raise ValueError(f"{y}: {len(g)} CPS households, fewer than {n_households}")
        g = g.iloc[np.sort(rng.choice(len(g), size=n_households, replace=False))]
        f = pd.DataFrame({"market_ids": y, "agent_ids": np.arange(n_households),
                          "weights": (g.weight / g.weight.sum()).to_numpy(float)})
        f["income"] = np.clip(g.hhincome.to_numpy(float) * cpi[2015] / cpi[y], *INCOME_WINSOR)
        fs = 1 + g.married.to_numpy() + g.numchildren.to_numpy()
        f["fs2"] = (fs == 2).astype(float)
        f["fs34"] = ((fs >= 3) & (fs <= 4)).astype(float)
        f["fs5"] = (fs >= 5).astype(float)
        f["fs_group"] = np.select([fs == 1, fs == 2, fs <= 4], [1, 2, 3], 4)
        f["age"] = g.age.to_numpy(float) / 100.0
        f["age_group"] = np.digitize(f.age, AGE_BINS) + 1
        f["rural"] = g.rural.to_numpy(float)
        f["college"] = g.bachelors.to_numpy(float)
        frames.append(f)
    a = pd.concat(frames, ignore_index=True)
    w_tot = a.groupby("market_ids").weights.transform("sum")   # 1 in every year
    avg = float((a.income * a.weights).sum() / w_tot.groupby(a.market_ids).first().sum())
    a["inc"] = (a.income - avg) / 1e4
    a["inc2"] = a.inc ** 2 / 1e2
    q = pd.read_csv(r / "data" / "income_quintile_cutoffs_by_year.csv").set_index("year")
    cut = q[["inc_p2", "inc_p4", "inc_p6", "inc_p8"]] * cpi[2015] / cpi[2018]
    a["inc_quintile"] = 0
    for y, idx in a.groupby("market_ids").groups.items():
        # MATLAB discretize on [-Inf; cutoffs; Inf]: bins closed on the left
        a.loc[idx, "inc_quintile"] = np.digitize(a.loc[idx, "income"], cut.loc[y].to_numpy(float)) + 1
    a["cex"] = a.market_ids.between(*CEX_YEARS).astype(float)
    a["mri"] = a.market_ids.between(*MRI_YEARS).astype(float)
    if years is not None:
        a = a[a.market_ids.isin(list(years))].reset_index(drop=True)
    return a


def load_micro_tables(root: str | pathlib.Path | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """load_micro_tables(root=None) -> {"cex": {"value", "variance"}, "mri": {...}}: the 44 demographics x
    characteristics tables of cleanMoments.m (CEX 44 x 75: 13 characteristics and 62 make groups; MRI 44 x 76, plus
    PurchaseProb). The MRI rows on urban / rural are NaN, as cleanMoments.m sets them."""
    from scipy.io import loadmat
    r = package_root(root)
    makes = pd.read_csv(r / "data" / "make_group_ids.csv").make.tolist()

    def fix(s: str) -> str:
        return s.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")

    cex = loadmat(r / "data" / "microMomentsCEX.mat")
    # cleanMoments.m labels both surveys with the CEX names (the MRI file's name arrays are not used)
    chars = [str(x[0]) for x in cex["car_char_names_cex"].ravel()][:-1]
    demog = [str(x[0]) for x in cex["demog_names_cex"].ravel()]
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for s in ("cex", "mri"):
        m = cex if s == "cex" else loadmat(r / "data" / "microMomentsMRI.mat")
        cols = [fix(c) for c in chars + makes + ["PurchaseProb"]]
        val, var = m[f"moment_values_{s}"], m[f"moment_variance_{s}"]
        cols = cols[:val.shape[1]]
        out[s] = {"value": pd.DataFrame(val, index=demog, columns=cols),
                  "variance": pd.DataFrame(var, index=demog, columns=cols)}
    urb = ["Urban", "Rural", "UrbanMINUSRural", "UrbanOVERRural"]
    out["mri"]["value"].loc[urb, :] = np.nan
    return out


def load_micro_moments(root: str | pathlib.Path | None = None) -> MicroMoments:
    """load_micro_moments(root=None) -> MicroMoments of spec 6 (33 rows)."""
    t = load_micro_tables(root)
    rows: list[dict[str, Any]] = []

    def add(demog_rows: Sequence[tuple[str, Any, Any]], chars: Sequence[str], surveys: Sequence[str], demographic: str,
            kind: str = "diff") -> None:
        # extractMom: per survey a (demog x char) block, surveys stacked, then column-major: char outermost
        for ch in chars:
            for s in surveys:
                for name, a, b in demog_rows:
                    rows.append({"name": f"{name} {ch} {s}", "survey": s, "char": ch, "kind": kind,
                                 "demographic": demographic, "a": a, "b": b,
                                 "value": float(t[s]["value"].loc[name, ch]),
                                 "variance": float(t[s]["variance"].loc[name, ch])})

    inc = [(f"Income{q}MINUSIncome1", q, 1) for q in (5, 4, 3, 2)]
    age = [("Age60+MINUSAge20-29", 5, 1), ("Age50-59MINUSAge20-29", 4, 1), ("Age40-49MINUSAge20-29", 3, 1),
           ("Age30-39MINUSAge20-29", 2, 1)]
    fam = [("FamilySize2MINUSFamilySize1", 2, 1), ("FamilySize3-4MINUSFamilySize1", 3, 1),
           ("FamilySize5+MINUSFamilySize1", 4, 1)]
    add(inc, ["Price"], ["cex", "mri"], "inc_quintile")
    add(age, ["Price"], ["cex", "mri"], "age_group")
    add(fam, ["Van", "logSizeLW"], ["cex", "mri"], "fs_group")
    add([("UrbanMINUSRural", 0, 1)], ["Truck"], ["cex"], "rural")
    add([(f"Income{q}OVERIncome1", q, 1) for q in (5, 4, 3, 2)], ["PurchaseProb"], ["mri"], "inc_quintile", "ratio")
    return MicroMoments(pd.DataFrame(rows))


def load_second_choices(root: str | pathlib.Path | None = None, products: pd.DataFrame | None = None
                        ) -> dict[int, dict[str, np.ndarray]]:
    """load_second_choices(root=None, products=None) -> {year: {"prob", "count"}} for 1991, 1999, 2005, 2015:
    (J_t, J_t) arrays in the year's product order (``position``), first choice in rows, second choice in columns,
    the package's matrices padded with zero trailing rows/columns to J_t (getSecondChoiceStructure.m). A row sums
    to 1 when the first choice has respondents and to 0 otherwise."""
    from scipy.io import loadmat
    r = package_root(root)
    products = load_products(r) if products is None else products
    m = loadmat(r / "data" / "secondChoiceOut.mat")
    out: dict[int, dict[str, np.ndarray]] = {}
    for y in SECOND_CHOICE_YEARS:
        J = int((products.market_ids == y).sum())
        res = {}
        for key, name in (("prob", "secondChoiceProb"), ("count", "secondChoiceN")):
            a = np.asarray(m[f"{name}_{y}"], dtype=float)
            if a.shape[0] > J or a.shape[1] > J:
                raise ValueError(f"{name}_{y} is {a.shape}, larger than the {J} products of {y}")
            full = np.zeros((J, J))
            full[:a.shape[0], :a.shape[1]] = a
            res[key] = full
        out[y] = res
    return out


@dataclasses.dataclass(frozen=True, eq=False)
class GMYData:
    """Everything one GMY replication script needs: products, agents, micro moments and second choices."""

    products: pd.DataFrame
    agents: pd.DataFrame
    micro: MicroMoments
    second_choices: Mapping[int, Mapping[str, np.ndarray]]
    root: pathlib.Path


def load_data(root: str | pathlib.Path | None = None, n_households: int = 1000, seed: int = SEED,
         years: Sequence[int] | None = None) -> GMYData:
    """load_data(root=None, n_households=1000, seed=SEED, years=None) -> GMYData; ``years`` keeps those markets only
    (the micro moments then pool over the years kept)."""
    r = package_root(root)
    p = load_products(r)
    sc = load_second_choices(r, p)
    if years is not None:
        keep = [int(y) for y in years]
        p = p[p.market_ids.isin(keep)].reset_index(drop=True)
        sc = {y: v for y, v in sc.items() if y in keep}
    a = load_agents(r, n_households, seed, years)
    return GMYData(products=p, agents=a, micro=load_micro_moments(r), second_choices=sc, root=r)


#: demographic agent column -> its levels, for the household sets of the micro moments
SET_DEMOGRAPHICS = {"inc_quintile": (1, 2, 3, 4, 5), "age_group": (1, 2, 3, 4, 5), "fs_group": (1, 2, 3, 4),
                    "rural": (0, 1)}
SURVEYS = ("cex", "mri")


def add_set_columns(agents: pd.DataFrame) -> pd.DataFrame:
    """``agents`` with one 0/1 column per (survey, demographic, level): demographic = level and the year in the
    survey's window (loadCPSMktHholds.m lines 109-147)."""
    return micro_sets.add_set_columns(agents, SURVEYS, SET_DEMOGRAPHICS)


def group_moment_rows(micro: MicroMoments) -> list[tuple[str, str | None, str, str]]:
    """(kind, product column or None, set column a, set column b) for each of the 33 rows of ``micro``, in order: the
    arguments of ``moments.GroupMomentRow``."""
    return micro_sets.group_moment_rows(micro, MICRO_CHAR)


def micro_evaluator(products: pd.DataFrame, agents: pd.DataFrame, micro: MicroMoments
                    ) -> micro_sets.MicroMomentEvaluator:
    """GMY's model counterparts of the 33 micro moments (ExjCondH.m) on ``products`` and ``agents``."""
    return micro_sets.MicroMomentEvaluator(products, agents, micro, MICRO_CHAR)


def merge_twins(products: pd.DataFrame, keys: Sequence[str] = TWIN_KEYS) -> tuple[pd.DataFrame, int]:
    """Rows equal on ``keys`` (market, firm, price and every kernel characteristic) are one product to every PCM
    consumer: keep the first, sum shares and quantity (``twins.merge_twins``)."""
    return twins.merge_twins(products, keys, ("shares", "quantity", "sales"))


def price_structure(products: pd.DataFrame, chars: Sequence[str] = TASTE[1:]) -> dict[int, dict[str, float]]:
    """Per market: products, median p / gap to the nearest-priced neighbour, exact price ties, and identical-x
    clusters (products sharing every kernel characteristic) with the products in them."""
    out = {}
    for y, g in products.groupby("market_ids"):
        p = np.sort(g.prices.to_numpy(float))
        d = np.diff(p)
        gap = np.minimum(np.r_[np.inf, d], np.r_[d, np.inf])
        with np.errstate(divide="ignore"):
            ratio = np.where(gap > 0, p / gap, np.inf)
        sizes = g.groupby(list(chars)).size()
        out[int(y)] = {"products": int(len(g)), "median_p_over_gap": float(np.median(ratio)),
                       "exact_price_ties": int((d == 0).sum()), "identical_x_clusters": int((sizes > 1).sum()),
                       "products_in_identical_x_clusters": int(sizes[sizes > 1].sum()),
                       "identical_x_and_price_pairs": int(g.duplicated(list(chars) + ["prices"]).sum())}
    return out


class GMYSpec:
    """The GMY PCM spec read by nfp_harness.pipeline (interface in its module docstring; GMY spec WP-G7)."""

    name = "gmy_pcm"
    firm_column = "firm_ids"
    groups = "veh_type"
    group_other = "car"
    binary_columns = ("van", "suv", "truck", "lux", "sport", "electric", "regionEU", "regionUS")
    prefit_max_iterations = 60
    extensive_free: tuple[str, ...] = ()
    # step 1 does not wait for a gate (D1): under the bound sigma_p is expected weakly identified (LEARNINGS 32), so the
    # gate is a diagnostic stage whose flat profile is reported, not a precondition
    gate_required: tuple[str, ...] = ()
    gate_concentrate_iterations = 20
    a_lo_band_parameter = "a_lo"
    a_lo_band_values = A_LO_BAND
    a_lo_band_moments = "all"
    band_parameter = "sigma_p"
    band_values = (2.0, 3.0)
    band_moments = "all"
    use_max_step = False
    # trust region and noise handling as the Petrin spec (LEARNINGS 25, 32), with the round-1 fix of the collapse on
    # the kinked second-choice and micro blocks: the radius floored at 0.05 natural units, a probe restart at least at
    # the probe's step, and probe restarts while a compass step lowers q by more than 1; round 2: and by more than 0.1%
    # of q (round-1 step 1 spent two hours on restarts gaining 1 to 3 on q near 3,870)
    optimizer_options = {"Delta_0": 0.1, "Delta_max": 0.5, "noise_probe": 1e-5, "noise_Delta": 1e-4,
                         "noise_rejections": 3, "lm_damping": True, "more_scaling": True, "cauchy_stop": True,
                         "noise_converge": True, "noise_Delta_floor": 0.05, "probe_restart_at_step": True,
                         "probe_restart_gain": 1.0, "probe_restart_gain_rel": 1e-3}

    def __init__(self, years: Sequence[int] = ALL_YEARS, households: int = 1000, blp_fit: pathlib.Path = BLP_FIT,
                 instruments: str = "gmy", moments: str = "gmy", robustness: Sequence[str] = (),
                 a_lo: float = A_LO, income_basis: str = "level", min_share: float = MIN_SHARE,
                 income_law: str = "gmy_blp") -> None:
        self.households = int(households)
        self.min_share = float(min_share)
        self.blp_fit = pathlib.Path(blp_fit)
        self.moments = "gmy"
        self.robustness: tuple[str, ...] = ()
        self.a_lo = float(a_lo)
        self.warm_delta: str | None = None
        self.start_theta: str | None = None
        self.start_theta_except: tuple[str, ...] = ()
        self.max_iterations: int | None = None
        self.cold_draws = 0
        self.prefit_rows = "price_purchase"
        self.antithetic_taste = False
        self.fixed: dict[str, float] = {}
        self.set_income_law(income_law)
        self._loaded: tuple[Any, ...] | None = None
        self.set_income_basis(income_basis)
        self.set_moments(moments)
        self.set_robustness(robustness)
        self.load(tuple(years), instruments)

    # ------------------------------------------------------------------ data

    def load(self, years: tuple[int, ...], instruments: str) -> None:
        """Products, agents, X1, Z and the supply design for ``years`` with the excluded set ``instruments``."""
        if instruments not in INSTRUMENT_SETS:
            raise ValueError(f"instruments must be one of {INSTRUMENT_SETS}, got {instruments!r}")
        years = tuple(sorted(int(y) for y in years))
        key = (years, instruments, self.households, self.kernel_chars, self.min_share)
        if self._loaded == key:
            return
        self.years = years
        self.instruments = instruments
        data = load_data(n_households=self.households, seed=SEED, years=self.years)
        self.raw_products = data.products
        self.structure_raw = price_structure(data.products, self.kernel_chars[1:])
        p, self.n_merged = merge_twins(data.products, self.twin_keys)
        keep = drop_small_shares(p.shares.to_numpy(float), p.market_ids.to_numpy(), self.min_share)
        self.dropped_small = {"products": int((~keep).sum()), "share_max_per_market": float(
            p[~keep].groupby("market_ids").shares.sum().max() if (~keep).any() else 0.0)}
        p = p[keep].reset_index(drop=True)
        p["prices_data"] = p.prices
        p["prices"] = break_price_ties(p.prices.to_numpy(float), p.market_ids.to_numpy(), MIN_GAP)
        p["log_hp"] = np.log(p.hp)
        p["log_curbweight"] = np.log(p.curbweight)
        self.products = p
        self.price_range = float(p.prices_data.max() - p.prices_data.min())
        self.structure = price_structure(p.assign(prices=p.prices_data), self.kernel_chars[1:])
        self.second_choices = {y: v for y, v in data.second_choices.items()}
        fit = json.loads(self.blp_fit.read_text())
        self.blp = {"theta": fit["theta"], "alpha": float(fit["alpha"]), "source": self.blp_fit.name}
        a = data.agents.copy()
        th = fit["theta"]
        alpha_i = (self.blp["alpha"] + th["prices x inc"] * a.inc + th["prices x inc2"] * a.inc2
                   + th["prices x age"] * a.age)
        if not (alpha_i < 0).all():
            raise ValueError("the BLP income law gives a non-negative price coefficient for some household")
        order = np.argsort(-alpha_i.to_numpy())
        cw = np.cumsum(a.weights.to_numpy()[order])
        self.alpha_median = float(-alpha_i.to_numpy()[order][np.searchsorted(cw, cw[-1] / 2)])
        a["log_c_gmy"] = np.log(-alpha_i / self.alpha_median)
        # log-income basis of phi_inc: log(income / pooled mean income), the mean of loadCPSMktHholds.m behind inc
        mean_income = float((a.income - a.inc * 1e4).iloc[0])
        a["log_inc_ratio"] = np.log(a.income / mean_income)
        # Petrin's bounded income law (--income-law petrin_bounded): income in $1000, ybar the pooled weighted median
        y = a.income.to_numpy(float) / 1e3
        order = np.argsort(y, kind="stable")
        cw = np.cumsum(a.weights.to_numpy(float)[order])
        self.income_law_ybar = float(y[order][np.searchsorted(cw, cw[-1] / 2)])
        a["log_ybar_inc_w"] = income_law_basis(y, self.income_law_ybar)
        self.agents = add_set_columns(a)
        self.micro = data.micro
        self.X1, self.X1_names = self.linear_design(p)
        self.linear_names = list(self.X1_names)
        Z_ex, ex_names = excluded_instruments(p, instruments)
        Z = np.column_stack([self.X1, Z_ex])
        keep = drop_collinear_columns(Z, keep=self.X1.shape[1])
        self.Z = Z[:, keep]
        self.Z_names = list(self.X1_names) + [ex_names[k - self.X1.shape[1]] for k in keep[self.X1.shape[1]:]]
        w = np.column_stack([np.ones(len(p)) if c == "const" else p[c].to_numpy(float) for c in COST_BASIS])
        self.cost_keep = keep_rank(w)
        self.cost_basis = w[:, self.cost_keep]
        # the excluded supply instruments standardised, as the demand side's: steel_weight is of order 6e5, so unscaled
        # its square set lambda_max of Z_s'Z_s / n to 4e11 and the 2SLS weight (Z_s'Z_s / n)^-1 of step 1 was refused
        # as not positive definite (lambda_min / lambda_max 1e-16; 3e-6 once scaled). The 2SLS supply block does not
        # depend on the column scale of Z_s.
        Zs = np.column_stack([self.cost_basis, standardise(np.column_stack([p[c].to_numpy(float)
                                                                            for c in SUPPLY_EXCLUDED]))])
        self.Z_supply = drop_collinear(Zs, keep=self.cost_basis.shape[1])
        # Clusters of the aggregate (and supply) block: the model (product_ids), not make2. linearDemand.do clusters
        # its standard errors by make2, but make2 has 56 levels against 112 aggregate moments, so the clustered S of
        # step 2 would be singular (rank <= 56) and W2 = S^-1 would not exist.
        self.clusters = p.product_ids.to_numpy()
        self._loaded = key

    @staticmethod
    def linear_design(p: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """X1 = [STAGE2_X, year dummies (first dropped), make2 dummies (first dropped), constant], rank-reduced."""
        cols = [p[c].to_numpy(float) for c in STAGE2_X]
        names = list(STAGE2_X)
        for c in ("market_ids", "make2_do"):
            lv = sorted(p[c].unique())
            for v in lv[1:]:
                cols.append((p[c] == v).to_numpy(float))
                names.append(f"{c}={v}")
        X = np.column_stack([np.ones(len(p))] + cols)
        names = ["const"] + names
        keep = keep_rank(X)
        return X[:, keep], [names[k] for k in keep]

    # ------------------------------------------------------------------ options

    def set_moments(self, moments: str) -> None:
        moments = MOMENT_ALIASES.get(moments, moments)
        if moments not in ESTIMATION_MOMENTS:
            raise ValueError(f"moments must be one of {ESTIMATION_MOMENTS} (or {tuple(MOMENT_ALIASES)}), got "
                             f"{moments!r}")
        self.moments = moments

    def set_robustness(self, robustness: Sequence[str] | None) -> None:
        r = tuple(sorted(set(robustness or ())))
        retired = [RETIRED_ROBUSTNESS[x] for x in r if x in RETIRED_ROBUSTNESS]
        if retired:
            raise ValueError("; ".join(retired))
        unknown = [x for x in r if x not in ROBUSTNESS]
        if unknown:
            raise ValueError(f"unknown robustness switches {unknown}; known: {ROBUSTNESS}")
        self.robustness = r

    def set_income_basis(self, basis: str) -> None:
        if basis not in INCOME_BASES:
            raise ValueError(f"income_basis must be one of {INCOME_BASES}, got {basis!r}")
        self.income_basis = basis
        self.prefit_variant: str | None = None   # the basis the prefit chose under "auto"

    @property
    def phi_inc_basis(self) -> str:
        """The basis phi_inc loads on now: the explicit choice, the prefit's under "auto", else the level."""
        if self.income_basis != "auto":
            return self.income_basis
        return self.prefit_variant or "level"

    @property
    def prefit_free(self) -> tuple[str, ...]:
        """The prefit's free parameters: phi_inc and phi_age, pi_const_inc when the layout has it (round 2: the
        purchase-rate rows need it beside phi_inc) and the age tastes when it has them (round 4). On the age rows alone
        (``--prefit-rows age``): phi_age and the age tastes."""
        age = tuple(f"pi_{c}_{d}" for c, d in PIS_AGE if (c, d) in self.pis)
        if self.prefit_rows == "age":
            return ("phi_age",) + age
        return ("phi_inc", "phi_age") + (("pi_const_inc",) if PI_CONST_INC in self.pis else ()) + age

    def set_income_law(self, law: str) -> None:
        if law not in INCOME_LAWS:
            raise ValueError(f"income_law must be one of {INCOME_LAWS}, got {law!r}")
        self.income_law = law

    def set_prefit_rows(self, rows: str) -> None:
        if rows not in PREFIT_ROWS:
            raise ValueError(f"prefit_rows must be one of {PREFIT_ROWS}, got {rows!r}")
        self.prefit_rows = rows

    def prefit_variants(self) -> tuple[str, ...]:
        """Layout variants the pipeline's prefit compares (the one with the lower prefit q is kept): the two phi_inc
        bases under ``--income-basis auto`` (round 1 item 4), none otherwise."""
        return ("level", "log") if self.income_basis == "auto" else ()

    def set_prefit_variant(self, name: str | None) -> None:
        if name is not None and name not in INCOME_BASIS_COLUMN:
            raise ValueError(f"unknown prefit variant {name!r}")
        self.prefit_variant = name

    @property
    def extra_spreads(self) -> tuple[str, ...]:
        return () if "gmy_spreads_only" in self.robustness else EXTRA_SPREADS

    @property
    def pis(self) -> tuple[tuple[str, str], ...]:
        return (PIS + (() if "no_pi_const_inc" in self.robustness else (PI_CONST_INC,))
                + (() if "no_age_taste" in self.robustness else PIS_AGE))

    @property
    def kernel_chars(self) -> tuple[str, ...]:
        """Columns of x in the kernel: GMY's 12 and the extra spread columns."""
        return TASTE + self.extra_spreads

    @property
    def twin_keys(self) -> tuple[str, ...]:
        """Products equal on these are one product to every PCM consumer (merged)."""
        return ("market_ids", "firm_ids", "prices") + self.kernel_chars[1:]

    @property
    def never_buyers(self) -> str:
        return "keep" if "never_buyers_gmy_code" in self.robustness else "drop"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--years", type=int, nargs="+", default=None,
                            help="markets (default all 39 years 1980-2018); the micro sets pool over the years present "
                                 "and the second-choice block uses those of 1991, 1999, 2005, 2015 among them")
        parser.add_argument("--instruments", choices=INSTRUMENT_SETS, default="gmy",
                            help="excluded demand instruments: gmy (GMY's 5 cost shifters and counts, default), gh "
                                 "(Gandhi-Houde local), gmy+gh")
        parser.add_argument("--moments", choices=ESTIMATION_MOMENTS + tuple(MOMENT_ALIASES), default="gmy",
                            help="gmy: Sargan + micro + second choice (default); gmy_nolinear: micro + second choice "
                                 "(GMY's nonlinear stage); demand_supply: gmy + the log-cost supply block")
        parser.add_argument("--robustness", action="append", choices=ROBUSTNESS, default=None,
                            help="fixed_w: fixed-W aggregate block (IVMoments) instead of Sargan; never_buyers_gmy_code: "
                                 "never-buyers keep their weight in the micro moments, as ExjCondH.m's code; "
                                 "no_pi_const_inc: GMY's constant x income taste fixed at 0 (the round-1 layout, D8); "
                                 "gmy_spreads_only: no spreads on height, curb weight and trims (D7); no_age_taste: "
                                 "no age x (footprint, lux) taste (the round-2 layout)")
        parser.add_argument("--prefit-rows", choices=PREFIT_ROWS, default="price_purchase",
                            help="micro rows of the prefit: price_purchase (the 20 price and purchase-rate rows, "
                                 "default) or age (the 8 age-price rows, phi_age and the age tastes free); part of the "
                                 "run key")
        parser.add_argument("--min-share", type=float, default=MIN_SHARE,
                            help="drop products whose data share is below this (default 1e-5, D9; 0 keeps all); "
                                 "part of the run key")
        parser.add_argument("--households", type=int, default=1000,
                            help="CPS households per year (default 1000, GMY's); part of the run key")
        parser.add_argument("--income-basis", choices=INCOME_BASES, default="level",
                            help="phi_inc on inc (level, the default), on log(income / mean) (log), or auto: the "
                                 "prefit fits both and keeps the lower q")
        parser.add_argument("--fix", action="append", default=None, metavar="NAME=VALUE",
                            help="fix a parameter at a value for every stage, e.g. --fix sigma_p=3.0 (repeatable; the "
                                 "bounds widen to the value); part of the run key")
        parser.add_argument("--antithetic-taste", action="store_true",
                            help="taste draws in antithetic pairs: each scrambled Sobol point followed by its negation "
                                 "on every taste dimension (a_i is integrated in closed form and has no draws); part "
                                 "of the run key")
        parser.add_argument("--income-law", choices=INCOME_LAWS, default="gmy_blp",
                            help="the fixed part of log alpha: gmy_blp (log c_i of GMY's BLP income-age law, default) "
                                 "or petrin_bounded (Petrin's law log(ybar / clip(y_i, 1, 150)), y_i in $1000; phi_inc, "
                                 "phi_age and pi_const_inc stay free); part of the run key")
        parser.add_argument("--start-theta", default=None,
                            help="a stored stage (<run dir>/stages/<stage>.npz) whose theta becomes the layout starts "
                                 "(clipped to the bounds), matched by parameter name through the run's history.csv "
                                 "(positional when the run has none and the layouts have the same length); a free "
                                 "parameter the stage does not have keeps its start; part of the run key")
        parser.add_argument("--start-theta-except", nargs="+", default=None, metavar="PARAMETER",
                            help="free parameters that keep this layout's starts under --start-theta (e.g. the three "
                                 "D7 spreads, to start them at 0.5); part of the run key")
        parser.add_argument("--a-lo", type=float, default=A_LO, help="bound on the unobserved factor a_i (default 0.2)")
        parser.add_argument("--warm-delta", default=None,
                            help="a stored stage (<run dir>/stages/<stage>.npz) of a run on other draws whose delta "
                                 "warm-starts every market of the same years (WP-G7 item 1)")
        parser.add_argument("--max-iterations", type=int, default=None,
                            help="trial-step limit of step1 and step2 (smoke runs); part of the run key")
        parser.add_argument("--cold-draws", type=int, default=0,
                            help="draw homotopy for markets with no warm start: invert first on every k-th draw (at "
                                 "least this many draws), then warm-start each finer level (PCMModel cold_draws; 0: "
                                 "off); part of the run key when set")

    def configure(self, args: Any) -> None:
        years = tuple(args.years) if getattr(args, "years", None) else ALL_YEARS
        self.set_robustness(getattr(args, "robustness", None))   # before load: the kernel columns decide the twins
        self.households = int(getattr(args, "households", None) or 1000)
        ms = getattr(args, "min_share", None)
        self.min_share = MIN_SHARE if ms is None else float(ms)
        self.set_income_basis(getattr(args, "income_basis", None) or "level")
        self.set_prefit_rows(getattr(args, "prefit_rows", None) or "price_purchase")
        self.set_income_law(getattr(args, "income_law", None) or "gmy_blp")
        self.antithetic_taste = bool(getattr(args, "antithetic_taste", False))
        self.fixed = parse_fixed(getattr(args, "fix", None))
        self.load(years, getattr(args, "instruments", None) or "gmy")
        self.set_moments(getattr(args, "moments", None) or "gmy")
        stt = getattr(args, "start_theta", None)
        self.start_theta = str(pathlib.Path(stt).resolve()) if stt else None
        self.start_theta_except = tuple(sorted(set(getattr(args, "start_theta_except", None) or ())))
        self.a_lo = float(getattr(args, "a_lo", A_LO))
        wd = getattr(args, "warm_delta", None)
        self.warm_delta = str(pathlib.Path(wd).resolve()) if wd else None
        mi = getattr(args, "max_iterations", None)
        self.max_iterations = None if mi is None else int(mi)
        self.cold_draws = int(getattr(args, "cold_draws", None) or 0)

    def run_key(self) -> dict[str, Any]:
        """Result-changing options beyond nu and seed: years (unless all 39), instruments, moments, robustness, a_lo and
        its kind, the warm delta's source and a smoke-run iteration limit."""
        out: dict[str, Any] = {"spec": "gmy_pcm", "a_lo": self.a_lo, "bound": "a"}
        if self.years != ALL_YEARS:
            out["years"] = list(self.years)
        if self.instruments != "gmy":
            out["instruments"] = self.instruments
        if self.moments != "gmy":
            out["moments"] = self.moments
        if self.robustness:
            out["robustness"] = list(self.robustness)
        if self.warm_delta:
            out["warm_delta"] = pathlib.Path(self.warm_delta).parent.parent.name + "/" + pathlib.Path(self.warm_delta).name
        if self.start_theta:
            out["start_theta"] = (pathlib.Path(self.start_theta).parent.parent.name + "/"
                                  + pathlib.Path(self.start_theta).name)
            if self.start_theta_except:
                out["start_theta_except"] = list(self.start_theta_except)
        if self.households != 1000:
            out["households"] = self.households
        out["min_share"] = self.min_share
        out["income_basis"] = self.income_basis   # round-1 layouts (D7, D8) never share a directory with older runs
        # round 2 (pi_const_inc free, D7 at 0.5) never shares a directory with round 1, round 4 (age taste) with either
        out["layout_round"] = 2 if "no_age_taste" in self.robustness else 4
        if self.prefit_rows != "price_purchase":
            out["prefit_rows"] = self.prefit_rows
        if self.income_law != "gmy_blp":
            out["income_law"] = self.income_law
        if self.antithetic_taste:
            out["antithetic_taste"] = True
        if self.fixed:
            out["fix"] = dict(sorted(self.fixed.items()))
        if self.max_iterations is not None:
            out["max_iterations"] = self.max_iterations
        if self.cold_draws:
            out["cold_draws"] = self.cold_draws
        return out

    @property
    def normalisations(self) -> list[str]:
        return [
            f"D9: products with a data share below {self.min_share:g} are dropped (their share goes to the outside good): "
            f"{self.dropped_small['products']} products, at most {self.dropped_small['share_max_per_market']:.2e} of a "
            f"market; such a product is bought by about one simulated consumer, so it adds only simulation noise",
            ("median a_i = 1 on the untruncated parent; log alpha_i = log c_i^GMY + phi_inc b_i + phi_age age_i + log "
             "a_i with c_i^GMY = |alpha + th_inc inc + th_inc2 inc2 + th_age age| / median from the BLP fit, "
             "coefficient fixed at 1" if self.income_law == "gmy_blp" else
             "median a_i = 1 on the untruncated parent; log alpha_i = log(ybar / clip(y_i, 1, 150)) + phi_inc b_i + "
             f"phi_age age_i + log a_i, Petrin's bounded income law (--income-law petrin_bounded, LEARNINGS 32) with y_i "
             f"in $1000 of 2015 and ybar = {self.income_law_ybar:.4g} the pooled weighted median, coefficient fixed at "
             "1 (phi_gmy fixed at 0)")
            + f", and b_i = {'inc_i' if self.phi_inc_basis == 'level' else 'log(income_i / mean income)'} "
            f"(D5; --income-basis {self.income_basis})",
            f"a_i >= a_lo = {self.a_lo:g}: the unobserved factor truncated below for every household, a calibration of "
            f"the welfare tail (LEARNINGS 32)",
            f"taste spreads non-negative, capped at {TASTE_CAP:g} ($10k per sd); u_i0 = 0 with year fixed effects in X1",
            *([f"fixed by --fix (a calibration, not an estimate): "
               f"{', '.join(f'{k} = {v:g}' for k, v in sorted(self.fixed.items()))}"] if self.fixed else []),
            "D1: objective = micro + second-choice moments + one aggregate block on xi with GMY's cost shifters; no Nash "
            "supply block (Nash markups and the Lerner index are post-estimation only)"
            + (f"; moments {self.moments}" if self.moments != "gmy" else ""),
            "D2: the micro moments condition on households with positive purchase probability"
            + (" (robustness: GMY's literal code, never-buyers keep their weight)" if self.never_buyers == "keep"
               else ""),
            "D3: the second-choice joint law is renormalised on both sides; data matrices merged over the same-firm "
            "twins, twin-to-twin pairs dropped",
            f"D4: excluded instruments {self.instruments} ({', '.join(self.Z_names[self.X1.shape[1]:])}), standardised",
            "D6: micro weight diag(1/variance), second-choice weight 30 per row, both fixed; step 2 updates the "
            "aggregate block only",
            "aggregate block clustered by model (product_ids): make2 has 56 levels against the aggregate moments, so a "
            "make2-clustered S is singular" + ("; fixed-W aggregate block (robustness fixed_w)"
                                               if "fixed_w" in self.robustness else "; Sargan form"),
            f"PCM taste draws are scrambled Sobol normals crossed with the CPS households ({self.households:,} per year)"
            + ("; in antithetic pairs on every taste dimension (--antithetic-taste)" if self.antithetic_taste else ""),
            (f"D7: PCM extension, taste spreads on z log height, z log curb weight and z log number of trims (GMY's "
             f"X1, not in the second-choice set), start {EXTRA_SPREAD_START:g}: the exact PCM has no per-product error "
             f"to dilute the taste sorting of second choices (GMY_ROUND1 section 3); twins are merged on these columns "
             f"too"
             if self.extra_spreads else "D7 off (robustness gmy_spreads_only): GMY's 11 spreads only"),
            ("D8 undone (round 2): GMY's constant x income taste is free and fitted in the prefit beside phi_inc: the "
             "purchase-rate rows need a shifter of the buy margin that leaves the price paid alone, which alpha "
             "(phi_inc) is not" if "no_pi_const_inc" not in self.robustness else
             "D8 (robustness no_pi_const_inc): GMY's constant x income taste is fixed at 0, income acts through alpha "
             "(phi_inc) alone"),
            ("round 4: intensive-margin age taste pi_age on z log footprint and lux (age / 100), start 0: phi_age "
             "shifts log alpha and under the bound acts on who buys, the taste on what is bought (GMY_ROUND4 item 2)"
             if "no_age_taste" not in self.robustness else
             "robustness no_age_taste: age enters through phi_age alone (the round-2 layout)"),
        ]

    @property
    def plausible(self) -> dict[str, Any]:
        return {"own_elasticity_median": (-30.0, -4.0), "margin_median": (0.03, 0.40),
                "own_elasticity_times_window_width": (15.0, 30.0)}

    # ------------------------------------------------------------------ parameters and moments

    def layout(self) -> ParameterLayout:
        th, am = self.blp["theta"], self.alpha_median
        specs = []
        for k, c in enumerate(UNOBS):
            specs.append(ParameterSpec(f"sigma_{c}", "taste", c, f"nu{k}", min(th[f"sigma {c}"] / am, TASTE_CAP), 0.0,
                                       TASTE_CAP, cap=True, unit=0.1,
                                       note="BLP sigma / median |alpha|; cap $100k per sd"))
        for k, c in enumerate(self.extra_spreads):
            specs.append(ParameterSpec(f"sigma_{c}", "taste", c, f"nu{len(UNOBS) + k}", EXTRA_SPREAD_START, 0.0,
                                       TASTE_CAP, cap=True, unit=0.1,
                                       note="PCM extension D7 (not in GMY); cap $100k per sd"))
        for c, d in self.pis:
            if (c, d) in PIS_AGE:   # not in GMY's BLP layout
                specs.append(ParameterSpec(f"pi_{c}_{d}", "taste", c, d, 0.0, -TASTE_CAP, TASTE_CAP, unit=0.1,
                                           note="round 4: intensive-margin age taste, start 0"))
                continue
            v = th[f"{c} x {d}"] / am
            specs.append(ParameterSpec(f"pi_{c}_{d}", "taste", c, d, v, -TASTE_CAP, TASTE_CAP, unit=0.1,
                                       note="BLP pi / median |alpha|"))
        petrin = self.income_law == "petrin_bounded"
        specs.append(ParameterSpec("phi_gmy", "scale", None, "log_c_gmy", 0.0 if petrin else 1.0, 0.0, 2.0, free=False,
                                   unit=0.1, note="log alpha on log c_i of the BLP income-age law; fixed at 1"
                                   + (" (0 under --income-law petrin_bounded)" if petrin else "")))
        if petrin:
            specs.append(ParameterSpec("phi_y", "scale", None, "log_ybar_inc_w", 1.0, -1.0, 3.0, free=False, unit=0.1,
                                       note="log alpha on log(ybar / clip(income, 1, 150)); fixed at 1, Petrin's "
                                            "income law"))
        for name, basis in (("phi_inc", INCOME_BASIS_COLUMN[self.phi_inc_basis]), ("phi_age", "age")):
            specs.append(ParameterSpec(name, "scale", None, basis, 0.0, -PHI_BOUND, PHI_BOUND, unit=0.1,
                                       note="log alpha shift beyond the BLP law; an estimate within 1e-3 units of a "
                                            "bound is flagged AT_BOUND"))
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, SIGMA_P["start"], SIGMA_P["lower"],
                                   SIGMA_P["upper"], unit=SIGMA_P["unit"], note="sd of log a (untruncated parent)"))
        a = self.a_lo
        specs.append(ParameterSpec("a_lo", "bound", None, None, a, a, a, unit=0.1, free=False,
                                   note="a_i >= a_lo; fixed (LEARNINGS 32)"))
        lay = ParameterLayout(specs)
        if self.start_theta:
            bad = sorted(set(self.start_theta_except) - set(lay.names))
            if bad:
                raise ValueError(f"--start-theta-except: {bad} are not parameters of this layout")
            src = stage_theta(self.start_theta, lay.names)
            lay = lay.replace({n: {"start": float(np.clip(src[n], sp.lower, sp.upper))}
                               for n, sp in zip(lay.names, lay.specs)
                               if sp.free and n in src and n not in self.start_theta_except})
        if self.fixed:
            bad = sorted(set(self.fixed) - set(lay.names))
            if bad:
                raise ValueError(f"--fix: {bad} are not parameters of this layout")
            lay = lay.fixed(self.fixed)
        return lay

    def pcm_spec(self) -> PCMSpec:
        return PCMSpec(self.kernel_chars, track_realised_mass=False, trial_cold_evals=0, bound_kind="a")

    def micro_moment(self, rows: str = "all") -> list[GroupMicroMoments]:
        """GMY's 33 micro moments (V3) as one GroupMicroMoments, in load_micro_moments order, fixed covariance
        diag(variance) (D6). Rows whose household sets are empty in ``--years`` are dropped. ``rows="prefit"`` keeps
        the price-by-income, price-by-age and purchase-rate rows (20 of 33), the rows that pin phi_inc and phi_age, or
        under ``--prefit-rows age`` the 8 price-by-age rows."""
        tab = self.micro.table
        keep = []
        for k, (kind, char, sa, sb) in enumerate(group_moment_rows(self.micro)):
            if rows == "prefit" and not (tab.char.iloc[k] in ("Price", "PurchaseProb")):
                continue
            if rows == "prefit" and self.prefit_rows == "age" and not (
                    tab.char.iloc[k] == "Price" and str(tab.name.iloc[k]).startswith("Age")):
                continue
            if self.agents[sa].sum() > 0 and self.agents[sb].sum() > 0:
                keep.append((k, GroupMomentRow(kind, char, sa, sb)))
        if not keep:
            return []
        idx = [k for k, _ in keep]
        mk = [y for y in self.years if CEX_YEARS[0] <= y <= CEX_YEARS[1]
              or MRI_YEARS[0] <= y <= MRI_YEARS[1]]
        return [GroupMicroMoments("micro_gmy", MicroDataset("GMY_CEX_MRI", 1, market_ids=mk), [r for _, r in keep],
                                  tab.value.to_numpy(float)[idx], tab.variance.to_numpy(float)[idx],
                                  labels=list(tab.name.iloc[idx]), never_buyers=self.never_buyers)]

    def aggregate_moment(self) -> Any:
        """The aggregate block on xi: Sargan form (default) or fixed W (robustness ``fixed_w``), clustered by model."""
        iv = IVMoments if "fixed_w" in self.robustness else SarganIVMoments
        return iv(self.Z, self.clusters)

    def supply_moment(self) -> SupplyMoments:
        return SupplyMoments(self.Z_supply, self.cost_basis, self.products.firm_ids.to_numpy(), self.clusters,
                             name="supply", w_names=[f"gamma[{c}]" for k, c in enumerate(COST_BASIS)
                                                     if k in self.cost_keep],
                             rcond_min=1e-12, mc_floor=0.05, fd_step=1e-3, fd_scale="unit")

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str = "all",
                      layout_changes: dict[str, Any] | None = None) -> NFPGMMProblem:
        """``moments``: ``all`` the spec's estimation set (``--moments``: gmy, gmy_nolinear, demand_supply),
        ``prefit`` the 20 price and purchase-rate micro rows alone, ``aggregate`` the Sargan block alone (evaluation
        only), or one of the estimation set names explicitly."""
        which = self.moments if moments == "all" else MOMENT_ALIASES.get(moments, moments)
        if which not in MOMENT_SETS:
            raise ValueError(f"moments must be 'all' or one of {MOMENT_SETS}, got {moments!r}")
        layout = self.layout()
        if which == "prefit":   # the prefit rows identify phi alone: the problem is built with the rest fixed
            layout = layout.replace({n: {"free": n in self.prefit_free and n not in self.fixed} for n in layout.names})
        if layout_changes:
            layout = layout.replace(layout_changes)
        model = PCMModel(self.pcm_spec(), {"products": self.products, "agents": self.agents}, layout, nu, SEED + seed,
                         total_threads=threads, antithetic=self.antithetic_taste, cold_draws=self.cold_draws)
        if self.warm_delta:
            delta_w, theta_w = self.read_warm_delta()
            model.warm_from_delta(delta_w, theta_w if theta_w is not None and theta_w.size == layout.P
                                  else layout.starts())
        ms: list[Any] = []
        if which in ("gmy", "demand_supply", "aggregate"):
            ms.append(self.aggregate_moment())
        if which == "prefit":
            ms += self.micro_moment("prefit")
        elif which != "aggregate":
            ms += self.micro_moment() + self.second_choice_moment()
        if which == "demand_supply":
            ms.append(self.supply_moment())
        prob = NFPGMMProblem(model, ms, LinearIV(self.X1, self.Z, names=self.X1_names), layout,
                             max_step=self.use_max_step, evaluate_only=which == "aggregate")
        prob.normalisations = self.normalisations   # type: ignore[attr-defined]   # read by the pipeline's reports
        return prob

    def read_warm_delta(self) -> tuple[np.ndarray, np.ndarray | None]:
        """((n,) delta in this spec's row order, the stage's theta) from ``--warm-delta``: the stage's stored model
        state matched by year, and the theta it was solved at in this layout's order (by parameter name)."""
        assert self.warm_delta is not None
        st = read_stage_file(self.warm_delta)
        model = (st.get("problem_state") or {}).get("model") or {}
        ids = [str(x) for x in model.get("market_ids", [])]
        out = np.full(len(self.products), np.nan)
        mids = self.products.market_ids.to_numpy()
        for y in self.years:
            if str(y) not in ids:
                continue
            k = ids.index(str(y))
            rows = self.market_rows(y)
            d = np.asarray(model["delta"][k], dtype=float)
            if d.size == rows.size:
                out[rows] = d
        if not np.isfinite(out[np.isin(mids, [int(i) for i in ids])]).any():
            raise ValueError(f"--warm-delta {self.warm_delta}: no usable market state")
        if st.get("theta") is None:
            return out, None
        # the theta the delta was solved at, in this layout's order: a taste or scale term the stage's layout lacks
        # was 0 there (e.g. pi_const_inc in a no_pi_const_inc run, phi_y in a gmy_blp run); any other missing
        # parameter takes this layout's start
        try:
            src = stage_theta(self.warm_delta, self._hash_layout().names)
        except ValueError:   # no names and another length: build_problem warms at the layout starts
            return out, None
        lay = self.layout()
        return out, np.array([src.get(n, 0.0 if sp.role in ("taste", "scale") else sp.start)
                              for n, sp in zip(lay.names, lay.specs)])

    def second_choice_data(self) -> dict[int, list[float]]:
        out = {}
        raw = self.raw_products
        for y, v in self.second_choices.items():
            pr = raw[raw.market_ids == y]
            out[int(y)] = first_second_correlations(pr.shares, v["prob"],
                                                    pr[list(SECOND_CHOICE_CHARS)].to_numpy(float)).tolist()
        return out

    def market_rows(self, year: int) -> np.ndarray:
        """Rows of ``self.products`` of market ``year`` in the PCM model's order (price order, stable)."""
        p = self.products
        pos = np.flatnonzero(p.market_ids.to_numpy() == year)
        return pos[np.argsort(p.prices.to_numpy(float)[pos], kind="stable")]

    def merged_second_choice_data(self, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(data correlations (K,), x (J, K), data shares (J,)) of ``year`` on the merged product set in the model's
        price order (WP-G4 item 4, D3): the data joint law pi = (s / S) prob is summed over each twin group in rows and
        columns, a twin-to-twin pair (now the product with itself) is dropped, and the law is renormalised."""
        raw = self.raw_products[self.raw_products.market_ids == year].sort_values("position")
        prob = np.asarray(self.second_choices[year]["prob"], dtype=float)
        rows = self.market_rows(year)
        merged = self.products.iloc[rows]
        keys = [c for c in self.twin_keys if c != "prices"]
        to = twins.twin_targets(raw, merged, keys, raw_price="prices", merged_price="prices_data")
        # a raw product whose merged product was dropped (D9, to = -1) leaves the joint law, as first and second
        pi = twins.merge_joint_law(prob, raw.shares.to_numpy(float), to, rows.size)
        s_m = merged.shares.to_numpy(float)
        a = s_m / s_m.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(a[:, None] > 0, pi / a[:, None], 0.0)
        x = merged[list(SECOND_CHOICE_CHARS)].to_numpy(float)
        return first_second_correlations(s_m, C, x), x, s_m

    def second_choice_moment(self, weight: float = SC_WEIGHT) -> list[SecondChoiceMoments]:
        """GMY's second-choice correlation moments (V4) for the second-choice years in ``--years``: 11 characteristics
        x year, NaN data rows dropped (41 rows on all four years), weight 30 (D6), merged product set (D3)."""
        years = [y for y in SECOND_CHOICE_YEARS if y in self.years]
        if not years:
            return []
        corr, x, s = {}, {}, {}
        for y in years:
            corr[y], x[y], s[y] = self.merged_second_choice_data(y)
        return [SecondChoiceMoments("second_choice", years, SECOND_CHOICE_CHARS, corr, x, s, weight=weight)]

    def step2_weighting(self, problem: Any, ev: Any) -> np.ndarray:
        """W2 at the step-1 point (D6): the micro and second-choice blocks keep their fixed (n V)^-1; the aggregate block
        is efficient on the excluded instruments only. Under the default concentration beta is OLS, so the X1 rows of
        Z'xi are identically zero and the clustered S of the full aggregate block is singular (a make2 or model with a
        single cluster carries a row that is exactly zero). In the basis Z~ = Z A = [X1, M_X1 Z_ex], A = [[I, -G], [0,
        I]] with G = (X1'X1)^-1 X1'Z_ex, W_aa = A blockdiag((X1'X1/n)^-1, S_h^-1) A' with S_h the clustered covariance
        of the residualised excluded rows (A' S A restricted to them). That W keeps beta at OLS (the excluded rows of
        Z~'(delta - X1 b) do not depend on b), so the aggregate block is n h' S_h^-1 h with h = Z~_ex' xi / n: the
        aggregate J on the excluded instruments' degrees of freedom."""
        from nfp_harness.weighting import weight_from_covariance

        S = problem.covariance(ev)
        W = np.zeros_like(S)
        agg = getattr(problem, "agg", None)
        for name, sl in ev.blocks.items():
            if agg is not None and name == agg.name:
                continue
            W[sl, sl] = weight_from_covariance(S[sl, sl])
        if agg is not None:
            a = problem.agg_rows
            Z = np.asarray(agg.Z, dtype=float)
            X1 = self.X1
            K = X1.shape[1]
            if not np.array_equal(Z[:, :K], X1):
                raise ValueError("step2_weighting: the aggregate Z does not start with X1")
            n = Z.shape[0]
            G = np.linalg.lstsq(X1, Z[:, K:], rcond=None)[0]
            A = np.eye(Z.shape[1])
            A[:K, K:] = -G
            St = A.T @ S[a, a] @ A
            B = np.zeros_like(St)
            B[:K, :K] = np.linalg.inv(X1.T @ X1 / n)
            B[K:, K:] = weight_from_covariance(St[K:, K:])
            W[a, a] = A @ B @ A.T
        return 0.5 * (W + W.T)

    def hash_payload(self) -> dict[str, Any]:
        return {"data": "GMY replication package (Dataverse doi:10.7910/DVN/CZGOKP), prices spread to a $10 gap",
                "years": list(self.years), "households": self.households, "taste": self.kernel_chars, "unobs": UNOBS,
                "pis": self.pis, "extra_spreads": self.extra_spreads, "income_basis": self.income_basis,
                "x1": list(self.X1_names), "instruments": self.instruments, "z": list(self.Z_names),
                "merge": {"keys": self.twin_keys, "merged": self.n_merged}, "min_share": self.min_share, "moments": self.moments,
                "robustness": list(self.robustness), "a_lo": {"value": self.a_lo, "kind": "a"},
                "micro": {"rows": len(self.micro.table), "weight": "diag(1/variance) fixed", "never_buyers":
                          self.never_buyers},
                "second_choice": {"years": [y for y in SECOND_CHOICE_YEARS if y in self.years],
                                  "weight": SC_WEIGHT, "chars": SECOND_CHOICE_CHARS},
                "aggregate": "IVMoments" if "fixed_w" in self.robustness else "SarganIVMoments",
                "clusters": "product_ids", "blp_start": self.blp["source"],
                "layout": [dataclasses.asdict(s) for s in self._hash_layout().specs],
                "start_theta": self.start_theta, "start_theta_except": list(self.start_theta_except),
                "pcm": dataclasses.asdict(self.pcm_spec()), "optimizer": self.optimizer_options,
                "prefit": {"free": self.prefit_free, "max_iterations": self.prefit_max_iterations,
                           "moments": "20 price and purchase-rate micro rows" if self.prefit_rows == "price_purchase"
                           else "8 age-price micro rows"},
                "a_lo_band": list(self.a_lo_band_values), "warm_delta": self.warm_delta,
                "max_iterations": self.max_iterations, "plausible": self.plausible,
                # options added in round 4, only when set, so the payload of an older run is unchanged
                **({"income_law": {"law": self.income_law, "ybar_1000usd": self.income_law_ybar, "clip": INCOME_LAW_CLIP}}
                   if self.income_law != "gmy_blp" else {}),
                **({"draws": {"antithetic": "every taste dimension"}} if self.antithetic_taste else {}),
                **({"fix": dict(sorted(self.fixed.items()))} if self.fixed else {}),
                **({"cold_draws": self.cold_draws} if self.cold_draws else {})}

    def _hash_layout(self) -> ParameterLayout:
        """The layout as hashed: under ``--income-basis auto`` the level variant whatever the prefit chose, so the hash
        of a run does not change when its prefit picks a basis."""
        chosen = self.prefit_variant
        self.prefit_variant = None
        try:
            return self.layout()
        finally:
            self.prefit_variant = chosen

    def named_directions(self, theta: np.ndarray, units: np.ndarray) -> dict[str, np.ndarray]:
        """The utility-scale direction (sigma_p up, every taste spread down in proportion, LEARNINGS 1)."""
        names = self.layout().names
        u = np.asarray(theta, dtype=float) / np.asarray(units, dtype=float)
        scale = np.zeros(len(names))
        for k, n in enumerate(names):
            if n == "sigma_p":
                scale[k] = u[k]
            elif n.startswith("sigma_"):
                scale[k] = -u[k]
        return {"scale_direction": scale}

    def gate_profiles(self) -> list[tuple[str, str, dict[str, Any]]]:
        """The sigma_p ray profile (section 3): lambda in {0.5, 1, 2} times the start spreads, sigma_p in {1.5, ..,
        3.5}, phi_inc and phi_age concentrated at each point; NotMonotoneDecreasing on q_fixed. A diagnostic: step 1
        does not require it (``gate_required`` is empty)."""
        lay = self.layout()
        sig = {n: float(lay.specs[k].start) for k, n in enumerate(lay.names) if n.startswith("sigma_") and n != "sigma_p"}
        rules = {"q_fixed": {"rule": NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed", rise=3.84,
                                                           only={"lambda": 1.0}),
                             "columns": ["q_fixed", "q_fixed:aggregate", "q_fixed:micro_gmy", "q_fixed:second_choice"]}}
        return [("sigma_ray", "all", {"direction": sig, "lambdas": [0.5, 1.0, 2.0],
                                      "grid": {"sigma_p": [1.5, 2.0, 2.5, 3.0, 3.5]}, "origin": 0.0,
                                      "concentrate": ("phi_inc", "phi_age"), "rules": rules})]

    # ------------------------------------------------------------------ postestimation (section 4)

    def summary(self, problem: Any, result: Any) -> dict[str, Any]:
        """Every number of the comparison table at an estimate (section 4), by the definitions of ``estimate_blp.py``'s
        ``post_estimation`` where they exist."""
        model, theta = problem.model, np.asarray(result.theta, dtype=float)
        ev = result.evaluation
        delta = np.asarray(ev.aux["delta"], dtype=float)
        post = pp.postestimation(model, theta, delta, firm_ids=self.firm_column, groups=self.groups,
                                 group_other=self.group_other)
        pr = post["products"]
        mids = np.asarray(pr["market_ids"]).astype(int)
        ok_mk = {int(m["market_id"]) for m in post["markets"] if m["nash_status"] == "ok"}
        out: dict[str, Any] = {"convergence": result.convergence.value, "stage": result.stage, "q": float(result.q),
                               "q_blocks": dict(ev.aux.get("q_blocks") or {}),
                               "theta": dict(zip(problem.layout.names, map(float, theta))),
                               "at_bound": at_bound_parameters(theta, problem.layout.specs, problem.layout.units()),
                               "beta": dict(zip(ev.aux.get("beta_names", ()), map(float, ev.aux.get("beta", [])))),
                               "sd_xi": float(ev.aux.get("sd_xi", float("nan")))}
        own = np.asarray(pr["own_elasticity"], dtype=float)
        margin = np.asarray(pr["margin"], dtype=float)
        out["own_elasticity_median"] = float(np.nanmedian(own))
        out["own_elasticity_trimmed_mean"] = trimmed_mean(own)
        out["own_elasticity_median_by_year"] = {int(y): float(np.nanmedian(own[mids == y])) for y in self.years}
        out["own_elasticity_mean_by_year"] = {int(y): float(np.nanmean(own[mids == y])) for y in self.years}
        out["own_elasticity_trimmed_mean_by_year"] = {int(y): trimmed_mean(own[mids == y]) for y in self.years}
        out["near_twins_by_year"] = near_twins(self.products)
        out["lerner_median_by_year"] = {int(y): float(np.nanmedian(margin[mids == y])) for y in self.years
                                        if y in ok_mk}
        out["nash_singular_markets"] = [int(m) for m in post["nash_singular"]]
        out["margin_median"] = float(np.nanmedian(margin[np.isin(mids, list(ok_mk))])) if ok_mk else float("nan")
        out["share_own_elasticity_above_-1"] = float(np.mean(own[np.isfinite(own)] > -1.0))
        out["products_own_elasticity_above_-1"] = int(np.sum(own[np.isfinite(own)] > -1.0))   # Nash margin above 1
        ok = np.isin(mids, list(ok_mk))
        out["margin_median_sales_weighted"] = comparison.weighted_median(
            margin[ok], self.products.quantity.to_numpy(float)[ok]) if ok_mk else float("nan")
        out["table5"] = self.quintile_elasticities(model, theta, delta)
        out["diversion_by_vehicle_type"] = {
            str(g): {**{str(h): float(post["group_matrix"][a, b]) for b, h in enumerate(post["group_names"])},
                     "outside": float(post["group_matrix"][a, -1])} for a, g in enumerate(post["group_names"])}
        out["diversion_to_own_type"] = {str(g): float(post["group_matrix"][a, a])
                                        for a, g in enumerate(post["group_names"])}
        cs = {}
        for t, m in enumerate(model.markets):
            r = pp.consumer_surplus(model, theta, delta, m.market_id)
            cs[int(m.market_id)] = float(r["total"]) * 1e4 / float(np.sum(r["weights"]))   # $ per household
        out["cs_per_household_usd_by_year"] = cs
        wa = pp.window_anatomy(model, theta, delta)
        out["window_anatomy"] = {k: wa[k] for k in ANATOMY_COLUMNS}
        out["own_elasticity_times_window_width"] = -out["own_elasticity_median"] * wa["window_width_log_alpha"]
        # micro and second-choice tables
        names = problem.layout.names
        mic = next((m for m in problem.moments if m.name == "micro_gmy"), None)
        if mic is not None:
            pred = ev.aux["micro_predicted"]["micro_gmy"]
            out["micro"] = [{"name": lab, "data": float(d), "pcm": float(v), "z": float((v - d) / np.sqrt(var))}
                            for lab, d, v, var in zip(mic.labels, mic.values, pred, mic.variances)]
            out["micro_never_buyers"] = ev.aux.get("micro_gmy")
        sc = next((m for m in problem.moments if m.name == "second_choice"), None)
        if sc is not None:
            pred = ev.aux["micro_predicted"]["second_choice"]
            rows, k = {}, 0
            for y in sc.years:
                model_c = np.full(len(sc.chars), np.nan)
                model_c[sc.keep[y]] = pred[k:k + sc.keep[y].size]
                k += sc.keep[y].size
                rows[int(y)] = {"data": sc.data_corr[y].tolist(), "pcm": model_c.tolist()}
            out["second_choice"] = rows
            res = np.concatenate([np.asarray(r["pcm"]) - np.asarray(r["data"]) for r in rows.values()])
            out["second_choice_rmse"] = float(np.sqrt(np.nanmean(res ** 2)))
        out["phi"] = {n: float(theta[names.index(n)]) for n in ("phi_inc", "phi_age", "sigma_p")}
        return out

    def quintile_elasticities(self, model: Any, theta: np.ndarray, delta: np.ndarray) -> dict[int, dict[str, Any]]:
        """Table 5 by ``estimate_blp.py``'s definition: quintile demand s_jq = sum_{i in q} w_i s_ij, e_jq = (p_j /
        s_jq) d s_jq / d p_j, mean (and median) over the products with s_jq > 0 (in the PCM a product can have no buyer
        in a quintile; the count is reported), for 1980 and 2018 when present."""
        k = model._k
        th, ph, sp = model.kernel_params(theta)
        kw = model.kernel_kwargs(theta)
        out: dict[int, dict[str, Any]] = {}
        for t, m in enumerate(model.markets):
            y = int(m.market_id)
            if y not in TABLE5_YEARS:
                continue
            km = model.kernel_market(t)
            md = model.market_draws(t)
            quint = self.agents.inc_quintile.to_numpy()[np.asarray(md["agent_rows"])][md["agent_of_draw"]]
            row, med, trim, zeros = [], [], [], []
            for q in range(1, 6):
                w = np.asarray(km.weights) * (quint == q)
                sub = k.Market(np.asarray(km.x), np.asarray(km.p), w, np.asarray(km.taste_basis), list(km.taste_col),
                               np.asarray(km.scale_basis))
                r = k.evaluate_market(delta[np.asarray(m.rows)], sub, th, ph, sp, derivatives=False,
                                      price_derivatives=True, n_threads=model.total_threads, **kw)
                s = np.asarray(r["s"])
                pos = s > 0
                e = np.asarray(km.p)[pos] / s[pos] * np.diag(np.asarray(r["ds_dp"]))[pos]
                row.append(float(e.mean()))
                med.append(float(np.median(e)))
                trim.append(trimmed_mean(e))
                zeros.append(int((~pos).sum()))
            out[y] = {"mean_over_products": row, "median_over_products": med, "trimmed_mean_over_products": trim,
                      "products_without_buyers": zeros}
        return out

    def identification_summary(self, problem: Any, result: Any) -> dict[str, Any]:
        """The weak-identification plan of WP-G2 item 4 at an estimate: (b) the Cragg-Donald-style statistic on the
        aggregate rows of G alone and on all rows, (c) the aggregate block's q (Sargan J at step 2), (d) for each
        parameter the shares of ||W^1/2 G_k||^2 by block (which block pins which parameter)."""
        from nfp_harness.diagnostics.identification import block_shares, weak_identification_block

        ev = result.evaluation
        out: dict[str, Any] = {}
        if ev is None or ev.G is None:
            return out
        W = np.asarray(result.W if result.W is not None else problem.W, dtype=float)
        free = [k for k, s in enumerate(problem.layout.specs) if s.free]
        names = [problem.layout.names[k] for k in free]
        G = ev.G[:, free]
        out["block_shares"] = block_shares(G, W, ev.blocks, names)
        units = np.asarray(problem.layout.units(), dtype=float)[free]
        try:
            S = problem.covariance(ev)
            for tag, rows in (("all", slice(0, ev.g.size)),) + tuple(
                    (f"block:{b}", s) for b, s in ev.blocks.items()):
                blk = weak_identification_block(G[rows], S[rows, rows], problem.n, units=units)
                out[f"cragg_donald:{tag}"] = {k: blk.values.get(k) for k in ("statistic", "dof", "critical_95",
                                                                            "verdict")}
        except Exception as exc:  # noqa: BLE001 - reported, not fatal for post
            out["cragg_donald_error"] = f"{type(exc).__name__}: {exc}"
        out["q_aggregate"] = float((ev.aux.get("q_blocks") or {}).get("aggregate", float("nan")))
        return out

    def post(self, pipe: Any, problem: Any, result: Any, out: pathlib.Path) -> dict[str, Any]:
        summ = self.summary(problem, result)
        summ["identification"] = self.identification_summary(problem, result)
        inst = {"set": self.instruments, "excluded": self.Z_names[self.X1.shape[1]:], "rank_Z": int(self.Z.shape[1]),
                "rank_X1": int(self.X1.shape[1]), "first_stage": first_stage_table(self.products, self.X1)}
        (out / "instruments.json").write_text(json.dumps(inst, indent=1))
        summ["instruments"] = inst
        blp = {label: json.loads(f.read_text()) for label, f in BLP_SUMMARIES.items() if f.exists()}
        table = comparison_table({"PCM a_lo %g" % self.a_lo: summ}, blp, PAPER)
        (out / "comparison.txt").write_text(table + "\n")
        pipeline.log.info("comparison with GMY's BLP:\n%s", table)
        return summ

    def headlines(self, problem: Any, result: Any) -> dict[str, float]:
        s = self.summary(problem, result)
        return {"own_elasticity_median": s["own_elasticity_median"], "margin_median": s["margin_median"]}

    def band_row(self, pipe: Any, problem: Any, result: Any, value: float) -> dict[str, Any]:
        s = self.summary(problem, result)
        row: dict[str, Any] = {k: s[k] for k in ("own_elasticity_median", "own_elasticity_trimmed_mean", "margin_median",
                                                  "own_elasticity_times_window_width", "second_choice_rmse")
                               if k in s}
        row["at_bound"] = ";".join(s["at_bound"])
        row.update({f"q:{b}": v for b, v in s["q_blocks"].items()})
        row.update({f"lerner_median_{y}": v for y, v in s["lerner_median_by_year"].items()})
        row.update({f"theta:{n}": v for n, v in s["theta"].items()})
        return row

    a_lo_band_row = band_row


def comparison_table(pcm: Mapping[str, Mapping[str, Any]], blp: Mapping[str, Mapping[str, Any]],
              paper: Mapping[str, Any] | None = None) -> str:
    """comparison_table(pcm, blp, paper=None) -> the PCM-vs-BLP comparison table of the GMY spec (section 4) as
    text.

    ``pcm`` maps a column label to a PCM summary (``GMYSpec.summary``), ``blp`` a label to an ``estimate_blp.py`` JSON
    (``table5``, ``lerner_median_by_year``, ``second_choice``, ``micro``, ``own_elasticity_mean_by_year``), ``paper``
    the paper's quoted numbers (definitions unverified). Every row uses one definition for every column; a missing
    entry prints as ``-``. Second-choice means are over the years of the first PCM column.
    """
    paper = dict(paper or {})
    cols = list(pcm) + list(blp) + (["paper (definition UNVERIFIED)"] if paper else [])
    first = next(iter(pcm.values()), {})
    years = sorted(int(y) for y in (first.get("lerner_median_by_year") or first.get("own_elasticity_median_by_year")
                                    or {}))
    lines: list[str] = []
    w = max(14, *(len(c) + 2 for c in cols))

    def row(label: str, vals: Sequence[Any], digits: int = 3) -> None:
        lines.append(f"{label:52s}" + "".join(f"{_fmt(v, digits):>{w}s}" for v in vals))

    lines.append(f"{'':52s}" + "".join(f"{c:>{w}s}" for c in cols))
    lines.append("Own-price elasticity by income quintile, mean over products (quintile demand s_jq, e_jq = p_j/s_jq "
                 "ds_jq/dp_j; PCM: mean / 5-95% trimmed mean [median])")
    for y in (1980, 2018):
        for q in range(5):
            vals: list[Any] = []
            for s in pcm.values():
                t = _year_get(s.get("table5"), y)
                trim = t.get("trimmed_mean_over_products") if t else None
                vals.append(f"{_fmt(t['mean_over_products'][q], 2)}"
                            + (f" / {_fmt(trim[q], 2)}" if trim else "")
                            + f" [{_fmt(t['median_over_products'][q], 2)}]"
                            if t and "median_over_products" in t else (t["mean_over_products"][q] if t else None))
            for b in blp.values():
                t = _year_get(b.get("table5"), y)
                vals.append(t["mean_over_products"][q] if t else None)
            if paper:
                t = _year_get(paper.get("table5"), y)
                vals.append(t[q] if t else None)
            row(f"  {y} Q{q + 1}", vals, 2)
    lines.append("Median Lerner index (p - mc)/p, Nash with parent ownership, median over products")
    for y in years:
        vals = [_year_get(s.get("lerner_median_by_year"), y) for s in pcm.values()]
        vals += [_year_get(b.get("lerner_median_by_year"), y) for b in blp.values()]
        if paper:
            vals.append(" / ".join(f"{_fmt(_year_get(v, y))}" for v in paper.get("lerner_median", {}).values())
                        if y in (1980, 2018) else None)
        row(f"  {y}", vals)
    na = [None] * (len(blp) + (1 if paper else 0))
    row("  all years: median over products", [s.get("margin_median") for s in pcm.values()] + na)
    row("  all years: sales-weighted median", [s.get("margin_median_sales_weighted") for s in pcm.values()] + na)
    row("  products with |own elasticity| < 1 (margin > 1)", [s.get("products_own_elasticity_above_-1")
                                                             for s in pcm.values()] + na, 0)
    lines.append("Own-price elasticity, mean over products (PCM: mean / 5-95% trimmed mean [median])")
    for y in years:
        vals: list[Any] = [f"{_fmt(_year_get(s.get('own_elasticity_mean_by_year'), y), 2)} / "
                           f"{_fmt(_year_get(s.get('own_elasticity_trimmed_mean_by_year'), y), 2)} "
                           f"[{_fmt(_year_get(s.get('own_elasticity_median_by_year'), y), 2)}]" for s in pcm.values()]
        vals += [_year_get(b.get("own_elasticity_mean_by_year"), y) for b in blp.values()]
        if paper:
            vals.append(None)
        row(f"  {y}", vals, 2)
    sc_years = sorted(int(y) for y in (first.get("second_choice") or {}))
    if sc_years:
        lines.append(f"Second-choice correlation of first and second choice, mean over {sc_years} (data: merged "
                     f"products; paper: Table 6 data / model)")
        chars = list((paper.get("table6_data") or {}).keys()) or [f"x{k}" for k in range(11)]
        for c_i, c in enumerate(chars):
            def mean_of(sc: Mapping[Any, Any] | None, key: str) -> Any:
                v = [(_year_get(sc, y) or {}).get(key, [None] * len(chars))[c_i] for y in sc_years] if sc else []
                v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
                return sum(v) / len(v) if v else None
            vals = [f"{_fmt(mean_of(s.get('second_choice'), 'pcm'))} ({_fmt(mean_of(s.get('second_choice'), 'data'))})"
                    for s in pcm.values()]
            vals += [mean_of(b.get("second_choice"), "model") for b in blp.values()]
            if paper:
                vals.append(f"{_fmt(paper['table6_data'].get(c), 2)} / {_fmt(paper['table6_model'].get(c), 2)}")
            row(f"  {c} model (data)", vals)
        row("  RMSE model - data (PCM years)", [s.get("second_choice_rmse") for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    mic = first.get("micro") or []
    if mic:
        lines.append("Micro moments (V3): data, then each model")
        bl = [{r["name"]: r.get("model") for r in (b.get("micro") or [])} for b in blp.values()]
        row("  name / data", ["data"] + [""] * (len(cols) - 1))
        for r in mic:
            vals = [f"{_fmt(next((x['pcm'] for x in (s.get('micro') or []) if x['name'] == r['name']), None))}"
                    for s in pcm.values()] + [d.get(r["name"]) for d in bl]
            if paper:
                vals.append(None)
            lines.append(f"  {r['name']:42s}{_fmt(r['data']):>10s}" + "".join(f"{_fmt(v):>{w}s}" for v in vals))
    lines.append("Diversion to the product's own vehicle type (share-weighted, pooled over markets)")
    for g in ("car", "van", "suv", "truck"):
        row(f"  {g}", [(s.get("diversion_to_own_type") or {}).get(g) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    lines.append("Consumer surplus per household, $ (PCM frontier formula, 1/alpha tail bounded by a_lo; not a headline)")
    for y in years:
        row(f"  {y}", [_year_get(s.get("cs_per_household_usd_by_year"), y) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)), 0)
    lines.append("Window anatomy and fit")
    for k in ("frontier_products_per_buying_draw", "log_alpha_range_buyers", "window_width_log_alpha"):
        row(f"  {k}", [(s.get("window_anatomy") or {}).get(k) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    bins = next((list(h) for s in pcm.values() if (h := (s.get("window_anatomy") or {}).get("frontier_size_mass"))),
                [])
    for b in bins:
        row(f"  buyer mass on frontiers of {b} product{'' if b == '1' else 's'}",
            [((s.get("window_anatomy") or {}).get("frontier_size_mass") or {}).get(b) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)))
    row("  own_elasticity_times_window_width", [s.get("own_elasticity_times_window_width") for s in pcm.values()]
        + [None] * (len(blp) + (1 if paper else 0)), 2)
    row("  convergence", [s.get("convergence") for s in pcm.values()] + [None] * (len(blp) + (1 if paper else 0)))
    row("  q", [s.get("q") for s in pcm.values()] + [b.get("q") for b in blp.values()] + ([None] if paper else []), 2)
    for blk in (first.get("q_blocks") or {}):
        row(f"  q:{blk}", [(s.get("q_blocks") or {}).get(blk) for s in pcm.values()]
            + [None] * (len(blp) + (1 if paper else 0)), 2)
    row("  AT_BOUND", [";".join(s.get("at_bound") or []) or "none" for s in pcm.values()]
        + [None] * (len(blp) + (1 if paper else 0)))
    return "\n".join(lines)


def stage_theta(path: str | pathlib.Path, names: Sequence[str]) -> dict[str, float]:
    """{parameter name: value} of a stored stage's theta, named by its run's history.csv (``stage_parameter_names``);
    a stage without names is read positionally against ``names``, which then must have its length."""
    th = np.asarray(read_stage_file(path)["theta"], dtype=float)
    src = stage_parameter_names(path)
    if src is None or len(src) != th.size:
        if th.size != len(names):
            raise ValueError(f"{path}: {th.size} parameters without names, this layout has {len(names)}")
        src = list(names)
    return dict(zip(src, map(float, th)))


def trimmed_mean(v: np.ndarray, trim: tuple[float, float] = TRIM) -> float:
    """Mean of the finite values between the ``trim`` quantiles (5% and 95% by default): the own-elasticity tail of
    near-twin products is a draw artefact (GMY_ROUND1 section 4), the trimmed mean is not."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    lo, hi = np.quantile(v, trim)
    return float(v[(v >= lo) & (v <= hi)].mean())


def near_twins(products: pd.DataFrame, gap: float = NEAR_TWIN_GAP) -> dict[int, int]:
    """Per year, the products whose data price is within ``gap`` ($100) of another product of the same vehicle type."""
    out = {}
    for y, g in products.groupby("market_ids"):
        n = 0
        for _, h in g.groupby("veh_type"):
            p = np.sort(h.prices_data.to_numpy(float))
            if p.size > 1:
                d = np.diff(p)
                near = np.minimum(np.r_[np.inf, d], np.r_[d, np.inf]) < gap
                n += int(near.sum())
        out[int(y)] = n
    return out


def blp_sums(p: pd.DataFrame) -> np.ndarray:
    """BLP sums of the style and segment dummies and of the three continuous characteristics over the firm's other
    products and over rivals (market = year, firm = parent), (n, 18)."""
    return iv.blp_sums(p, BLP_DUMMIES + GH_CONTINUOUS)


def gh_instruments(p: pd.DataFrame, version: str = "local") -> np.ndarray:
    """Gandhi-Houde differentiation instruments on the three continuous characteristics, own-firm and rival, (n, 6)."""
    return iv.gh_instruments(p, GH_CONTINUOUS, version)


def excluded_instruments(p: pd.DataFrame, which: str = "gmy") -> tuple[np.ndarray, list[str]]:
    """(Z_ex (n, k), names) for ``--instruments``: ``gmy`` = GMY_EXCLUDED (5), ``gh`` = Gandhi-Houde local (6),
    ``gmy+gh`` = both (11), every column standardised so the rank tolerance of ``drop_collinear`` is scale-free."""
    if which not in INSTRUMENT_SETS:
        raise ValueError(f"unknown instrument set {which!r}")
    return iv.excluded_instruments(p, GMY_EXCLUDED if which in ("gmy", "gmy+gh") else (),
                                   GH_CONTINUOUS if which in ("gh", "gmy+gh") else ())


def first_stage_table(p: pd.DataFrame, X1: np.ndarray) -> list[dict[str, Any]]:
    """Relevance screen of each candidate excluded set (WP-G2 facts): columns, rank added to X1, the homoskedastic
    first-stage F of price on the set after partialling X1, and the R^2 it adds. Data only."""
    sets: list[tuple[str, np.ndarray]] = [
        ("lag_pl_con (GMY baseline)", p[["lag_pl_con"]].to_numpy(float)),
        ("+ iv_prod", p[["lag_pl_con", "iv_prod"]].to_numpy(float)),
        ("+ steel_weight", p[["lag_pl_con", "iv_prod", "steel_weight"]].to_numpy(float)),
        ("gmy: + IV_N, IV_N_weight (default)", excluded_instruments(p, "gmy")[0]),
        ("lag_pl_con, iv_prod, IV_N, IV_N_weight (stub)", p[["lag_pl_con", "iv_prod", "IV_N", "IV_N_weight"]]
         .to_numpy(float)),
        ("BLP sums", blp_sums(p)),
        ("gh: Gandhi-Houde local", gh_instruments(p, "local")),
        ("Gandhi-Houde quadratic", gh_instruments(p, "quadratic")),
        ("gmy+gh", excluded_instruments(p, "gmy+gh")[0]),
    ]
    return iv.first_stage_table(p.prices.to_numpy(float), X1, sets)


def evaluate(args: argparse.Namespace) -> None:
    if args.threads > 64:
        raise SystemExit("at most 64 threads")
    RESULTS.mkdir(exist_ok=True)
    spec = GMYSpec(tuple(args.years), instruments=args.instruments)
    prob = spec.build_problem(args.nu, 0, args.threads, args.moments)
    instruments = {"set": args.instruments, "excluded": spec.Z_names[spec.X1.shape[1]:], "rank_Z": spec.Z.shape[1],
                   "rank_X1": spec.X1.shape[1], "first_stage": first_stage_table(spec.products, spec.X1)}
    model = prob.model
    theta = spec.layout().starts()
    t0 = time.time()
    ev = prob.evaluate(theta, need_jacobian=False)
    wall = time.time() - t0
    sol = ev.aux["solution"]
    delta = np.asarray(sol.delta)
    W = np.eye(ev.g.size)
    per = []
    for m, pm in zip(model.markets, sol.per_market):
        rows = np.asarray(m.rows)
        s_model = np.asarray(sol.outputs["shares"])[rows]
        s_data = spec.products.shares.to_numpy(float)[rows]
        pm = dict(pm)
        pm["max_rel_share_error"] = float(np.max(np.abs(s_model / s_data - 1))) if np.all(np.isfinite(s_model)) \
            else float("nan")
        pm["products_inverted"] = int(np.sum(np.isfinite(np.asarray(delta)[rows])))
        per.append(pm)
    anatomy = pp.window_anatomy(model, theta, delta, n_threads=args.threads) if ev.status.value == "ok" else {}
    micro_rows = []
    if ev.ok and "micro_gmy" in ev.aux.get("micro_predicted", {}):
        mom = next(m for m in prob.moments if m.name == "micro_gmy")
        blp = {r["name"]: r["model"] for r in json.loads(BLP_FIT.read_text()).get("micro", [])} if BLP_FIT.exists() \
            else {}
        for lab, d, v in zip(mom.labels, ev.aux["micro_data"]["micro_gmy"], ev.aux["micro_predicted"]["micro_gmy"]):
            micro_rows.append({"name": lab, "data": float(d), "pcm": float(v), "blp_fit": blp.get(lab)})
            print(f"{lab:45s} data {d:9.4f}  PCM {v:9.4f}  BLP {blp.get(lab, float('nan')):9.4f}")
    out = {"years": spec.years, "nu": args.nu, "draws_per_market": [m.R for m in model._mk], "status": ev.status.value,
           "wall_seconds": wall, "q_identity_w": float(ev.g @ W @ ev.g) if ev.ok else float("nan"), "theta": dict(zip(spec.layout().names,
                                                                                           theta.tolist())),
           "alpha_median_blp": spec.alpha_median, "blp_source": spec.blp["source"],
           "twins_merged": spec.n_merged, "price_structure_raw": spec.structure_raw,
           "price_structure_after_merge": spec.structure, "per_market": per, "window_anatomy": anatomy,
           "second_choice_data": spec.second_choice_data(),
           "moments": {m.name: getattr(m, "k", None) for m in prob.moments}, "instruments": instruments,
           "identification": prob.identification, "micro": micro_rows,
           "micro_never_buyers": ev.aux.get("micro_gmy") if ev.ok else None}
    tag = "_".join(str(y) for y in spec.years)
    path = RESULTS / f"pcm_evaluate_nu{args.nu}_{args.moments}_{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("status", "wall_seconds", "twins_merged", "per_market", "window_anatomy")},
                     indent=1, default=str))
    print("wrote", path)


def main() -> int:
    """``evaluate`` (one cold evaluation at the starting theta) or the pipeline's ``run`` / ``status``."""
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        ap = argparse.ArgumentParser(description="evaluate the GMY PCM problem once at the starting theta (cold)")
        ap.add_argument("command")
        ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
        ap.add_argument("--nu", type=int, default=4)
        ap.add_argument("--threads", type=int, default=32)
        ap.add_argument("--moments", choices=MOMENT_SETS + tuple(MOMENT_ALIASES), default="gmy",
                        help="gmy: Sargan + micro + second choice; gmy_nolinear: no aggregate block; demand_supply: "
                             "gmy + supply; aggregate: Sargan block alone (evaluation only)")
        ap.add_argument("--instruments", choices=INSTRUMENT_SETS, default="gmy",
                        help="excluded demand instruments: gmy (GMY's 5 cost shifters and counts), gh (Gandhi-Houde "
                             "local), gmy+gh")
        evaluate(ap.parse_args())
        return 0
    return pipeline.main(GMYSpec(), default_nu=16, default_base=HARNESS)


if __name__ == "__main__":
    sys.exit(main())

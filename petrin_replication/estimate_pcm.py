"""Petrin (2002) minivans, exact pure characteristics model (PCM), as a spec on nfp_harness.

    u_ijt = delta_jt + x_jt . t_i - alpha_i p_jt,   u_i0t = 0
    x     = [1, hpwt, space, air, mpd, fwd, fam],   fam = max(mi, sw, su, pv)   (the taste columns, TASTE)
    t_ik  = sigma_k nu_ik on the 7 taste columns  +  pi_b log(fs_i) fv_i on b in {mi, sw, su, pv}
    log alpha_i = phi_mid mid_i + phi_high high_i + sigma_p nu_ip        (low income is the reference, c = 1)
    delta_jt = X1_jt beta + xi_jt,  X1 = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv, pgnp, trend, trend2]
    Z     = [X1, demand_instruments0..21]  (35 columns; LEARNINGS 16)

Moments: the 35 aggregate IV moments and Petrin's 10 CEX micro moments (pyblp's Petrin data), with the
micro block of W1 and S fixed at Petrin's sample covariances (PETRIN_COVARIANCES_LOCATION): 45 moments for
13 linear and 14 nonlinear parameters. Normalisations: median alpha = 1 in the low-income group (mu = 0,
LEARNINGS 11); sigma_fam capped at $25k; taste spreads non-negative; u_i0 = 0 in every year with pgnp and
trends in X1. Data: the 17 same-firm exact twins (identical price and characteristics) are merged into one
row with summed shares, then the remaining price ties are spread to a $10 gap. Consumers: the 1,000 CEX
agents per market crossed with scrambled Sobol standard normal taste draws, 2^ceil(log2(1000 nu)) draws per
market. Exact PCM shares, no smoothing (LEARNINGS 15). Specification: LEARNINGS 19. Stages and their
files: nfp_harness.pipeline (DESIGN 6.4); the gate is a sigma_p profile concentrated in (phi_mid, phi_high).

Robustness switches (each changes the run directory): ``--robustness body_sigma`` frees taste spreads on
mi, sw, su, pv (cap $25k); ``--robustness income_winsor`` adds phi_y log(ybar / clip(income, 1, 150)) to
log alpha with phi_y fixed at 1 (Petrin's alpha ~ 1/income, winsorised).

Run (results under petrin_replication/results/harness/, untracked):
  /home/istomin/anaconda3/bin/python petrin_replication/estimate_pcm.py run --stage prefit --nu 4
  /home/istomin/anaconda3/bin/python petrin_replication/estimate_pcm.py run --stage gate --nu 4 [--detach]
  /home/istomin/anaconda3/bin/python petrin_replication/estimate_pcm.py run --stage all --nu 16 --detach
  /home/istomin/anaconda3/bin/python petrin_replication/estimate_pcm.py status --nu 16
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
import json  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyblp  # noqa: E402

from nfp_harness import comparison, pipeline  # noqa: E402
from nfp_harness.data import break_price_ties  # noqa: E402
from nfp_harness.linear import LinearIV  # noqa: E402
from nfp_harness.models import pcm_post as pp  # noqa: E402
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.moments import IVMoments, MicroDataset, MicroPart, RatioMicroMoment  # noqa: E402
from nfp_harness.params import ParameterLayout, ParameterSpec  # noqa: E402
from nfp_harness.problem import NFPGMMProblem  # noqa: E402
from nfp_harness.profile import NotMonotoneDecreasing  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results" / "harness"
CHARS = ("const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")   # the X1 basis
X1_COLS = CHARS + ("pgnp", "trend", "trend2")
BODY = ("mi", "sw", "su", "pv")
TASTE = ("const", "hpwt", "space", "air", "mpd", "fwd", "fam")   # columns with a taste spread
KERNEL_CHARS = TASTE + BODY   # columns of x in the kernel: the body dummies carry pi_b (and body sigmas)
YEAR = 1984
SIG_REF = {"sigma_const": 2.0, "sigma_hpwt": 2.0, "sigma_space": 2.0, "sigma_air": 1.0, "sigma_mpd": 1.0,
           "sigma_fwd": 2.0, "sigma_fam": 2.0}
SIGMA_STARTS = {"const": 3.0, "hpwt": 2.0, "space": 1.0, "air": 0.5, "mpd": 1.0, "fwd": 1.0, "fam": 2.0}
PI_STARTS = {"mi": 0.75, "sw": 0.28, "su": 0.20, "pv": 0.41}   # pyblp BLP-micro pi / 0.65 (SPEC 3.2)
FAM_CAP = 25.0       # $1000: 2 (1 - Phi(66.1 / 25)) = 0.8% of consumers value fam beyond the price range
BODY_SIGMA_CAP = 25.0
INCOME_CLIP = (1.0, 150.0)
TWIN_KEYS = ("market_ids", "firm_ids", "prices", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")
ROBUSTNESS = ("body_sigma", "income_winsor")
MICRO_COVARIANCE_SOURCE = "pyblp.data.PETRIN_COVARIANCES_LOCATION, fixed"
BLP_SUMMARIES = {"BLP micro": "blp_micro_summary.json", "BLP no micro": "blp_nomicro_summary.json",
                 "BLP demand micro": "blp_demand_micro_summary.json"}


def merge_twins(products: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows equal on TWIN_KEYS (same market, firm, price and characteristics) are one product to every
    consumer in the PCM: keep the first row with the summed shares (and quantities). Returns (products,
    number of rows merged away)."""
    g = products.groupby(list(TWIN_KEYS), sort=False)
    first = (g.cumcount() == 0).to_numpy()
    out = products.copy()
    for c in ("shares", "q"):
        out[c] = g[c].transform("sum")
    return out[first].reset_index(drop=True), int((~first).sum())


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
    agents["logfs_fv"] = np.log(agents.fs) * agents.fv
    # Price scale c_i = alpha_i / a_i: log c_i = phi_mid mid_i + phi_high high_i, so c_i takes three values
    # and is bounded by construction (LEARNINGS 18, 19). The income_winsor robustness run adds
    # phi_y log(ybar / income_i) with income clipped to [1, 150] ($1000), so c_i spans at most a factor 150.
    agents["log_ybar_inc_w"] = np.log(float(agents.income.median()) / agents.income.clip(*INCOME_CLIP))
    agents["income_group"] = np.select([agents.low == 1, agents.mid == 1, agents.high == 1], ["low", "mid", "high"], "none")
    agents["fs_group"] = np.where(agents.fs >= 5, "5+", agents.fs.astype(int).astype(str))
    return products, agents, price_range, n_merged


class PetrinSpec:
    """The Petrin PCM spec read by nfp_harness.pipeline (interface in its module docstring)."""

    name = "petrin_pcm"
    firm_column = "firm_ids"
    groups = "body"
    group_other = "car"
    binary_columns = ("air", "fwd", "fam") + BODY
    prefit_free = ("phi_mid", "phi_high")
    prefit_max_iterations = 10
    extensive_free = ("phi_mid", "phi_high", "sigma_p", "sigma_const")
    gate_concentrate_iterations = 20
    gate_required = "micro"

    def __init__(self, robustness: tuple[str, ...] = ()) -> None:
        self.robustness: tuple[str, ...] = tuple(sorted(set(robustness)))
        self.products, self.agents, self.price_range, self.n_merged_twins = load_data()
        p = self.products
        self.X1 = np.column_stack([np.ones(len(p))] + [p[c].to_numpy(float) for c in X1_COLS[1:]])
        self.Z = np.column_stack([self.X1] + [p[f"demand_instruments{k}"].to_numpy(float) for k in range(22)])
        self.linear_names = list(X1_COLS)
        self.clusters = p.clustering_ids.to_numpy()
        self.values = dict(pd.read_csv(pyblp.data.PETRIN_VALUES_LOCATION).iloc[:, :2].to_numpy())
        self.micro_covariances = pd.read_csv(pyblp.data.PETRIN_COVARIANCES_LOCATION, index_col=0)
        self.normalisations = [
            "median alpha = 1 in the low-income group: log alpha_i = phi_mid mid_i + phi_high high_i + log a_i, "
            "a_i ~ LogNormal(0, sigma_p), no constant (mu = 0; LEARNINGS 11)",
            f"taste spread on the family-hauler factor fam = max(mi, sw, su, pv) capped at {FAM_CAP:g} ($1000), "
            f"so under 1% of consumers value it beyond the pooled price range {self.price_range:.1f}",
            "",   # body sigmas, set by set_robustness
            "taste spreads non-negative (draws are symmetric)",
            "u_i0 = 0 in every year, with pgnp, trend and trend2 in X1 (harmless for the 1984 counterfactual)",
            "PCM taste draws are scrambled Sobol standard normals crossed with pyblp's CEX agents; pyblp's nodes "
            "are not used (owner decision 3)",
        ]
        self.set_robustness(self.robustness)
        self.plausible = {   # SPEC section 4
            "own_elasticity_median": (-8.0, -3.0), "share_own_elasticity_below_-20": (None, 0.05),
            "nonzero_cross_median": (10.0, None), "diversion_leaving_group_median:mi": (0.15, None),
            "margin_median": (0.15, 0.40), "inside_share_elasticity_median": (-2.0, -0.5),
            "sd_xi_over_sd_delta": (None, 1.0), "xi_kurtosis": (None, 10.0),
            "quality_dispersion_buyer_dollars_median": (1.5, 6.0),
            "spread_buyer_dollars": {"sigma_fam": (None, FAM_CAP), "sigma_const": (None, 15.0)},
        }

    # ------------------------------------------------------------------ robustness switches and CLI

    def set_robustness(self, robustness: Any) -> None:
        r = tuple(sorted(set(robustness or ())))
        unknown = [x for x in r if x not in ROBUSTNESS]
        if unknown:
            raise ValueError(f"unknown robustness switches {unknown}; known: {ROBUSTNESS}")
        self.robustness = r
        self.normalisations[2] = (
            "taste spreads on the body dummies mi, sw, su, pv fixed at 0 (only second-choice data would pin "
            "them, LEARNINGS 14)" if "body_sigma" not in r else
            f"taste spreads on the body dummies mi, sw, su, pv free and capped at {BODY_SIGMA_CAP:g} ($1000) "
            f"(robustness body_sigma)")

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--robustness", action="append", choices=ROBUSTNESS, default=None,
                            help="robustness switch (repeatable): body_sigma frees the 4 body taste spreads "
                                 "(cap 25); income_winsor adds phi_y = 1 on log(ybar / clip(income, 1, 150))")

    def configure(self, args: Any) -> None:
        self.set_robustness(getattr(args, "robustness", None))

    def run_key(self) -> dict[str, Any]:
        """Result-changing options beyond nu and seed: the robustness switches (none: the main run)."""
        return {"robustness": list(self.robustness)} if self.robustness else {}

    # ------------------------------------------------------------------ parameters and moments

    def layout(self) -> ParameterLayout:
        """Nonlinear parameters (SPEC 2, change list 3-6). Body sigmas exist only under ``body_sigma``, so the
        Sobol dimension of the main run is the 7 taste columns."""
        r, p = self.price_range, self.products
        specs = []
        for k, c in enumerate(TASTE):
            name, start = f"sigma_{c}", SIGMA_STARTS[c]
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
        specs += [ParameterSpec("phi_mid", "scale", None, "mid", 0.15, -3.0, 3.0,
                                note="log alpha shift for mid income (low = 0)"),
                  ParameterSpec("phi_high", "scale", None, "high", -0.30, -3.0, 3.0,
                                note="log alpha shift for high income (low = 0)")]
        winsor = "income_winsor" in self.robustness
        specs.append(ParameterSpec("phi_y", "scale", None, "log_ybar_inc_w", 1.0 if winsor else 0.0, -1.0, 3.0,
                                   free=False, note="log alpha on log(ybar / clip(income, 1, 150)); fixed, 1 under "
                                                    "income_winsor, else 0"))
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, 1.0, 0.1, 2.5, unit=0.1, note="sd of log alpha"))
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
    optimizer_options = {"Delta_0": 0.1, "Delta_max": 0.5}

    def pcm_spec(self) -> PCMSpec:
        # At a trial phi step from the pre-fit's second point, markets 1990-1993 failed the warm stage (2,500
        # evaluations, 60-73 s), the theta path (stopped at tau 0.25-0.5) and then a 40,000-evaluation cold
        # stage (860-1,380 s): 26 minutes to learn that the step must be rejected. Trial points therefore get
        # no cold stage; the first evaluation and the cold profile points keep the full budget.
        return PCMSpec(KERNEL_CHARS, track_realised_mass=self.use_max_step, trial_cold_evals=0)

    def hash_payload(self) -> dict[str, Any]:
        return {"data": "pyblp 1.3.0 Petrin, prices spread to a $10 gap", "chars": CHARS, "taste": TASTE,
                "kernel_chars": KERNEL_CHARS, "x1": X1_COLS,
                "merge": {"rule": "same TWIN_KEYS: keep first row, sum shares and q", "keys": TWIN_KEYS},
                "instruments": "petrin", "z": "X1 + demand_instruments0..21",
                "layout": [dataclasses.asdict(s) for s in self.layout().specs],
                "micro": {"values": self.values, "observations": 29125, "covariance": MICRO_COVARIANCE_SOURCE},
                "caps": {"sigma_fam": FAM_CAP, "body_sigma": BODY_SIGMA_CAP, "income_clip": INCOME_CLIP},
                "robustness": list(self.robustness), "pcm": dataclasses.asdict(self.pcm_spec()),
                "draws": "Sobol normals, seed + market position, crossed with agents", "max_step": self.use_max_step,
                "optimizer": self.optimizer_options,
                "prefit": {"free": self.prefit_free, "max_iterations": self.prefit_max_iterations},
                "extensive_free": self.extensive_free, "gate_concentrate_iterations": self.gate_concentrate_iterations}

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str,
                      layout_changes: dict[str, Any] | None = None) -> NFPGMMProblem:
        layout = self.layout()
        if layout_changes:
            layout = layout.replace(layout_changes)
        model = PCMModel(self.pcm_spec(), {"products": self.products, "agents": self.agents}, layout, nu, seed,
                         total_threads=threads)
        ms: list[Any] = [IVMoments(self.Z, self.clusters)]
        if moments in ("all", "prefit"):
            ms.append(self.micro_moment("all" if moments == "all" else "income"))
        return NFPGMMProblem(model, ms, LinearIV(self.X1, self.Z, names=X1_COLS), layout, max_step=self.use_max_step)

    def named_directions(self, theta: np.ndarray, units: np.ndarray) -> dict[str, np.ndarray]:
        """Known flat directions in natural units at theta (DESIGN 5.2): the utility-scale direction
        (sigma_p up, every taste spread down in proportion, LEARNINGS 1) and the body-level direction
        (sigma_fam, any body sigmas and the family-size pis up in proportion, LEARNINGS 14)."""
        names = self.layout().names
        u = np.asarray(theta, dtype=float) / np.asarray(units, dtype=float)
        scale, body = np.zeros(len(names)), np.zeros(len(names))
        for k, n in enumerate(names):
            if n == "sigma_p":
                scale[k] = u[k]
            elif n.startswith("sigma_"):
                scale[k] = -u[k]
            if n in {f"sigma_{b}" for b in BODY + ("fam",)} | {f"pi_{b}" for b in BODY}:
                body[k] = u[k]
        return {"scale_direction": scale, "body_level_direction": body}

    # ------------------------------------------------------------------ gate (DESIGN 6.4)

    def gate_profiles(self) -> list[tuple[str, str, dict[str, Any]]]:
        """One sigma ray profile with all moments evaluated (DESIGN 6.4), concentrated in (phi_mid, phi_high):
        at every grid point the two income shifts are re-solved with everything else held, so the income purchase
        rates do not drift with sigma_p (LEARNINGS 19). Read twice: the micro verdict on
        q_fixed, and the no-micro verdict on its aggregate block, which with block-diagonal W1 is the
        aggregate-only objective with Z = [X1, Z_excl] at the same inversions (owner decision 1)."""
        fail = ("Profile check failed: q is still monotone in sigma_p{what}. LEARNINGS 14 predicted otherwise with "
                "micro moments; the plan needs the owner before any long estimation.")
        econ = ["sd_xi", "median_own_elasticity", "nonzero_cross_median", "diversion_leaving_body_median",
                "inside_share_mean", "inside_share_semi_elasticity", "phi_mid", "phi_high"]
        rules = {
            "no_micro": {"rule": NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed:aggregate",
                                                       rise=3.84, fail_message=fail.format(what=" without micro moments")),
                         "columns": ["q_fixed:aggregate", "q_cue:aggregate"] + econ},
            "micro": {"rule": NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed", rise=3.84,
                                                    fail_message=fail.format(what=" with micro moments")),
                      "columns": ["q_fixed", "q_cue", "q_fixed:aggregate", "q_fixed:micro"] + econ
                      + ["E[1{j>0}|mid_i]", "E[1{j>0}|high_i]"]},
        }
        kw = {"direction": SIG_REF, "lambdas": [0.5, 1.0, 2.0],
              "grid": {"sigma_p": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]}, "origin": 0.0,
              "concentrate": ("phi_mid", "phi_high"), "economics": self.gate_economics, "rules": rules}
        return [("sigma_ray", "all", kw)]

    def gate_economics(self, problem: Any, ev: Any) -> dict[str, float]:
        """Per profile point: the aggregate-only CUE objective, elasticities, diversion, inside shares, the
        inside-share elasticity to a uniform 1% price rise (mean over markets) and the re-solved phi."""
        sol, fit = ev.aux["solution"], ev.aux["linear_fit"]
        g_a = ev.g[problem.agg_rows]
        S_a = problem.agg.covariance_block(sol, fit)
        out = {"q_cue:aggregate": float(problem.n * g_a @ np.linalg.solve(S_a, g_a))}
        post = pp.postestimation(problem.model, ev.theta, ev.aux["delta"], firm_ids=self.firm_column,
                                 groups=self.groups, group_other=self.group_other)
        pr = post["products"]
        out.update({"median_own_elasticity": float(np.nanmedian(pr["own_elasticity"])),
                    "nonzero_cross_median": float(np.nanmedian(pr["nonzero_cross"])),
                    "diversion_leaving_body_median": float(np.nanmedian(pr["diversion_leaving_group"])),
                    "inside_share_mean": float(np.mean([m["inside_share"] for m in post["markets"]])),
                    "inside_share_semi_elasticity": float(np.mean([m["inside_share_elasticity"]
                                                                   for m in post["markets"]]))})
        names = problem.layout.names
        for n in ("phi_mid", "phi_high"):
            out[n] = float(np.asarray(ev.theta)[names.index(n)])
        pred = ev.aux.get("micro_predicted", {}).get("micro")
        if pred is not None:
            for lab, v in zip(problem.moments[1].labels, pred):
                if "1{j>0}" in lab:
                    out[lab.replace(" ", "")] = float(v)
        return out

    # ------------------------------------------------------------------ postestimation (DESIGN 6.5)

    def _draw_labels(self, model: Any, t: int, column: str) -> np.ndarray:
        md = model.market_draws(t)
        return self.agents[column].to_numpy()[np.asarray(md["agent_rows"])[md["agent_of_draw"]]].astype(str)

    def counterfactual(self, model: Any, theta: np.ndarray, delta: np.ndarray, resolve_prices: bool
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
        """1984 without minivans (Petrin's headline): CS change at fixed prices and, if asked, with prices
        re-solved to a Nash equilibrium at the FOC costs. Gains in $M = households (thousands) x CS ($1000)."""
        mi = self.products.mi.to_numpy() == 1
        cf = pp.drop_products_counterfactual(model, theta, delta, YEAR, mi, firm_ids=self.firm_column,
                                             resolve_prices=resolve_prices, tol=1e-7, max_iterations=500)
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
            out.update(nash_status_1984=nash["status"], nash_iterations_1984=nash["iterations"],
                       nash_fixed_point_steps_1984=nash["fixed_point_steps"], nash_max_residual_1984=nash["step"],
                       nash_foc_residual_1984=nash["foc_residual"])
            if nash.get("per_draw") is not None:
                out["minivan_cs_gain_1984_musd"] = hh * (cf["observed"]["cs"] - nash["cs"])
                out["inside_share_1984_nash"] = nash["inside_share"]
                per["nash"] = nash["per_draw"]
            out["nash_note_1984"] = (
                "Nash prices converged; the Nash gain is at an equilibrium." if nash["status"] == "converged" else
                f"Nash re-solve ended '{nash['status']}' after {nash['iterations']} iterations (max |r| "
                f"{nash['step']:.3g}); the 'Nash' gain is at the last iterate, not at an equilibrium. At finite draws "
                f"an exact PCM price equilibrium need not exist (WP8).")
        t = pp.market_position(model, YEAR)
        w = np.asarray(model.kernel_market(t).weights)
        tables = {}
        for name, col in (("income", "income_group"), ("fs", "fs_group")):
            rows = comparison.cs_gain_by_group(per, w, self._draw_labels(model, t, col), hh)
            tables[name] = rows
            for r in rows:
                if "gain_nash_musd" in r:
                    out[f"cs_gain_1984_{name}_{r['group']}_musd"] = r["gain_nash_musd"]
                out[f"cs_gain_1984_fixed_prices_{name}_{r['group']}_musd"] = r["gain_fixed_prices_musd"]
        return out, {"tables": tables, "raw": cf}

    def post(self, pipe: Any, problem: Any, result: Any, out: pathlib.Path) -> dict[str, Any]:
        model, theta, delta = problem.model, np.asarray(result.theta), np.asarray(result.evaluation.aux["delta"])
        post = pp.postestimation(model, theta, delta, firm_ids=self.firm_column, groups=self.groups,
                                 group_other=self.group_other)
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
        prod.to_csv(out / "product_level.csv", index=False)
        summary["inside_share_semi_elasticity"] = float(np.mean([m["inside_share_elasticity"]
                                                                 for m in post["markets"]]))
        summary.update(minivan_diversion(summary["group_diversion"]))
        summary["cs_tables_1984"] = extra["tables"]
        for name, rows in extra["tables"].items():
            pd.DataFrame(rows).to_csv(out / f"cs_1984_by_{name}.csv", index=False)
        blp = {}
        for label, fname in BLP_SUMMARIES.items():
            f = HERE / "results" / fname
            if f.exists():
                blp[label] = json.loads(f.read_text())
        note = ("BLP micro and BLP no micro estimate a log-cost supply side jointly; BLP demand micro and the PCM use "
                "demand moments only (aggregate + micro) and recover costs from the Nash FOC. "
                f"PCM estimate: stage {result.stage}, {result.convergence.value}, {pipe.cfg.nu} draws per agent. "
                f"PCM Nash status 1984: {summary.get('nash_status_1984')}. {summary.get('nash_note_1984', '')}")
        rows = {k: v for k, v in comparison.scalar_items(summary).items() if k != "nash_note_1984"}
        table = pp.compare_blp(rows, {k: comparison.scalar_items(v) for k, v in blp.items()}, note=note)
        gd = ["", "Diversion by body type (share-weighted mean, rows: from, columns: to):"]
        for src, dst in summary["group_diversion"].items():
            gd.append(f"  PCM {src:>4}: " + "  ".join(f"{k} {v:.3f}" for k, v in dst.items()))
        for v in blp.values():
            if v.get("group_diversion"):
                v.update(minivan_diversion(v["group_diversion"]))
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
        post = pp.postestimation(model, theta, delta, firm_ids=self.firm_column, groups=self.groups,
                                 group_other=self.group_other)
        cf, _ = self.counterfactual(model, theta, delta, resolve_prices=False)
        return {"own_elasticity_median": float(np.nanmedian(post["products"]["own_elasticity"])),
                "minivan_cs_gain_1984_fixed_prices_musd": cf["minivan_cs_gain_1984_fixed_prices_musd"]}


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

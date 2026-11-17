"""Petrin (2002) minivans, exact pure characteristics model (PCM), as a spec on nfp_harness.

    u_ijt = delta_jt + x_jt . t_i - c_i a_i p_jt,   u_i0t = 0
    x     = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv]
    t_i   = sigma_k nu_ik on all 10 columns (body sigmas on mi, sw, su, pv)  +  pi_b log(fs_i) fv_i on mi, sw, su, pv
    log c_i = phi_low low_i/income_i + phi_mid mid_i/income_i + phi_high high_i/income_i
    a_i   ~ LogNormal(0, sigma_p)
    delta_jt = X1_jt beta + xi_jt,  X1 = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv, pgnp, trend, trend2]
    Z     = [X1, demand_instruments0..21]  (35 columns; LEARNINGS 16)

Moments: the 35 aggregate IV moments and Petrin's 10 CEX micro moments (pyblp's Petrin data), 45 in all
for 13 linear and 18 nonlinear parameters. Normalisations: median alpha = 1 at c_i = 1 (mu = 0,
LEARNINGS 11); body-type taste spreads capped at the pooled price range ($66.1k); taste spreads
non-negative; u_i0 = 0 in every year with pgnp and trends in X1. Consumers: the 1,000 CEX agents per
market crossed with scrambled Sobol standard normal taste draws, 2^ceil(log2(1000 nu)) draws per market.
Exact PCM shares, no smoothing (LEARNINGS 15). Stages and their files: nfp_harness.pipeline (DESIGN 6.4).

Run (results under petrin_replication/results/harness/, untracked):
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
import math  # noqa: E402
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
CHARS = ("const", "hpwt", "space", "air", "mpd", "fwd", "mi", "sw", "su", "pv")
X1_COLS = CHARS + ("pgnp", "trend", "trend2")
BODY = ("mi", "sw", "su", "pv")
YEAR = 1984
SIG_REF = {"sigma_const": 2.0, "sigma_hpwt": 2.0, "sigma_space": 2.0, "sigma_air": 1.0, "sigma_mpd": 1.0,
           "sigma_fwd": 2.0, "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0}
SIGMA_STARTS = {"const": 3.0, "hpwt": 2.0, "space": 1.0, "air": 0.5, "mpd": 1.0, "fwd": 1.0,
                "mi": 2.0, "sw": 2.0, "su": 2.0, "pv": 2.0}
BLP_SUMMARIES = {"BLP micro": "blp_micro_summary.json", "BLP no micro": "blp_nomicro_summary.json",
                 "BLP demand micro": "blp_demand_micro_summary.json"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agents = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
    price_range = float(products.prices.max() - products.prices.min())   # pooled, before tie breaking: 66.1
    products["prices"] = break_price_ties(products.prices.to_numpy(float), products.market_ids.to_numpy(), 0.01)
    # body type label for diversion by group: the first of mi, sw, su, pv whose dummy is 1, else car (two
    # products, 1988 and 1989, carry both mi and pv and count as minivans, as in estimate_blp.py)
    body = np.full(len(products), "car", dtype=object)
    for b in reversed(BODY):
        body[products[b].to_numpy() == 1] = b
    products["body"] = body.astype(str)
    agents["logfs_fv"] = np.log(agents.fs) * agents.fv
    for lvl in ("low", "mid", "high"):
        agents[f"{lvl}_inc"] = agents[lvl] / agents.income
    agents["income_group"] = np.select([agents.low == 1, agents.mid == 1, agents.high == 1], ["low", "mid", "high"], "none")
    agents["fs_group"] = np.where(agents.fs >= 5, "5+", agents.fs.astype(int).astype(str))
    return products, agents, price_range


class PetrinSpec:
    """The Petrin PCM spec read by nfp_harness.pipeline (interface in its module docstring)."""

    name = "petrin_pcm"
    firm_column = "firm_ids"
    groups = "body"
    group_other = "car"
    binary_columns = ("air", "fwd") + BODY
    prefit_free = ("phi_low", "phi_mid", "phi_high")
    prefit_max_iterations = 30
    gate_required = "micro"

    def __init__(self) -> None:
        self.products, self.agents, self.price_range = load_data()
        p = self.products
        self.X1 = np.column_stack([np.ones(len(p))] + [p[c].to_numpy(float) for c in X1_COLS[1:]])
        self.Z = np.column_stack([self.X1] + [p[f"demand_instruments{k}"].to_numpy(float) for k in range(22)])
        self.linear_names = list(X1_COLS)
        self.clusters = p.clustering_ids.to_numpy()
        self.values = dict(pd.read_csv(pyblp.data.PETRIN_VALUES_LOCATION).iloc[:, :2].to_numpy())
        self.normalisations = [
            "median alpha = 1 at c_i = 1: log alpha_i = phi . scale basis + log a_i, a_i ~ LogNormal(0, sigma_p), "
            "no constant (mu = 0; LEARNINGS 11)",
            f"taste spreads on the body dummies mi, sw, su, pv capped at the pooled price range "
            f"{self.price_range:.1f} ($1000), a normalisation of a direction the data leave quiet",
            "taste spreads non-negative (draws are symmetric)",
            "u_i0 = 0 in every year, with pgnp, trend and trend2 in X1 (harmless for the 1984 counterfactual)",
            "PCM taste draws are scrambled Sobol standard normals crossed with pyblp's CEX agents; pyblp's nodes "
            "are not used (owner decision 3)",
        ]
        r = self.price_range
        self.plausible = {
            "own_elasticity_median": (-12.0, -2.0), "share_own_elasticity_below_-50": (None, 0.01),
            "nonzero_cross_median": (10.0, None), "diversion_leaving_group_median": (0.05, None),
            "margin_median": (0.05, 0.60), "sd_xi_over_sd_delta": (None, 1.0),
            "spread_buyer_dollars": {f"sigma_{b}": (None, r) for b in BODY},
        }

    # ------------------------------------------------------------------ parameters and moments

    def layout(self) -> ParameterLayout:
        r, p = self.price_range, self.products
        specs = []
        for k, c in enumerate(CHARS):
            if c == "const":
                specs.append(ParameterSpec("sigma_const", "taste", c, f"nu{k}", SIGMA_STARTS[c], 0.0, 10 * r, unit=1.0,
                                           note="upper = 10 x price range, domain guard"))
            elif c in BODY:
                specs.append(ParameterSpec(f"sigma_{c}", "taste", c, f"nu{k}", SIGMA_STARTS[c], 0.0, r, cap=True,
                                           unit=1.0, note="cap = price range, normalisation"))
            else:
                specs.append(ParameterSpec(f"sigma_{c}", "taste", c, f"nu{k}", SIGMA_STARTS[c], 0.0,
                                           10 * r / float(np.std(p[c])), note="upper = 10 x price range, domain guard"))
        specs += [ParameterSpec(f"pi_{b}", "taste", b, "logfs_fv", 0.3, -math.inf, math.inf, note="x log(fs) fv")
                  for b in BODY]
        specs += [ParameterSpec(f"phi_{lvl}", "scale", None, f"{lvl}_inc", 0.0, -math.inf, math.inf,
                                note=f"log alpha on {lvl}/income") for lvl in ("low", "mid", "high")]
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, 1.0, 0.05, 3.0, note="sd of log alpha"))
        return ParameterLayout(specs)

    def micro_moment(self, which: str) -> RatioMicroMoment:
        """Petrin's CEX micro moments as in estimate_blp.build_micro_moments; ``which`` = "all" or "income"."""
        nums, dens, vals, labels = [], [], [], []
        if which == "all":
            for demo in ("age", "fs"):
                for b in BODY:
                    nums.append(MicroPart(f"E[{demo}*{b}]", (demo,), (b,)))
                    dens.append(MicroPart(f"E[{b}]", (), (b,)))
                    vals.append(self.values[f"E[{demo} | {b}]"])
                    labels.append(f"E[{demo}_i | {b}_j]")
        for inc in ("mid", "high"):
            nums.append(MicroPart(f"E[inside*{inc}]", (inc,), ()))
            dens.append(MicroPart(f"E[{inc}]", (inc,), (), outside_value=1.0))
            vals.append(self.values[f"E[new | {inc}]"])
            labels.append(f"E[1{{j>0}} | {inc}_i]")
        return RatioMicroMoment("micro", MicroDataset("CEX", 29125), nums, dens, np.array(vals), labels)

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
        return PCMSpec(CHARS, track_realised_mass=self.use_max_step, trial_cold_evals=0)

    def hash_payload(self) -> dict[str, Any]:
        return {"data": "pyblp 1.3.0 Petrin, prices spread to a $10 gap", "chars": CHARS, "x1": X1_COLS,
                "z": "X1 + demand_instruments0..21", "layout": [dataclasses.asdict(s) for s in self.layout().specs],
                "micro": {"values": self.values, "observations": 29125}, "pcm": dataclasses.asdict(self.pcm_spec()),
                "draws": "Sobol normals, seed + market position, crossed with agents", "max_step": self.use_max_step,
                "optimizer": self.optimizer_options}

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
        (body sigmas and their family-size pis up in proportion, LEARNINGS 14)."""
        names = self.layout().names
        u = np.asarray(theta, dtype=float) / np.asarray(units, dtype=float)
        scale, body = np.zeros(len(names)), np.zeros(len(names))
        for k, n in enumerate(names):
            if n == "sigma_p":
                scale[k] = u[k]
            elif n.startswith("sigma_"):
                scale[k] = -u[k]
            if n in {f"sigma_{b}" for b in BODY} | {f"pi_{b}" for b in BODY}:
                body[k] = u[k]
        return {"scale_direction": scale, "body_level_direction": body}

    # ------------------------------------------------------------------ gate (DESIGN 6.4)

    def gate_profiles(self) -> list[tuple[str, str, dict[str, Any]]]:
        """One sigma ray profile with all moments evaluated (DESIGN 6.4), read twice: the micro verdict on
        q_fixed, and the no-micro verdict on its aggregate block, which with block-diagonal W1 is the
        aggregate-only objective with Z = [X1, Z_excl] at the same inversions (owner decision 1)."""
        fail = ("Profile check failed: q is still monotone in sigma_p{what}. LEARNINGS 14 predicted otherwise with "
                "micro moments; the plan needs the owner before any long estimation.")
        econ = ["sd_xi", "median_own_elasticity", "nonzero_cross_median", "diversion_leaving_body_median",
                "inside_share_mean"]
        rules = {
            "no_micro": {"rule": NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed:aggregate",
                                                       rise=3.84, fail_message=fail.format(what=" without micro moments")),
                         "columns": ["q_fixed:aggregate", "q_cue:aggregate"] + econ},
            "micro": {"rule": NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed", rise=3.84,
                                                    fail_message=fail.format(what=" with micro moments")),
                      "columns": ["q_fixed", "q_cue", "q_fixed:aggregate", "q_fixed:micro"] + econ
                      + ["E[1{j>0}|mid_i]", "E[1{j>0}|high_i]"]},
        }
        kw = {"direction": SIG_REF, "lambdas": [0.5, 1.0, 2.0], "grid": {"sigma_p": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
              "origin": 0.0, "layout_changes": {"sigma_p": {"upper": 3.5}}, "economics": self.gate_economics,
              "rules": rules}
        return [("sigma_ray", "all", kw)]

    def gate_economics(self, problem: Any, ev: Any) -> dict[str, float]:
        """Per profile point: the aggregate-only CUE objective, elasticities, diversion, inside shares."""
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
                    "inside_share_mean": float(np.mean([m["inside_share"] for m in post["markets"]]))})
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
        prod.to_csv(out / "product_level.csv", index=False)
        summary = comparison.summarise(post["products"], post["group_matrix"], post["group_names"])
        cf, extra = self.counterfactual(model, theta, delta, resolve_prices=True)
        summary.update(cf)
        summary["inside_share_by_market"] = {str(m["market_id"]): m["inside_share"] for m in post["markets"]}
        summary["nash_singular_markets"] = [str(m) for m in post["nash_singular"]]
        sb = pp.spread_in_buyer_dollars(model, theta, delta)
        summary["mean_alpha_buyers"] = sb["mean_alpha_buyers"]
        summary["spread_buyer_dollars"] = sb["spreads"]
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
        for label, v in blp.items():
            for src, dst in (v.get("group_diversion") or {}).items():
                gd.append(f"  {label} {src:>4}: " + "  ".join(f"{k} {x:.3f}" for k, x in dst.items()))
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


if __name__ == "__main__":
    sys.exit(pipeline.main(PetrinSpec(), default_nu=16, default_base=RESULTS))

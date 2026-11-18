"""Grieco, Murry & Yurukoglu (2024, QJE) US automobiles, exact pure characteristics model (PCM): the intended spec
on nfp_harness, mirroring petrin_replication/estimate_pcm.py. This is a stub: it builds the problem and evaluates it
at one theta (a cold inversion, no estimation).

    u_ijt = delta_jt + x_jt . t_i - alpha_i p_jt,   u_i0t = 0,   p in $10k of 2015
    x     = [1, van, suv, truck, z log footprint, z log hp, z log mpg, lux, sport, electric, EuroBrand, USBrand]
    t_ik  = sigma_k nu_ik on the 11 GMY random-coefficient columns
            + pi (van, z log footprint) x (fs2, fs34, fs5) + pi truck x rural + pi const x inc    (GMY's demographics)
    log alpha_i = log c_i^GMY + phi_inc inc_i + phi_age age_i + log a_i                         (GMY's income law)
    c_i^GMY = |alpha_bar + th_inc inc + th_inc2 inc2 + th_age age| / median, from the BLP fit (fixed, coefficient 1)
    a_i ~ LogNormal(0, sigma_p) truncated below at a_lo = 0.2                                      (LEARNINGS 32)
    delta_jt = X1_jt beta + xi_jt,  X1 = [GMY's 13 second-stage characteristics, year and make2 dummies]
    Z     = [X1, lag_pl_con, iv_prod, IV_N, IV_N_weight, BLP sums of van, suv, truck, lux, sport, electric, z log footprint,
             hp, mpg over own and rival products, Gandhi-Houde local and quadratic differentiation IVs on the last three]

Moments declared: Sargan aggregate block; micro block in the harness's form (purchase-weighted E[x | set], levels:
E[price | income quintile], E[price | age group] for CEX and MRI, E[van | family size], E[log footprint | family
size], E[truck | rural / urban] from CEX), fixed diagonal covariance from the package's variances; log-cost supply
block, Nash markups from the PCM's ds/dp with ownership by parent, w = [1, log hp, log curb weight, log mpg, log
footprint, van, suv, truck, electric, regionASIA, regionEU, trend], Z_s = [w, lag_pl_con, steel_weight, NAFTA and
Korea trade dummies]; second-choice block: proposed, not in the harness (the data correlations are reported).

Deviations from GMY that the harness forces (proposals in the replication report): GMY's micro moments are
differences of household-averaged conditional means and ratios of purchase rates (gmy_data.MicroMomentEvaluator),
which RatioMicroMoment cannot express, so the stub targets the levels in the harness's purchase-weighted form and
drops the four purchase-rate ratios; the second-choice block is absent.

Starting theta (``--start blp``): every taste and pi of the BLP fit (results/blp_fit_h1000.json) divided by the median
|alpha_i|, so tastes are in $10k of the median consumer; phi at 0 (the BLP law sits in c_i); sigma_p 2.5 and a_lo 0.2
(Petrin's starts). Data: same-firm exact twins merged (same price and x, shares summed), remaining price ties spread
to a $10 gap. Consumers: CPS households (gmy_data.load_agents, 1,000 per year) crossed with scrambled Sobol normal
taste draws, 2^ceil(log2(1000 nu)) per market.

Run: /home/istomin/anaconda3/bin/python gmy_replication/estimate_pcm.py evaluate [--years 1980 1999 2018] [--nu 4]
     [--threads 32]
Output (untracked): gmy_replication/results/pcm_evaluate_nu<nu>_<years>.json
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
import json  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfp_harness import gmy_data  # noqa: E402
from nfp_harness.data import break_price_ties  # noqa: E402
from nfp_harness.linear import LinearIV, matrix_rank  # noqa: E402
from nfp_harness.models import pcm_post as pp  # noqa: E402
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.moments import MicroDataset, MicroPart, RatioMicroMoment, SarganIVMoments, SupplyMoments  # noqa: E402
from nfp_harness.params import ParameterLayout, ParameterSpec  # noqa: E402
from nfp_harness.problem import NFPGMMProblem  # noqa: E402
from nfp_harness.second_choice import first_second_correlations  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 8673320
YEARS = (1980, 1999, 2018)
TASTE = ("const", "van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport", "electric",
         "regionEU", "regionUS")
UNOBS = ("van", "suv", "truck", "z_log_footprint", "z_log_hp", "z_log_mpg", "lux", "sport", "electric", "regionEU",
         "regionUS")
PIS = (("van", "fs2"), ("van", "fs34"), ("van", "fs5"), ("z_log_footprint", "fs2"), ("z_log_footprint", "fs34"),
       ("z_log_footprint", "fs5"), ("truck", "rural"), ("const", "inc"))
TWIN_KEYS = ("market_ids", "firm_ids", "prices") + TASTE[1:]
MIN_GAP = 0.001          # $10 in $10k
SIGMA_P = {"start": 2.5, "lower": 1.0, "upper": 4.0, "unit": 0.25}
A_LO = 0.2
PHI_BOUND = 4.0
TASTE_CAP = 10.0         # $100k per sd: above the 1980-2018 price range ($11k-$100k)
BLP_FIT = RESULTS / "blp_fit_h1000.json"
COST_BASIS = ("const", "log_hp", "log_curbweight", "log_mpg", "log_footprint", "van", "suv", "truck", "electric",
              "regionASIA", "regionEU", "trend")
SUPPLY_EXCLUDED = ("lag_pl_con", "steel_weight", "NAFTA_D", "NAFTA_T", "KOREATRDE_D", "KOREATRDE_T")
DEMAND_EXCLUDED = ("lag_pl_con", "iv_prod", "IV_N", "IV_N_weight")
BLP_DUMMIES = ("van", "suv", "truck", "lux", "sport", "electric")
GH_CONTINUOUS = ("z_log_footprint", "z_log_hp", "z_log_mpg")
MICRO_LEVELS = {   # (survey, demographic column, data row, level, characteristic data column, product column)
    "price_income": [(s, "inc_quintile", f"Income{q}", q, "Price", "prices") for s in ("cex", "mri") for q in range(1, 6)],
    "price_age": [(s, "age_group", r, k, "Price", "prices") for s in ("cex", "mri")
                  for k, r in enumerate(("Age20-29", "Age30-39", "Age40-49", "Age50-59", "Age60+"), 1)],
    "van_fs": [(s, "fs_group", r, k, "Van", "van") for s in ("cex", "mri")
               for k, r in enumerate(("FamilySize1", "FamilySize2", "FamilySize3-4", "FamilySize5+"), 1)],
    "size_fs": [(s, "fs_group", r, k, "logSizeLW", "log_footprint") for s in ("cex", "mri")
                for k, r in enumerate(("FamilySize1", "FamilySize2", "FamilySize3-4", "FamilySize5+"), 1)],
    "truck_rural": [("cex", "rural", "Urban", 0, "Truck", "truck"), ("cex", "rural", "Rural", 1, "Truck", "truck")],
}


def merge_twins(products: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows equal on TWIN_KEYS are one product to every PCM consumer: keep the first, sum shares and quantity."""
    g = products.groupby(list(TWIN_KEYS), sort=False)
    first = (g.cumcount() == 0).to_numpy()
    out = products.copy()
    for c in ("shares", "quantity", "sales"):
        out[c] = g[c].transform("sum")
    return out[first].reset_index(drop=True), int((~first).sum())


def price_structure(products: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Per market: products, median p / gap to the nearest-priced neighbour, exact price ties, and identical-x
    clusters (products sharing every kernel characteristic) with the products in them."""
    out = {}
    for y, g in products.groupby("market_ids"):
        p = np.sort(g.prices.to_numpy(float))
        d = np.diff(p)
        gap = np.minimum(np.r_[np.inf, d], np.r_[d, np.inf])
        with np.errstate(divide="ignore"):
            ratio = np.where(gap > 0, p / gap, np.inf)
        sizes = g.groupby(list(TASTE[1:])).size()
        out[int(y)] = {"products": int(len(g)), "median_p_over_gap": float(np.median(ratio)),
                       "exact_price_ties": int((d == 0).sum()), "identical_x_clusters": int((sizes > 1).sum()),
                       "products_in_identical_x_clusters": int(sizes[sizes > 1].sum()),
                       "identical_x_and_price_pairs": int(g.duplicated(list(TASTE[1:]) + ["prices"]).sum())}
    return out


class GMYSpec:
    """The GMY PCM spec (module docstring). Builds the product and agent tables, layout, moments and problem."""

    name = "gmy_pcm"

    def __init__(self, years: tuple[int, ...] = YEARS, households: int = 1000, blp_fit: pathlib.Path = BLP_FIT) -> None:
        self.years = tuple(int(y) for y in years)
        data = gmy_data.load(n_households=households, seed=SEED, years=self.years)
        self.raw_products = data.products
        self.structure_raw = price_structure(data.products)
        p, self.n_merged = merge_twins(data.products)
        p["prices_data"] = p.prices
        p["prices"] = break_price_ties(p.prices.to_numpy(float), p.market_ids.to_numpy(), MIN_GAP)
        p["log_hp"] = np.log(p.hp)
        p["log_curbweight"] = np.log(p.curbweight)
        self.products = p
        self.structure = price_structure(p.assign(prices=p.prices_data))
        self.second_choices = {y: v for y, v in data.second_choices.items()}
        self.micro_tables = gmy_data.load_micro_tables(data.root)
        fit = json.loads(blp_fit.read_text())
        self.blp = {"theta": fit["theta"], "alpha": float(fit["alpha"]), "source": blp_fit.name}
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
        for s in ("cex", "mri"):
            for col, levels in (("inc_quintile", range(1, 6)), ("age_group", range(1, 6)), ("fs_group", range(1, 5)),
                                ("rural", (0, 1))):
                for v in levels:
                    a[f"{s}_{col}_{v}"] = ((a[col] == v) & (a[s] == 1)).astype(float)
        self.agents = a
        self.X1, self.X1_names = self.linear_design(p)
        exog = self.X1
        Z = np.column_stack([exog] + [p[c].to_numpy(float) for c in DEMAND_EXCLUDED] + [rival_instruments(p)])
        self.Z = drop_collinear(Z, keep=exog.shape[1])
        w = np.column_stack([np.ones(len(p)) if c == "const" else p[c].to_numpy(float) for c in COST_BASIS])
        self.cost_keep = keep_rank(w)
        self.cost_basis = w[:, self.cost_keep]
        Zs = np.column_stack([self.cost_basis] + [p[c].to_numpy(float) for c in SUPPLY_EXCLUDED])
        self.Z_supply = drop_collinear(Zs, keep=self.cost_basis.shape[1])
        self.clusters = p.clustering_ids.to_numpy()

    @staticmethod
    def linear_design(p: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """X1 = [STAGE2_X, year dummies (first dropped), make2 dummies (first dropped), constant], rank-reduced."""
        cols = [p[c].to_numpy(float) for c in gmy_data.STAGE2_X]
        names = list(gmy_data.STAGE2_X)
        for c in ("market_ids", "make2_do"):
            lv = sorted(p[c].unique())
            for v in lv[1:]:
                cols.append((p[c] == v).to_numpy(float))
                names.append(f"{c}={v}")
        X = np.column_stack([np.ones(len(p))] + cols)
        names = ["const"] + names
        keep = keep_rank(X)
        return X[:, keep], [names[k] for k in keep]

    def layout(self) -> ParameterLayout:
        th, am = self.blp["theta"], self.alpha_median
        specs = []
        for k, c in enumerate(UNOBS):
            specs.append(ParameterSpec(f"sigma_{c}", "taste", c, f"nu{k}", min(th[f"sigma {c}"] / am, TASTE_CAP), 0.0,
                                       TASTE_CAP, cap=True, unit=0.1,
                                       note="BLP sigma / median |alpha|; cap $100k per sd"))
        for c, d in PIS:
            v = th[f"{c} x {d}"] / am
            specs.append(ParameterSpec(f"pi_{c}_{d}", "taste", c, d, v, -TASTE_CAP, TASTE_CAP, unit=0.1,
                                       note="BLP pi / median |alpha|"))
        specs.append(ParameterSpec("phi_gmy", "scale", None, "log_c_gmy", 1.0, 0.0, 2.0, free=False,
                                   unit=0.1, note="log alpha on log c_i of the BLP income-age law; fixed at 1"))
        for name, basis in (("phi_inc", "inc"), ("phi_age", "age")):
            specs.append(ParameterSpec(name, "scale", None, basis, 0.0, -PHI_BOUND, PHI_BOUND, unit=0.1,
                                       note="log alpha shift beyond the BLP law"))
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, SIGMA_P["start"], SIGMA_P["lower"],
                                   SIGMA_P["upper"], unit=SIGMA_P["unit"], note="sd of log a (untruncated parent)"))
        specs.append(ParameterSpec("a_lo", "bound", None, None, A_LO, A_LO, A_LO, unit=0.1, free=False,
                                   note="a_i >= a_lo; fixed (LEARNINGS 32)"))
        return ParameterLayout(specs)

    def pcm_spec(self) -> PCMSpec:
        return PCMSpec(TASTE, track_realised_mass=False, trial_cold_evals=0, bound_kind="a")

    def micro_moment(self) -> list[RatioMicroMoment]:
        """One RatioMicroMoment per survey: E[x | set, inside] levels with the package's variances (fixed, diagonal)."""
        out = []
        for s, years in (("cex", gmy_data.CEX_YEARS), ("mri", gmy_data.MRI_YEARS)):
            mk = [y for y in self.years if years[0] <= y <= years[1]]
            if not mk:
                continue
            nums, dens, vals, var, labels = [], [], [], [], []
            for rows in MICRO_LEVELS.values():
                for sv, col, row, lvl, ch, pc in rows:
                    if sv != s:
                        continue
                    h = f"{s}_{col}_{lvl}"
                    nums.append(MicroPart(f"{h}*{pc}", (h,), (pc,)))
                    dens.append(MicroPart(f"{h}*inside", (h,), ()))
                    vals.append(float(self.micro_tables[s]["value"].loc[row, ch]))
                    var.append(float(self.micro_tables[s]["variance"].loc[row, ch]))
                    labels.append(f"E[{pc} | {row}, inside] {s}")
            out.append(RatioMicroMoment(f"micro_{s}", MicroDataset(s.upper(), 1, market_ids=mk), nums, dens,
                                        np.array(vals), labels, covariance=np.diag(var)))
        return out

    def supply_moment(self) -> SupplyMoments:
        return SupplyMoments(self.Z_supply, self.cost_basis, self.products.firm_ids.to_numpy(), self.clusters,
                             name="supply", w_names=[f"gamma[{c}]" for k, c in enumerate(COST_BASIS)
                                                     if k in self.cost_keep],
                             rcond_min=1e-12, mc_floor=0.05, fd_step=1e-3, fd_scale="unit")

    def build_problem(self, nu: int, threads: int, moments: str = "all") -> NFPGMMProblem:
        layout = self.layout()
        model = PCMModel(self.pcm_spec(), {"products": self.products, "agents": self.agents}, layout, nu, SEED,
                         total_threads=threads)
        ms: list[Any] = [SarganIVMoments(self.Z, self.clusters)]
        if moments in ("all", "demand"):
            ms += self.micro_moment()
        if moments == "all":
            ms.append(self.supply_moment())
        return NFPGMMProblem(model, ms, LinearIV(self.X1, self.Z, names=self.X1_names), layout, max_step=False)

    def second_choice_data(self) -> dict[int, list[float]]:
        out = {}
        raw = self.raw_products
        for y, v in self.second_choices.items():
            pr = raw[raw.market_ids == y]
            out[int(y)] = first_second_correlations(pr.shares, v["prob"],
                                                    pr[list(gmy_data.SECOND_CHOICE_CHARS)].to_numpy(float)).tolist()
        return out


def rival_instruments(p: pd.DataFrame) -> np.ndarray:
    """Excluded demand instruments beyond GMY's: BLP sums of the style and segment dummies and of the three continuous
    characteristics over the firm's other products and over rivals, and Gandhi-Houde local and quadratic
    differentiation instruments on the three continuous characteristics
    (pyblp 1.3.0 builders, market = year, firm = parent)."""
    import pyblp
    d = pd.DataFrame({"market_ids": p.market_ids, "firm_ids": p.firm_ids})
    for c in GH_CONTINUOUS + BLP_DUMMIES:
        d[c] = p[c].to_numpy(float)
    blp = pyblp.build_blp_instruments(pyblp.Formulation("0 + " + " + ".join(BLP_DUMMIES + GH_CONTINUOUS)), d)
    gh = [pyblp.build_differentiation_instruments(pyblp.Formulation("0 + " + " + ".join(GH_CONTINUOUS)), d,
                                                  version=v) for v in ("local", "quadratic")]
    return np.column_stack([blp] + gh)


def keep_rank(A: np.ndarray, tol: float = 1e-10) -> list[int]:
    """Columns of A kept greedily left to right while each raises the rank."""
    keep: list[int] = []
    for k in range(A.shape[1]):
        if np.std(A[:, k]) == 0 and k > 0:
            continue
        if matrix_rank(A[:, keep + [k]], rtol=tol) == len(keep) + 1:
            keep.append(k)
    return keep


def drop_collinear(Z: np.ndarray, keep: int) -> np.ndarray:
    """Z with columns past the first ``keep`` dropped when they add no rank."""
    cols = list(range(keep))
    for k in range(keep, Z.shape[1]):
        if matrix_rank(Z[:, cols + [k]]) == len(cols) + 1:
            cols.append(k)
    return Z[:, cols]


def evaluate(args: argparse.Namespace) -> None:
    if args.threads > 32:
        raise SystemExit("at most 32 threads")
    RESULTS.mkdir(exist_ok=True)
    spec = GMYSpec(tuple(args.years))
    prob = spec.build_problem(args.nu, args.threads, args.moments)
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
    out = {"years": spec.years, "nu": args.nu, "draws_per_market": [m.R for m in model._mk], "status": ev.status.value,
           "wall_seconds": wall, "q_identity_w": float(ev.g @ W @ ev.g) if ev.ok else float("nan"), "theta": dict(zip(spec.layout().names,
                                                                                           theta.tolist())),
           "alpha_median_blp": spec.alpha_median, "blp_source": spec.blp["source"],
           "twins_merged": spec.n_merged, "price_structure_raw": spec.structure_raw,
           "price_structure_after_merge": spec.structure, "per_market": per, "window_anatomy": anatomy,
           "second_choice_data": spec.second_choice_data(),
           "moments": {m.name: getattr(m, "k", None) for m in prob.moments}}
    tag = "_".join(str(y) for y in spec.years)
    path = RESULTS / f"pcm_evaluate_nu{args.nu}_{args.moments}_{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("status", "wall_seconds", "twins_merged", "per_market", "window_anatomy")},
                     indent=1, default=str))
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate", help="build the problem and evaluate it once at the starting theta (cold)")
    e.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    e.add_argument("--nu", type=int, default=4)
    e.add_argument("--threads", type=int, default=32)
    e.add_argument("--moments", choices=("aggregate", "demand", "all"), default="aggregate",
                   help="aggregate: Sargan block only (the feasibility check); demand: + micro; all: + supply")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

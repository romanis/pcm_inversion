"""Nevo (2000) cereal demand with the exact pure characteristics model (PCM), as a spec on nfp_harness.

Same data, instruments, fixed effects and demographic interactions as the pyblp tutorial specification of
estimate_blp.py, with the logit error removed:

    u_ijt = delta_jt + x_j . t_i - c_i a_i p_jt,   u_i0t = 0,   x_j = [1, sugar_j, mushy_j]
    t_i   = sigma nu_i + Pi D_i                    (tastes for the constant, sugar and mushy; D = income, age)
    c_i   = exp(phi . [income, income^2, child])   (demographic shifters of price sensitivity)
    a_i   ~ LogNormal(0, sigma_p)                   (integrated in closed form, no bound)
    delta_jt = d_j (brand fixed effect, absorbed) + xi_jt

PCM utility has no scale of its own: the median price sensitivity at mean demographics is 1, so delta is in dollars
and there is no linear price coefficient. Consumers: each market's 20 CPS draws crossed with a Gauss-Hermite product
grid over nu with ``--gh-nodes`` x ``--nu`` nodes per dimension (``PaperSpec.taste_nodes``), or with ``--draws sobol``
the harness's scrambled Sobol draws (``--nu`` per agent, required by ``--firm-taste``, which adds one N(0,
sigma_firm_f^2) taste per firm's brands). Moments: Z = the 20 excluded instruments demeaned by brand, aggregate IV
moments on xi = delta demeaned by brand (``Absorb``), W1 = (Z'Z/n)^-1 (pyblp's one-step). Stages: nfp_harness.pipeline
(prefit of the three taste spreads, step1 = the one-step GMM, step2, post, stability); no gate is required.

Run (results under nevo_replication/results/harness/, untracked):
  python nevo_replication/estimate_pcm.py run --stage step1 [--threads 32]
  python nevo_replication/estimate_pcm.py run --stage all --detach
Options of this paper: --gh-nodes, --draws {gh,sobol}, --firm-taste, --drop DSUGAR,DPRICE,RATIO; the shared ones
(--start-theta, --fix, ...) come from nfp_harness.paper.PaperSpec.
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

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyblp  # noqa: E402

from nfp_harness import comparison, pipeline  # noqa: E402
from nfp_harness.linear import Absorb  # noqa: E402
from nfp_harness.models.pcm import PCMSpec  # noqa: E402
from nfp_harness.moments import IVMoments  # noqa: E402
from nfp_harness.paper import PaperSpec  # noqa: E402
from nfp_harness.params import ParameterLayout, ParameterSpec  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results" / "harness"
CHARS = ("const", "sugar", "mushy")
FIRMS = (1, 2, 3, 4, 6)
# (name, characteristic, basis, start, unit): basis nu<k> is a taste draw, an agent column a demographic interaction
TASTE = (("sigma_const", "const", "nu0", 0.05, 0.05), ("sigma_sugar", "sugar", "nu1", 0.003, 0.003),
         ("sigma_mushy", "mushy", "nu2", 0.03, 0.03)) + tuple(
    (f"pi_{c}_{d}", c, d, 0.0, u) for c, u in (("const", 0.05), ("sugar", 0.003), ("mushy", 0.03))
    for d in ("income", "age"))
SCALE = (("phi_price_income", "income", 0.3), ("phi_price_income_sq", "income_squared", 0.02),
         ("phi_price_child", "child", 0.3))


def gauss_hermite(dim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The n^dim product grid of probabilists' Gauss-Hermite nodes, weights summing to 1."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    nodes = np.stack(np.meshgrid(*[z] * dim, indexing="ij"), -1).reshape(-1, dim)
    return nodes, np.prod(np.stack(np.meshgrid(*[w] * dim, indexing="ij"), -1).reshape(-1, dim), axis=1)


def drop_near_duplicates(products: pd.DataFrame, max_dsugar: float, max_rel_dprice: float,
                         min_share_ratio: float) -> pd.DataFrame:
    """Within a market, of two brands with equal mushy, sugar within max_dsugar grams and prices within
    max_rel_dprice, drop the smaller when the larger outsells it by more than min_share_ratio (its share goes to the
    outside good)."""
    drop = set()
    for _, d in products.groupby("market_ids"):
        r = d.to_dict("records")
        for a, ra in zip(d.index, r):
            for b, rb in zip(d.index, r):
                if (a < b and ra["mushy"] == rb["mushy"] and abs(ra["sugar"] - rb["sugar"]) <= max_dsugar
                        and abs(ra["prices"] - rb["prices"]) / min(ra["prices"], rb["prices"]) < max_rel_dprice):
                    big, small = (a, b) if ra["shares"] >= rb["shares"] else (b, a)
                    if products.shares[big] / products.shares[small] > min_share_ratio:
                        drop.add(small)
    return products.drop(index=sorted(drop)).reset_index(drop=True)


class NevoSpec(PaperSpec):
    """The Nevo PCM spec read by nfp_harness.pipeline (module docstring)."""

    name = "nevo_pcm"
    binary_columns = ("mushy",)
    prefit_free = ("sigma_const", "sigma_sugar", "sigma_mushy")
    gate_required = ()
    a_lo_default = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.gh_nodes, self.draws, self.firm_taste, self.drop = 5, "gh", False, None
        self.load()

    def load(self) -> None:
        p = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
        if self.drop:
            p = drop_near_duplicates(p, *self.drop)
        for f in FIRMS:
            p[f"firm{f}"] = (p.firm_ids == f).astype(float)
        self.products, self.agents = p, pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
        self.fe = Absorb(p.product_ids.to_numpy())
        self.Z = self.fe.demean(p[[f"demand_instruments{k}" for k in range(20)]].to_numpy(float))
        self.X1, self.linear_names = np.zeros((len(p), 0)), []
        self.price_range = float(p.prices.max() - p.prices.min())
        self.plausible: dict[str, Any] = {}

    def add_paper_arguments(self, parser: Any) -> None:
        parser.add_argument("--gh-nodes", type=int, default=5, help="Gauss-Hermite nodes per taste dimension at nu 1")
        parser.add_argument("--draws", choices=("gh", "sobol"), default="gh", help="taste draws (module docstring)")
        parser.add_argument("--firm-taste", action="store_true", help="one taste spread per firm (needs sobol)")
        parser.add_argument("--drop", default=None, metavar="DSUGAR,DPRICE,RATIO", help="drop near-duplicate brands")

    def configure_paper(self, args: Any) -> None:
        self.gh_nodes, self.draws = int(args.gh_nodes), str(args.draws)
        self.firm_taste = bool(args.firm_taste)
        self.drop = tuple(float(v) for v in args.drop.split(",")) if args.drop else None
        if self.firm_taste and self.draws != "sobol":
            raise ValueError("--firm-taste needs --draws sobol (a product grid in 8 dimensions is too large)")
        self.load()

    def paper_run_key(self) -> dict[str, Any]:
        out: dict[str, Any] = {"draws": self.draws}
        if self.draws == "gh":
            out["gh_nodes"] = self.gh_nodes
        if self.firm_taste:
            out["firm_taste"] = True
        if self.drop:
            out["drop"] = list(self.drop)
        return out

    def payload(self) -> dict[str, Any]:
        return {"spec": self.name, **self.paper_run_key(), "layout": [s.name for s in self.base_layout().specs]}

    @property
    def normalisations(self) -> list[str]:
        return ["median a_i = 1 at mean demographics (mu = 0): delta in dollars, no linear price coefficient",
                "brand fixed effects absorbed; Z demeaned by brand", "taste spreads non-negative"]

    def taste_nodes(self, nu: int) -> tuple[np.ndarray, np.ndarray] | None:
        return gauss_hermite(3, self.gh_nodes * int(nu)) if self.draws == "gh" else None

    def base_layout(self) -> ParameterLayout:
        specs = [ParameterSpec(n, "taste", c, b, s, 0.0 if b.startswith("nu") else -20 * u, 20 * u, unit=u)
                 for n, c, b, s, u in TASTE]
        if self.firm_taste:
            specs += [ParameterSpec(f"sigma_firm{f}", "taste", f"firm{f}", f"nuf{f}", 0.02, 0.0, 0.6, unit=0.03)
                      for f in FIRMS]
        specs += [ParameterSpec(n, "scale", None, b, 0.0, -20 * u, 20 * u, unit=u) for n, b, u in SCALE]
        specs.append(ParameterSpec("sigma_p", "sigma_p", None, None, 0.5, 0.05, 5.0, unit=0.5))
        return ParameterLayout(specs)

    def pcm_spec(self) -> PCMSpec:
        chars = CHARS + (tuple(f"firm{f}" for f in FIRMS) if self.firm_taste else ())
        return PCMSpec(chars, tol=1e-13, warm_evals=3000, cold_evals=20000, path_evals=1000, path_seconds=60.0)

    def linear(self) -> Absorb:
        return self.fe

    def moment_list(self, moments: str, **kw: Any) -> list[Any]:
        return [IVMoments(self.Z, name="aggregate")]

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str = "all",
                      layout_changes: dict[str, Any] | None = None, **kw: Any) -> Any:
        """PaperSpec's problem with W = (Z'Z/n)^-1 (``Absorb`` has no aggregate weighting of its own)."""
        prob = super().build_problem(nu, seed, threads, moments, layout_changes, **kw)
        prob.set_weighting(np.linalg.inv(self.Z.T @ self.Z / len(self.Z)))
        return prob

    def gate_profiles(self) -> list[Any]:
        return []

    def post(self, pipe: Any, problem: Any, result: Any, out: pathlib.Path) -> dict[str, Any]:
        theta, delta = np.asarray(result.theta), np.asarray(result.evaluation.aux["delta"])
        post = self.postestimation(problem.model, theta, delta)
        pd.DataFrame(post["products"]).to_csv(out / "product_level.csv", index=False)
        summary = comparison.summarise(post["products"], post["group_matrix"], post["group_names"])
        summary["theta"] = dict(zip(problem.layout.names, theta.tolist()))
        return summary

    def headlines(self, problem: Any, result: Any) -> dict[str, float]:
        out = {n: float(v) for n, v in zip(problem.layout.names, result.theta)}
        out["q"] = float(result.q)
        return out


if __name__ == "__main__":
    sys.exit(pipeline.main(NevoSpec(), default_nu=1, default_base=RESULTS))

"""Grieco, Murry & Yurukoglu (2024, QJE 139(2)) US automobile data: products, CPS households, micro moments
and second-choice matrices, read from the authors' replication package (Harvard Dataverse doi:10.7910/DVN/CZGOKP,
``GMY_Replication.tar.gz``, MIT licence). The raw files are not committed: unpack the package so that
``<root>/data/data_blp.csv`` exists (default root ``gmy_replication/data/GMY_Replication`` of this repo).

Every rule below is the authors' code in ``codes/`` of the package, cited by file:

* products (``loadWardsSalesData.m``): one market per year (1980-2018, 165-323 models, 9,694 rows, all with
  ``FINAL_SAMPLE`` = 1); price ``msrp / 10`` in $10,000 of 2015 (msrp is already real, in $1,000 of 2015); market size
  ``number_households / 2.5`` (average holding 5 years, 2 cars per household), shares ``sales / market size``; the
  file order is the authors' product order (sorted by year, then make), which the second-choice matrices index.
* ownership (``makeOwnership.m``): products with the same ``parent`` are jointly priced.
* utility interactions (``loadWardsSalesData.m`` xTerms, ``getMu_lowMem.m``): the 15 columns ``XTERMS``; within mu the
  three continuous ones (log footprint, log hp, log mpg) are z-scored over the pooled products (MATLAB zscore, sd with
  n - 1), and the micro moments use them raw.
* second stage (``linearDemand.do``): delta on price (instrument ``lag_pl_con``) and ``STAGE2_X`` with year and
  ``make2`` fixed effects, clustered by ``make2``; the do-file's differentiation counts ``IV_N`` and ``IV_N_weight`` are
  built here too (the do-file builds them and does not use them in the baseline).
* households (``loadCPSMktHholds.m``): per year a uniform draw without replacement of CPS households, weights the CPS
  weights normalised to sum to 1 in each year; income in 2015 dollars, winsorised to [5,000, 225,000]; ``inc`` =
  (income - pooled mean) / 10,000, ``inc2`` = inc^2 / 100; family size 1 + married + children; age / 100; income
  quintiles from the national cutoffs (2018 dollars converted to 2015); CEX sets live in 1983-2004, MRI in 1992-2014.
* micro moments (``loadMicMomData.m``, ``getMicroMoments.m``, ``ExjCondH.m``): the 33 moments of specification 6
  (``NEWPATTERN``), weight 1 / variance. The model counterpart of E[x | H] is the household-weighted mean over the
  households in H (pooled over the set's years, each year's weights summing to 1) of that household's expected x
  given a purchase: ``MicroMomentEvaluator``. The "CEX" and "MRI" column labels of the working paper's Table 8 are
  swapped relative to the files the code reads (microMomentsCEX.mat holds the values Table 8 prints under MRI, and
  the reverse; the urban-rural truck row, which the code takes from the CEX file, sits under MRI in the table).
* second choices (``getSecondChoiceStructure.m``): ``secondChoiceProb_<year>`` has first choices in rows and second
  choices in columns, in the year's product order, short of the year's product count by up to two trailing
  rows/columns, which are padded with zeros.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import urllib.request
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DATAVERSE_DOI", "DEFAULT_ROOT", "XTERMS", "Z_XTERMS", "STAGE2_X", "SECOND_CHOICE_YEARS", "SECOND_CHOICE_CHARS",
    "GMYData", "MicroMoments", "package_root", "fetch_package", "load_products", "load_agents", "load_micro_tables",
    "load_micro_moments", "load_second_choices", "load", "zscore", "MicroMomentEvaluator", "model_micro_moments",
]

DATAVERSE_DOI = "doi:10.7910/DVN/CZGOKP"
DATAVERSE_API = "https://dataverse.harvard.edu/api"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "gmy_replication" / "data" / "GMY_Replication"

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
                                f"(nfp_harness.gmy_data.fetch_package downloads it from {DATAVERSE_DOI})")
    return r


def fetch_package(dest: str | pathlib.Path) -> pathlib.Path:
    """fetch_package(dest) -> path of ``GMY_Replication.tar.gz`` downloaded into ``dest`` from the Dataverse API
    (dataset ``DATAVERSE_DOI``, latest version). Unpack it with ``tar xzf`` so ``package_root`` finds it."""
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
                                f"?persistentId={DATAVERSE_DOI}", timeout=60) as fh:
        files = json.load(fh)["data"]
    ids = [f["dataFile"]["id"] for f in files if f["dataFile"]["filename"] == "GMY_Replication.tar.gz"]
    if len(ids) != 1:
        raise RuntimeError(f"expected one GMY_Replication.tar.gz in {DATAVERSE_DOI}, found {len(ids)}")
    out = dest / "GMY_Replication.tar.gz"
    urllib.request.urlretrieve(f"{DATAVERSE_API}/access/datafile/{ids[0]}", out)
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    """MATLAB zscore: (v - mean) / sd with the n - 1 sd."""
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=1)


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


def load_agents(root: str | pathlib.Path | None = None, n_households: int = 1000, seed: int = 8673320,
                years: Sequence[int] | None = None) -> pd.DataFrame:
    """load_agents(root=None, n_households=1000, seed=8673320, years=None) -> household table, ``n_households``
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


@dataclasses.dataclass(frozen=True, eq=False)
class MicroMoments:
    """The 33 micro moments of GMY's specification 6 (``NEWPATTERN``) in loadMicMomData.m order.

    ``table`` has one row per moment: ``name``, ``survey`` (cex / mri), ``char`` (data column), ``kind`` ("diff":
    E[x | set a] - E[x | set b]; "ratio": P(buy | a) / P(buy | b)), ``demographic`` (agent column of the sets),
    ``a`` and ``b`` (its values), ``value`` and ``variance``. ``weight`` is diag(1 / variance)."""

    table: pd.DataFrame

    @property
    def values(self) -> np.ndarray:
        return self.table.value.to_numpy(float)

    @property
    def variances(self) -> np.ndarray:
        return self.table.variance.to_numpy(float)

    @property
    def weight(self) -> np.ndarray:
        return np.diag(1.0 / self.variances)


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


def load(root: str | pathlib.Path | None = None, n_households: int = 1000, seed: int = 8673320,
         years: Sequence[int] | None = None) -> GMYData:
    """load(root=None, n_households=1000, seed=8673320, years=None) -> GMYData; ``years`` keeps those markets only
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


class MicroMomentEvaluator:
    """MicroMomentEvaluator(products, agents, micro)(probabilities) -> (33,) model counterparts (ExjCondH.m).

    ``probabilities[t]`` is the (J_t, H_t) purchase probability matrix of the t-th market in sorted year order, rows
    in ``products`` order within the year, columns in ``agents`` order within the year. For a "diff" row each
    household's expected x given a purchase, sum_j x_j P_jh / sum_j P_jh, is averaged with the household weights over
    the households of set a (and of set b) in the survey's years, pooled across years (each year's weights sum to 1,
    so each year counts equally). For a "ratio" row the household's purchase probability sum_j P_jh is averaged the
    same way. This is GMY's definition; it is not pyblp's purchase-weighted E[x | H, inside].
    """

    def __init__(self, products: pd.DataFrame, agents: pd.DataFrame, micro: MicroMoments) -> None:
        self.years = np.unique(products.market_ids.to_numpy())
        self.micro = micro
        self.J = [int((products.market_ids == y).sum()) for y in self.years]
        self.H = [int((agents.market_ids == y).sum()) for y in self.years]
        chars = sorted(set(MICRO_CHAR.values()))
        self.chars = chars
        self.x = [products.loc[products.market_ids == y, chars].to_numpy(float) for y in self.years]
        # per moment row and side: one (H_t,) weight vector per market, zero outside the set
        self.sides: list[tuple[np.ndarray, np.ndarray]] = []
        w_t = [agents.loc[agents.market_ids == y] for y in self.years]
        for row in micro.table.itertuples():
            pair = []
            for level in (row.a, row.b):
                vec = [np.where((ag[row.demographic].to_numpy() == level) & (ag[row.survey].to_numpy() == 1),
                                ag.weights.to_numpy(float), 0.0) for ag in w_t]
                pair.append(np.concatenate(vec))
            self.sides.append((pair[0], pair[1]))
        self.kind = micro.table.kind.to_numpy()
        self.col = [chars.index(MICRO_CHAR[c]) if k == "diff" else -1
                    for c, k in zip(micro.table.char, micro.table.kind)]

    def __call__(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        if len(probabilities) != self.years.size:
            raise ValueError(f"{len(probabilities)} probability matrices for {self.years.size} markets")
        inside, cond = [], []
        for t, P in enumerate(probabilities):
            P = np.asarray(P, dtype=float)
            if P.shape != (self.J[t], self.H[t]):
                raise ValueError(f"{self.years[t]}: probabilities {P.shape}, expected ({self.J[t]}, {self.H[t]})")
            s = P.sum(axis=0)
            inside.append(s)
            cond.append((self.x[t].T @ P) / s)       # (C, H_t)
        S = np.concatenate(inside)
        X = np.concatenate(cond, axis=1)
        out = np.empty(len(self.sides))
        for k, (wa, wb) in enumerate(self.sides):
            v = S if self.kind[k] == "ratio" else X[self.col[k]]
            a, b = (wa @ v) / wa.sum(), (wb @ v) / wb.sum()
            out[k] = a / b if self.kind[k] == "ratio" else a - b
        return out


def model_micro_moments(probabilities: Sequence[np.ndarray], products: pd.DataFrame, agents: pd.DataFrame,
                        micro: MicroMoments) -> np.ndarray:
    """model_micro_moments(probabilities, products, agents, micro) -> (33,): ``MicroMomentEvaluator`` in one call."""
    return MicroMomentEvaluator(products, agents, micro)(probabilities)


#: demographic agent column -> its levels, for the household sets of the micro moments
SET_DEMOGRAPHICS = {"inc_quintile": (1, 2, 3, 4, 5), "age_group": (1, 2, 3, 4, 5), "fs_group": (1, 2, 3, 4),
                    "rural": (0, 1)}


def set_column(survey: str, demographic: str, level: Any) -> str:
    """Name of the 0/1 agent column of a household set: ``<survey>_<demographic>_<level>``."""
    return f"{survey}_{demographic}_{int(level)}"


def add_set_columns(agents: pd.DataFrame) -> pd.DataFrame:
    """``agents`` with one 0/1 column per (survey, demographic, level): demographic = level and the year in the
    survey's window (loadCPSMktHholds.m lines 109-147)."""
    a = agents.copy()
    for s in ("cex", "mri"):
        for col, levels in SET_DEMOGRAPHICS.items():
            for v in levels:
                a[set_column(s, col, v)] = ((a[col] == v) & (a[s] == 1)).astype(float)
    return a


def group_moment_rows(micro: MicroMoments) -> list[tuple[str, str | None, str, str]]:
    """(kind, product column or None, set column a, set column b) for each of the 33 rows of ``micro``, in order: the
    arguments of ``moments.GroupMomentRow``."""
    out = []
    for r in micro.table.itertuples():
        char = MICRO_CHAR[r.char] if r.kind == "diff" else None
        out.append((r.kind, char, set_column(r.survey, r.demographic, r.a), set_column(r.survey, r.demographic, r.b)))
    return out

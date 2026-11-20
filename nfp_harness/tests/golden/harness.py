"""Golden-master cases of the refactor safety net: what each paper's code computes at one fixed point.

Every case evaluates one paper's estimation code, unchanged, at a fixed theta, a fixed small number of draws and a fixed
seed, and returns its outputs as a flat ``{key: array}`` dict ("pins"). ``nfp_harness/tests/test_golden_eval.py``
recomputes each case and requires every pin to equal the stored golden bit for bit (dtype, shape and value, NaN equal
to NaN). Wall-clock fields and thread counts are not pinned.

Cases (``CASES``):

- ``petrin_pcm``: petrin_replication/estimate_pcm.py, ``PetrinSpec`` on all 13 markets (fewer than 11 leave the
  demand or supply instruments rank-deficient and the spec refuses), ``build_problem(nu=1, seed=0, moments="all")``
  = Sargan aggregate + CEX micro + log-cost supply, ``evaluate`` at the layout starts with the Jacobian from a stored
  warm start (``harness_case``), then the post-estimation outputs at that point.
- ``gmy_pcm`` (round-4 layout, age tastes on) and ``gmy_pcm_no_age_taste`` (round-2 layout): gmy_replication/
  estimate_pcm.py, ``GMYSpec(GMY_YEARS, households=GMY_HOUSEHOLDS)``, ``build_problem(nu=1, seed=0,
  moments="demand_supply")`` = Sargan aggregate + group micro + second choice + supply, as Petrin. The round-4 case
  sets the two age tastes (start 0) to ``GMY_AGE_TASTES``; at 0 the two layouts give identical g and delta.
- ``nevo_pcm``: nevo_replication/estimate_pcm.py's ``NevoSpec`` on the harness, 5-node Gauss-Hermite grid,
  ``evaluate`` at the layout starts: q, the moments g = Z'xi/n, the gradient of q, delta, xi, G and W. The golden was
  written on 0e95f4a by the script's own pre-harness ``PCMProblem`` (legacy ``python_pcm_demand`` binding) at its
  ``start_values()``, the same theta. delta, xi and g stay bit-identical: the same consumers, the same cold
  ``pcm_invert::invert`` and the same brand demeaning. Four pins are compared to a relative tolerance because the
  harness computes them by a different but equivalent sequence of operations: W (``NFPGMMProblem.set_weighting``
  stores 0.5 (W + W'), and ``inv(Z'Z/n)`` is symmetric only to 5e-16), q (through W), G (the kernel's LU solve of
  ds/ddelta D = -ds/dtheta and (Z'M/n) D, where the old code used numpy's solve and Z'D/n) and the gradient of q.
- ``gmy_blp``: gmy_replication/estimate_blp.py ``main()`` stage ``published`` (GMY's Table 4 theta, no fit) on
  ``GMY_YEARS`` with ``GMY_HOUSEHOLDS`` households; writes go to a temporary directory.
- ``kernel_synthetic``: the two synthetic markets of ``tests/fixtures/gmy_baseline.py`` (every ``pcm_kernel`` function,
  data-free), so the kernel baseline is tracked and always checked.

Not pinned: petrin_replication/estimate_blp.py. Its pyblp evaluation at the initial sigma and pi (``main("micro")`` with
``Optimization("return")``) takes about 210 s and is not reproducible run to run: two runs on c96d4b8 with identical
environments differed by up to 5e-8 in beta and the objective.

Each case runs in its own interpreter (``compute_isolated``): the three paper scripts share the module name
``estimate_pcm`` and set process-wide thread variables at import. The child gets ``OMP_NUM_THREADS`` = the kernel thread
count and one BLAS thread (``OPENBLAS``/``MKL``/``NUMEXPR_NUM_THREADS`` = 1) with ``MKL_CBWR=COMPATIBLE``, so the BLAS
reduction order is the same in every run and only the kernel's thread count varies; CBWR makes MKL's kernel choice
independent of array alignment, which a refactor may change without changing any arithmetic.

CLI (used by ``compute_isolated``): ``python -m nfp_harness.tests.golden.harness compute <case> --threads N --out F``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
GOLDEN_DIR = pathlib.Path(__file__).resolve().parent
VENDOR = ROOT / "nevo_replication" / ".vendor"
MAX_THREADS = 24          # cores this suite may occupy; the Nevo pool is sized to stay within it
TIMEOUT = 900             # seconds per isolated case run
GMY_YEARS = (1980, 1999)                  # 1980: a Table 5 year; 1999: a second-choice and MRI micro year
GMY_HOUSEHOLDS = 200                      # 256 draws per market at nu = 1
NEVO_GH_NODES = 5
GMY_AGE_TASTES = {"pi_z_log_footprint_age": 0.5, "pi_lux_age": 0.5}   # start 0; x age/100, about $2.5k at age 50
WARM_OFFSET = 1e-4       # warm start = solution + 1e-4 sin(row), in utility units ($1000 Petrin, $10k GMY)
UNPINNED = frozenset({"seconds", "wall_seconds", "threads", "wall", "elapsed"})

Pins = dict[str, np.ndarray]


@dataclass(frozen=True)
class Case:
    name: str
    paper: str
    build: Callable[[int, Pins | None], Pins]   # (threads, inputs: the golden's input.* pins or None)
    needs: tuple[str, ...]        # paths under ROOT, "module:<name>" (importable), "kernel", "legacy_bindings"
    threads: tuple[int, ...]      # kernel thread counts the test checks
    note: str
    #: pins compared to a relative tolerance instead of bit for bit: (pin, rtol) pairs, each with its reason in the
    #: case's docstring entry (module docstring)
    approx: tuple[tuple[str, float], ...] = ()


# ------------------------------------------------------------------ flattening and comparison

def _as_array(v: Any) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype == object:
        a = np.asarray([str(x) for x in a.ravel()]).reshape(a.shape)
    return a


def flatten(prefix: str, obj: Any, out: Pins | None = None) -> Pins:
    """Nested dicts / lists / arrays / scalars -> ``{"prefix.key.sub": array}``. Keys in ``UNPINNED`` are skipped;
    a list of dicts is indexed (``prefix.0.key``); a list of numbers becomes one array."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in UNPINNED:
                continue
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)) and any(isinstance(v, (dict, list, tuple)) for v in obj):
        for i, v in enumerate(obj):
            flatten(f"{prefix}.{i}", v, out)
    elif obj is None:
        out[prefix] = np.asarray("None")
    else:
        out[prefix] = _as_array(obj)
    return out


def digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest_all(pins: Pins) -> str:
    h = hashlib.sha256()
    for k in sorted(pins):
        h.update(k.encode())
        h.update(digest(pins[k]).encode())
    return h.hexdigest()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-identity: same dtype, shape and bytes (so -0.0 differs from 0.0 and a NaN equals the same NaN)."""
    return digest(np.asarray(a)) == digest(np.asarray(b))


def close(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    """Same dtype and shape, finite, and max |a - b| <= rtol max |a|."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape or a.dtype.kind != "f":
        return False
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return same(a, b)
    return bool(np.max(np.abs(a - b), initial=0.0) <= rtol * np.max(np.abs(a), initial=0.0))


def differences(ref: Pins, got: Pins, approx: dict[str, float] | None = None) -> list[str]:
    """Human-readable list of every pin that is missing, extra or not bit-identical (pins named in ``approx``: not
    within their relative tolerance)."""
    approx = approx or {}
    msgs = [f"missing {k}" for k in sorted(set(ref) - set(got))]
    msgs += [f"extra {k}" for k in sorted(set(got) - set(ref))]
    for k in sorted(set(ref) & set(got)):
        a, b = ref[k], got[k]
        if same(a, b) or (k in approx and close(a, b, approx[k])):
            continue
        if a.dtype.kind in "fc" and b.dtype.kind in "fc" and a.shape == b.shape:
            with np.errstate(invalid="ignore"):
                d = np.abs(a.astype(float) - b.astype(float))
            msgs.append(f"{k}: {a.dtype}{a.shape}, {int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))} entries "
                        f"differ, max |diff| {np.nanmax(d) if d.size else 0:.3e}, sha256 {digest(a)[:12]} -> "
                        f"{digest(b)[:12]}")
        else:
            msgs.append(f"{k}: {a.dtype}{a.shape} {np.ravel(a)[:3]} -> {b.dtype}{b.shape} {np.ravel(b)[:3]}")
    return msgs


# ------------------------------------------------------------------ files

def golden_dir() -> pathlib.Path:
    """``$NFP_GOLDEN_DIR`` when set, else this directory (the tracked goldens)."""
    env = os.environ.get("NFP_GOLDEN_DIR")
    return pathlib.Path(env) if env else GOLDEN_DIR


def golden_path(case: str, directory: pathlib.Path | None = None) -> pathlib.Path:
    return (directory or golden_dir()) / f"{case}.npz"


def save(path: pathlib.Path, pins: Pins, meta: dict[str, Any]) -> None:
    meta = dict(meta, sha256=digest_all(pins), pins=len(pins))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"pin:{k}": v for k, v in pins.items()}, **{"meta:json": np.asarray(json.dumps(meta))})
    os.replace(tmp, path)


def load(path: pathlib.Path) -> tuple[Pins, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        pins = {k[4:]: z[k] for k in z.files if k.startswith("pin:")}
        meta = json.loads(str(z["meta:json"]))
    return pins, meta


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"),
            "dirty_tracked": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ------------------------------------------------------------------ environment of a case run

def child_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               MKL_CBWR="COMPATIBLE", OMP_WAIT_POLICY="PASSIVE", OMP_DYNAMIC="FALSE", PYTHONHASHSEED="0",
               PYTHONPATH=os.pathsep.join([str(ROOT)] + [p for p in [env.get("PYTHONPATH")] if p]))
    return env


def compute_isolated(case: str, threads: int, golden: pathlib.Path | None = None) -> tuple[Pins, dict[str, Any]]:
    """Run ``case`` at ``threads`` kernel threads in a fresh interpreter; (pins, meta). With ``golden`` the run takes
    its inputs (the ``input.*`` pins, e.g. a warm-start delta) from that file; without, it computes them."""
    with tempfile.TemporaryDirectory(prefix=f"golden_{case}_") as d:
        out = pathlib.Path(d) / "pins.npz"
        cmd = [sys.executable, "-m", "nfp_harness.tests.golden.harness", "compute", case, "--threads", str(threads),
               "--out", str(out)] + (["--inputs", str(golden)] if golden is not None else [])
        r = subprocess.run(cmd, cwd=ROOT, env=child_env(threads), capture_output=True, text=True, timeout=TIMEOUT)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"golden case {case} at {threads} threads failed (exit {r.returncode}):\n"
                               f"{r.stderr[-4000:]}")
        return load(out)


def missing_needs(case: Case) -> list[str]:
    missing = []
    for need in case.needs:
        if need.startswith("module:"):
            mod = need.split(":", 1)[1]
            with _vendor_path():
                if importlib.util.find_spec(mod) is None:
                    missing.append(f"python module {mod}")
        elif need == "kernel":
            if not any((ROOT / "nfp_harness" / "_native").glob("pcm_kernel*.so")):
                missing.append("nfp_harness/_native/pcm_kernel*.so (build the CMake target pcm_kernel_py)")
        elif not (ROOT / need).exists():
            missing.append(str(ROOT / need))
    return missing


@contextlib.contextmanager
def _vendor_path() -> Iterator[None]:
    added = [p for p in (str(ROOT), str(VENDOR)) if p not in sys.path]
    sys.path[:0] = added
    try:
        yield
    finally:
        for p in added:
            sys.path.remove(p)


def load_script(rel: str, name: str) -> Any:
    """Import a paper script by path under a unique module name."""
    path = ROOT / rel
    for p in (str(VENDOR), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ generic pins of the harness path

def pin_evaluation(prob: Any, ev: Any) -> Pins:
    """g, G, q, q by block, delta, xi, beta, D, supply gamma / omega / log mc, micro predictions, model shares, and
    the per-market inversion record (residual, share evaluations, rcond, status, stages without timings)."""
    from nfp_harness.tests.fixtures.gmy_baseline import strip_timings

    aux = ev.aux
    pins: Pins = {"eval.status": np.asarray(ev.status.value), "eval.theta": np.asarray(ev.theta),
                  "eval.g": np.asarray(ev.g), "eval.G": np.asarray(ev.G), "eval.q": np.asarray(aux["q"]),
                  "eval.W": np.asarray(prob.W)}
    names = list(ev.blocks)
    pins["eval.block_names"] = np.asarray(names)
    pins["eval.block_rows"] = np.asarray([[ev.blocks[n].start, ev.blocks[n].stop] for n in names])
    pins["eval.q_blocks"] = np.asarray([aux["q_blocks"][n] for n in names])
    for k in ("delta", "xi", "beta", "D", "gamma", "omega", "log_mc", "linear_coefficients", "sd_xi"):
        if aux.get(k) is not None:
            pins[f"eval.{k}"] = np.asarray(aux[k])
    for k in ("beta_names", "linear_names_all"):
        if aux.get(k) is not None:
            pins[f"eval.{k}"] = np.asarray(list(aux[k]))
    for blk, v in aux.get("micro_predicted", {}).items():
        pins[f"eval.micro_predicted.{blk}"] = np.asarray(v)
    flatten("eval.supply", aux.get("supply") or {}, pins)
    for name in names:   # moment summaries (second choice, groups, margin): whatever the block reports
        if isinstance(aux.get(name), dict):
            flatten(f"eval.aux.{name}", aux[name], pins)
    sol = aux["solution"]
    if "shares" in sol.outputs:
        pins["eval.shares"] = np.asarray(sol.outputs["shares"])
    pm = sol.per_market
    pins["eval.market_ids"] = np.asarray([str(m["market_id"]) for m in pm])
    for k in ("residual", "evaluations", "rcond", "products", "draws"):
        pins[f"eval.per_market.{k}"] = np.asarray([m[k] for m in pm])
    for k in ("status", "kernel_status"):
        pins[f"eval.per_market.{k}"] = np.asarray([str(m[k]) for m in pm])
    pins["eval.per_market.stages"] = np.asarray([strip_timings(str(m["stages"])) for m in pm])
    return pins


def pin_post(model: Any, theta: np.ndarray, delta: np.ndarray, spec: Any, threads: int) -> Pins:
    """Post-estimation at (theta, delta): product-level own elasticities, Nash markups / costs / margins, diversion
    (by group), per-market inside share and its elasticity, the group diversion matrix, consumer surplus per market
    (total and per draw) and the window anatomy."""
    from nfp_harness.models import pcm_post as pp

    post = pp.postestimation(model, theta, delta, firm_ids=spec.firm_column, groups=spec.groups,
                             group_other=spec.group_other)
    pins: Pins = {}
    flatten("post.products", post["products"], pins)
    flatten("post.markets", post["markets"], pins)
    pins["post.group_names"] = np.asarray(post["group_names"])
    pins["post.group_matrix"] = np.asarray(post["group_matrix"])
    pins["post.own_elasticity_median"] = np.asarray(float(np.nanmedian(post["products"]["own_elasticity"])))
    for m in model.markets:
        cs = pp.consumer_surplus(model, theta, delta, m.market_id, n_threads=threads)
        flatten(f"post.cs.{m.market_id}", cs, pins)
    flatten("post.window_anatomy", pp.window_anatomy(model, theta, delta, n_threads=threads), pins)
    return pins


def evaluate_and_post(prob: Any, spec: Any, theta: np.ndarray, threads: int) -> Pins:
    ev = prob.evaluate(theta, need_jacobian=True)
    if not ev.ok:
        raise RuntimeError(f"evaluation at the golden theta ended {ev.status.value}: {ev.aux.get('note')} "
                           f"{[pm.get('stages') for pm in ev.aux.get('per_market', [])]}")
    pins = pin_evaluation(prob, ev)
    pins.update(pin_post(prob.model, np.asarray(ev.theta), np.asarray(ev.aux["delta"]), spec, threads))
    return pins


def _close(prob: Any) -> None:
    close = getattr(prob.model, "close", None)
    if callable(close):
        close()


def harness_case(make: Callable[[int], tuple[Any, Any]], threads: int, inputs: Pins | None, warm: bool,
                 theta_changes: dict[str, float] | None = None) -> Pins:
    """The pins of one ``NFPGMMProblem`` (``make(threads)`` -> (problem, spec)) at its layout starts, with the
    entries named in ``theta_changes`` replaced (e.g. a taste that starts at 0 and would otherwise be inert).

    ``warm``: the evaluation starts every market from ``input.warm_delta``, the delta solved at the same theta (a
    cold ``model.solve`` when the golden is generated) moved by ``WARM_OFFSET`` sin(row), so the warm stage runs real
    Newton steps on the paper's data (a start at the exact solution would stop at the first residual check, and the
    inversion's bits would not reach any output). The stored start is reused when the golden is checked, so the pinned
    path is the warm inversion, the derivatives, the supply secants and the post-estimation, not the long cold stage
    (5.5 minutes for Petrin at 16 threads; cold inversion is pinned on the synthetic markets of ``kernel_synthetic``).
    Otherwise the evaluation is cold. ``layout.starts`` is pinned, so a changed start fails although theta is an
    input."""
    from nfp_harness.types import KernelRequests

    def golden_theta(names: list[str], starts: np.ndarray) -> np.ndarray:
        theta = starts.copy()
        for n, v in (theta_changes or {}).items():
            theta[names.index(n)] = v
        return theta

    prob, spec = make(threads)
    try:
        starts = np.asarray(prob.layout.starts(), dtype=float)
        names = list(prob.layout.names)
        pins: Pins = {"layout.names": np.asarray(names), "layout.starts": starts,
                      "golden.theta": golden_theta(names, starts),
                      "input.theta": golden_theta(names, starts) if inputs is None
                      else np.asarray(inputs["input.theta"])}
        theta = pins["input.theta"]
        if warm:
            if inputs is None:
                sol = prob.model.solve(theta, KernelRequests(), False)
                if sol.status.value != "ok":
                    raise RuntimeError(f"cold solve for the warm delta ended {sol.status.value}")
                d = np.asarray(sol.delta, dtype=float)
                pins["input.warm_delta"] = d + WARM_OFFSET * np.sin(np.arange(d.size, dtype=float))
                _close(prob)
                prob, spec = make(threads)
            else:
                pins["input.warm_delta"] = np.asarray(inputs["input.warm_delta"])
            if prob.model.warm_from_delta(pins["input.warm_delta"], theta) != len(prob.model.markets):
                raise RuntimeError("the warm delta does not cover every market")
        pins.update(evaluate_and_post(prob, spec, theta, threads))
    finally:
        _close(prob)
    return pins


# ------------------------------------------------------------------ cases

def build_petrin_pcm(threads: int, inputs: Pins | None) -> Pins:
    def make(t: int) -> tuple[Any, Any]:
        mod = sys.modules.get("golden_petrin_pcm") or load_script("petrin_replication/estimate_pcm.py",
                                                                  "golden_petrin_pcm")
        spec = mod.PetrinSpec()
        return spec.build_problem(1, 0, t, "all"), spec
    return harness_case(make, threads, inputs, warm=True)


def build_gmy_pcm(robustness: tuple[str, ...], theta_changes: dict[str, float] | None = None
                  ) -> Callable[[int, Pins | None], Pins]:
    def build(threads: int, inputs: Pins | None) -> Pins:
        def make(t: int) -> tuple[Any, Any]:
            mod = sys.modules.get("golden_gmy_pcm") or load_script("gmy_replication/estimate_pcm.py", "golden_gmy_pcm")
            spec = mod.GMYSpec(GMY_YEARS, households=GMY_HOUSEHOLDS, robustness=robustness)
            return spec.build_problem(1, 0, t, "demand_supply"), spec
        return harness_case(make, threads, inputs, warm=True, theta_changes=theta_changes)
    return build


def build_nevo_pcm(threads: int, inputs: Pins | None) -> Pins:
    del inputs
    mod = load_script("nevo_replication/estimate_pcm.py", "golden_nevo_pcm")
    spec = mod.NevoSpec()
    spec.gh_nodes = NEVO_GH_NODES
    prob = spec.build_problem(1, 0, threads, "all")
    try:
        theta = np.asarray(prob.layout.starts(), dtype=float)
        ev = prob.evaluate(theta, need_jacobian=True)
        if not ev.ok:
            raise RuntimeError(f"Nevo evaluation at the layout starts ended {ev.status.value}")
        g, G = np.asarray(ev.g), np.asarray(ev.G)
        pins: Pins = {"nevo.names": np.asarray(list(prob.layout.names)), "nevo.theta": theta,
                      "nevo.q": np.asarray(ev.aux["q"]), "nevo.g": g, "nevo.dq": 2 * prob.n * (G.T @ prob.W @ g),
                      "nevo.ok": np.asarray(True), "nevo.delta": np.asarray(ev.aux["delta"]),
                      "nevo.xi": np.asarray(ev.aux["xi"]), "nevo.G": G, "nevo.W": np.asarray(prob.W)}
    finally:
        _close(prob)
    return pins


def build_gmy_blp(threads: int, inputs: Pins | None) -> Pins:
    del inputs
    mod = load_script("gmy_replication/estimate_blp.py", "golden_gmy_blp")
    with tempfile.TemporaryDirectory(prefix="golden_gmy_blp_") as d:
        mod.OUT = pathlib.Path(d)
        argv = sys.argv
        sys.argv = ["estimate_blp.py", "published", "--households", str(GMY_HOUSEHOLDS), "--threads",
                    str(max(1, min(threads, len(GMY_YEARS)))), "--years", *map(str, GMY_YEARS)]
        try:
            mod.main()
        finally:
            sys.argv = argv
        tag = f"blp_published_h{GMY_HOUSEHOLDS}_{len(GMY_YEARS)}y"
        pins = flatten("summary", json.loads((mod.OUT / f"{tag}.json").read_text()))
        with np.load(mod.OUT / f"{tag}.npz", allow_pickle=False) as z:
            pins.update({f"npz.{k}": z[k] for k in z.files})
    return pins


def build_kernel_synthetic(threads: int, inputs: Pins | None) -> Pins:
    del inputs
    from nfp_harness.tests.fixtures import gmy_baseline

    pins: Pins = {}
    with tempfile.TemporaryDirectory(prefix="golden_kernel_") as d:
        for path in gmy_baseline.build(pathlib.Path(d), threads, skip_gmy=True):
            inp, out, _ = gmy_baseline.load(path)
            flatten(f"{path.stem}.in", inp, pins)
            flatten(f"{path.stem}.out", out, pins)
    return pins


PCM_DATA = ("module:pyblp", "kernel")
GMY_DATA = ("gmy_replication/data/GMY_Replication", "gmy_replication/results/blp_fit_h1000.json")

CASES: dict[str, Case] = {c.name: c for c in (
    Case("petrin_pcm", "Petrin", build_petrin_pcm, PCM_DATA, (1, 8),
         "PetrinSpec on all 13 markets, nu 1, seed 0, moments all, layout starts"),
    Case("gmy_pcm", "GMY", build_gmy_pcm((), GMY_AGE_TASTES), PCM_DATA + GMY_DATA, (1, 8),
         f"GMYSpec round-4 layout (age tastes on) on {GMY_YEARS}, {GMY_HOUSEHOLDS} households, nu 1, seed 0, "
         f"moments demand_supply, layout starts with {GMY_AGE_TASTES}"),
    Case("gmy_pcm_no_age_taste", "GMY", build_gmy_pcm(("no_age_taste",)), PCM_DATA + GMY_DATA, (1, 8),
         f"GMYSpec round-2 layout (robustness no_age_taste) on {GMY_YEARS}, {GMY_HOUSEHOLDS} households, nu 1, "
         "seed 0, moments demand_supply, layout starts"),
    Case("nevo_pcm", "Nevo", build_nevo_pcm, PCM_DATA, (1, 8),
         f"nevo_replication NevoSpec, {NEVO_GH_NODES}-node GH grid, evaluate at the layout starts (golden written "
         f"by the pre-harness PCMProblem at its start_values)",
         approx=(("nevo.W", 1e-15), ("nevo.q", 1e-13), ("nevo.G", 1e-12), ("nevo.dq", 1e-11))),
    Case("gmy_blp", "GMY", build_gmy_blp, ("module:pyblp",) + GMY_DATA, (1,),
         f"gmy estimate_blp main() stage published on {GMY_YEARS}, {GMY_HOUSEHOLDS} households"),
    Case("kernel_synthetic", "kernel", build_kernel_synthetic, ("kernel",), (1, 8),
         "tests/fixtures/gmy_baseline.py synthetic markets (a_lo and alpha_lo bounds), every pcm_kernel function"),
)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compute one golden case in this interpreter")
    ap.add_argument("command", choices=("compute",))
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--inputs", type=pathlib.Path, default=None, help="golden file whose input.* pins to use")
    a = ap.parse_args(argv)
    if not 1 <= a.threads <= MAX_THREADS:
        raise SystemExit(f"--threads must lie in [1, {MAX_THREADS}]")
    inputs = None
    if a.inputs is not None:
        inputs = {k: v for k, v in load(a.inputs)[0].items() if k.startswith("input.")}
    t0 = time.time()
    pins = CASES[a.case].build(a.threads, inputs)
    native = sys.modules.get("nfp_harness._native.pcm_kernel")
    meta = {"case": a.case, "threads": a.threads, "seconds": time.time() - t0, "numpy": np.__version__,
            "kernel_module": getattr(native, "__file__", None), **git_state()}
    save(a.out, pins, meta)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())

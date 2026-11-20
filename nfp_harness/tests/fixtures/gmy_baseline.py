"""Baseline fixtures of the compiled kernel: every existing ``pcm_kernel`` function on three markets, saved so that
additive kernel work (new functions, unchanged existing ones) can be proven bit-identical (GMY spec WP-G0).

Markets:

- ``synthetic_a``: 8 products, 2 taste columns, 50 draws, one scale basis column, ``a_lo = 0.2`` (bound on a_i);
- ``synthetic_alpha``: the same market with the bound on alpha_i (``alpha_lo = 0.2``);
- ``gmy_1980``: GMY 1980 after the twin merge and the price-tie spread, 1,024 draws (nu = 1), theta at the GMY
  spec's starting point, delta from one cold ``solve_market``.

For each market the file holds the inputs (market arrays, theta, phi, sigma_p, the bound keyword, data shares,
delta) and the outputs of ``solve_market`` (cold for the synthetic markets, warm from the saved delta for GMY 1980),
``evaluate_market`` with a three-part micro request and price derivatives, ``frontier_signature``, ``max_step``
along a fixed random direction, ``affected_mass``, ``frontier_anatomy`` and ``consumer_surplus``.

Build (writes ``<dir>/<market>.npz``; the directory is untracked, e.g. under ``$KIROCREW_SCRATCH``):

    python -m nfp_harness.tests.fixtures.gmy_baseline <dir> [--threads 8] [--skip-gmy]

``nfp_harness/tests/test_kernel_regression.py`` reloads the files and checks every output for bit-identity.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any

import numpy as np

MARKET_FIELDS = ("x", "p", "weights", "taste_basis", "taste_col", "scale_basis")
SOLVE_KW = dict(tol=1e-9, warm_evals=2500, cold_evals=40000, path_evals=1500, path_seconds=90.0, rcond_min=1e-13)
TIMING = re.compile(r"\d+(\.\d+)?(e[+-]?\d+)?s\b")


def _kernel() -> Any:
    from nfp_harness._native import pcm_kernel
    return pcm_kernel


def strip_timings(stages: str) -> str:
    """The ``stages`` string of solve_market without its wall-clock numbers ("warm ok 0.01s 4ev" -> "warm ok Ts 4ev")."""
    return TIMING.sub("Ts", stages)


def synthetic_inputs(bound: str) -> dict[str, Any]:
    """The synthetic market with its parameters; ``bound`` is "a" (a_lo = 0.2) or "alpha" (alpha_lo = 0.2)."""
    rng = np.random.default_rng(20261118)
    n, r = 8, 50
    x = np.column_stack([np.ones(n), rng.normal(size=n), (rng.random(n) < 0.5).astype(float)])
    p = 1.0 + 0.4 * np.arange(n) + 0.05 * rng.random(n)
    taste_basis = rng.normal(size=(r, 2))
    scale_basis = rng.normal(scale=0.5, size=(r, 1))
    weights = np.full(r, 1.0 / r)
    theta = np.array([0.15, 0.1])
    phi = np.array([0.3])
    sigma_p = 1.5
    delta = 1.8 * p - 0.15 * p ** 2 - 0.3 + 0.002 * rng.normal(size=n)   # concave, slopes above the bound
    return {"x": x, "p": p, "weights": weights, "taste_basis": taste_basis, "taste_col": np.array([1, 2]),
            "scale_basis": scale_basis, "theta": theta, "phi": phi, "sigma_p": sigma_p, "delta_true": delta,
            "bound_key": "a_lo" if bound == "a" else "alpha_lo", "bound_value": 0.2}


def gmy_1980_inputs(threads: int) -> dict[str, Any]:
    """GMY 1980 at nu = 1 from the GMY spec, theta at its starting point, delta from one cold solve_market."""
    root = pathlib.Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "gmy_replication"), str(root / "nevo_replication" / ".vendor")]
    import estimate_pcm as gmy  # gmy_replication/estimate_pcm.py

    spec = gmy.GMYSpec((1980,), robustness=("no_age_taste",))   # the round-2 layout the fixtures were built on
    layout = spec.layout()
    from nfp_harness.models.pcm import PCMModel
    model = PCMModel(spec.pcm_spec(), {"products": spec.products, "agents": spec.agents}, layout, 1, gmy.SEED,
                     total_threads=threads)
    try:
        km = model.kernel_market(0)
        th, ph, sp = model.kernel_params(layout.starts())
        kw = model.kernel_kwargs(layout.starts())
        (key, value), = kw.items()
        out = {f: np.asarray(getattr(km, f)) for f in MARKET_FIELDS}
        out.update(theta=th, phi=ph, sigma_p=sp, bound_key=key, bound_value=value,
                   s_data=np.asarray(model._mk[0].s_data, dtype=float))
    finally:
        model.close()
    return out


def market_of(inp: dict[str, Any]) -> Any:
    k = _kernel()
    return k.Market(np.asarray(inp["x"], dtype=float), np.asarray(inp["p"], dtype=float),
                    np.asarray(inp["weights"], dtype=float), np.asarray(inp["taste_basis"], dtype=float),
                    [int(c) for c in np.asarray(inp["taste_col"]).ravel()], np.asarray(inp["scale_basis"], dtype=float))


def micro_request(inp: dict[str, Any]) -> dict[str, Any]:
    """Three micro parts: inside share, E[x_1 s] with h = 1, E[x_2 s] with h = the first taste basis column."""
    x = np.asarray(inp["x"], dtype=float)
    tb = np.asarray(inp["taste_basis"], dtype=float)
    H = np.column_stack([np.ones(tb.shape[0]), tb[:, 0]])
    U = np.column_stack([np.ones(x.shape[0]), x[:, 1], x[:, 2] - 0.5])
    return {"H": H, "U": U, "h_of_part": [0, 0, 1]}


def outputs(inp: dict[str, Any], n_threads: int, delta_warm: np.ndarray | None) -> dict[str, Any]:
    """Every existing kernel function at the fixture's inputs, as a flat dict of arrays and scalars."""
    k = _kernel()
    mk = market_of(inp)
    th, ph, sp = np.asarray(inp["theta"], dtype=float), np.asarray(inp["phi"], dtype=float), float(inp["sigma_p"])
    bkw = {str(inp["bound_key"]): float(inp["bound_value"])}
    s_data = np.asarray(inp["s_data"], dtype=float)
    out: dict[str, Any] = {}
    theta_warm = None if delta_warm is None else np.r_[th, ph, sp]
    sol = k.solve_market(s_data, mk, th, ph, sp, delta_warm, theta_warm, micro_request(inp), n_threads=n_threads,
                         derivatives=True, price_derivatives=True, **SOLVE_KW, **bkw)
    for key, v in sol.items():
        if key == "seconds":
            continue
        out[f"solve.{key}"] = strip_timings(v) if key == "stages" else v
    delta = np.asarray(sol["delta"], dtype=float)
    ev = k.evaluate_market(delta, mk, th, ph, sp, micro_request(inp), derivatives=True, price_derivatives=True,
                           n_threads=n_threads, **bkw)
    out.update({f"evaluate.{key}": v for key, v in ev.items()})
    sig = np.asarray(k.frontier_signature(delta, mk, th, ph, sp, n_threads, **bkw))
    out["frontier_signature"] = sig
    rng = np.random.default_rng(7)
    P = th.size + ph.size + 1
    dtheta = 0.05 * rng.normal(size=P)
    ddot = 0.05 * rng.normal(size=delta.size)
    ms = k.max_step(delta, ddot, mk, th, ph, sp, dtheta, 1.5, n_threads, **bkw)
    out.update({f"max_step.{key}": v for key, v in ms.items()})
    tau = 0.5 * min(1.0, float(ms["tau_first"]) * 4 if np.isfinite(ms["tau_first"]) else 1.0)
    d2 = delta + tau * ddot
    sig2 = np.asarray(k.frontier_signature(d2, mk, th + tau * dtheta[:th.size], ph + tau * dtheta[th.size:P - 1],
                                           sp + tau * dtheta[P - 1], n_threads, **bkw))
    changed = sig2 != sig
    out["affected.changed"] = changed
    out["affected_mass"] = np.asarray(k.affected_mass(delta, mk, th, ph, sp, changed, n_threads, **bkw))
    an = k.frontier_anatomy(delta, mk, th, ph, sp, n_threads, **bkw)
    out.update({f"anatomy.{key}": v for key, v in an.items()})
    cs = k.consumer_surplus(delta, mk, th, ph, sp, n_threads, **bkw)
    out.update({f"cs.{key}": v for key, v in cs.items()})
    return out


def build(directory: pathlib.Path, threads: int, skip_gmy: bool) -> list[pathlib.Path]:
    k = _kernel()
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    cases: list[tuple[str, dict[str, Any], bool]] = []
    for bound in ("a", "alpha"):
        inp = synthetic_inputs(bound)
        mk = market_of(inp)
        s = np.asarray(k.evaluate_market(inp["delta_true"], mk, inp["theta"], inp["phi"], inp["sigma_p"],
                                         derivatives=False, **{inp["bound_key"]: inp["bound_value"]})["s"])
        if not ((s > 0).all() and s.sum() < 1):
            raise RuntimeError(f"synthetic fixture ({bound}): shares {s} are not all positive")
        inp["s_data"] = s
        cases.append((f"synthetic_{bound}", inp, False))
    if not skip_gmy:
        inp = gmy_1980_inputs(threads)
        cold = k.solve_market(inp["s_data"], market_of(inp), inp["theta"], inp["phi"], inp["sigma_p"],
                              n_threads=threads, derivatives=False, **SOLVE_KW, **{inp["bound_key"]: inp["bound_value"]})
        if cold["status"] != "ok":
            raise RuntimeError(f"GMY 1980 cold solve ended {cold['status']}: {cold['stages']}")
        inp["delta_cold"] = np.asarray(cold["delta"], dtype=float)
        cases.append(("gmy_1980", inp, True))
    for name, inp, warm in cases:
        out = outputs(inp, threads, inp["delta_cold"] if warm else None)
        arrays = {f"in.{key}": np.asarray(v) for key, v in inp.items()}
        arrays.update({f"out.{key}": np.asarray(v) for key, v in out.items()})
        arrays["meta.warm"] = np.asarray(warm)
        path = directory / f"{name}.npz"
        np.savez(path, **arrays)
        written.append(path)
    return written


def load(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """(inputs, outputs, warm) of one fixture file."""
    with np.load(path, allow_pickle=False) as z:
        inp = {key[3:]: z[key] for key in z.files if key.startswith("in.")}
        out = {key[4:]: z[key] for key in z.files if key.startswith("out.")}
        warm = bool(z["meta.warm"])
    for key in ("sigma_p", "bound_value"):
        inp[key] = float(inp[key])
    inp["bound_key"] = str(inp["bound_key"])
    return inp, out, warm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the kernel baseline fixtures (GMY spec WP-G0).")
    ap.add_argument("directory", type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-gmy", action="store_true", help="synthetic markets only (no GMY data needed)")
    a = ap.parse_args(argv)
    for p in build(a.directory, a.threads, a.skip_gmy):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

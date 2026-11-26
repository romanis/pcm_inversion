"""Flat directions, units versus identification, weak identification (DESIGN 5.2, 5.3). Owned by WP6.

Also the order condition of LEARNINGS 16: the number of moments must be at least the number of
unknowns, concentrated linear coefficients included.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from ..types import DiagnosticBlock

__all__ = [
    "IDENTIFICATION_DEFAULTS",
    "WEAK_ID_DEFAULTS",
    "identification_block",
    "weak_identification_block",
    "order_condition_block",
    "sqrt_psd",
    "block_shares",
    "scale_direction",
    "proportional_direction",
    "identification_summary",
]

IDENTIFICATION_DEFAULTS: dict[str, float] = {
    "small_ratio": 1e-4,  # s_l / s_1 below this in either matrix: a direction is reported
    "zero_ratio": 1e-6,  # below this: UNITS (natural) or IDENTIFICATION (Jacobi)
    "flat_dq": 1.0,  # dq_per_unit below this: FLAT_IN_NATURAL_UNITS
    "loading_min": 0.2,  # loadings printed above this absolute value
    "named_cosine": 0.8,  # |cosine| to a declared direction above this: reported under its name
    "dedupe_cosine": 0.9,  # directions from the two SVDs closer than this are one direction
}

WEAK_ID_DEFAULTS: dict[str, float] = {"level": 0.95}


def _merge(defaults: Mapping[str, Any], thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(defaults)
    if thresholds:
        out.update({k: v for k, v in thresholds.items() if k in defaults})
    return out


def sqrt_psd(W: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semi-definite matrix (negative eigenvalues clipped to 0)."""
    W = 0.5 * (np.asarray(W, dtype=float) + np.asarray(W, dtype=float).T)
    vals, vecs = np.linalg.eigh(W)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def _loadings(v: np.ndarray, names: Sequence[str], minimum: float) -> dict[str, float]:
    order = np.argsort(-np.abs(v))
    return {names[i]: float(v[i]) for i in order if abs(v[i]) > minimum}


def _format_loadings(loadings: Mapping[str, float]) -> str:
    return ", ".join(f"{v:+.2f} {k}" for k, v in loadings.items()) or "no loading above threshold"


def _sign_normalise(v: np.ndarray) -> np.ndarray:
    k = int(np.argmax(np.abs(v)))
    return -v if v[k] < 0 else v


def identification_block(G: np.ndarray, W: np.ndarray, units: np.ndarray, names: Sequence[str], n: int,
                         named_directions: Mapping[str, np.ndarray] | None = None,
                         thresholds: Mapping[str, Any] | None = None) -> DiagnosticBlock:
    """identification_block(G, W, units, names, n, named_directions=None, thresholds=None) -> DiagnosticBlock.

    SVD of ``A_nat = W^{1/2} G diag(unit)`` and of the column-normalised ``A_jac``. For each direction
    with ``s_l/s_1 < 1e-4`` in either: ``dq_per_unit = n s_l^2``, loadings above 0.2 in named
    parameters, verdict UNITS / IDENTIFICATION / WEAK, and the declared named direction with
    |cosine| above 0.8.

    ``G`` holds the free parameters only, in the order of ``names``. Named directions are vectors in
    natural units over the same parameters. A direction found in both SVDs is reported once.
    Verdicts: IDENTIFICATION if the direction has ``s/s_1 < 1e-6`` in ``A_jac``; WEAK if it is
    between 1e-6 and 1e-4 there; UNITS if it is small only in ``A_nat`` (rescaling removes it).
    """
    th = _merge(IDENTIFICATION_DEFAULTS, thresholds)
    G = np.asarray(G, dtype=float)
    units = np.asarray(units, dtype=float)
    names = list(names)
    P = G.shape[1]
    block = DiagnosticBlock(name="identification", thresholds=dict(th))
    if P == 0:
        block.status = "not_run"
        block.messages.append("Identification: no free parameters.")
        return block
    if not np.all(np.isfinite(G)):
        block.status = "fail"
        block.messages.append("FAIL: Identification not computed: G has non-finite entries.")
        return block

    A_nat = sqrt_psd(W) @ G * units[None, :]
    col = np.linalg.norm(A_nat, axis=0)
    safe = np.where(col > 0, col, 1.0)
    A_jac = A_nat / safe[None, :]

    s_nat = np.linalg.svd(A_nat, compute_uv=False)
    _, s_jac, Vt_jac = np.linalg.svd(A_jac)
    _, _, Vt_nat = np.linalg.svd(A_nat)
    s1_nat = float(s_nat[0]) if s_nat.size else 0.0
    s1_jac = float(s_jac[0]) if s_jac.size else 0.0
    # with k < P the SVD returns only k values; the missing ones are exactly zero
    s_nat_full = np.concatenate([s_nat, np.zeros(P - s_nat.size)])
    s_jac_full = np.concatenate([s_jac, np.zeros(P - s_jac.size)])

    def ratio(x: float, s1: float) -> float:
        return float(x / s1) if s1 > 0 else 0.0

    rc_nat = ratio(s_nat_full[-1], s1_nat)
    rc_jac = ratio(s_jac_full[-1], s1_jac)
    block.values.update(rc_natural=rc_nat, rc_jacobi=rc_jac,
                        singular_values_natural=s_nat_full.tolist(),
                        singular_values_jacobi=s_jac_full.tolist(),
                        column_norms=dict(zip(names, col.tolist())))

    # candidate directions, all stored as unit vectors in Jacobi coordinates u (A_jac u = A_nat v)
    cands: list[np.ndarray] = []
    for l in range(P):
        if ratio(s_jac_full[l], s1_jac) < th["small_ratio"]:
            cands.append(Vt_jac[l])
    for l in range(P):
        if ratio(s_nat_full[l], s1_nat) < th["small_ratio"]:
            u = Vt_nat[l] * safe
            cands.append(u / np.linalg.norm(u))
    # one direction per dimension of the small subspace: a candidate already (mostly) in the span
    # of the directions kept so far is dropped, otherwise its orthogonal remainder is kept
    unique: list[np.ndarray] = []
    min_rest = float(np.sqrt(max(0.0, 1.0 - th["dedupe_cosine"] ** 2)))
    for u in cands:
        rest = u.copy()
        for w in unique:
            rest -= (rest @ w) * w
        nr = float(np.linalg.norm(rest))
        if nr > min_rest:
            unique.append(rest / nr)

    named = {k: np.asarray(v, dtype=float) for k, v in (named_directions or {}).items()}
    directions: list[dict[str, Any]] = []
    worst = "ok"
    for u in unique:
        v = u / safe
        v = _sign_normalise(v / np.linalg.norm(v))
        r_jac = ratio(float(np.linalg.norm(A_jac @ (v * safe / np.linalg.norm(v * safe)))), s1_jac)
        a_nat = float(np.linalg.norm(A_nat @ v))
        r_nat = ratio(a_nat, s1_nat)
        dq = float(n * a_nat ** 2)
        if r_jac < th["zero_ratio"]:
            verdict = "IDENTIFICATION"
        elif r_jac < th["small_ratio"]:
            verdict = "WEAK"
        else:
            verdict = "UNITS"
        loads = _loadings(v, names, th["loading_min"])
        match, best = None, 0.0
        for nm, d in named.items():
            nd = np.linalg.norm(d)
            if nd > 0:
                c = abs(float(v @ d) / nd)
                if c > th["named_cosine"] and c > best:
                    match, best = nm, c
        entry: dict[str, Any] = {
            "index": len(directions) + 1, "verdict": verdict, "ratio_jacobi": r_jac, "ratio_natural": r_nat,
            "dq_per_unit": dq, "flat_in_natural_units": dq < th["flat_dq"], "loadings": loads,
            "vector": dict(zip(names, v.tolist())), "named": match,
            "named_cosine": best if match else None,
        }
        if verdict == "UNITS":
            k = int(np.argmax(np.abs(v)))
            entry["suggested_unit"] = {names[k]: float(units[k] * col.max() / safe[k])}
        msg = (f"Flat direction {entry['index']}: {verdict} (s/s1 = {r_jac:.1e} Jacobi, {r_nat:.1e} natural; "
               f"1 unit moves q by {dq:.2f}"
               f"{', FLAT_IN_NATURAL_UNITS' if entry['flat_in_natural_units'] else ''}); "
               f"loads {_format_loadings(loads)}"
               f"{f'; matches {match}' if match else ''}.")
        if verdict == "UNITS":
            (pn, su), = entry["suggested_unit"].items()
            msg += f" Rescaling removes it: suggested unit for {pn} is {su:.3g}."
        level = "fail" if verdict == "IDENTIFICATION" else "flag"
        entry["message"] = f"{level.upper()}: {msg}"
        block.messages.append(entry["message"])
        directions.append(entry)
        if level == "fail" or worst == "ok":
            worst = level
    block.values["directions"] = directions
    block.status = worst  # type: ignore[assignment]
    if not directions:
        block.messages.insert(0, f"Identification: no flat direction (relative condition {rc_jac:.1e} Jacobi, "
                                 f"{rc_nat:.1e} natural units).")
    else:
        n_id = sum(d["verdict"] == "IDENTIFICATION" for d in directions)
        block.messages.insert(0, f"Identification: {len(directions)} flat direction(s), {n_id} unidentified "
                                 f"(relative condition {rc_jac:.1e} Jacobi, {rc_nat:.1e} natural units).")
    return block


def order_condition_block(k: int, p_free: int, k_linear: int) -> DiagnosticBlock:
    """order_condition_block(k, p_free, k_linear) -> DiagnosticBlock (LEARNINGS 16).

    Fails with UNDER_IDENTIFIED when there are fewer moments than unknowns
    (free nonlinear parameters plus concentrated linear coefficients).
    """
    unknowns = int(p_free) + int(k_linear)
    overid = int(k) - unknowns
    block = DiagnosticBlock(name="order_condition",
                            values={"moments": int(k), "unknowns": unknowns, "p_free": int(p_free),
                                    "k_linear": int(k_linear), "overidentifying": overid})
    if overid < 0:
        block.status = "fail"
        block.values["verdict"] = "UNDER_IDENTIFIED"
        block.messages.append(
            f"FAIL: UNDER_IDENTIFIED: {k} moments for {unknowns} unknowns ({p_free} nonlinear + {k_linear} "
            f"linear); at least {-overid} directions are unidentified by the moment conditions themselves "
            f"(LEARNINGS 16).")
    else:
        block.status = "ok"
        block.values["verdict"] = "EXACTLY_IDENTIFIED" if overid == 0 else "OVERIDENTIFIED"
        block.messages.append(f"Order condition: {k} moments for {unknowns} unknowns ({p_free} nonlinear + "
                              f"{k_linear} linear), {overid} overidentifying restriction{'s' if overid != 1 else ''}.")
    return block


def weak_identification_block(G: np.ndarray, S: np.ndarray, n: int,
                              thresholds: Mapping[str, Any] | None = None,
                              units: np.ndarray | None = None) -> DiagnosticBlock:
    """weak_identification_block(G, S, n, thresholds=None, units=None) -> DiagnosticBlock.

    ``CD = n lambda_min(Gs' S^{-1} Gs)`` with ``Gs = G diag(1 / sd_k)`` against the chi-square(k - P + 1)
    95% critical value; below it WEAK_IDENTIFICATION. Ignores the sampling variance of G (stated).

    The column scale ``sd_k`` is one over the natural unit when ``units`` is given (CD is then the
    precision, in chi-square units, of the worst one-natural-unit move), else the root mean square
    of column k of G.
    """
    th = _merge(WEAK_ID_DEFAULTS, thresholds)
    G = np.asarray(G, dtype=float)
    k, P = G.shape
    dof = k - P + 1
    block = DiagnosticBlock(name="weak_identification", thresholds={"level": th["level"]})
    note = "ignores the sampling variance of G, so it is a screen, not a test with exact size"
    if dof < 1:
        block.status = "fail"
        block.values.update(statistic=None, dof=dof, critical_95=None, verdict="UNDER_IDENTIFIED")
        block.messages.append(f"FAIL: Weak identification statistic undefined: {k} moments for {P} parameters.")
        return block
    if units is not None:
        Gs = G * np.asarray(units, dtype=float)[None, :]
        scale = "natural units"
    else:
        rms = np.sqrt(np.mean(G ** 2, axis=0))
        Gs = G / np.where(rms > 0, rms, 1.0)[None, :]
        scale = "column root mean square"
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    Sinv_G = np.linalg.lstsq(S, Gs, rcond=None)[0]
    M = Gs.T @ Sinv_G
    lam = float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])
    cd = float(n * max(lam, 0.0))
    crit = float(stats.chi2.ppf(th["level"], dof))
    weak = cd < crit
    block.values.update(statistic=cd, dof=dof, critical_95=crit, column_scale=scale, note=note,
                        verdict="WEAK_IDENTIFICATION" if weak else "ok")
    block.status = "flag" if weak else "ok"
    text = (f"Cragg-Donald-style statistic {cd:.3g} against the chi-square({dof}) 95% value {crit:.3g} "
            f"(columns scaled by {scale}; {note}).")
    block.messages.append(f"FLAG: WEAK_IDENTIFICATION: {text}" if weak else f"Weak identification: ok. {text}")
    return block


def block_shares(G: np.ndarray, W: np.ndarray, blocks: Mapping[str, slice],
                 names: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    """block_shares(G, W, blocks, names=None) -> {parameter: {block: share}}: for each column k of the (k, P) Jacobian,
    the fraction of ``||W^{1/2} G[:, k]||^2`` that comes from each moment block's rows (GMY spec WP-G2 4(d): which block
    pins which parameter). Rows of ``W^{1/2} G`` are split by ``blocks``; with block-diagonal W this is
    ``G_b[:, k]' W_bb G_b[:, k]`` per block. A column with zero norm gets NaN shares."""
    G = np.asarray(G, dtype=float)
    A = sqrt_psd(W) @ G
    tot = np.sum(A ** 2, axis=0)
    names = list(names) if names is not None else [f"theta{k}" for k in range(G.shape[1])]
    out: dict[str, dict[str, float]] = {}
    for k, nm in enumerate(names):
        out[nm] = {b: (float(np.sum(A[s, k] ** 2) / tot[k]) if tot[k] > 0 else float("nan")) for b, s in blocks.items()}
    return out


def scale_direction(names: Sequence[str], theta: np.ndarray, units: np.ndarray) -> np.ndarray:
    """The utility-scale direction in natural units at theta (DESIGN 5.2, LEARNINGS 1): sigma_p up and every taste
    spread (a parameter named ``sigma_<column>``) and every factor loading (``lambda_<group>_f<k>``, signed) down in
    proportion."""
    u = np.asarray(theta, dtype=float) / np.asarray(units, dtype=float)
    out = np.zeros(len(names))
    for k, n in enumerate(names):
        if n == "sigma_p":
            out[k] = u[k]
        elif n.startswith(("sigma_", "lambda_")):
            out[k] = -u[k]
    return out


def proportional_direction(names: Sequence[str], theta: np.ndarray, units: np.ndarray,
                           members: Sequence[str]) -> np.ndarray:
    """The direction in natural units at theta along which the parameters ``members`` move up in proportion."""
    u = np.asarray(theta, dtype=float) / np.asarray(units, dtype=float)
    members = set(members)
    return np.array([u[k] if n in members else 0.0 for k, n in enumerate(names)])


def identification_summary(problem: Any, result: Any) -> dict[str, Any]:
    """Weak identification at an estimate: the Cragg-Donald-style statistic (``weak_identification_block``) on all
    rows of G and on each moment block, the aggregate block's q (Sargan J at step 2), and for each free parameter the
    shares of ||W^1/2 G_k||^2 by block (``block_shares``: which block pins which parameter)."""
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

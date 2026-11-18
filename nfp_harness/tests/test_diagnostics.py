"""Tests for diagnostics (WP6; DESIGN 5.2-5.5, 5.7-5.10 and the WP6 acceptance list in DESIGN 7).

Every test plants a known defect in a synthetic Evaluation or history and checks that the report
detects it and names it.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from nfp_harness.diagnostics import DiagnosticInputs, build_report, render_text, run_diagnostics, validate
from nfp_harness.diagnostics.collinearity import collinearity_block
from nfp_harness.diagnostics.degeneracies import (
    DEGENERACIES,
    detect_all,
    normalisation_cap_active,
    run_to_bound,
    run_to_infinity,
    scale_artifact,
    sd_xi_blowup,
    segmented_corner,
    vertical_collapse,
)
from nfp_harness.diagnostics.identification import (
    identification_block,
    order_condition_block,
    weak_identification_block,
)
from nfp_harness.diagnostics.inference import sandwich_covariance, se_gate
from nfp_harness.diagnostics.overid import hansen_j
from nfp_harness.diagnostics.report import to_jsonable, write_report
from nfp_harness.diagnostics.stability import integration_stability, multistart_summary
from nfp_harness.types import (
    ConvergenceClass,
    EvalStatus,
    Evaluation,
    HistoryRecord,
    OptimizeResult,
    ParameterSpec,
)

RNG = np.random.default_rng(20261116)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def rec(i: int, q: float, sd_xi: float = 1.0, q_fixed: float | None = None, q_cue: float | None = None,
        theta=(0.0,), accepted: bool = True) -> HistoryRecord:
    return HistoryRecord(iteration=i, accepted=accepted, q=q, q_fixed=q if q_fixed is None else q_fixed,
                         q_cue=q if q_cue is None else q_cue, sd_xi=sd_xi, rho=0.5, Delta=0.3, tau_cap=1.0,
                         status=EvalStatus.OK, seconds=0.1, theta=np.asarray(theta, dtype=float))


def result(theta, q, cls=ConvergenceClass.CONVERGED, history=(), ev=None, weighting="two_step", W=None,
           stage="step2") -> OptimizeResult:
    return OptimizeResult(theta=np.asarray(theta, dtype=float), q=q, convergence=cls, evaluation=ev,
                          history=list(history), accepted_steps=len(history), evaluations=len(history) + 1,
                          wall_seconds=60.0, projected_gradient=1e-5, pg_tol=1e-3, gn_decrease=1e-9,
                          gn_tol=1e-6, Delta=0.3, q_mass=0.2, stage=stage, weighting=weighting, W=W)


def spec(name, role="taste", column=None, start=1.0, lower=0.0, upper=np.inf, cap=False, unit=1.0, free=True):
    return ParameterSpec(name=name, role=role, column=column, basis="nu", start=start, lower=lower, upper=upper,
                         cap=cap, unit=unit, free=free)


def concentrated_G(X1: np.ndarray, Z: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G = Z' M D / n with beta concentrated out by IV on Z with W = (Z'Z/n)^-1 (DESIGN 2.4)."""
    n = X1.shape[0]
    W = np.linalg.inv(Z.T @ Z / n)
    A = X1.T @ Z @ W @ Z.T
    M = np.eye(n) - X1 @ np.linalg.solve(A @ X1, A)
    return Z.T @ M @ D / n, W


# ------------------------------------------------------------------------------------------------
# 5.2 identification: acceptance cases
# ------------------------------------------------------------------------------------------------


def test_column_scaled_by_1e_minus_7_is_units() -> None:
    G = RNG.normal(size=(10, 3))
    G[:, 2] *= 1e-7
    b = identification_block(G, np.eye(10), np.ones(3), ["sigma_a", "sigma_b", "sigma_p"], n=500)
    dirs = b.values["directions"]
    assert len(dirs) == 1
    d = dirs[0]
    assert d["verdict"] == "UNITS"
    assert list(d["loadings"]) == ["sigma_p"] and d["loadings"]["sigma_p"] == pytest.approx(1.0, abs=1e-6)
    assert d["ratio_natural"] < 1e-6 and d["ratio_jacobi"] >= 1e-4
    assert "suggested_unit" in d and d["suggested_unit"]["sigma_p"] > 1e5
    assert b.status == "flag" and "UNITS" in b.messages[1]


def test_two_identical_columns_is_identification_with_loadings() -> None:
    G = RNG.normal(size=(10, 4))
    G[:, 3] = G[:, 1]
    names = ["sigma_const", "sigma_hpwt", "sigma_space", "sigma_p"]
    b = identification_block(G, np.eye(10), np.ones(4), names, n=500,
                             named_directions={"scale_direction": np.array([0, 1.0, 0, -1.0])})
    dirs = b.values["directions"]
    assert len(dirs) == 1
    d = dirs[0]
    assert d["verdict"] == "IDENTIFICATION"
    assert set(d["loadings"]) == {"sigma_hpwt", "sigma_p"}
    assert d["loadings"]["sigma_hpwt"] == pytest.approx(-d["loadings"]["sigma_p"], abs=1e-8)
    assert abs(d["loadings"]["sigma_hpwt"]) == pytest.approx(2 ** -0.5, abs=1e-8)
    assert d["named"] == "scale_direction" and d["flat_in_natural_units"]
    assert b.status == "fail" and b.messages[1].startswith("FAIL: Flat direction 1: IDENTIFICATION")


def test_weak_direction_and_clean_jacobian() -> None:
    G = RNG.normal(size=(12, 3))
    b = identification_block(G, np.eye(12), np.ones(3), ["a", "b", "c"], n=100)
    assert b.status == "ok" and not b.values["directions"]
    G2 = G.copy()
    G2[:, 2] = G2[:, 1] + 1e-5 * RNG.normal(size=12)
    b2 = identification_block(G2, np.eye(12), np.ones(3), ["a", "b", "c"], n=100)
    assert [d["verdict"] for d in b2.values["directions"]] == ["WEAK"]


# ------------------------------------------------------------------------------------------------
# 5.3 collinearity, weak instruments, order condition (LEARNINGS 16)
# ------------------------------------------------------------------------------------------------


def petrin_like(n: int = 400, K: int = 5, L: int = 6, P: int = 4):
    X1 = np.column_stack([np.ones(n), RNG.normal(size=(n, K - 1))])
    Zx = RNG.normal(size=(n, L)) + 0.3 * X1[:, 1:2]
    D = RNG.normal(size=(n, P)) + Zx @ RNG.normal(size=(L, P))
    return X1, Zx, D


def test_z_without_x1_fails_zx1_rank_check() -> None:
    X1, Zx, _ = petrin_like(K=5, L=3)
    names = ["const", "hpwt", "space", "air", "mpd"]
    b = collinearity_block(X1, Zx, None, [], x1_names=names)
    assert b.status == "fail"
    assert b.values["ZX1"]["rank"] == 3 and b.values["ZX1"]["required"] == 5
    assert any(m.startswith("FAIL: rank(Z'X1) = 3 < 5") for m in b.messages)
    assert set(b.values["ZX1"]["x1_outside_span_Z"]) == set(names)
    good = collinearity_block(X1, np.column_stack([X1, Zx]), None, [], x1_names=names)
    assert good.status == "ok" and good.values["ZX1"]["x1_in_span_Z"]
    assert good.values["ZX1"]["min_canonical_correlation"] == pytest.approx(1.0, abs=1e-10)


def test_collinear_x1_and_g_pairs() -> None:
    X1, Zx, _ = petrin_like()
    X1b = np.column_stack([X1, 2 * X1[:, 1]])
    A = RNG.normal(size=(8, 3))
    A[:, 2] = A[:, 0] * 1.0001 + 1e-6
    b = collinearity_block(X1b, np.column_stack([X1b, Zx]), A, ["s1", "s2", "s3"])
    assert b.status == "fail"
    assert any(m.startswith("FAIL: X1 has rank 5 with 6 columns") for m in b.messages)
    assert b.values["G_pairs"][0]["a"] == "s1" and b.values["G_pairs"][0]["b"] == "s3"


def test_order_condition_and_weak_identification() -> None:
    oc = order_condition_block(22, 18, 13)
    assert oc.status == "fail" and oc.values["verdict"] == "UNDER_IDENTIFIED"
    assert "22 moments for 31 unknowns (18 nonlinear + 13 linear)" in oc.messages[0]
    assert order_condition_block(45, 18, 13).values["overidentifying"] == 14
    G = RNG.normal(size=(10, 3))
    S = np.eye(10)
    strong = weak_identification_block(G, S, n=2000)
    assert strong.status == "ok" and strong.values["dof"] == 8
    weak = weak_identification_block(1e-4 * G, S, n=2000, units=np.ones(3))
    assert weak.status == "flag" and weak.values["verdict"] == "WEAK_IDENTIFICATION"
    assert "sampling variance of G" in weak.messages[0]


def test_learnings_16_planted_report_names_under_identification() -> None:
    """Z = excluded instruments only (no X1): 6 moments for 5 linear + 4 nonlinear unknowns."""
    X1, Zx, D = petrin_like(n=400, K=5, L=6, P=4)
    n = X1.shape[0]
    G, W = concentrated_G(X1, Zx, D)
    g = G @ np.full(4, 0.01)
    specs = [spec("sigma_const"), spec("sigma_hpwt"), spec("sigma_mi"), spec("sigma_p", role="sigma_p", unit=0.1)]
    theta = np.array([2.0, 1.0, 2.0, 1.0])
    ev = Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK, blocks={"aggregate": slice(0, 6)}, aux={},
                    seconds=1.0)
    res = result(theta, float(n * g @ W @ g), ev=ev, weighting="fixed", W=W, stage="step1")
    linear_names = ["const", "hpwt", "space", "air", "mpd"]
    rep = build_report(DiagnosticInputs(result=res, specs=specs, units=np.array([1, 1, 1, 0.1]), n=n,
                                        run_id="planted-l16", spec_name="planted", spec_hash="x", stage="step1",
                                        S=np.eye(6) * 0.1, X1=X1, Z=Zx, linear_names=linear_names,
                                        beta=np.zeros(5)))
    d = to_jsonable(rep.to_dict())
    validate(d)
    oc = d["identification"]["order_condition"]
    assert oc["verdict"] == "UNDER_IDENTIFIED" and oc["moments"] == 6 and oc["unknowns"] == 9
    assert d["identification"]["status"] == "fail"
    # concentrating 5 linear coefficients with Z'X1 of full rank leaves rank(G) = 6 - 5 = 1,
    # so 3 of the 4 nonlinear directions are unidentified
    assert sum(x["verdict"] == "IDENTIFICATION" for x in d["identification"]["directions"]) == 3
    assert set(d["collinearity"]["ZX1"]["x1_outside_span_Z"]) == set(linear_names)
    text = render_text(rep)
    assert "FAIL: UNDER_IDENTIFIED: 6 moments for 9 unknowns (4 nonlinear + 5 linear)" in text
    assert "FLAG: X1_NOT_IN_Z" in text
    assert "Standard errors omitted" in text
    # the fix of DESIGN F1: Z = [X1, Z_excl] has 11 moments for 9 unknowns and no unidentified direction
    Zf = np.column_stack([X1, Zx])
    Gf, Wf = concentrated_G(X1, Zf, D)
    ev2 = Evaluation(theta=theta, g=Gf @ np.full(4, 0.01), G=Gf, status=EvalStatus.OK,
                     blocks={"aggregate": slice(0, 11)}, aux={}, seconds=1.0)
    rep2 = build_report(DiagnosticInputs(result=result(theta, 1.0, ev=ev2, W=Wf), specs=specs,
                                         units=np.array([1, 1, 1, 0.1]), n=n, S=np.linalg.inv(Wf), X1=X1, Z=Zf,
                                         linear_names=linear_names, beta=np.zeros(5)))
    d2 = to_jsonable(rep2.to_dict())
    assert d2["identification"]["order_condition"]["verdict"] == "OVERIDENTIFIED"
    assert not any(x["verdict"] == "IDENTIFICATION" for x in d2["identification"]["directions"])
    assert d2["collinearity"]["status"] == "ok"


# ------------------------------------------------------------------------------------------------
# 5.4 degeneracies
# ------------------------------------------------------------------------------------------------


def scale_history() -> list[HistoryRecord]:
    """LEARNINGS 1: sd(xi) 0.48 -> 0.044, q_fixed tracks sd(xi)^2, q_cue stays near 30."""
    sd = np.geomspace(0.48, 0.044, 12)
    return [rec(i, q=18.2 * (s / 0.48) ** 2, sd_xi=s, q_cue=33.0 - 0.3 * i) for i, s in enumerate(sd)]


def blowup_history(theta=(0.0,)) -> list[HistoryRecord]:
    """LEARNINGS 13: q 39.6 -> 0.8 while sd(xi) rises 1.4 -> 7 (5x)."""
    q = np.geomspace(39.6, 0.8, 12)
    sd = np.geomspace(1.4, 7.0, 12)
    return [rec(i, q=qq, sd_xi=s, q_fixed=100 * s ** 2, theta=theta) for i, (qq, s) in enumerate(zip(q, sd))]


def test_scale_artifact_planted() -> None:
    r = scale_artifact(scale_history())
    assert r["detected"] and r["evidence"]["q_fixed_fall_factor"] > 10
    assert "fixed-W objective is rewarding a small xi" in r["message"]
    assert not scale_artifact(blowup_history())["detected"]


def test_sd_xi_blowup_planted() -> None:
    r = sd_xi_blowup(blowup_history(), None)
    assert r["detected"] and r["evidence"]["sd_xi_rise_factor"] == pytest.approx(5.0)
    assert "noisy moments, not by fit" in r["message"]
    assert not sd_xi_blowup(scale_history(), None)["detected"]
    xi = RNG.normal(size=2000)
    assert not sd_xi_blowup([], xi)["detected"]
    xi[:10] = 80.0  # a few huge residuals: lumpy xi
    lumpy = sd_xi_blowup([], xi)
    assert lumpy["detected"] and lumpy["evidence"]["kurtosis"] > 10 and lumpy["evidence"]["top_share"] > 0.25


def test_bounds_caps_and_infinity() -> None:
    specs = [spec("sigma_const"), spec("sigma_sw", column="sw", upper=66.1, cap=True), spec("sigma_p", lower=0.05,
             upper=3.0, unit=0.1, role="sigma_p")]
    theta = np.array([0.0, 66.1, 3.0])
    rb = run_to_bound(theta, np.array([1.0, -1.0, -2.0]), specs)
    assert rb["detected"] and [h["name"] for h in rb["evidence"]["parameters"]] == ["sigma_const", "sigma_p"]
    assert not run_to_bound(theta, np.array([-1.0, -1.0, 2.0]), specs)["detected"]  # gradients point inward
    cap = normalisation_cap_active(theta, specs)
    assert cap["detected"] and cap["evidence"]["parameters"][0]["name"] == "sigma_sw"
    # r = exp(-theta): q falls slowly toward 0 while theta runs away (DESIGN 4.8 test 4)
    hist = [rec(i, q=float(np.exp(-2 * t)), theta=(t,)) for i, t in enumerate(np.linspace(0.0, 9.0, 16))]
    ri = run_to_infinity(hist, [spec("theta", role="scale", start=0.0, lower=-np.inf)])
    assert ri["detected"] and ri["evidence"]["parameters"][0]["to"] == pytest.approx(9.0)


def test_segmented_corner_and_vertical_collapse() -> None:
    specs = [spec("sigma_const", column="const", unit=1.0), spec("sigma_sw", column="sw"),
             spec("pi_sw", column="sw", lower=-np.inf, start=0.3)]
    theta = np.array([3.0, 117.0, 0.3])
    sc = segmented_corner(theta, specs, price_range=66.1, cross_group_diversion=None, binary_columns=("sw",))
    assert sc["detected"] and sc["evidence"]["parameters"][0]["name"] == "sigma_sw"
    assert "separate sw pools (57% have taste beyond the price range" in sc["message"]
    assert not segmented_corner(np.array([3.0, 2.0, 0.3]), specs, 66.1, 0.2, binary_columns=("sw",))["detected"]
    assert segmented_corner(np.array([3.0, 2.0, 0.3]), specs, 66.1, 0.003, binary_columns=("sw",))["detected"]
    vc = vertical_collapse(np.array([0.01, 0.02, 0.3]), specs, {"own_elasticity": {"median": -219.0}})
    assert vc["detected"] and "median own elasticity -219" in vc["message"]
    assert not vertical_collapse(np.array([3.0, 2.0, 0.3]), specs, {"own_elasticity": {"median": -4.0},
                                                                     "nonzero_cross": {"median": 40}})["detected"]


def test_detect_all_returns_every_name() -> None:
    out = detect_all(scale_history(), {"theta": np.array([1.0]), "layout": [spec("theta")]}, None)
    assert [d["name"] for d in out] == list(DEGENERACIES)
    assert [d["name"] for d in out if d["detected"]] == ["SCALE_ARTIFACT"]


# ------------------------------------------------------------------------------------------------
# 5.5 Hansen J, 5.9 standard errors
# ------------------------------------------------------------------------------------------------


def test_hansen_j() -> None:
    g = np.array([0.1, -0.2, 0.05])
    S = np.diag([0.5, 1.0, 2.0])
    b = hansen_j(g, S, n=100, dof=1, weighting="two_step")
    assert b.values["J"] == pytest.approx(100 * (0.01 / 0.5 + 0.04 + 0.0025 / 2))
    assert b.status == "flag"
    fixed = hansen_j(g, S, n=100, dof=1, weighting="fixed")
    assert fixed.status == "not_run" and "not chi-square" in fixed.messages[0]
    nc = hansen_j(g, S, n=100, dof=1, weighting="cue", converged=False)
    assert "Not interpreted: objective did not converge" in nc.messages[0]


def test_se_gate_and_sandwich() -> None:
    ok, why = se_gate(ConvergenceClass.CONVERGED, 1e-3, [])
    assert ok and why is None
    ok, why = se_gate(ConvergenceClass.FLAT_PROGRESS, 1e-9, [{"name": "RUN_TO_BOUND", "detected": True}])
    assert not ok
    assert why.startswith("Standard errors omitted: objective did not converge (class FLAT_PROGRESS)")
    assert "ill conditioned" in why and "RUN_TO_BOUND detected" in why
    G = RNG.normal(size=(6, 2))
    A = RNG.normal(size=(6, 6))
    S = A @ A.T + np.eye(6)
    V = sandwich_covariance(G, np.linalg.inv(S), S, n=50)
    assert np.allclose(V, np.linalg.inv(G.T @ np.linalg.solve(S, G)) / 50)


# ------------------------------------------------------------------------------------------------
# 5.7, 5.8 stability
# ------------------------------------------------------------------------------------------------


def test_integration_stability_and_multistart() -> None:
    base = result([1.0, 2.0], 10.0)
    stable = integration_stability(base, lambda t: result(t + 0.01, 10.2, history=[rec(0, 10.1, theta=t)]),
                                   np.ones(2), names=["a", "b"])
    assert stable.status == "ok" and stable.values["status"] == "STABLE"
    assert stable.values["dq"] == pytest.approx(0.1)
    unstable = integration_stability(base, lambda t: result(t + np.array([0.0, 0.5]), 9.0), np.ones(2),
                                     se=np.array([0.1, 0.1]), names=["a", "b"])
    assert unstable.values["status"] == "UNSTABLE" and unstable.values["unstable"] == ["b"]
    assert unstable.messages[0].startswith("FLAG: Integration UNSTABLE")

    def boom(_t):
        raise RuntimeError("inversion failed in market 1990")

    assert integration_stability(base, boom, np.ones(2)).status == "fail"
    ms = multistart_summary([result([1.0, 2.0], 5.0), result([1.05, 2.0], 5.1),
                             result([3.0, 2.0], 7.0, cls=ConvergenceClass.STALLED)], np.ones(2))
    assert ms["distinct_endpoints"] == 2 and ms["q_spread"] == pytest.approx(2.0) and ms["converged_runs"] == 2


# ------------------------------------------------------------------------------------------------
# 5.10 report: schema validation and text rendering of a degenerate run
# ------------------------------------------------------------------------------------------------


def degenerate_report():
    """A CUE run of the LEARNINGS 13 kind: sd(xi) 5x up, sigma_sw beyond the price range, at bound, flat."""
    specs = [spec("sigma_const", column="const"), spec("sigma_hpwt", column="hpwt"),
             spec("sigma_sw", column="sw", upper=66.1, cap=True),
             spec("sigma_p", role="sigma_p", lower=0.05, upper=3.0, unit=0.1)]
    theta = np.array([0.36, 1.0, 66.1, 0.05])
    k, n = 10, 300
    G = RNG.normal(size=(k, 4))
    G[:, 1] = G[:, 0]  # sigma_const and sigma_hpwt move the moments identically
    g = 0.01 * RNG.normal(size=k)
    W = np.eye(k)
    ev = Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK,
                    blocks={"aggregate": slice(0, 8), "micro": slice(8, 10)}, aux={}, seconds=2.0)
    hist = blowup_history(theta)
    res = result(theta, 0.8, cls=ConvergenceClass.FLAT_PROGRESS, history=hist, ev=ev, weighting="cue", W=W)
    xi = RNG.normal(size=n)
    xi[:5] = 60.0
    X1 = np.column_stack([np.ones(n), RNG.normal(size=(n, 2))])
    Z = np.column_stack([X1, RNG.normal(size=(n, 5))])  # 8 aggregate instruments
    inp = DiagnosticInputs(result=res, specs=specs, units=np.array([1, 1, 1, 0.1]), n=n, run_id="planted-cue",
                           spec_name="toy_pcm", spec_hash="abc123", stage="step2",
                           normalisations=["median alpha = 1 (mu = 0)", "sigma_sw capped at the price range 66.1"],
                           S=np.eye(k), X1=X1, Z=Z, linear_names=["const", "hpwt", "sw"],
                           beta=np.array([-8.9, 1.0, -0.5]), xi=xi, price_range=66.1, binary_columns=["sw"],
                           economics={"own_elasticity": {"median": -219.0}, "nonzero_cross": {"median": 7},
                                      "status": "flag", "messages": ["FLAG: median own elasticity = -219 is outside "
                                                                     "the declared plausible range [-12, -2]."]},
                           files={"history": "history.csv"})
    return build_report(inp)


def test_report_validates_and_text_contains_every_flag(tmp_path) -> None:
    rep = degenerate_report()
    d = to_jsonable(rep.to_dict())
    validate(d)
    json.dumps(d, allow_nan=False)
    detected = {x["name"] for x in d["degeneracies"] if x["detected"]}
    assert {"SD_XI_BLOWUP", "RUN_TO_BOUND", "NORMALISATION_CAP_ACTIVE", "SEGMENTED_CORNER",
            "VERTICAL_COLLAPSE"} <= detected
    text = render_text(rep)
    assert text.startswith("RESULT: FLAT_PROGRESS at q = 0.8 (cue, 10 moments, 7 parameters).")
    for name in detected:
        assert name in text
    flags = []

    def walk(x):
        if isinstance(x, str) and x.startswith(("FLAG:", "FAIL:")):
            flags.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(d)
    assert flags
    for f in flags:
        assert f in text
    assert "Hansen J = " in text and "Not interpreted: objective did not converge" in text
    assert "Economics not interpreted: objective did not converge." in text
    assert "FAIL: Flat direction 1: IDENTIFICATION" in text
    write_report(rep, tmp_path)
    assert json.loads((tmp_path / "report.json").read_text())["run_id"] == "planted-cue"
    assert (tmp_path / "report.txt").read_text() == text


def test_validate_rejects_broken_reports() -> None:
    d = to_jsonable(degenerate_report().to_dict())
    bad = json.loads(json.dumps(d))
    del bad["convergence"]["pg_tol"]
    with pytest.raises(ValueError, match="convergence.'pg_tol'"):
        validate(bad)
    bad = json.loads(json.dumps(d))
    bad["degeneracies"][0]["name"] = "MADE_UP"
    with pytest.raises(ValueError, match=r"degeneracies\[0\].name"):
        validate(bad)
    bad = json.loads(json.dumps(d))
    bad["identification"]["directions"][0]["verdict"] = "MAYBE"
    with pytest.raises(ValueError, match="verdict"):
        validate(bad)
    bad = json.loads(json.dumps(d))
    bad["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        validate(bad)


# ------------------------------------------------------------------------------------------------
# run_diagnostics on a fake problem (the orchestration path the scripts use)
# ------------------------------------------------------------------------------------------------


def test_run_diagnostics_end_to_end() -> None:
    k, n = 9, 200
    specs = [spec("sigma_a"), spec("sigma_b"), spec("phi", role="scale", lower=-np.inf, start=0.0)]
    G = RNG.normal(size=(k, 3))
    g = 0.02 * RNG.normal(size=k)
    S = np.eye(k) * 0.5
    theta = np.array([1.0, 0.5, 0.1])
    ev = Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK, blocks={"aggregate": slice(0, k)},
                    aux={"xi": RNG.normal(size=n), "beta": np.array([1.0, 2.0])}, seconds=1.0)
    W = np.linalg.inv(S)
    res = result(theta, float(n * g @ W @ g), ev=ev, W=W, history=[rec(0, 5.0, theta=theta)])
    problem = types.SimpleNamespace(layout=types.SimpleNamespace(specs=specs), covariance=lambda e: S)
    sp = types.SimpleNamespace(name="toy", hash="h", normalisations=["mu = 0"], linear_names=["c", "x"],
                               diagnostics={"pair_corr": 0.999})
    rep = run_diagnostics(problem, res, sp,
                          {"units": np.ones(3), "reevaluate": lambda t: result(t, res.q, history=[])})
    d = to_jsonable(rep.to_dict())
    validate(d)
    assert d["convergence"]["class"] == "CONVERGED" and d["standard_errors"]["computed"]
    assert all(p["se"] is not None and p["se"] > 0 for p in d["parameters"])
    assert d["hansen_j"]["dof"] == k - 3 - 2
    assert d["integration"]["status"] == "STABLE"
    assert d["identification"]["order_condition"]["verdict"] == "OVERIDENTIFIED"
    assert render_text(rep).startswith("RESULT: CONVERGED")


def test_scale_artifact_fires_on_a_fall_in_mean_alpha_of_buyers() -> None:
    """LEARNINGS 25: buyers moving into the low-alpha tail (mean alpha of buyers down more than 2x) is the
    scale artifact's composition route, even when q_fixed, q_cue and sd(xi) do not show the pure rescaling."""
    import dataclasses

    flat = [rec(i, 10.0 - 0.1 * i) for i in range(4)]
    alphas = (0.24, 0.20, 0.11, 0.053)
    hist = [dataclasses.replace(h, mean_alpha_buyers=a) for h, a in zip(flat, alphas)]
    out = scale_artifact(hist)
    assert out["detected"] and out["evidence"]["mean_alpha_buyers_fall_factor"] == pytest.approx(0.24 / 0.053)
    assert "low-alpha tail" in out["message"]
    mild = [dataclasses.replace(h, mean_alpha_buyers=a) for h, a in zip(flat, (0.24, 0.22, 0.2, 0.15))]
    assert not scale_artifact(mild)["detected"]
    assert not scale_artifact(flat)["detected"]   # no alpha column: the old rule alone


def test_vertical_tail_flag_above_the_plausible_range() -> None:
    from nfp_harness.diagnostics.economics import plausible_flags

    flags, _, checked = plausible_flags({"own_elasticity_times_sigma_p": 58.8},
                                        {"own_elasticity_times_sigma_p": (17.0, 55.0)})
    assert flags and "own_elasticity_times_sigma_p" in flags[0]
    assert checked["own_elasticity_times_sigma_p"]["hi"] == 55.0


def test_at_bound_exempts_zero_taste_spread() -> None:
    from nfp_harness.diagnostics import at_bound_parameters
    from nfp_harness.types import ParameterSpec

    specs = [
        ParameterSpec("sigma_fwd", "taste", "fwd", "nu5", 1.0, 0.0, 25.0, cap=True),
        ParameterSpec("phi_high", "scale", None, "high", 1.0, -4.0, 4.0),
        ParameterSpec("pi_mi", "taste", "mi", "logfs", 1.0, 0.0, 10.0),
    ]
    theta = np.array([0.0, -4.0, 0.0])
    assert at_bound_parameters(theta, specs, np.ones(3)) == ["phi_high", "pi_mi"]

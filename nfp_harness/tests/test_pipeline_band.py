"""Tests for the pipeline's sigma_p_band stage (set identification of one parameter, the gate's fallback).

A toy spec on the synthetic PCM of ``test_pcm_model`` (2 markets, 8 candidates, 256 draws): the band fixes
sigma_p at each value, runs fixed-W step 1 and step 2 from a stored prefit point, writes band.csv, and a
second run on the same directory reads every point back instead of estimating it again.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import numpy as np
import pytest

from nfp_harness.linear import LinearIV
from nfp_harness.moments import IVMoments, MicroDataset, RatioMicroMoment
from nfp_harness.pipeline import GateFailed, Pipeline, RunConfig
from nfp_harness.problem import NFPGMMProblem

pytest.importorskip("nfp_harness._native")
from nfp_harness.models.pcm import PCMModel, PCMSpec  # noqa: E402
from nfp_harness.tests.test_pcm_model import (  # noqa: E402
    CHARS, THETA_TRUE, make_synthetic, micro_parts, synthetic_layout, true_micro_values,
)


class ToySpec:
    name = "toy_band"
    normalisations: list[str] = []
    band_parameter = "sigma_p"
    band_values = (0.8, 1.0)
    band_moments = "all"
    max_iterations = 8
    optimizer_options: dict[str, Any] = {}

    def __init__(self, syn: dict[str, Any]) -> None:
        self.syn = syn
        self.calls = 0

    def hash_payload(self) -> dict[str, Any]:
        return {"toy": 1}

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str,
                      layout_changes: dict[str, Any] | None = None) -> NFPGMMProblem:
        syn = self.syn
        layout = synthetic_layout(THETA_TRUE * np.array([1.05, 0.95, 1.1, 0.9, 1.0]))
        model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
        nums, dens, labels = micro_parts()
        micro = RatioMicroMoment("micro", MicroDataset("synthetic", 10_000), nums, dens, true_micro_values(syn), labels,
                                 covariance=np.eye(3) * 1e-4)
        return NFPGMMProblem(model, [IVMoments(syn["Z"]), micro], LinearIV(syn["X1"], syn["Z"]), layout)

    def band_row(self, pipe: Any, problem: Any, result: Any, value: float) -> dict[str, Any]:
        self.calls += 1
        return {"sigma_p_echo": float(result.theta[problem.layout.names.index("sigma_p")])}

    def band_reference_rows(self) -> list[dict[str, Any]]:
        return [{"label": "reference", "sigma_p_echo": -1.0}]

    def band_finish(self, pipe: Any, problem: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Headline = the last point; rebuild its estimate through the pipeline and echo its q."""
        self.finish_calls = getattr(self, "finish_calls", 0) + 1
        layout = problem.layout
        try:
            res, stage = pipe.band_result(problem, rows[-1]["sigma_p"])
        finally:
            problem.layout = layout
        rows[-1]["label"] = "headline"
        rows[-1]["rebuilt_q"] = float(res.q)
        rows[-1]["rebuilt_stage"] = stage
        return rows


def test_sigma_p_band_rows_and_resume(tmp_path: pathlib.Path) -> None:
    syn = make_synthetic(2, 8, 32, 8, seed=2)
    spec = ToySpec(syn)
    cfg = RunConfig(base=tmp_path, nu=syn["nu"], seed=syn["seed"], threads=8)
    pipe = Pipeline(spec, cfg)
    pipe.rd.save_stage("prefit", {"theta": synthetic_layout().starts() * np.array([1.05, 0.95, 1.1, 0.9, 1.3])})
    try:
        rows = pipe.sigma_p_band()
    finally:
        pipe.close()
    assert [r["sigma_p"] for r in rows] == [0.8, 1.0]
    assert [r["sigma_p_echo"] for r in rows] == [0.8, 1.0]   # the band parameter stays fixed
    assert all(r["estimate_stage"] in ("step1", "step2") and np.isfinite(r["q"]) for r in rows)
    assert spec.calls == 2
    with open(pipe.path / "sigma_p_band" / "band.csv") as fh:
        table = list(csv.DictReader(fh))
    assert [t["label"] for t in table] == ["", "headline", "reference"]
    assert rows[-1]["rebuilt_q"] == rows[-1]["q"] and rows[-1]["rebuilt_stage"] == rows[-1]["estimate_stage"]
    assert spec.finish_calls == 1
    assert pipe.rd.stage_done("band.0.8.row") and pipe.rd.stage_done("band.1.step1")

    spec2 = ToySpec(syn)
    pipe2 = Pipeline(spec2, cfg)
    try:
        rows2 = pipe2.sigma_p_band()
    finally:
        pipe2.close()
    assert spec2.calls == 0 and rows2 == rows and getattr(spec2, "finish_calls", 0) == 0


def test_gate_failure_is_a_typed_refusal(tmp_path: pathlib.Path) -> None:
    import json

    syn = make_synthetic(2, 8, 32, 8, seed=2)
    spec = ToySpec(syn)
    spec.gate_required = ("a", "b")  # type: ignore[attr-defined]
    spec.fallback_stage = "sigma_p_band"  # type: ignore[attr-defined]
    cfg = RunConfig(base=tmp_path, nu=syn["nu"], seed=syn["seed"], threads=8, gate_nu=syn["nu"])
    pipe = Pipeline(spec, cfg)
    try:
        (pipe.path / "gate").mkdir(exist_ok=True)
        (pipe.path / "gate" / "verdict.json").write_text(json.dumps(
            {"spec_hash": pipe.spec_hash, "verdict": {"a": False, "b": False}}))
        with pytest.raises(GateFailed, match="sigma_p_band"):
            pipe._check_gate()
        (pipe.path / "gate" / "verdict.json").write_text(json.dumps(
            {"spec_hash": pipe.spec_hash, "verdict": {"a": False, "b": True}}))
        pipe._check_gate()   # one passing verdict of the tuple is enough
    finally:
        pipe.close()


class ToyMarginSpec:
    """Synthetic PCM with aggregate (Sargan), supply and an optional margin row: the margin_band stage."""

    name = "toy_margin_band"
    normalisations: list[str] = ["toy"]
    margin_band_values = (0.25, 0.3)
    max_iterations = 4
    optimizer_options: dict[str, Any] = {"noise_probe": 1e-5, "lm_damping": True}
    start_from_gate = True
    start_concentrate = ("pi_x1_demo",)
    start_slice = {"lambda": 1.0}
    start_column = "q_fixed"
    gate_concentrate_iterations = 3

    def __init__(self, syn: dict[str, Any], gate: pathlib.Path) -> None:
        self.syn, self.start_gate, self.calls = syn, str(gate), 0

    def hash_payload(self) -> dict[str, Any]:
        return {"toy": 2}

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str,
                      layout_changes: dict[str, Any] | None = None, margin: float | None = None) -> NFPGMMProblem:
        from nfp_harness.moments import MarginMoment, SarganIVMoments, SupplyMoments
        from nfp_harness.tests.test_supply import _supply_inputs

        syn = self.syn
        n = syn["data"]["products"]["prices"].size
        firms = np.arange(n) // 2
        syn["data"]["products"]["firm_ids"] = firms
        layout = synthetic_layout(THETA_TRUE)
        model = PCMModel(PCMSpec(CHARS, tol=1e-12), syn["data"], layout, syn["nu"], syn["seed"], total_threads=8)
        Zs, w = _supply_inputs(syn)
        sup = SupplyMoments(Zs, w, firms, w_names=("c0", "c1", "c2"), fd_step=1e-3, fd_scale="unit")
        ms: list[Any] = [SarganIVMoments(syn["Z"]), sup]
        if margin is not None:
            ms.append(MarginMoment(sup, margin))
        prob = NFPGMMProblem(model, ms, LinearIV(syn["X1"], syn["Z"]), layout)
        prob.normalisations = ["toy", f"margin calibration m* = {margin}"]  # type: ignore[attr-defined]
        return prob

    def binding_weighting(self, problem: Any, W: np.ndarray) -> np.ndarray:
        W = np.array(W, dtype=float)
        for m in getattr(problem, "calibration", ()):
            r = problem.blocks[m.name]
            W[r, :] = 0.0
            W[:, r] = 0.0
            W[r, r] = m.binding_block()
        return W

    def named_directions(self, theta: np.ndarray, units: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def margin_band_row(self, pipe: Any, problem: Any, result: Any, value: float) -> dict[str, Any]:
        self.calls += 1
        return {"margin_trimmed_mean": result.evaluation.aux["margin"]["margin_trimmed_mean"]}

    def band_reference_rows(self) -> list[dict[str, Any]]:
        return [{"label": "reference"}]


def test_margin_band_rows_start_and_resume(tmp_path: pathlib.Path) -> None:
    """margin_band: the start is the gate minimum on lambda = 1 with pi re-solved, one problem per m* with a
    binding margin row, rows in band.csv, a report per stage, and a second run reads every point back."""
    syn = make_synthetic(2, 8, 32, 8, seed=2, price_shift=12.0, delta_shift=2.0)
    names = synthetic_layout().names
    gate = tmp_path / "gate" / "profile_sigma_ray"
    gate.mkdir(parents=True)
    start = THETA_TRUE * np.array([1.02, 0.98, 1.2, 1.0, 1.03])
    other = THETA_TRUE * 1.1
    with open(gate / "profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda", "sigma_p", "status", "q_fixed"] + [f"theta:{n}" for n in names])
        w.writerow([1.0, start[4], "ok", 5.0] + list(start))
        w.writerow([0.5, other[4], "ok", 1.0] + list(other))    # lower q, other slice: not the start
        w.writerow([1.0, other[4], "inversion_failed", 0.1] + list(other))
    spec = ToyMarginSpec(syn, tmp_path / "gate")
    cfg = RunConfig(base=tmp_path / "runs", nu=syn["nu"], seed=syn["seed"], threads=8)
    pipe = Pipeline(spec, cfg)
    try:
        rows = pipe.margin_band()
        g0 = pipe.rd.load_stage("margin_band.gate_minimum")
        np.testing.assert_allclose(g0["theta"], start)
        st = pipe.rd.load_stage("margin_band.start")["theta"]
        assert np.array_equal(np.delete(st, 2), np.delete(start, 2)) and st[2] != start[2]   # only pi moved
    finally:
        pipe.close()
    assert [r["margin_target"] for r in rows] == [0.25, 0.3] and spec.calls == 2
    for r in rows:
        assert r["step1_convergence"] and np.isfinite(r["q"]) and np.isfinite(r["J"])
    W1 = pipe.rd.load_stage("weights.mband.0.25.W1")["W"]
    assert W1[-1, -1] == pytest.approx(1e4 / (syn["data"]["products"]["prices"].size * 1e-4))
    rep = json.loads((pipe.path / "reports" / "margin_band.0.25.step1" / "report.json").read_text())
    assert "margin calibration m* = 0.25" in rep["spec"]["normalisations"]
    with open(pipe.path / "margin_band" / "band.csv") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 3 and table[-1]["label"] == "reference"

    spec2 = ToyMarginSpec(syn, tmp_path / "gate")
    pipe2 = Pipeline(spec2, cfg)
    try:
        rows2 = pipe2.margin_band()
    finally:
        pipe2.close()
    assert spec2.calls == 0 and rows2 == rows

"""Tests for the pipeline's sigma_p_band stage (set identification of one parameter, the gate's fallback).

A toy spec on the synthetic PCM of ``test_pcm_model`` (2 markets, 8 candidates, 256 draws): the band fixes
sigma_p at each value, runs fixed-W step 1 and step 2 from a stored prefit point, writes band.csv, and a
second run on the same directory reads every point back instead of estimating it again.
"""

from __future__ import annotations

import csv
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
    assert [t["label"] for t in table][-1] == "reference" and len(table) == 3
    assert pipe.rd.stage_done("band.0.8.row") and pipe.rd.stage_done("band.1.step1")

    spec2 = ToySpec(syn)
    pipe2 = Pipeline(spec2, cfg)
    try:
        rows2 = pipe2.sigma_p_band()
    finally:
        pipe2.close()
    assert spec2.calls == 0 and rows2 == rows


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

"""Staged estimation runs from a paper spec: CLI, run directory, profile gate, two-step GMM, postestimation,
integration check (DESIGN 4.7, 5, 6.4). Model-agnostic: the paper's spec supplies the problems and the
paper-specific postestimation.

Stages, each checkpointed in the run directory ``<base>/<spec.name>_<run_key[:10]>/`` (run key = hash of
nu, seed and the spec's ``run_key()`` options, e.g. robustness switches) and each ending with a diagnostics report in ``reports/<stage>/report.{json,txt}`` (the
latest one is also copied to the run directory root):

- ``prefit``: the spec's ``prefit_free`` parameters alone, all others fixed, on the spec's prefit
  moments with W1, at most ``prefit_max_iterations`` trial steps (DESIGN 6.2).
- ``gate``: the spec's ray profiles (``spec.gate_profiles()``) from the prefit point, cold inversions on
  parallel workers, with the pass rule. Writes ``gate/report.{json,txt}`` and ``gate/verdict.json``. A
  profile whose ``concentrate`` is a tuple of parameter names re-optimises only those at every point (the
  others held), with the spec's optimizer options and at most ``spec.gate_concentrate_iterations`` (20)
  trial steps per point.
- ``extensive``: only if the spec has ``extensive_free`` and no ``start_from_gate``: those parameters alone from
  the prefit point with W1 on all moments (refused unless the gate passed, as step1). step1 then starts from
  its endpoint.
- ``step1``: fixed W1 = ``initial_weighting(theta_prefit)`` from the extensive (else prefit) point. Refused unless the gate
  tagged ``spec.gate_required`` passed in the run with ``--gate-nu`` draws per agent (same seed). With
  ``spec.start_from_gate`` step1 starts instead from the gate minimum (``start_point``: the gate profile's
  lowest ``spec.start_column`` on ``spec.start_slice``, read from ``spec.start_gate`` or else from the gate of
  the ``--gate-nu`` run, then ``spec.start_concentrate`` re-solved alone with
  ``spec.start_concentrate_weighting``), and W1 is formed there.
- ``step2``: W2 = S(theta_hat_1)^{-1} from theta_hat_1. Refused unless step 1 converged (CONVERGED_NOISE
  counts) and its report shows none of TwoStep's blocking degeneracies (DESIGN 4.5).
- ``post``: at the last finished estimate (step2, else step1): the report again and ``spec.post``, whose
  summary goes to ``post/summary.json``.
- ``stability``: at most 20 optimizer iterations from the estimate with ``2 nu`` draws per agent and new
  seeds, then the report with the integration block (DESIGN 5.7) and ``spec.headlines`` compared.
- ``sigma_p_band`` (set identification of one parameter, the fallback when the gate fails): for each value in
  ``spec.band_values`` of ``spec.band_parameter``, that parameter fixed, fixed-W step 1 (W1) and step 2 (W2 =
  S^{-1} at the step-1 point, only if step 1 converged) on ``spec.band_moments`` from the prefit point, then
  ``spec.band_row(pipeline, problem, result, value) -> dict`` at the last estimate. Rows go to
  ``sigma_p_band/band.csv`` (plus ``spec.band_reference_rows()``, e.g. BLP) and ``band.json``; each point is
  checkpointed (stages ``band.<value>.step1|step2|row``), so a killed run resumes at the point it was in. When
  every point is in, ``spec.band_finish(pipeline, problem, rows) -> rows`` (optional) labels the headline and
  may add columns (it can rebuild a point's estimate with ``pipeline.band_result``); its rows are checkpointed
  as stage ``band.finish``.
- ``a_lo_band`` (ROUND5 5.1 item 3): the same procedure over ``spec.a_lo_band_values`` of
  ``spec.a_lo_band_parameter`` (the bound a_lo, a calibration of the welfare tail), rows from
  ``spec.a_lo_band_row`` (default ``band_row``) to ``a_lo_band/band.csv``, stages ``aband.<value>.step1|step2|row``,
  ``spec.a_lo_band_finish`` as ``aband.finish``. A fixed band parameter's bounds widen to the band value.
  ``--band-values`` runs a subset of a band's values (finish waits for all), ``--band-step1-only`` stops each point
  after step 1 and its report (``reports/<band>.<value>.step1``).
- ``margin_band`` (a calibrated markup level, round-3 analysis 5.3): for each m* in ``spec.margin_band_values``
  the problem ``spec.build_problem(..., margin=m*)`` (all nonlinear parameters free), step 1 with W1 from
  ``start_point`` (shared by every point), step 2 with W2 = S^{-1} at the step-1 point when step 1 converged
  and showed no blocking degeneracy, a report per stage (``reports/margin_band.<m*>.step1|step2``), then
  ``spec.margin_band_row(pipeline, problem, result, m*)``. ``J`` is n g'W g over the non-calibration rows at
  the last estimate (Hansen's J at step 2). Rows go to ``margin_band/band.csv`` (plus
  ``spec.band_reference_rows()``); each point is checkpointed (``mband.<m*>.step1|step2|row``) and the points
  run one after another, each on all threads.

Every W the estimation stages minimise (W1, W2, band W) passes through ``spec.binding_weighting(problem, W)``
when the spec has it, e.g. to make a calibration row bind.

``--stage all`` runs extensive, step1, step2, post and stability in order; a refused stage is reported and the
chain continues with what it can do (post and stability fall back to the step-1 estimate). When extensive or
step1 is refused because the gate failed and the spec names a ``fallback_stage``, that stage runs instead. Resume by
rerunning the identical command: finished stages are skipped, the stage in progress resumes from its
last accepted step (``optimizer.minimize``), a profile from its last finished point. A second process
on the same run directory is refused while the first is alive.

Spec interface (duck-typed): ``name``; ``hash_payload() -> dict`` (what changes results);
``normalisations``; the economics and diagnostics fields read by ``diagnostics.run_diagnostics`` and
``economics_block`` (``firm_column``, ``groups``, ``group_other``, ``price_range``, ``plausible``,
``binary_columns``, ``X1``, ``Z``, ``linear_names``); ``build_problem(nu, seed, threads, moments,
layout_changes=None) -> NFPGMMProblem`` with ``moments`` in {"all", "demand", "prefit", "aggregate"};
``prefit_free``, ``prefit_max_iterations``; ``gate_profiles() -> [(name, moments, kwargs)]`` with
``ray_profile`` keyword arguments (direction, lambdas, grid, origin, economics) plus ``layout_changes``
and ``rules`` ({tag: {"rule": callable(ProfileResult) -> (passed, message), "columns": [...]}});
``gate_required`` (the tag step1 needs, or a tuple of tags of which one must pass); ``named_directions(theta, units) -> {name: vector}``;
``post(pipeline, problem, result, out_dir) -> dict``; ``headlines(problem, result) -> {name: float}``;
optionally ``optimizer_options`` (OptimizerOptions overrides for every stage), ``max_iterations``,
``extensive_free``, ``gate_concentrate_iterations``, ``run_key() -> dict`` (result-changing options that
name the run directory), ``add_arguments(parser)`` and ``configure(args)`` (spec-specific CLI options).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from . import runtime
from .checkpoint import RunDirectory, SpecMismatchError, atomic_write, code_commit, hash_options
from .diagnostics import run_diagnostics, write_report
from .diagnostics.economics import economics_block
from .diagnostics.report import to_jsonable
from .optimizer import OptimizerOptions, minimize
from .profile import ray_profile
from .types import OptimizeResult
from .weighting import Fixed, TwoStep, weight_from_covariance

__all__ = ["Pipeline", "RunConfig", "StageRefused", "GateFailed", "STAGES", "CHAIN", "main", "run_dir_path", "run_key_options"]

log = logging.getLogger("nfp_harness.pipeline")

STAGES = ("prefit", "gate", "extensive", "step1", "step2", "post", "stability", "sigma_p_band", "a_lo_band",
          "margin_band")
CHAIN = ("extensive", "step1", "step2", "post", "stability")
STABILITY_SEED_OFFSET = 100_000
STABILITY_ITERATIONS = 20


class StageRefused(RuntimeError):
    """A stage's precondition does not hold; the message says which and what to do."""


class GateFailed(StageRefused):
    """step1 (or extensive) is refused because the required gate verdict failed."""


@dataclass(frozen=True)
class RunConfig:
    base: pathlib.Path
    nu: int
    seed: int = 0
    threads: int | None = None
    workers: int = 4
    gate_nu: int = 4
    outer: str = "nested"   # OptimizerOptions.outer of every stage: "nested" or "mpec" (demand-only problems)
    band_values: tuple[float, ...] = ()   # run only these values of a band stage (empty: all); results unchanged
    band_step1_only: bool = False         # band stages stop after each point's step 1 (no step 2, no row)


def run_key_options(nu: int, seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The CLI options that change results and so name the run directory: nu, seed and the spec's
    ``run_key()`` options (left out when empty, so specs without them keep their directory names)."""
    out: dict[str, Any] = {"nu": int(nu), "seed": int(seed)}
    if extra:
        out.update(extra)
    return out


def spec_run_key(spec: Any) -> dict[str, Any]:
    """The spec's result-changing options beyond nu and seed (``spec.run_key()``), {} without the hook."""
    fn = getattr(spec, "run_key", None)
    return dict(fn()) if callable(fn) else {}


def run_dir_path(base: pathlib.Path, spec: Any, nu: int, seed: int) -> pathlib.Path:
    """``<base>/<spec.name>_<run key[:10]>``."""
    return base / f"{spec.name}_{hash_options(run_key_options(nu, seed, spec_run_key(spec)))[:10]}"


def _block_diagonal_like(W: np.ndarray, blocks: dict[str, slice],
                          coupled: Sequence[Sequence[str]] = ()) -> np.ndarray:
    """W with every entry outside the moment blocks' diagonal blocks set to zero (S is block diagonal), except
    between blocks named together in ``coupled`` (the aggregate and supply blocks, whose S has cross terms)."""
    out = np.zeros_like(W)
    groups = [list(g) for g in coupled]
    grouped = {b for g in groups for b in g}
    groups += [[b] for b in blocks if b not in grouped]
    for g in groups:
        idx = np.concatenate([np.arange(blocks[b].start, blocks[b].stop) for b in g if b in blocks])
        out[np.ix_(idx, idx)] = W[np.ix_(idx, idx)]
    return 0.5 * (out + out.T)


def _table(res: Any, cols: Sequence[str]) -> str:
    """Fixed-width table of a profile's points: axis values, status, the given columns, seconds."""
    head = list(res.axis_names) + ["status"] + list(cols) + ["sec"]
    widths = [max(10, len(h)) for h in head]
    lines = ["  ".join(h.rjust(w) for h, w in zip(head, widths))]
    for p in res.points:
        cells = [f"{p.coords[a]:.6g}" for a in res.axis_names] + [p.status]
        cells += [f"{p.values.get(c, float('nan')):.6g}" for c in cols] + [f"{p.seconds:.1f}"]
        lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


class Pipeline:
    """One run directory of one spec at one (nu, seed)."""

    def __init__(self, spec: Any, cfg: RunConfig, new_run: bool = False) -> None:
        self.spec = spec
        self.cfg = cfg
        self.spec_hash = hash_options(spec.hash_payload())
        key = run_key_options(cfg.nu, cfg.seed, spec_run_key(spec))
        self.rd = RunDirectory(cfg.base, spec.name, self.spec_hash, hash_options(key), new_run=new_run, options=key)
        self.path = self.rd.path
        self._problems: dict[tuple[Any, ...], Any] = {}

    # ---------------------------------------------------------------- problems and reports

    def problem(self, moments: str = "all", nu: int | None = None, seed: int | None = None) -> Any:
        key = (moments, nu or self.cfg.nu, self.cfg.seed if seed is None else seed)
        if key not in self._problems:
            problem = self.spec.build_problem(key[1], key[2], self.cfg.threads, moments)
            if self.cfg.outer == "mpec":
                from .mpec import refusal

                reason = refusal(problem)
                if reason:
                    close = getattr(getattr(problem, "model", None), "close", None)
                    if close is not None:
                        close()
                    raise StageRefused(f"--outer mpec with moments {moments!r}: {reason}")
            self._problems[key] = problem
        return self._problems[key]

    def close(self) -> None:
        for p in self._problems.values():
            close = getattr(getattr(p, "model", None), "close", None)
            if close is not None:
                close()
        self._problems.clear()

    def report(self, problem: Any, result: OptimizeResult, stage: str, **extra: Any) -> dict[str, Any]:
        """Run the diagnostics at a stage's endpoint and write reports/<stage>/report.{json,txt} and the root copy."""
        ev = result.evaluation
        units = problem.layout.units()
        options: dict[str, Any] = {"units": units, "n": problem.n, "run_id": self.path.name, "hash": self.spec_hash,
                                   "code_commit": code_commit(),
                                   "normalisations": list(getattr(problem, "normalisations", None)
                                                          or self.spec.normalisations),
                                   "named_directions": self.spec.named_directions(result.theta, units),
                                   "files": {"history": "history.csv", "checkpoint": "checkpoint.npz",
                                             "log": "log.txt"}}
        if ev is not None and ev.ok and ev.G is not None:
            try:
                G_full, names = problem.full_jacobian(ev)
                options.update(G_full=G_full, linear_names=names[:G_full.shape[1] - problem.P])
                if ev.aux.get("linear_coefficients") is not None:   # beta and the supply side's gamma
                    options["beta"] = np.asarray(ev.aux["linear_coefficients"], dtype=float)
            except (ValueError, AttributeError):
                pass
            try:
                eco = economics_block(problem, result, self.spec)
                options["economics"] = eco
                options["cross_group_diversion"] = eco.values.get("cross_group_diversion")
            except Exception as exc:  # noqa: BLE001 - the report says the block failed instead of losing the stage
                log.warning("economics block failed at %s: %s: %s", stage, type(exc).__name__, exc)
        options.update(extra)
        rep = run_diagnostics(problem, result, self.spec, options)
        out = self.path / "reports" / stage
        write_report(rep, out)
        for name in ("report.json", "report.txt"):
            shutil.copyfile(out / name, self.path / name)
        log.info("stage %s report: %s", stage, out / "report.txt")
        return json.loads((out / "report.json").read_text())

    def _stage_state(self, stage: str) -> dict[str, Any]:
        if not self.rd.stage_done(stage):
            raise StageRefused(f"stage {stage} has not finished in {self.path}")
        return self.rd.load_stage(stage)

    def _saved_weight(self, name: str, make: Callable[[], np.ndarray]) -> np.ndarray:
        """A weighting matrix computed once and stored in the run directory, so a resume uses it bit for bit."""
        key = f"weights.{name}"
        if self.rd.stage_done(key):
            return np.asarray(self.rd.load_stage(key)["W"], dtype=float)
        W = make()
        self.rd.save_stage(key, {"W": W})
        return W

    def _warm(self, problem: Any, stage: str) -> None:
        """Warm-start problem's model from a finished stage's model state (same markets and draws)."""
        if self.rd.stage_done(stage):
            ps = self.rd.load_stage(stage).get("problem_state") or {}
            if ps.get("model") is not None and hasattr(problem.model, "load_state_dict"):
                try:
                    problem.model.load_state_dict(ps["model"])
                except (ValueError, KeyError) as exc:
                    log.info("warm start from %s not used: %s", stage, exc)

    def _binding(self, problem: Any, W: np.ndarray) -> np.ndarray:
        """``spec.binding_weighting(problem, W)`` when the spec has it, else W."""
        hook = getattr(self.spec, "binding_weighting", None)
        return np.asarray(W, dtype=float) if hook is None else np.asarray(hook(problem, W), dtype=float)

    def _gate_minimum(self, tag: str, names: Sequence[str]) -> np.ndarray:
        """theta at the gate minimum (module docstring, ``step1``), stored once as stage ``<tag>.gate_minimum`` so a
        resumed run starts where it started even if the gate files change."""
        key = f"{tag}.gate_minimum"
        if self.rd.stage_done(key):
            return np.asarray(self.rd.load_stage(key)["theta"], dtype=float)
        import csv

        src = getattr(self.spec, "start_gate", None)
        path = pathlib.Path(src) if src else (
            run_dir_path(self.cfg.base, self.spec, self.cfg.gate_nu, self.cfg.seed) / "gate")
        if path.is_dir():
            files = sorted(path.glob("profile_*/profile.csv")) or sorted(path.glob("profile.csv"))
            if len(files) != 1:
                raise StageRefused(f"{tag}: expected one gate profile.csv under {path}, found {len(files)}")
            path = files[0]
        if not path.exists():
            raise StageRefused(f"{tag}: no gate profile at {path}; run the gate or pass the spec's start gate")
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        col = str(getattr(self.spec, "start_column", "q_fixed"))
        sl = dict(getattr(self.spec, "start_slice", None) or {})

        def ok(r: dict[str, str]) -> bool:
            try:
                return (r.get("status") == "ok" and np.isfinite(float(r[col]))
                        and all(abs(float(r[k]) - float(v)) < 1e-12 for k, v in sl.items()))
            except (KeyError, ValueError):
                return False

        good = [r for r in rows if ok(r)]
        if not good:
            raise StageRefused(f"{tag}: no evaluated point with a finite {col} on {sl} in {path}")
        best = min(good, key=lambda r: float(r[col]))
        missing = [n for n in names if f"theta:{n}" not in best]
        if missing:
            raise StageRefused(f"{tag}: the gate profile {path} has no columns for {missing}")
        theta = np.array([float(best[f"theta:{n}"]) for n in names])
        log.info("%s: start at the gate minimum of %s (%s %s, %s = %s)", tag, path, sl, {a: best.get(a) for a in
                 ("lambda", "sigma_p")}, col, best[col])
        self.rd.save_stage(key, {"theta": theta, "source": str(path), "row": json.dumps(best)})
        return theta

    def start_point(self, problem: Any, tag: str) -> np.ndarray:
        """The gate minimum with ``spec.start_concentrate`` re-solved alone (stage ``<tag>.start``) on
        ``spec.start_concentrate_weighting(problem, W1)``, at most ``spec.gate_concentrate_iterations`` steps."""
        theta_g = self._gate_minimum(tag, problem.layout.names)
        conc = tuple(getattr(self.spec, "start_concentrate", None) or ())
        if not conc:
            return theta_g
        layout = problem.layout
        unknown = [c for c in conc if c not in layout.names]
        if unknown:
            raise ValueError(f"start_concentrate names unknown parameters {unknown}")
        hook = getattr(self.spec, "start_concentrate_weighting", None)
        W = self._saved_weight(f"{tag}.start_W", lambda: (problem.initial_weighting(theta_g) if hook is None
                                                          else hook(problem, problem.initial_weighting(theta_g))))
        problem.layout = layout.replace({n: {"free": n in conc} for n in layout.names})
        try:
            res = minimize(problem, Fixed(W), self._options(
                stage=f"{tag}.start", theta0=theta_g, noise_probe=None,
                max_iterations=int(getattr(self.spec, "gate_concentrate_iterations", 20))), checkpoint=self.rd)
        finally:
            problem.layout = layout
        log.info("%s: start concentrated in %s: %s, q %.6g", tag, conc, res.convergence.value, res.q)
        return np.asarray(res.theta, dtype=float)

    # ---------------------------------------------------------------- stages

    def prefit(self) -> np.ndarray:
        """Theta at the end of the prefit stage (runs it if needed)."""
        if self.rd.stage_done("prefit"):
            return np.asarray(self._stage_state("prefit")["theta"], dtype=float)
        problem = self.problem("prefit")
        free = set(self.spec.prefit_free)
        layout = problem.layout
        prefit_layout = layout.replace({n: {"free": n in free} for n in layout.names})
        problem.layout = prefit_layout
        try:
            theta0 = prefit_layout.starts()
            W1 = self._saved_weight("prefit_W1", lambda: problem.initial_weighting(theta0))
            hook = getattr(self.spec, "concentrate_weighting", None)
            W_pre = W1 if hook is None else hook(problem, W1)
            res = minimize(problem, Fixed(W_pre), self._options(stage="prefit", theta0=theta0,
                                                               max_iterations=self.spec.prefit_max_iterations),
                           checkpoint=self.rd)
            self.report(problem, res, "prefit")
        finally:
            problem.layout = layout
        return np.asarray(res.theta, dtype=float)

    def gate(self) -> dict[str, Any]:
        """The spec's ray profiles from the prefit point; writes gate/report.{json,txt} and gate/verdict.json.

        Each profile is evaluated once; each of its ``rules`` (tag -> {"rule": pass rule, "columns": table
        columns}) gives one verdict. Rules on different moment blocks of the same profile share its inner
        inversions: with block-diagonal W1 the aggregate block of q_fixed is the aggregate-only objective.
        """
        theta = self.prefit()
        out = self.path / "gate"
        out.mkdir(exist_ok=True)
        doc: dict[str, Any] = {"spec_hash": self.spec_hash, "nu": self.cfg.nu, "seed": self.cfg.seed,
                               "theta0": theta.tolist(), "profiles": {}, "verdict": {}}
        texts = []
        for name, moments, kw in self.spec.gate_profiles():
            kw = dict(kw)
            changes = kw.pop("layout_changes", None)
            rules = kw.pop("rules")

            def factory(worker: int, threads: int, moments: str = moments, changes: Any = changes) -> Any:
                return self.spec.build_problem(self.cfg.nu, self.cfg.seed, threads, moments, changes)

            problem = factory(0, self.cfg.threads or os.cpu_count() or 1)
            conc = kw.get("concentrate")
            if isinstance(conc, (tuple, list)):
                unknown = [c for c in conc if c not in problem.layout.names]
                if unknown:
                    raise ValueError(f"profile {name}: concentrate names unknown parameters {unknown}")
                kw["concentrate"] = True
                kw["hold"] = [n for n in problem.layout.names if n not in conc]
                kw.setdefault("optimizer_options", self._options(
                    max_iterations=int(getattr(self.spec, "gate_concentrate_iterations", 20)), record_cue=False))
            try:
                self._warm(problem, "prefit")  # W1 needs one evaluation at the prefit point; the points start cold
                W1 = problem.initial_weighting(theta)
                hook = getattr(self.spec, "concentrate_weighting", None)
                if kw.get("concentrate") and hook is not None:
                    kw["concentrate_weighting"] = hook(problem, W1)
                res = ray_profile(problem, theta, kw.pop("direction"), kw.pop("lambdas"), kw.pop("grid", None),
                                  weighting=W1, run_dir=out / f"profile_{name}", tag=name, workers=self.cfg.workers,
                                  problem_factory=factory, **kw)
            finally:
                problem.model.close()
            along = res.axis_names[-1]
            failed = [p.coords for p in res.points if not p.ok]
            doc["profiles"][name] = {"moments": moments, "rows": res.rows(), "failed_points": failed,
                                     "csv": str(out / f"profile_{name}" / "profile.csv"), "verdicts": {}}
            for tag, r in rules.items():
                cols = [c for c in r["columns"] if c in res.columns]
                passed, message = r["rule"](res)
                shapes = {c: {",".join(f"{v:g}" for v in key): v.to_dict() for key, v in res.classify(along, c).items()}
                          for c in cols if c.startswith("q_")}
                doc["profiles"][name]["verdicts"][tag] = {"passed": passed, "message": message, "columns": cols,
                                                          "shapes": shapes}
                doc["verdict"][tag] = passed
                text = [f"== {tag}: profile {name} ({moments} moments evaluated), {len(res.points)} points, "
                        f"{res.workers} workers ==", _table(res, cols)]
                text += [v["message"] for c in cols if c in shapes for v in shapes[c].values()]
                if failed:
                    text.append(f"FLAG: {len(failed)} point(s) did not evaluate: {failed}")
                text.append(("PASS: " if passed else "FAIL: ") + message)
                texts.append("\n".join(text))
                log.info("gate %s: %s", tag, "PASS" if passed else "FAIL")
        js = json.dumps(to_jsonable(doc), indent=1, allow_nan=False)
        atomic_write(out / "report.json", js.encode())
        atomic_write(out / "report.txt", ("\n\n".join(texts) + "\n").encode())
        atomic_write(out / "verdict.json", json.dumps({"spec_hash": self.spec_hash, "verdict": doc["verdict"],
                                                       "nu": self.cfg.nu}).encode())
        rep_dir = self.path / "reports" / "gate"
        rep_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("report.json", "report.txt"):
            shutil.copyfile(out / fname, rep_dir / fname)
        return doc

    def _check_gate(self) -> None:
        required = self.spec.gate_required
        gate_dir = run_dir_path(self.cfg.base, self.spec, self.cfg.gate_nu, self.cfg.seed) / "gate" / "verdict.json"
        if not gate_dir.exists():
            raise StageRefused(f"step1 needs a passed gate: no {gate_dir}. Run `run --stage gate --nu "
                               f"{self.cfg.gate_nu}` first.")
        v = json.loads(gate_dir.read_text())
        if v.get("spec_hash") != self.spec_hash:
            raise StageRefused(f"the gate in {gate_dir.parent} was run with another spec (hash {v.get('spec_hash')}); "
                               f"rerun it with the current spec.")
        tags = (required,) if isinstance(required, str) else tuple(required)
        if not any(v["verdict"].get(t) for t in tags):
            raise GateFailed(f"the {' and '.join(tags)} gate failed ({gate_dir.parent / 'report.txt'}); the long "
                               f"estimation does not start (DESIGN 6.4)." + (
                                   f" Fallback: run --stage {self.spec.fallback_stage}."
                                   if getattr(self.spec, "fallback_stage", None) else ""))

    def _options(self, **kw: Any) -> OptimizerOptions:
        """OptimizerOptions for a stage: the spec's ``optimizer_options`` overrides (e.g. trust radii), then ``kw``;
        ``outer`` from ``RunConfig.outer``."""
        return OptimizerOptions(**{**dict(getattr(self.spec, "optimizer_options", None) or {}), "outer": self.cfg.outer,
                                   **kw})

    def _limit(self) -> int | None:
        """Trial-step limit of step1 and step2: ``spec.max_iterations`` if set (smoke tests), else none."""
        v = getattr(self.spec, "max_iterations", None)
        return None if v is None else int(v)

    def _w1(self, problem: Any, theta: np.ndarray) -> np.ndarray:
        return self._saved_weight("W1", lambda: self._binding(problem, problem.initial_weighting(theta)))

    def extensive(self) -> np.ndarray:
        """Theta at the end of the extensive-margin stage: ``spec.extensive_free`` alone from the prefit
        point with W1 on all moments (SPEC 3.5a). The prefit theta when the spec has no such stage."""
        free_names = tuple(getattr(self.spec, "extensive_free", None) or ())
        if getattr(self.spec, "start_from_gate", False):
            log.info("extensive: skipped, step1 starts from the gate minimum (spec.start_from_gate)")
            return None  # type: ignore[return-value]
        if not free_names:
            return self.prefit()
        if self.rd.stage_done("extensive"):
            return np.asarray(self._stage_state("extensive")["theta"], dtype=float)
        self._check_gate()
        theta_w = self.prefit()
        problem = self.problem("all")
        layout = problem.layout
        unknown = [n for n in free_names if n not in layout.names]
        if unknown:
            raise ValueError(f"extensive_free names unknown parameters {unknown}")
        self._warm(problem, "prefit")
        W1 = self._w1(problem, theta_w)
        problem.layout = layout.replace({n: {"free": n in free_names} for n in layout.names})
        try:
            res = minimize(problem, Fixed(W1), self._options(stage="extensive", theta0=theta_w,
                                                               max_iterations=self._limit()), checkpoint=self.rd)
            if not self.rd.stage_done("report.extensive"):
                self.report(problem, res, "extensive")
                self.rd.save_stage("report.extensive", {"done": True})
        finally:
            problem.layout = layout
        return np.asarray(res.theta, dtype=float)

    def step1(self) -> OptimizeResult:
        self._check_gate()
        problem = self.problem("all")
        if getattr(self.spec, "start_from_gate", False):
            # a finished step1 is read back by minimize; W1 is stored, so theta_w only matters the first time
            theta0 = (np.asarray(self._stage_state("step1")["theta"], dtype=float) if self.rd.stage_done("step1")
                      else self.start_point(problem, "step1"))
            theta_w = theta0
        else:
            theta_w = self.prefit()
            theta0 = self.extensive()
            if not self.rd.stage_done("step1"):
                self._warm(problem, "extensive" if self.rd.stage_done("extensive") else "prefit")
        W1 = self._w1(problem, theta_w)
        res = minimize(problem, Fixed(W1), self._options(stage="step1", theta0=theta0, max_iterations=self._limit()),
                       checkpoint=self.rd)
        if not self.rd.stage_done("report.step1"):
            self.report(problem, res, "step1")
            self.rd.save_stage("report.step1", {"done": True})
        return res

    def step2(self) -> OptimizeResult:
        if not self.rd.stage_done("step1"):
            raise StageRefused("step2 needs a finished step1; run `run --stage step1` (or `all`) first.")
        problem = self.problem("all")
        st1 = self._stage_state("step1")
        W1 = np.asarray(st1["W"], dtype=float)
        r1 = minimize(problem, Fixed(W1), self._options(stage="step1"), checkpoint=self.rd)
        rep1 = json.loads((self.path / "reports" / "step1" / "report.json").read_text())
        detected = [d["name"] for d in rep1.get("degeneracies", []) if d.get("detected")]
        reason = None
        if not r1.convergence.converged:
            reason = f"step 1 ended {r1.convergence.value}, not converged"
        else:
            ok, why = TwoStep(W1).gate(detected)
            reason = None if ok else why
        if reason:
            msg = f"Step 2 not started: {reason}. Post and stability use the step-1 point."
            atomic_write(self.path / "reports" / "step2_refused.txt", (msg + "\n").encode())
            raise StageRefused(msg)
        if r1.evaluation is None:
            raise StageRefused("step2: the step-1 endpoint could not be re-evaluated")

        ev1 = r1.evaluation

        def make_w2() -> np.ndarray:
            return self._binding(problem, _block_diagonal_like(weight_from_covariance(problem.covariance(ev1)),
                                                               ev1.blocks, getattr(problem, "coupled_blocks", ())))

        W2 = self._saved_weight("W2", make_w2)
        res = minimize(problem, Fixed(W2), self._options(stage="step2", theta0=r1.theta, max_iterations=self._limit()),
                       checkpoint=self.rd)
        if not self.rd.stage_done("report.step2"):
            self.report(problem, res, "step2")
            self.rd.save_stage("report.step2", {"done": True})
        return res

    def estimate(self) -> tuple[Any, OptimizeResult, str]:
        """(problem, result, stage) at the last finished estimation stage, re-evaluated with its own W."""
        stage = "step2" if self.rd.stage_done("step2") else "step1"
        st = self._stage_state(stage)
        problem = self.problem("all")
        res = minimize(problem, Fixed(np.asarray(st["W"], dtype=float)), self._options(stage=stage),
                       checkpoint=self.rd)
        if res.evaluation is None:
            raise StageRefused(f"the {stage} estimate could not be re-evaluated")
        return problem, res, stage

    def post(self) -> dict[str, Any]:
        if self.rd.stage_done("post"):
            return json.loads((self.path / "post" / "summary.json").read_text())
        problem, res, stage = self.estimate()
        rep = self.report(problem, res, "post")
        out = self.path / "post"
        out.mkdir(exist_ok=True)
        summary = {"estimate_stage": stage, "convergence": res.convergence.value, "q": res.q,
                   "theta": dict(zip(problem.layout.names, map(float, res.theta))),
                   "flags": [m for blk in ("economics",) for m in rep.get(blk, {}).get("messages", [])
                             if m.startswith(("FLAG:", "FAIL:"))]}
        summary.update(self.spec.post(self, problem, res, out))
        atomic_write(out / "summary.json", json.dumps(to_jsonable(summary), indent=1, allow_nan=False).encode())
        self.rd.save_stage("post", {"estimate_stage": stage})
        return summary

    def stability(self) -> dict[str, Any]:
        if self.rd.stage_done("stability.report"):
            return json.loads((self.path / "reports" / "stability" / "report.json").read_text())
        problem, res, stage = self.estimate()
        W = np.asarray(self._stage_state(stage)["W"], dtype=float)
        nu2 = 2 * self.cfg.nu
        prob2 = self.problem("all", nu=nu2, seed=self.cfg.seed + STABILITY_SEED_OFFSET)
        res2 = minimize(prob2, Fixed(W), self._options(stage="stability", theta0=res.theta,
                                                          max_iterations=STABILITY_ITERATIONS), checkpoint=self.rd)
        head0 = self.spec.headlines(problem, res)
        head1 = self.spec.headlines(prob2, res2) if res2.evaluation is not None else {}
        headlines = {k: (head0[k], head1[k]) for k in head0 if k in head1}
        rep = self.report(problem, res, "stability", reevaluate=lambda theta: res2, headlines=headlines)
        self.rd.save_stage("stability.report", {"estimate_stage": stage, "nu_doubled": nu2})
        return rep

    def sigma_p_band(self) -> list[dict[str, Any]]:
        """The sigma_p band stage (module docstring). Returns the rows of sigma_p_band/band.csv."""
        return self._band("sigma_p_band")

    def a_lo_band(self) -> list[dict[str, Any]]:
        """The a_lo band stage (module docstring, ROUND5 5.1 item 3). Returns the rows of a_lo_band/band.csv."""
        return self._band("a_lo_band")

    # band name -> (spec attribute prefix, checkpoint tag prefix, default parameter)
    _BANDS = {"sigma_p_band": ("band", "band", "sigma_p"), "a_lo_band": ("a_lo_band", "aband", "a_lo")}

    def _band_fixed(self, layout: Any, par: str, v: float) -> Any:
        """``layout`` with ``par`` fixed at ``v``; its bounds widen to ``v`` when ``v`` lies outside them (a fixed
        parameter such as the bound a_lo has lower = upper = its value)."""
        sp = layout.specs[layout.names.index(par)]
        ch: dict[str, Any] = {"free": False, "start": v}
        if not sp.lower <= v <= sp.upper:
            ch.update(lower=min(sp.lower, v), upper=max(sp.upper, v))
        return layout.replace({par: ch})

    def _band(self, name: str) -> list[dict[str, Any]]:
        spec = self.spec
        attr, prefix, default_par = self._BANDS[name]
        par = str(getattr(spec, f"{attr}_parameter", default_par))
        values = [float(v) for v in getattr(spec, f"{attr}_values")]
        only = self.cfg.band_values
        if only:
            missing = [v for v in only if not any(abs(v - x) < 1e-12 for x in values)]
            if missing:
                raise StageRefused(f"{name}: --band-values {missing} are not among the spec's values {values}")
            values = [x for x in values if any(abs(v - x) < 1e-12 for v in only)]
        complete = len(values) == len(getattr(spec, f"{attr}_values"))
        moments = str(getattr(spec, f"{attr}_moments", "all"))
        row_fn = getattr(spec, f"{attr}_row", None) or spec.band_row
        out = self.path / name
        out.mkdir(exist_ok=True)
        theta_w = self.prefit()
        problem = self.problem(moments)
        layout = problem.layout
        if par not in layout.names:
            raise ValueError(f"band parameter {par!r} is not a parameter: {layout.names}")
        k = layout.names.index(par)
        rows: list[dict[str, Any]] = []
        try:
            for v in values:
                tag = f"{prefix}.{v:g}"
                if self.rd.stage_done(f"{tag}.row"):
                    rows.append(dict(json.loads(str(self.rd.load_stage(f"{tag}.row")["row"]))))
                    continue
                log.info("%s: %s = %g", name, par, v)
                problem.layout = self._band_fixed(layout, par, v)
                theta0 = np.array(theta_w, dtype=float)
                theta0[k] = v
                if not self.rd.stage_done(f"{tag}.step1"):
                    self._warm(problem, "prefit")
                W1 = self._saved_weight(f"{tag}.W1", lambda: self._binding(problem, problem.initial_weighting(theta0)))
                r1 = minimize(problem, Fixed(W1), self._options(stage=f"{tag}.step1", theta0=theta0,
                                                                  max_iterations=self._limit()), checkpoint=self.rd)
                res, stage = r1, "step1"
                if self.cfg.band_step1_only:
                    log.info("%s: %s = %g, step 1 only (--band-step1-only): %s, q %.10g", name, par, v,
                             r1.convergence.value, r1.q)
                    self._band_report(problem, r1, f"{name}.{v:g}.step1")
                    continue
                if r1.convergence.converged and r1.evaluation is not None:
                    ev1 = r1.evaluation

                    def make_w2(problem: Any = problem, ev1: Any = ev1) -> np.ndarray:
                        return self._binding(problem, _block_diagonal_like(
                            weight_from_covariance(problem.covariance(ev1)), ev1.blocks,
                            getattr(problem, "coupled_blocks", ())))

                    W2 = self._saved_weight(f"{tag}.W2", make_w2)
                    res = minimize(problem, Fixed(W2), self._options(stage=f"{tag}.step2", theta0=r1.theta,
                                                                       max_iterations=self._limit()), checkpoint=self.rd)
                    stage = "step2"
                row: dict[str, Any] = {"label": "", par: v, "estimate_stage": stage,
                                       "step1_convergence": r1.convergence.value, "convergence": res.convergence.value,
                                       "q": float(res.q)}
                if res.evaluation is not None and res.evaluation.ok:
                    row.update(row_fn(self, problem, res, v))
                else:
                    row["note"] = "no successful evaluation at the estimate"
                rows.append(row)
                self.rd.save_stage(f"{tag}.row", {"row": json.dumps(to_jsonable(row), allow_nan=False)})
                self._write_band(out, rows, par)
            finish = getattr(spec, f"{attr}_finish", None)
            if callable(finish) and complete and len(rows) == len(values):
                if self.rd.stage_done(f"{prefix}.finish"):
                    rows = [dict(r) for r in json.loads(str(self.rd.load_stage(f"{prefix}.finish")["rows"]))]
                else:
                    problem.layout = layout
                    rows = list(finish(self, problem, rows))
                    self.rd.save_stage(f"{prefix}.finish", {"rows": json.dumps(to_jsonable(rows), allow_nan=False)})
        finally:
            problem.layout = layout
        self._write_band(out, rows, par)
        return rows

    def band_result(self, problem: Any, value: float, name: str = "sigma_p_band") -> tuple[OptimizeResult, str]:
        """(result, stage) of the finished point at ``value`` of band ``name`` (``sigma_p_band`` or ``a_lo_band``):
        the stored stage (step2 if it ran, else step1) rebuilt with one evaluation at its endpoint.
        ``problem.layout`` is left with the band parameter fixed at ``value``; the caller restores it."""
        attr, prefix, default_par = self._BANDS[name]
        par = str(getattr(self.spec, f"{attr}_parameter", default_par))
        tag = f"{prefix}.{float(value):g}"
        stage = "step2" if self.rd.stage_done(f"{tag}.step2") else "step1"
        if not self.rd.stage_done(f"{tag}.{stage}"):
            raise StageRefused(f"band point {par} = {value:g} has not finished")
        problem.layout = self._band_fixed(problem.layout, par, float(value))
        W = np.asarray(self._stage_state(f"{tag}.{stage}")["W"], dtype=float)
        res = minimize(problem, Fixed(W), self._options(stage=f"{tag}.{stage}"), checkpoint=self.rd)
        return res, stage

    def margin_band(self) -> list[dict[str, Any]]:
        """The margin band stage (module docstring). Returns the rows of margin_band/band.csv."""
        spec = self.spec
        values = [float(v) for v in getattr(spec, "margin_band_values")]
        out = self.path / "margin_band"
        out.mkdir(exist_ok=True)
        rows: list[dict[str, Any]] = []
        theta_s: np.ndarray | None = None
        for v in values:
            tag = f"mband.{v:g}"
            if self.rd.stage_done(f"{tag}.row"):
                rows.append(dict(json.loads(str(self.rd.load_stage(f"{tag}.row")["row"]))))
                continue
            if theta_s is None:
                theta_s = self.start_point(self.problem("all"), "margin_band")
            log.info("margin_band: m* = %g", v)
            problem = spec.build_problem(self.cfg.nu, self.cfg.seed, self.cfg.threads, "all", margin=v)
            try:
                if not self.rd.stage_done(f"{tag}.step1"):
                    self._warm(problem, "margin_band.start")
                ts = theta_s
                W1 = self._saved_weight(f"{tag}.W1", lambda: self._binding(problem, problem.initial_weighting(ts)))
                r1 = minimize(problem, Fixed(W1), self._options(stage=f"{tag}.step1", theta0=theta_s,
                                                                  max_iterations=self._limit()), checkpoint=self.rd)
                rep1 = self._band_report(problem, r1, f"margin_band.{v:g}.step1")
                res, stage = r1, "step1"
                detected = [d["name"] for d in rep1.get("degeneracies", []) if d.get("detected")]
                ok2, why = TwoStep(W1).gate(detected)
                step2_note = "" if (r1.convergence.converged and ok2) else (
                    f"step 2 not started: step 1 ended {r1.convergence.value}" if not r1.convergence.converged
                    else why)
                if not step2_note and r1.evaluation is not None:
                    ev1 = r1.evaluation

                    def make_w2(problem: Any = problem, ev1: Any = ev1) -> np.ndarray:
                        return self._binding(problem, _block_diagonal_like(
                            weight_from_covariance(problem.covariance(ev1)), ev1.blocks,
                            getattr(problem, "coupled_blocks", ())))

                    W2 = self._saved_weight(f"{tag}.W2", make_w2)
                    res = minimize(problem, Fixed(W2), self._options(stage=f"{tag}.step2", theta0=r1.theta,
                                                                       max_iterations=self._limit()),
                                   checkpoint=self.rd)
                    self._band_report(problem, res, f"margin_band.{v:g}.step2")
                    stage = "step2"
                row: dict[str, Any] = {"label": "", "margin_target": v, "estimate_stage": stage,
                                       "step1_convergence": r1.convergence.value,
                                       "convergence": res.convergence.value, "q": float(res.q),
                                       "step2_note": step2_note}
                ev = res.evaluation
                if ev is not None and ev.ok:
                    calib = np.zeros(ev.g.size, dtype=bool)
                    for m in getattr(problem, "calibration", ()):
                        calib[ev.blocks[m.name]] = True
                    g, W = ev.g[~calib], np.asarray(res.W)[np.ix_(~calib, ~calib)]
                    row["J"] = float(problem.n * g @ W @ g)
                    row["J_weighting"] = "S^-1 (Hansen J)" if stage == "step2" else "W1 (not a J statistic)"
                    row.update(spec.margin_band_row(self, problem, res, v))
                else:
                    row["note"] = "no successful evaluation at the estimate"
                rows.append(row)
                self.rd.save_stage(f"{tag}.row", {"row": json.dumps(to_jsonable(row), allow_nan=False)})
                self._write_band(out, rows, "margin_target")
            finally:
                problem.model.close()
        self._write_band(out, rows, "margin_target")
        return rows

    def _band_report(self, problem: Any, result: OptimizeResult, stage: str) -> dict[str, Any]:
        """The diagnostics report of one band stage, written once (``report.<stage>``) and read back after."""
        if self.rd.stage_done(f"report.{stage}"):
            return json.loads((self.path / "reports" / stage / "report.json").read_text())
        rep = self.report(problem, result, stage)
        self.rd.save_stage(f"report.{stage}", {"done": True})
        return rep

    def _write_band(self, out: pathlib.Path, rows: list[dict[str, Any]], par: str) -> None:
        import csv
        import io

        ref = getattr(self.spec, "band_reference_rows", None)
        all_rows = list(rows) + (list(ref()) if callable(ref) else [])
        cols: list[str] = []
        for r in all_rows:
            cols += [c for c in r if c not in cols]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})
        atomic_write(out / "band.csv", buf.getvalue().encode())
        atomic_write(out / "band.json", json.dumps(to_jsonable({"parameter": par, "rows": all_rows}), indent=1,
                                                   allow_nan=False).encode())

    def run(self, stage: str) -> int:
        """Run one stage, or the chain for ``all``. Returns the process exit status."""
        stages = CHAIN if stage == "all" else (stage,)
        status = 0
        for s in stages:
            runtime.heartbeat(self.path, 0, stage=s)
            log.info("stage %s: start (nu %d, seed %d, run %s)", s, self.cfg.nu, self.cfg.seed, self.path)
            try:
                getattr(self, s)()
                log.info("stage %s: done", s)
            except StageRefused as exc:
                log.warning("stage %s refused: %s", s, exc)
                status = 3
                fallback = getattr(self.spec, "fallback_stage", None)
                if stage == "all" and s in ("extensive", "step1") and fallback and isinstance(exc, GateFailed):
                    log.info("stage %s: the gate failed; running the fallback stage %s", s, fallback)
                    runtime.heartbeat(self.path, 0, stage=fallback)
                    getattr(self, fallback)()
                    log.info("stage %s: done", fallback)
                if s in ("extensive", "step1") or stage != "all":
                    break
        return status


# ------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------


def _parser(default_nu: int, default_base: pathlib.Path, spec: Any = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Staged NFP-GMM estimation (nfp_harness.pipeline).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--nu", type=int, default=default_nu, help="draws per agent (draws per market = next power of two)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--results", type=pathlib.Path, default=default_base, help="base directory of run directories")
        if name == "run":
            p.add_argument("--stage", required=True, choices=STAGES + ("all",))
            p.add_argument("--gate-nu", type=int, default=4, help="nu of the gate run that step1 requires")
            p.add_argument("--threads", type=int, default=None, help="OpenMP threads in total (default all cores)")
            p.add_argument("--workers", type=int, default=4, help="parallel profile points in the gate")
            p.add_argument("--detach", action="store_true", help="re-execute in a new session, output to log.txt")
            p.add_argument("--new-run", action="store_true", help="archive the run directory and start over")
            p.add_argument("--band-values", default=None,
                           help="comma-separated subset of a band stage's values to run (default all); the run "
                                "directory is the same, so the remaining points can run later")
            p.add_argument("--band-step1-only", action="store_true",
                           help="band stages: run step 1 of each point and its report, no step 2 and no band row")
        add = getattr(spec, "add_arguments", None)
        if callable(add):
            add(p)
    return ap


def main(spec: Any, argv: Sequence[str] | None = None, default_nu: int = 16,
         default_base: str | pathlib.Path = "results") -> int:
    """main(spec, argv=None) -> exit status. ``run --stage <s> [--nu N] [--detach] [--new-run]`` and ``status``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser(default_nu, pathlib.Path(default_base), spec).parse_args(argv)
    configure = getattr(spec, "configure", None)
    if callable(configure):
        configure(args)
    cfg_base = args.results.resolve()
    if args.command == "status":
        path = run_dir_path(cfg_base, spec, args.nu, args.seed)
        print(json.dumps(runtime.status(path), indent=1, default=str))
        return 0

    cfg = RunConfig(base=cfg_base, nu=args.nu, seed=args.seed, threads=args.threads, workers=args.workers,
                    gate_nu=args.gate_nu, outer=str(getattr(args, "outer", None) or "nested"),
                    band_values=tuple(float(x) for x in str(args.band_values).split(",") if x.strip())
                    if getattr(args, "band_values", None) else (),
                    band_step1_only=bool(getattr(args, "band_step1_only", False)))
    path = run_dir_path(cfg_base, spec, cfg.nu, cfg.seed)
    st = runtime.status(path)
    if st.get("alive") and int(st.get("pid", -1)) != os.getpid():
        print(f"run directory {path} is in use by live pid {st['pid']} (stage {st.get('stage')}); not starting a "
              f"second process. Check it with `status`.", file=sys.stderr)
        return 4
    try:
        pipe = Pipeline(spec, cfg, new_run=args.new_run and os.environ.get(runtime.DETACHED_ENV) != "1")
    except SpecMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    if args.detach:
        child_argv = [sys.argv[0]] + [a for a in argv if a not in ("--detach", "--new-run")]
        (pipe.path / "cores").mkdir(parents=True, exist_ok=True)
        runtime.detach(pipe.path / "log.txt", argv=child_argv, cwd=pipe.path / "cores")
    logger = runtime.setup_logging(pipe.path / "log.txt")
    if os.environ.get(runtime.DETACHED_ENV) != "1" and not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    runtime.check_environment()
    crash = runtime.enable_crash_dumps(pipe.path)
    log.info("crash dumps: core limit %s, core file %s, Python stacks in %s", crash["core_limit"],
             crash["expected_core"], crash["faulthandler"])
    log.info("command: %s", " ".join([sys.executable, sys.argv[0], *argv]))
    try:
        return pipe.run(args.stage)
    finally:
        pipe.close()

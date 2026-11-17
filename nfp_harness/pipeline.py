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
- ``extensive``: only if the spec has ``extensive_free``: those parameters alone from the prefit point with
  W1 on all moments (refused unless the gate passed, as step1). step1 then starts from its endpoint.
- ``step1``: fixed W1 = ``initial_weighting(theta_prefit)`` from the extensive (else prefit) point. Refused unless the gate
  tagged ``spec.gate_required`` passed in the run with ``--gate-nu`` draws per agent (same seed).
- ``step2``: W2 = S(theta_hat_1)^{-1} from theta_hat_1. Refused unless step 1 converged and its report
  shows none of TwoStep's blocking degeneracies (DESIGN 4.5).
- ``post``: at the last finished estimate (step2, else step1): the report again and ``spec.post``, whose
  summary goes to ``post/summary.json``.
- ``stability``: at most 20 optimizer iterations from the estimate with ``2 nu`` draws per agent and new
  seeds, then the report with the integration block (DESIGN 5.7) and ``spec.headlines`` compared.

``--stage all`` runs extensive, step1, step2, post and stability in order; a refused stage is reported and the
chain continues with what it can do (post and stability fall back to the step-1 estimate). Resume by
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

__all__ = ["Pipeline", "RunConfig", "StageRefused", "STAGES", "CHAIN", "main", "run_dir_path", "run_key_options"]

log = logging.getLogger("nfp_harness.pipeline")

STAGES = ("prefit", "gate", "extensive", "step1", "step2", "post", "stability")
CHAIN = ("extensive", "step1", "step2", "post", "stability")
STABILITY_SEED_OFFSET = 100_000
STABILITY_ITERATIONS = 20


class StageRefused(RuntimeError):
    """A stage's precondition does not hold; the message says which and what to do."""


@dataclass(frozen=True)
class RunConfig:
    base: pathlib.Path
    nu: int
    seed: int = 0
    threads: int | None = None
    workers: int = 4
    gate_nu: int = 4


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
            self._problems[key] = self.spec.build_problem(key[1], key[2], self.cfg.threads, moments)
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
                                   "code_commit": code_commit(), "normalisations": list(self.spec.normalisations),
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
            raise StageRefused(f"the {' and '.join(tags)} gate failed ({gate_dir.parent / 'report.txt'}); the long "
                               f"estimation does not start (DESIGN 6.4)." + (
                                   f" Fallback: run --stage {self.spec.fallback_stage}."
                                   if getattr(self.spec, "fallback_stage", None) else ""))

    def _options(self, **kw: Any) -> OptimizerOptions:
        """OptimizerOptions for a stage: the spec's ``optimizer_options`` overrides (e.g. trust radii), then ``kw``."""
        return OptimizerOptions(**{**dict(getattr(self.spec, "optimizer_options", None) or {}), **kw})

    def _limit(self) -> int | None:
        """Trial-step limit of step1 and step2: ``spec.max_iterations`` if set (smoke tests), else none."""
        v = getattr(self.spec, "max_iterations", None)
        return None if v is None else int(v)

    def _w1(self, problem: Any, theta: np.ndarray) -> np.ndarray:
        return self._saved_weight("W1", lambda: problem.initial_weighting(theta))

    def extensive(self) -> np.ndarray:
        """Theta at the end of the extensive-margin stage: ``spec.extensive_free`` alone from the prefit
        point with W1 on all moments (SPEC 3.5a). The prefit theta when the spec has no such stage."""
        free_names = tuple(getattr(self.spec, "extensive_free", None) or ())
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
        theta_w = self.prefit()
        theta0 = self.extensive()
        problem = self.problem("all")
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
            return _block_diagonal_like(weight_from_covariance(problem.covariance(ev1)), ev1.blocks,
                                        getattr(problem, "coupled_blocks", ()))

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
                    gate_nu=args.gate_nu)
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

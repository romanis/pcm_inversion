"""The generic part of a paper's spec on ``nfp_harness.pipeline`` (DESIGN 6.0): the command-line options every paper
takes, their run-key and spec-hash entries, the starting theta, fixed parameters, warm starts from another run and the
assembly of a problem from the paper's declarative parts.

A paper subclasses ``PaperSpec`` and writes only what is its own:

- data: ``self.products`` and ``self.agents`` (loaded in ``__init__`` or ``configure_paper``), ``X1``, ``Z``, ``linear_names``,
  ``clusters``, ``price_range`` and the economics fields the diagnostics read (``firm_column``, ``groups``, ...);
- ``base_layout() -> ParameterLayout``: the nonlinear parameters with their starts, bounds and units, nothing applied;
- ``pcm_spec() -> PCMSpec``: the kernel's columns and solve settings;
- ``moment_list(moments, **kw) -> [Moment]``: the moments of a moment set (``"all"``, ``"prefit"``, ...);
- ``normalisations`` (a list of report lines) and ``payload() -> dict`` (what changes results, hashed);
- ``paper_run_key() -> dict``: the paper's own result-changing options; ``add_paper_arguments`` / ``configure_paper``;
- the pipeline's declarative fields (``prefit_free``, ``gate_profiles``, band values, ``post``, ``headlines``, ...).

Shared options (``add_arguments``, read by ``configure``), each in the run key only when it differs from its default, so
a run without it keeps its directory: ``--robustness`` (choices ``robustness_choices``; ``a_lo_robustness`` maps a
switch to a value of the bound), ``--a-lo`` (default ``a_lo_default``), ``--fix NAME=VALUE``, ``--start-theta`` and
``--start-theta-except``, ``--warm-delta``, ``--cold-draws``, ``--max-iterations``, ``--outer`` and ``--start-gate``.
``a_lo`` and the bound kind ``bound`` are always in the run key.

Declarative weighting fields read by the pipeline: ``concentrate_rows`` and ``start_concentrate_rows`` ({moment block:
(lo, hi)} rows of the block up-weighted by ``upweight`` in the prefit / gate concentration and in the gate-minimum
start; the start also zeroes the calibration rows), ``step2_weighting`` (a rule name of ``weighting.STEP2_RULES``).
``flat_directions`` names parameter sets that move together on a known flat direction (beside the utility-scale
direction every PCM spec has).
"""

from __future__ import annotations

import pathlib
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import read_stage_file, stage_theta
from .diagnostics.identification import proportional_direction, scale_direction
from .linear import LinearIV
from .params import ParameterLayout, parse_fixed
from .problem import NFPGMMProblem

__all__ = ["PaperSpec", "stage_label"]


def stage_label(path: str | pathlib.Path) -> str:
    """``<run dir name>/<stage file name>`` of a stored stage: how a run key names another run's stage."""
    p = pathlib.Path(path)
    return p.parent.parent.name + "/" + p.name


class PaperSpec:
    """Base class of a paper's spec (module docstring)."""

    name = "paper"
    firm_column = "firm_ids"
    groups: str | None = None
    group_other: str | None = None
    binary_columns: tuple[str, ...] = ()
    prefit_free: tuple[str, ...] = ()
    prefit_max_iterations = 10
    extensive_free: tuple[str, ...] = ()
    gate_required: Any = ()
    gate_concentrate_iterations = 20
    use_max_step = False
    optimizer_options: Mapping[str, Any] = {}
    #: the bound parameter's kind (``PCMSpec.bound_kind``), always in the run key
    bound_kind = "a"
    a_lo_default = 0.2
    #: robustness switches the paper accepts, the help text, switches that set the bound, and retired names -> reason
    robustness_choices: tuple[str, ...] = ()
    robustness_help = "robustness switch (repeatable)"
    a_lo_robustness: Mapping[str, float] = {}
    retired_robustness: Mapping[str, str] = {}
    #: parameters that keep the layout's start under --start-theta whatever the stage holds
    start_theta_keep: tuple[str, ...] = ()
    #: how --start-theta enters the run key: "source" (the stage's label) or "values" (label and the values taken)
    start_theta_key = "source"
    #: added to the pipeline's seed for the taste draws
    draw_seed_offset = 0
    #: prefit / gate and start concentration rows, {block: (lo, hi)}, up-weighted by ``upweight`` (module docstring)
    concentrate_rows: Mapping[str, tuple[int | None, int | None]] = {}
    start_concentrate_rows: Mapping[str, tuple[int | None, int | None]] = {}
    upweight = 1e4
    #: named flat directions beyond the utility scale: {name: parameters moving up in proportion}
    flat_directions: Mapping[str, Sequence[str]] = {}
    #: the problem's moment sets that only evaluate (no estimation)
    evaluate_only_moments: tuple[str, ...] = ()
    prefit_variant: str | None = None

    def __init__(self) -> None:
        self.robustness: tuple[str, ...] = ()
        self.a_lo = float(self.a_lo_default)
        self.a_lo_switch: str | None = None   # the robustness switch that set a_lo, if any
        self.fixed: dict[str, float] = {}
        self.start_theta_path: str | None = None
        self.start_theta_except: tuple[str, ...] = ()
        self.warm_delta: str | None = None
        self.cold_draws = 0
        self.max_iterations: int | None = None
        self.outer = "nested"
        self.start_gate: str | None = None

    # ------------------------------------------------------------------ hooks a paper fills in

    def base_layout(self) -> ParameterLayout:
        raise NotImplementedError

    def pcm_spec(self) -> Any:
        raise NotImplementedError

    def moment_list(self, moments: str, **kw: Any) -> list[Any]:
        raise NotImplementedError

    def payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def paper_run_key(self) -> dict[str, Any]:
        return {}

    def add_paper_arguments(self, parser: Any) -> None:
        pass

    def configure_paper(self, args: Any) -> None:
        pass

    def check_robustness(self, robustness: tuple[str, ...]) -> None:
        """Paper-specific exclusions between switches (raise ValueError)."""

    def prefit_variants(self) -> tuple[str, ...]:
        return ()

    def set_prefit_variant(self, name: str | None) -> None:
        if name is not None and name not in self.prefit_variants():
            raise ValueError(f"unknown prefit variant {name!r}")
        self.prefit_variant = name

    def layout_for(self, layout: ParameterLayout, moments: str) -> ParameterLayout:
        """The layout of a moment set's problem (default: the spec's layout)."""
        return layout

    def problem_normalisations(self, moments: str, **kw: Any) -> list[str]:
        return list(self.normalisations)

    @property
    def normalisations(self) -> list[str]:
        return []

    # ------------------------------------------------------------------ shared options

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--robustness", action="append", choices=self.robustness_choices or None, default=None,
                            help=self.robustness_help)
        parser.add_argument("--a-lo", type=float, default=None,
                            help=f"bound on the unobserved factor a_i (default {self.a_lo_default:g})")
        parser.add_argument("--fix", action="append", default=None, metavar="NAME=VALUE",
                            help="fix a parameter at a value for every stage, e.g. --fix sigma_p=3.0 (repeatable; the "
                                 "bounds widen to the value); part of the run key")
        parser.add_argument("--start-theta", default=None,
                            help="a stored stage (<run dir>/stages/<stage>.npz) whose theta becomes the layout starts "
                                 "(clipped to the bounds), matched by parameter name through the run's history.csv "
                                 "(positional when the run has none and the layouts have the same length); a free "
                                 "parameter the stage does not have keeps its start"
                                 + (f", and so do {', '.join(self.start_theta_keep)}" if self.start_theta_keep else "")
                                 + "; part of the run key")
        parser.add_argument("--start-theta-except", nargs="+", default=None, metavar="PARAMETER",
                            help="free parameters that keep this layout's starts under --start-theta; part of the run "
                                 "key")
        parser.add_argument("--warm-delta", default=None,
                            help="a stored stage (<run dir>/stages/<stage>.npz) of a run on other draws whose delta "
                                 "warm-starts every market it holds, at that stage's theta (another bound is a warm "
                                 "start too: the kernel's theta path moves it); part of the run key")
        parser.add_argument("--max-iterations", type=int, default=None,
                            help="trial-step limit of step1 and step2 (smoke runs); part of the run key")
        parser.add_argument("--cold-draws", type=int, default=0,
                            help="draw homotopy for markets with no warm start: invert first on every k-th draw (at "
                                 "least this many draws), then warm-start each finer level (PCMModel cold_draws; 0: "
                                 "off); part of the run key when set")
        parser.add_argument("--outer", choices=("nested", "mpec"), default="nested",
                            help="outer solver: nested (default) or mpec (IPOPT on theta and delta, demand-only "
                                 "moments; refused for any stage whose problem has the supply block or a margin row; "
                                 "needs the .envs/mpec python, see nfp_harness/mpec.py); part of the run key when set")
        parser.add_argument("--start-gate", default=None,
                            help="gate directory or profile.csv whose minimum starts step1 and margin_band when the "
                                 "spec starts from the gate; default the gate of the --gate-nu run")
        self.add_paper_arguments(parser)

    def configure(self, args: Any) -> None:
        """Read the shared options, then the paper's (``configure_paper``)."""
        self.set_robustness(getattr(args, "robustness", None))
        if self.a_lo_switch is None:
            self.a_lo = float(self.a_lo_default)
        self.set_a_lo(getattr(args, "a_lo", None))
        self.fixed = parse_fixed(getattr(args, "fix", None))
        stt = getattr(args, "start_theta", None)
        self.start_theta_path = str(pathlib.Path(stt).resolve()) if stt else None
        self.start_theta_except = tuple(sorted(set(getattr(args, "start_theta_except", None) or ())))
        wd = getattr(args, "warm_delta", None)
        self.warm_delta = str(pathlib.Path(wd).resolve()) if wd else None
        mi = getattr(args, "max_iterations", None)
        self.max_iterations = None if mi is None else int(mi)
        self.cold_draws = int(getattr(args, "cold_draws", None) or 0)
        self.outer = str(getattr(args, "outer", None) or "nested")
        sg = getattr(args, "start_gate", None)
        self.start_gate = str(pathlib.Path(sg).resolve()) if sg else None
        self.configure_paper(args)

    def set_robustness(self, robustness: Sequence[str] | None) -> None:
        r = tuple(sorted(set(robustness or ())))
        retired = [self.retired_robustness[x] for x in r if x in self.retired_robustness]
        if retired:
            raise ValueError("; ".join(retired))
        unknown = [x for x in r if x not in self.robustness_choices]
        if unknown:
            raise ValueError(f"unknown robustness switches {unknown}; known: {self.robustness_choices}")
        self.check_robustness(r)
        bounds = [x for x in r if x in self.a_lo_robustness]
        if len(bounds) > 1:
            raise ValueError(f"robustness switches {bounds} set a_lo twice; pick one")
        self.robustness = r
        if bounds:
            self.a_lo = float(self.a_lo_robustness[bounds[0]])
        elif self.a_lo_switch is not None:   # the switch that set the bound is gone
            self.a_lo = float(self.a_lo_default)
        self.a_lo_switch = bounds[0] if bounds else None

    def set_a_lo(self, a_lo: float | None) -> None:
        """The bound from --a-lo (None: the robustness switch's value, else ``a_lo_default``)."""
        if a_lo is None:
            return
        if self.a_lo_switch is not None:
            raise ValueError(f"--a-lo and robustness {self.a_lo_switch} both set a_lo; pick one")
        self.a_lo = float(a_lo)

    # ------------------------------------------------------------------ run key and hash

    def start_theta_values(self) -> dict[str, float]:
        """{parameter: value} of the --start-theta stage, named by its run's history (positional on the base layout
        when it has no names)."""
        assert self.start_theta_path is not None
        return stage_theta(self.start_theta_path, self.base_layout().names)

    @property
    def start_theta_source(self) -> str | None:
        return stage_label(self.start_theta_path) if self.start_theta_path else None

    def run_key(self) -> dict[str, Any]:
        """Result-changing options beyond nu and seed: the paper's (``paper_run_key``) and the shared ones (module
        docstring)."""
        out = dict(self.paper_run_key())
        if self.robustness:
            out["robustness"] = list(self.robustness)
        out["a_lo"] = self.a_lo
        out["bound"] = self.bound_kind
        if self.warm_delta:
            out["warm_delta"] = stage_label(self.warm_delta)
        if self.start_theta_path:
            if self.start_theta_key == "values":
                keep = set(self.start_theta_keep)
                vals = {n: v for n, v in self.start_theta_values().items() if n not in keep}
                out["start_theta"] = {"source": self.start_theta_source,
                                      "theta": {k: float(v).hex() for k, v in sorted(vals.items())}}
            else:
                out["start_theta"] = self.start_theta_source
            if self.start_theta_except:
                out["start_theta_except"] = list(self.start_theta_except)
        if self.fixed:
            out["fix"] = dict(sorted(self.fixed.items()))
        if self.max_iterations is not None:
            out["max_iterations"] = self.max_iterations
        if self.cold_draws:
            out["cold_draws"] = self.cold_draws
        if self.outer != "nested":
            out["outer"] = self.outer
        return out

    def hash_payload(self) -> dict[str, Any]:
        """The paper's ``payload()`` and the shared options that are set and that the payload does not already hold."""
        out = self.payload()
        shared = {"fix": dict(sorted(self.fixed.items())) if self.fixed else None,
                  "cold_draws": self.cold_draws or None, "warm_delta": self.warm_delta,
                  "max_iterations": self.max_iterations,
                  "start_theta_except": list(self.start_theta_except) if self.start_theta_except else None}
        out.update({k: v for k, v in shared.items() if v is not None and k not in out})
        return out

    # ------------------------------------------------------------------ layout and problems

    def layout(self) -> ParameterLayout:
        """``base_layout()`` with the --start-theta starts (free parameters it names, except ``start_theta_keep`` and
        --start-theta-except, clipped to their bounds) and the --fix values applied."""
        lay = self.base_layout()
        if self.start_theta_path:
            bad = sorted(set(self.start_theta_except) - set(lay.names))
            if bad:
                raise ValueError(f"--start-theta-except: {bad} are not parameters of this layout")
            src = self.start_theta_values()
            keep = set(self.start_theta_keep) | set(self.start_theta_except)
            lay = lay.replace({n: {"start": float(np.clip(src[n], sp.lower, sp.upper))}
                               for n, sp in zip(lay.names, lay.specs) if sp.free and n in src and n not in keep})
        if self.fixed:
            bad = sorted(set(self.fixed) - set(lay.names))
            if bad:
                raise ValueError(f"--fix: {bad} are not parameters of this layout")
            lay = lay.fixed(self.fixed)
        return lay

    def hash_layout(self) -> ParameterLayout:
        """The layout as hashed: with no prefit variant chosen, so a run's hash does not change when its prefit picks
        one."""
        chosen = self.prefit_variant
        self.prefit_variant = None
        try:
            return self.layout()
        finally:
            self.prefit_variant = chosen

    def linear(self) -> LinearIV:
        return LinearIV(self.X1, self.Z, names=self.linear_names)

    def build_model(self, layout: ParameterLayout, nu: int, seed: int, threads: int | None) -> Any:
        """The PCM model on the paper's products and agents, warm-started from --warm-delta when given."""
        from .models.pcm import PCMModel

        model = PCMModel(self.pcm_spec(), {"products": self.products, "agents": self.agents}, layout, nu,
                         self.draw_seed_offset + seed, total_threads=threads, antithetic=self.antithetic_draws,
                         cold_draws=self.cold_draws, taste_nodes=self.taste_nodes(nu))
        if self.warm_delta:
            delta_w, theta_w = self.read_warm_delta()
            model.warm_from_delta(delta_w, theta_w if theta_w is not None and theta_w.size == layout.P
                                  else layout.starts())
        return model

    @property
    def antithetic_draws(self) -> Sequence[str] | bool:
        return ()

    def taste_nodes(self, nu: int) -> tuple[np.ndarray, np.ndarray] | None:
        """A quadrature rule (nodes, weights) over the taste draw dimensions at ``nu`` (``PCMModel`` taste_nodes),
        or None (the default) for the Sobol draws."""
        return None

    def build_problem(self, nu: int, seed: int, threads: int | None, moments: str = "all",
                      layout_changes: dict[str, Any] | None = None, **kw: Any) -> NFPGMMProblem:
        """The problem of moment set ``moments`` (``moment_list``), with ``layout_changes`` on the layout; ``kw`` goes to
        ``moment_list`` and ``problem_normalisations`` (e.g. a calibration target)."""
        layout = self.layout_for(self.layout(), moments)
        if layout_changes:
            layout = layout.replace(layout_changes)
        model = self.build_model(layout, nu, seed, threads)
        ms = self.moment_list(moments, **kw)
        prob = NFPGMMProblem(model, ms, self.linear_for(moments), layout, max_step=self.use_max_step,
                             evaluate_only=moments in self.evaluate_only_moments)
        prob.normalisations = self.problem_normalisations(moments, **kw)   # type: ignore[attr-defined]
        self.finish_problem(prob, moments)
        return prob

    def linear_for(self, moments: str) -> LinearIV:
        """The linear IV of moment set ``moments`` (default ``linear()`` for every set; a paper whose sets differ in
        the aggregate block's instruments overrides it)."""
        return self.linear()

    def finish_problem(self, problem: NFPGMMProblem, moments: str) -> None:
        """Hook run on every problem ``build_problem`` returns (default: nothing), e.g. ``problem.set_joint(mode)``
        for a joint product-level weighting."""

    def market_rows(self, market: Any) -> np.ndarray:
        """Rows of ``self.products`` of ``market`` in the PCM model's order (price order, stable)."""
        ps = self.pcm_spec()
        p = self.products
        pos = np.flatnonzero(p[ps.market_column].to_numpy() == market)
        return pos[np.argsort(p[ps.price_column].to_numpy(float)[pos], kind="stable")]

    def read_warm_delta(self) -> tuple[np.ndarray, np.ndarray | None]:
        """((n,) delta in this spec's row order, the stage's theta) from --warm-delta: the stage's stored model state
        matched by market, and the theta it was solved at in this layout's order (by parameter name; a taste or scale
        term the stage's layout lacks was 0 there, any other missing parameter takes this layout's start)."""
        assert self.warm_delta is not None
        st = read_stage_file(self.warm_delta)
        model = (st.get("problem_state") or {}).get("model") or {}
        ids = [str(x) for x in model.get("market_ids", [])]
        out = np.full(len(self.products), np.nan)
        mids = self.products[self.pcm_spec().market_column].to_numpy()
        for m in dict.fromkeys(mids.tolist()):
            if str(m) not in ids:
                continue
            k = ids.index(str(m))
            rows = self.market_rows(m)
            d = np.asarray(model["delta"][k], dtype=float)
            if d.size == rows.size:
                out[rows] = d
        if not np.isfinite(out[np.isin(mids.astype(str), ids)]).any():
            raise ValueError(f"--warm-delta {self.warm_delta}: no usable market state")
        if st.get("theta") is None:
            return out, None
        try:
            src = stage_theta(self.warm_delta, self.hash_layout().names)
        except ValueError:   # no names and another length: the model warms at the layout starts
            return out, None
        lay = self.layout()
        return out, np.array([src.get(n, 0.0 if sp.role in ("taste", "scale") else sp.start)
                              for n, sp in zip(lay.names, lay.specs)])

    # ------------------------------------------------------------------ diagnostics and postestimation

    def postestimation(self, model: Any, theta: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
        """``pcm_post.postestimation`` with the spec's firm and group columns."""
        from .models.pcm_post import postestimation

        return postestimation(model, theta, delta, firm_ids=self.firm_column, groups=self.groups,
                              group_other=self.group_other)

    def named_directions(self, theta: np.ndarray, units: np.ndarray) -> dict[str, np.ndarray]:
        """Known flat directions in natural units at theta (DESIGN 5.2): the utility-scale direction (sigma_p up, every
        taste spread down in proportion, LEARNINGS 1) and each of ``flat_directions``."""
        names = self.layout().names
        out = {"scale_direction": scale_direction(names, theta, units)}
        for k, members in self.flat_directions.items():
            out[k] = proportional_direction(names, theta, units, members)
        return out

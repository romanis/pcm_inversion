"""Ray and grid profiles of the objective, and the shape classifier used as a gate (DESIGN 6.4, 7 WP9a).

A profile evaluates a ``Problem`` on a grid of parameter values. The grid is the product of axes:

- ``ParamAxis(name, values)`` sets one named parameter;
- ``RayAxis(direction, values, name="lambda", origin=None)`` sets the parameters named in
  ``direction`` to ``origin + lambda * direction``. ``origin=None`` takes them from theta0, so the
  points are ``theta0 + lambda d`` (a ray through theta0); ``origin=0.0`` gives ``lambda d`` (a ray
  from zero in that subspace, DESIGN 6.4: ``sigma = lambda x SIG_REF``).

Parameters not touched by an axis stay at theta0 (``concentrate=False``), or are re-optimised at
every grid point by ``optimizer.minimize`` with the touched parameters (and ``hold``) fixed
(``concentrate=True``, the concentrated profile). Concentrated points are warm-started from the
nearest completed grid point unless ``cold=True``.

At every point the profile records the objective under each weighting mode side by side:
``q_fixed = n g' W g`` with the profile's fixed W (``weighting``, default the problem's current W)
and ``q_cue = n g' S(theta)^{-1} g`` with S from ``problem.covariance`` at the same point (scale
free, LEARNINGS 1), the per-block parts of ``q_fixed``, ``sd(xi)``, and named economic summaries
from an ``economics(problem, evaluation) -> {name: value}`` callback.

``classify_profile(x, q)`` names the shape of a one-dimensional profile: MONOTONE_DECREASING,
MONOTONE_INCREASING, INTERIOR_MINIMUM or FLAT, with the evidence. ``NotMonotoneDecreasing`` is the
DESIGN 6.4 pass rule built on it.

Run directory (``run_dir``): ``profile.json`` (configuration and its hash), ``points/pNNNNN.json``
(one per completed grid point, written atomically) plus ``points/pNNNNN.state.npz`` (the problem's
warm-start state, for chained profiles), ``profile.csv`` and ``profile.txt`` (rewritten after every
point). Rerunning the same call resumes: completed points are read back, the rest are computed. A
directory written by a different configuration is refused (``ProfileMismatchError``) unless
``new_run=True``, which renames it to ``<name>.old-<UTC timestamp>``.

Parallelism. A model that parallelises internally (the PCM runs markets in threads with OpenMP teams)
must not be oversubscribed, so points run one after another on the given problem unless the
problem (or its ``model``) has a ``spare_capacity() -> int`` hook saying how many evaluations may
run at once, or ``workers`` is passed. More than one worker needs ``problem_factory(worker, threads)
-> Problem``, which builds an independent problem for each worker with ``threads`` =
``os.cpu_count() // workers``; points then run in a thread pool (the kernels release the GIL). Only
cold profiles run in parallel, since a chained profile depends on the order of its points.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import datetime as _dt
import enum
import io
import itertools
import json
import logging
import math
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .checkpoint import _join, _split, atomic_write, code_commit, hash_options
from .types import Evaluation, Problem

__all__ = [
    "ParamAxis",
    "RayAxis",
    "ProfileShape",
    "ShapeVerdict",
    "ProfilePoint",
    "ProfileResult",
    "ProfileMismatchError",
    "NotMonotoneDecreasing",
    "RisesAtBothEnds",
    "classify_profile",
    "profile",
    "ray_profile",
    "grid_profile",
]

log = logging.getLogger("nfp_harness.profile")

PROFILE_FORMAT = 1


class ProfileMismatchError(RuntimeError):
    """The run directory holds a profile with another configuration; pass new_run=True to start over."""


# ------------------------------------------------------------------------------------------------
# axes
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamAxis:
    """One named parameter set to each of ``values``. The axis is named after the parameter."""

    parameter: str
    values: tuple[float, ...]

    def __init__(self, parameter: str, values: Iterable[float]) -> None:
        object.__setattr__(self, "parameter", str(parameter))
        object.__setattr__(self, "values", tuple(float(v) for v in values))
        if not self.values:
            raise ValueError(f"axis {parameter!r} has no values")

    @property
    def name(self) -> str:
        return self.parameter

    def parameters(self) -> tuple[str, ...]:
        return (self.parameter,)

    def apply(self, theta: np.ndarray, theta0: np.ndarray, index: Mapping[str, int], value: float) -> None:
        theta[index[self.parameter]] = value

    def describe(self) -> dict[str, Any]:
        return {"kind": "param", "name": self.name, "parameter": self.parameter, "values": list(self.values)}


@dataclass(frozen=True)
class RayAxis:
    """Parameters in ``direction`` set to ``origin + value * direction`` for each value.

    ``origin`` None takes the origin from theta0 (points ``theta0 + lambda d``); a float uses it for
    every parameter of the ray; a mapping gives it per parameter.
    """

    direction: tuple[tuple[str, float], ...]
    values: tuple[float, ...]
    name: str = "lambda"
    origin: tuple[tuple[str, float], ...] | None = None

    def __init__(self, direction: Mapping[str, float], values: Iterable[float], name: str = "lambda",
                 origin: float | Mapping[str, float] | None = None) -> None:
        d = tuple((str(k), float(v)) for k, v in direction.items())
        if not d:
            raise ValueError("ray direction is empty")
        object.__setattr__(self, "direction", d)
        object.__setattr__(self, "values", tuple(float(v) for v in values))
        object.__setattr__(self, "name", str(name))
        if origin is None:
            o = None
        elif isinstance(origin, Mapping):
            missing = [k for k, _ in d if k not in origin]
            if missing:
                raise ValueError(f"ray origin lacks {missing}")
            o = tuple((k, float(origin[k])) for k, _ in d)
        else:
            o = tuple((k, float(origin)) for k, _ in d)
        object.__setattr__(self, "origin", o)
        if not self.values:
            raise ValueError(f"axis {name!r} has no values")

    def parameters(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.direction)

    def apply(self, theta: np.ndarray, theta0: np.ndarray, index: Mapping[str, int], value: float) -> None:
        origin = dict(self.origin) if self.origin is not None else None
        for k, d in self.direction:
            base = theta0[index[k]] if origin is None else origin[k]
            theta[index[k]] = base + value * d

    def describe(self) -> dict[str, Any]:
        return {"kind": "ray", "name": self.name, "direction": [list(p) for p in self.direction],
                "origin": None if self.origin is None else [list(p) for p in self.origin],
                "values": list(self.values)}


Axis = ParamAxis | RayAxis


# ------------------------------------------------------------------------------------------------
# shape classifier
# ------------------------------------------------------------------------------------------------


class ProfileShape(enum.Enum):
    """Shape of a one-dimensional profile over its grid."""

    MONOTONE_DECREASING = "MONOTONE_DECREASING"  # minimum at the right end of the grid
    MONOTONE_INCREASING = "MONOTONE_INCREASING"  # minimum at the left end
    INTERIOR_MINIMUM = "INTERIOR_MINIMUM"  # q rises from the minimum towards both ends
    FLAT = "FLAT"  # range of q within the tolerance, or fewer than 3 evaluated points


@dataclass(frozen=True)
class ShapeVerdict:
    """Shape of q(x) with the evidence it rests on."""

    shape: ProfileShape
    x_min: float  # grid value at the minimum (NaN without evaluated points)
    q_min: float
    rise_left: float  # q at the left end minus q_min
    rise_right: float  # q at the right end minus q_min
    tol: float  # rises at or below tol count as zero
    strict: bool  # every step between evaluated points moves q in one direction (or none)
    evidence: dict[str, Any]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"shape": self.shape.value, "x_min": self.x_min, "q_min": self.q_min, "rise_left": self.rise_left,
                "rise_right": self.rise_right, "tol": self.tol, "strict": self.strict,
                "evidence": _plain(self.evidence), "message": self.message}


def classify_profile(x: Sequence[float], q: Sequence[float], *, rel_tol: float = 1e-6, abs_tol: float = 1e-10,
                     label: str = "q", axis: str = "x") -> ShapeVerdict:
    """classify_profile(x, q, rel_tol=1e-6, abs_tol=1e-10) -> ShapeVerdict.

    Points with non-finite q (failed evaluations) are left out and counted in the evidence. With
    ``tol = max(abs_tol, rel_tol (1 + |q_min|))`` over the remaining points sorted by x:

    - FLAT if fewer than 3 points remain or ``max q - min q <= tol``;
    - INTERIOR_MINIMUM if q at both ends exceeds q_min by more than tol;
    - MONOTONE_DECREASING if q at the right end is within tol of q_min (q falls toward the right
      end; ``strict`` says whether every step falls);
    - MONOTONE_INCREASING otherwise (q at the left end is within tol of q_min).

    The classifier names the shape only. Whether a rise is large enough to matter (e.g. 3.84, the
    5% chi-square(1) value) is a pass rule's business (``NotMonotoneDecreasing``).
    """
    xa = np.asarray(x, dtype=float).ravel()
    qa = np.asarray(q, dtype=float).ravel()
    if xa.shape != qa.shape:
        raise ValueError(f"x has {xa.size} values, q has {qa.size}")
    order = np.argsort(xa, kind="stable")
    xa, qa = xa[order], qa[order]
    ok = np.isfinite(qa)
    failed = [float(v) for v in xa[~ok]]
    xs, qs = xa[ok], qa[ok]
    evidence: dict[str, Any] = {"axis": axis, "x": [float(v) for v in xs], "q": [float(v) for v in qs],
                                "failed_x": failed, "n_points": int(xa.size), "n_failed": len(failed)}
    if xs.size < 3:
        evidence["insufficient_points"] = True
        msg = (f"{label} along {axis}: {xs.size} evaluated point(s) of {xa.size}, fewer than the 3 needed to tell "
               f"a shape; reported as FLAT.")
        nan = float("nan")
        return ShapeVerdict(ProfileShape.FLAT, float(xs[np.argmin(qs)]) if xs.size else nan,
                            float(qs.min()) if qs.size else nan, nan, nan, nan, False, evidence, msg)
    k = int(np.argmin(qs))
    q_min = float(qs[k])
    tol = max(abs_tol, rel_tol * (1.0 + abs(q_min)))
    rise_left, rise_right = float(qs[0] - q_min), float(qs[-1] - q_min)
    rng = float(qs.max() - q_min)
    steps = np.diff(qs)
    n_up, n_down = int(np.sum(steps > tol)), int(np.sum(steps < -tol))
    evidence.update(argmin=k, range=rng, steps_up=n_up, steps_down=n_down)
    where = f"{axis} = {xs[k]:.6g}"
    ends = (f"{label} = {qs[0]:.6g} at {axis} = {xs[0]:.6g}, {q_min:.6g} at the minimum ({where}), "
            f"{qs[-1]:.6g} at {axis} = {xs[-1]:.6g}")
    if rng <= tol:
        shape, strict = ProfileShape.FLAT, True
        msg = f"{label} along {axis} is FLAT: range {rng:.3g} within tolerance {tol:.3g} ({ends})."
    elif rise_left > tol and rise_right > tol:
        shape = ProfileShape.INTERIOR_MINIMUM
        strict = n_down == 0 or n_up == 0 or bool(np.all(steps[:k] <= tol) and np.all(steps[k:] >= -tol))
        msg = (f"{label} along {axis} has an INTERIOR_MINIMUM at {where}: rises {rise_left:.4g} to the left end "
               f"and {rise_right:.4g} to the right end ({ends}).")
    elif rise_right <= tol:
        shape, strict = ProfileShape.MONOTONE_DECREASING, n_up == 0
        msg = (f"{label} along {axis} is MONOTONE_DECREASING: it falls by {rise_left:.4g} toward the right end of "
               f"the grid ({ends}); every step falls: {'yes' if strict else 'no'}.")
    else:
        shape, strict = ProfileShape.MONOTONE_INCREASING, n_down == 0
        msg = (f"{label} along {axis} is MONOTONE_INCREASING: it rises by {rise_right:.4g} toward the right end of "
               f"the grid ({ends}); every step rises: {'yes' if strict else 'no'}.")
    if failed:
        msg += f" {len(failed)} grid point(s) failed to evaluate and are left out: {axis} = {failed}."
    return ShapeVerdict(shape, float(xs[k]), q_min, rise_left, rise_right, tol, bool(strict), evidence, msg)


# ------------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------------


@dataclass(eq=False)
class ProfilePoint:
    """One grid point: where it is, what was evaluated there and how."""

    index: int  # position in the grid's iteration order (last axis fastest)
    grid_index: tuple[int, ...]  # position along each axis
    coords: dict[str, float]  # axis name -> value
    theta: np.ndarray  # (P,) the evaluated theta (the optimum for concentrated points)
    status: str  # EvalStatus value, or the optimizer's convergence class for concentrated points
    values: dict[str, float]  # q_fixed, q_cue, q_fixed:<block>, sd_xi, <economics names>
    seconds: float
    evaluations: int = 1
    convergence: str | None = None  # optimizer class (concentrated points only)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" or self.convergence in ("CONVERGED", "CONVERGED_AT_CAP", "CONVERGED_NOISE")

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "grid_index": list(self.grid_index), "coords": _floats_out(self.coords),
                "theta": [_float_out(float(v)) for v in self.theta], "status": self.status,
                "values": _floats_out(self.values), "seconds": self.seconds, "evaluations": self.evaluations,
                "convergence": self.convergence, "note": self.note}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> ProfilePoint:
        return cls(index=int(d["index"]), grid_index=tuple(int(i) for i in d["grid_index"]),
                   coords=_floats_in(d["coords"]), theta=np.array([_float_in(v) for v in d["theta"]], dtype=float),
                   status=str(d["status"]), values=_floats_in(d["values"]), seconds=float(d["seconds"]),
                   evaluations=int(d["evaluations"]), convergence=d.get("convergence"), note=str(d.get("note", "")))


@dataclass(eq=False)
class ProfileResult:
    """All grid points of a profile, in grid order, with helpers to read curves and classify them."""

    axes: list[Axis]
    names: list[str]  # parameter names (theta order)
    points: list[ProfilePoint]
    columns: list[str]  # value columns, in table order
    concentrate: bool
    objective: str  # which q a concentrated profile minimises ("fixed" or "cue")
    run_dir: pathlib.Path | None = None
    resumed_points: int = 0  # points read back from the run directory instead of computed
    workers: int = 1
    verdict: tuple[bool, str] | None = None  # (passed, message) of the pass rule, if one was given
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def axis_names(self) -> list[str]:
        return [a.name for a in self.axes]

    def column(self, name: str) -> np.ndarray:
        """Values of a column (axis name, value column or parameter name) over the points, NaN where missing."""
        if name in self.axis_names:
            return np.array([p.coords[name] for p in self.points])
        if name in self.names and name not in self.columns:
            k = self.names.index(name)
            return np.array([p.theta[k] for p in self.points])
        return np.array([p.values.get(name, float("nan")) for p in self.points])

    def rows(self) -> list[dict[str, Any]]:
        """One dict per point: index, axis values, status, value columns, theta, seconds, convergence, note."""
        out = []
        for p in self.points:
            row: dict[str, Any] = {"index": p.index}
            row.update({a: p.coords[a] for a in self.axis_names})
            row["status"] = p.status
            row.update({c: p.values.get(c, float("nan")) for c in self.columns})
            row.update({f"theta:{n}": float(v) for n, v in zip(self.names, p.theta)})
            row.update(seconds=p.seconds, evaluations=p.evaluations, convergence=p.convergence or "", note=p.note)
            out.append(row)
        return out

    def classify(self, along: str, column: str = "q_fixed", by: Sequence[str] | None = None, **kw: Any
                 ) -> dict[tuple[float, ...], ShapeVerdict]:
        """classify(along, column="q_fixed", by=None, **classify_profile kwargs) -> {slice key: ShapeVerdict}.

        One verdict per combination of the other axes (``by``, default all axes except ``along``);
        the key is the tuple of their values, () for a one-axis profile.
        """
        if along not in self.axis_names:
            raise KeyError(f"no axis {along!r}; axes are {self.axis_names}")
        by = [a for a in self.axis_names if a != along] if by is None else list(by)
        unknown = [a for a in by if a not in self.axis_names or a == along]
        if unknown:
            raise KeyError(f"cannot slice by {unknown}; axes are {self.axis_names}")
        x, q = self.column(along), self.column(column)
        keys = [tuple(p.coords[a] for a in by) for p in self.points]
        out: dict[tuple[float, ...], ShapeVerdict] = {}
        for key in dict.fromkeys(keys):
            sel = np.array([k == key for k in keys])
            label = column + ("" if not by else " (" + ", ".join(f"{a} = {v:g}" for a, v in zip(by, key)) + ")")
            out[key] = classify_profile(x[sel], q[sel], label=label, axis=along, **kw)
        return out

    def table(self, along: str | None = None, columns: Sequence[str] | None = None) -> str:
        """Fixed-width text table of the points, then the shape of q_fixed and q_cue along ``along``
        (default the last axis) and the pass rule's verdict."""
        cols = list(self.columns if columns is None else columns)
        head = self.axis_names + ["status"] + cols + ["sec"]
        lines = []
        widths = [max(10, len(h)) for h in head]
        lines.append("  ".join(h.rjust(w) for h, w in zip(head, widths)))
        for p in self.points:
            cells = [_fmt(p.coords[a]) for a in self.axis_names]
            cells.append(p.convergence if (self.concentrate and p.convergence) else p.status)
            cells += [_fmt(p.values.get(c, float("nan"))) for c in cols]
            cells.append(f"{p.seconds:.1f}")
            lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
        text = "\n".join(lines) + "\n"
        along = along or self.axis_names[-1]
        if len(self.points) == math.prod(len(a.values) for a in self.axes):
            text += "\n"
            for column in ("q_fixed", "q_cue"):
                if column in self.columns:
                    for v in self.classify(along, column).values():
                        text += v.message + "\n"
        if self.verdict is not None:
            text += ("\nPASS: " if self.verdict[0] else "\nFAIL: ") + self.verdict[1] + "\n"
        return text


class NotMonotoneDecreasing:
    """Pass rule of DESIGN 6.4: along ``along``, for every slice of the other axes, q is not monotone
    decreasing over the grid: either its minimum is interior, or q at the right end of the grid
    exceeds the minimum by at least ``rise`` (3.84, the 5% chi-square(1) value).

    Calling the rule on a ``ProfileResult`` returns ``(passed, message)``; ``message`` lists each
    slice's shape, and on failure starts with ``fail_message``. ``only`` ({axis: value}) restricts the rule
    to the slices with those values of the ``by`` axes (e.g. ``{"lambda": 1.0}``).
    """

    rule_name = "NotMonotoneDecreasing"

    def __init__(self, along: str, column: str = "q_fixed", by: Sequence[str] | None = None, rise: float = 3.84,
                 fail_message: str = "Profile check failed: q is still monotone in {along}.",
                 only: Mapping[str, float] | None = None, **classify_kw: Any) -> None:
        self.along, self.column, self.by, self.rise = along, column, by, float(rise)
        self.fail_message = fail_message
        self.only = dict(only or {})
        self.classify_kw = classify_kw

    def verdicts(self, result: ProfileResult) -> dict[tuple[float, ...], ShapeVerdict]:
        out = result.classify(self.along, self.column, self.by, **self.classify_kw)
        if not self.only:
            return out
        by = [a for a in result.axis_names if a != self.along] if self.by is None else list(self.by)
        unknown = [a for a in self.only if a not in by]
        if unknown:
            raise KeyError(f"{self.rule_name}: 'only' names axes {unknown} that are not slicing axes {by}")
        return {key: v for key, v in out.items()
                if all(math.isclose(key[by.index(a)], float(x)) for a, x in self.only.items())}

    def slice_ok(self, v: ShapeVerdict) -> bool:
        return v.shape is ProfileShape.INTERIOR_MINIMUM or (math.isfinite(v.rise_right) and v.rise_right >= self.rise)

    def head(self, passed: bool) -> str:
        where = "" if not self.only else " (slices " + ", ".join(f"{a} = {x:g}" for a, x in self.only.items()) + ")"
        return (f"{self.column} is not monotone decreasing in {self.along} in any slice{where} (rule: interior "
                f"minimum, or q at the right end at least {self.rise:g} above the minimum)." if passed
                else self.fail_message.format(along=self.along, column=self.column))

    def __call__(self, result: ProfileResult) -> tuple[bool, str]:
        verdicts = self.verdicts(result)
        lines, passed = [], bool(verdicts) or not self.only
        for v in verdicts.values():
            ok = self.slice_ok(v)
            passed &= ok
            lines.append(("ok: " if ok else "fails: ") + v.message)
        if not verdicts and self.only:
            lines.append(f"fails: no slice of {self.column} matches {self.only}.")
        return passed, self.head(passed) + "\n" + "\n".join(lines)

    def describe(self) -> dict[str, Any]:
        return {"rule": self.rule_name, "along": self.along, "column": self.column,
                "by": None if self.by is None else list(self.by), "rise": self.rise, "only": self.only}


class RisesAtBothEnds(NotMonotoneDecreasing):
    """Pass rule: in every selected slice, q at both ends of the grid exceeds its minimum by at least ``rise``
    (an interior minimum that the data resolve, e.g. the supply block's profile in sigma_p)."""

    rule_name = "RisesAtBothEnds"

    def __init__(self, along: str, column: str = "q_fixed", by: Sequence[str] | None = None, rise: float = 3.84,
                 fail_message: str = "Profile check failed: {column} does not rise at both ends in {along}.",
                 only: Mapping[str, float] | None = None, **classify_kw: Any) -> None:
        super().__init__(along, column, by, rise, fail_message, only, **classify_kw)

    def slice_ok(self, v: ShapeVerdict) -> bool:
        return bool(math.isfinite(v.rise_left) and math.isfinite(v.rise_right) and v.rise_left >= self.rise
                    and v.rise_right >= self.rise)

    def head(self, passed: bool) -> str:
        where = "" if not self.only else " (slices " + ", ".join(f"{a} = {x:g}" for a, x in self.only.items()) + ")"
        return (f"{self.column} rises by at least {self.rise:g} from its minimum to both ends of {self.along}{where}."
                if passed else self.fail_message.format(along=self.along, column=self.column))


# ------------------------------------------------------------------------------------------------
# JSON helpers (no NaN in files)
# ------------------------------------------------------------------------------------------------


def _float_out(v: float) -> Any:
    return v if math.isfinite(v) else repr(float(v))


def _float_in(v: Any) -> float:
    return float(v)


def _floats_out(d: Mapping[str, float]) -> dict[str, Any]:
    return {str(k): _float_out(float(v)) for k, v in d.items()}


def _floats_in(d: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): _float_in(v) for k, v in d.items()}


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return _float_out(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _fmt(v: float) -> str:
    return "nan" if not math.isfinite(v) else f"{v:.6g}"


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------------------------------------
# problem plumbing
# ------------------------------------------------------------------------------------------------


class _FixedView:
    """The problem with some parameters fixed (``free=False``) in its layout; everything else delegates.

    The optimizer reads free flags from ``problem.layout``, so a concentrated profile point is a
    ``minimize`` call on this view.
    """

    def __init__(self, problem: Any, layout: Any) -> None:
        self._problem = problem
        self.layout = layout

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


def _hook(obj: Any, name: str) -> Callable[..., Any] | None:
    fn = getattr(obj, name, None)
    return fn if callable(fn) else None


def _spare_capacity(problem: Any) -> int:
    for obj in (problem, getattr(problem, "model", None)):
        fn = _hook(obj, "spare_capacity") if obj is not None else None
        if fn is not None:
            try:
                return max(1, int(fn()))
            except NotImplementedError:
                pass
    return 1


def _n_of(problem: Any) -> float:
    for name in ("n", "n_products"):
        v = getattr(problem, name, None)
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return float(v)
    return 1.0


def _current_W(problem: Any) -> np.ndarray | None:
    W = getattr(problem, "W", None)
    return None if W is None else np.array(W, dtype=float)


def _q_cue(problem: Any, ev: Evaluation, n: float) -> tuple[float, str]:
    cov = _hook(problem, "covariance")
    if cov is None:
        return float("nan"), ""
    try:
        S = np.asarray(cov(ev), dtype=float)
        a = np.linalg.solve(S, ev.g)
        return float(n * ev.g @ a), ""
    except (np.linalg.LinAlgError, ValueError, RuntimeError, NotImplementedError) as exc:
        return float("nan"), f"q_cue not computed: {exc}"


@dataclass
class _Setup:
    names: list[str]
    index: dict[str, int]
    theta0: np.ndarray
    axes: list[Axis]
    fixed_names: list[str]
    W: np.ndarray | None
    modes: tuple[str, ...]
    concentrate: bool
    objective: str
    cold: bool
    economics: Callable[[Any, Evaluation], Mapping[str, float]] | None
    optimizer_options: Any
    W_concentrate: np.ndarray | None = None


def _point_theta(s: _Setup, grid_index: tuple[int, ...], start: np.ndarray) -> np.ndarray:
    theta = np.array(start, dtype=float)
    for axis, i in zip(s.axes, grid_index):
        axis.apply(theta, s.theta0, s.index, axis.values[i])
    return theta


def _record_values(problem: Any, s: _Setup, ev: Evaluation) -> tuple[dict[str, float], str]:
    n = _n_of(problem)
    values: dict[str, float] = {}
    notes = []
    if "fixed" in s.modes and s.W is not None:
        values["q_fixed"] = float(n * ev.g @ s.W @ ev.g)
        for b, rows in ev.blocks.items():
            values[f"q_fixed:{b}"] = float(n * ev.g[rows] @ s.W[rows, rows] @ ev.g[rows])
    if "cue" in s.modes:
        values["q_cue"], note = _q_cue(problem, ev, n)
        if note:
            notes.append(note)
    if "sd_xi" in ev.aux:
        values["sd_xi"] = float(ev.aux["sd_xi"])
    elif ev.aux.get("xi") is not None:
        values["sd_xi"] = float(np.std(ev.aux["xi"]))
    if ev.aux.get("delta") is not None:
        values["sd_delta"] = float(np.std(ev.aux["delta"]))
    if s.economics is not None:
        try:
            for k, v in dict(s.economics(problem, ev)).items():
                values[str(k)] = float(v)
        except Exception as exc:  # a broken summary must not lose the point
            notes.append(f"economics callback failed: {type(exc).__name__}: {exc}")
    return values, " ".join(notes)


def _evaluate_point(problem: Any, s: _Setup, index: int, grid_index: tuple[int, ...], start: np.ndarray,
                    snapshot: dict[str, Any] | None) -> ProfilePoint:
    """Evaluate (or concentrate) one grid point on ``problem``. Leaves the problem at the point's state
    (chained) or at ``snapshot`` (cold)."""
    t0 = time.perf_counter()
    theta = _point_theta(s, grid_index, start)
    coords = {a.name: a.values[i] for a, i in zip(s.axes, grid_index)}
    load = _hook(problem, "load_state_dict")
    if s.cold and snapshot is not None and load is not None:
        load(snapshot)
    convergence, evaluations, note = None, 1, ""
    if not s.concentrate:
        ev = problem.evaluate(theta, need_jacobian=False)
        status = ev.status.value
    else:
        from .optimizer import OptimizerOptions, minimize

        layout = problem.layout.replace({n: {"free": False} for n in s.fixed_names})
        view = _FixedView(problem, layout)
        weighting: Any = "cue" if s.objective == "cue" else (s.W if s.W_concentrate is None else s.W_concentrate)
        base_opts = s.optimizer_options or OptimizerOptions(max_iterations=200, record_cue=False)
        opts = dataclasses.replace(base_opts, stage=f"profile.{index}", theta0=theta)
        res = minimize(view, weighting, opts, checkpoint=None)
        theta = np.array(res.theta, dtype=float)
        convergence, evaluations, note = res.convergence.value, int(res.evaluations), res.message
        ev = res.evaluation
        if ev is None:
            ev = problem.evaluate(theta, need_jacobian=False)
            evaluations += 1
        status = ev.status.value
    values: dict[str, float] = {}
    if ev.ok:
        commit = _hook(problem, "commit")
        if commit is not None and not s.concentrate:  # minimize has already committed its optimum
            commit(theta)  # the covariance re-solve and economics then start from this point's delta
        values, vnote = _record_values(problem, s, ev)
        note = " ".join(x for x in (note, vnote) if x)
    else:
        restore = _hook(problem, "restore")
        if restore is not None:
            restore(theta)
        note = " ".join(x for x in (note, str(ev.aux.get("note", "")) if ev.aux else "") if x)
    if s.cold and snapshot is not None and load is not None:
        load(snapshot)
    return ProfilePoint(index=index, grid_index=grid_index, coords=coords, theta=theta, status=status, values=values,
                        seconds=time.perf_counter() - t0, evaluations=evaluations, convergence=convergence, note=note)


# ------------------------------------------------------------------------------------------------
# run directory
# ------------------------------------------------------------------------------------------------


class _Store:
    def __init__(self, path: pathlib.Path | None, config: dict[str, Any], new_run: bool) -> None:
        self.path = path
        self.hash = hash_options(config)
        if path is None:
            return
        meta = path / "profile.json"
        if path.exists() and new_run:
            os.replace(path, path.with_name(f"{path.name}.old-{_utc_stamp()}"))
        if meta.exists():
            have = json.loads(meta.read_text())
            if have.get("config_hash") != self.hash:
                raise ProfileMismatchError(
                    f"profile directory {path} was written by another configuration (hash "
                    f"{have.get('config_hash')!r}, now {self.hash!r}); its points cannot be reused. Pass "
                    f"new_run=True to archive it and start over.")
        else:
            (path / "points").mkdir(parents=True, exist_ok=True)
            doc = {"format": PROFILE_FORMAT, "config_hash": self.hash, "config": _plain(config),
                   "created": _utc_stamp(), "code_commit": code_commit()}
            atomic_write(meta, json.dumps(doc, indent=1).encode())
        (path / "points").mkdir(parents=True, exist_ok=True)

    def _stem(self, index: int) -> pathlib.Path:
        assert self.path is not None
        return self.path / "points" / f"p{index:05d}"

    def load_points(self) -> dict[int, ProfilePoint]:
        if self.path is None:
            return {}
        out = {}
        for js in sorted((self.path / "points").glob("p*.json")):
            d = json.loads(js.read_text())
            if d.get("config_hash") != self.hash:
                continue
            p = ProfilePoint.from_json(d["point"])
            out[p.index] = p
        return out

    def save_point(self, p: ProfilePoint, state: dict[str, Any] | None) -> None:
        if self.path is None:
            return
        stem = self._stem(p.index)
        if state is not None:
            arrays: dict[str, np.ndarray] = {}
            skeleton = _split(state, "", arrays)
            buf = io.BytesIO()
            np.savez(buf, __skeleton__=np.array(json.dumps(skeleton)), **arrays)
            atomic_write(stem.with_name(stem.name + ".state.npz"), buf.getvalue())
        doc = {"config_hash": self.hash, "point": p.to_json()}
        atomic_write(stem.with_name(stem.name + ".json"), json.dumps(doc, indent=1).encode())

    def load_state(self, index: int) -> dict[str, Any] | None:
        if self.path is None:
            return None
        f = self._stem(index)
        f = f.with_name(f.name + ".state.npz")
        if not f.exists():
            return None
        with np.load(f, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "__skeleton__"}
            return _join(json.loads(str(z["__skeleton__"])), arrays)

    def write_outputs(self, result: ProfileResult) -> None:
        if self.path is None:
            return
        rows = result.rows()
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: (_fmt_csv(v) if isinstance(v, float) else v) for k, v in r.items()})
        atomic_write(self.path / "profile.csv", buf.getvalue().encode())
        atomic_write(self.path / "profile.txt", result.table().encode())


def _fmt_csv(v: float) -> str:
    return repr(float(v)) if math.isfinite(v) else ("nan" if math.isnan(v) else repr(float(v)))


# ------------------------------------------------------------------------------------------------
# public entry points
# ------------------------------------------------------------------------------------------------


def _nearest_done(grid_index: tuple[int, ...], done: Mapping[tuple[int, ...], ProfilePoint]) -> ProfilePoint | None:
    best, best_d = None, None
    for gi, p in done.items():
        if not p.ok:
            continue
        d = sum(abs(a - b) for a, b in zip(gi, grid_index))
        if best_d is None or d < best_d or (d == best_d and p.index > best.index):  # type: ignore[union-attr]
            best, best_d = p, d
    return best


def profile(problem: Problem, theta0: np.ndarray | None, axes: Sequence[Axis], *,
            weighting: np.ndarray | None = None, modes: Sequence[str] = ("fixed", "cue"),
            concentrate: bool = False, objective: str = "fixed", hold: Sequence[str] = (), cold: bool = True,
            economics: Callable[[Any, Evaluation], Mapping[str, float]] | None = None,
            run_dir: str | pathlib.Path | None = None, new_run: bool = False, workers: int | None = None,
            problem_factory: Callable[[int, int], Problem] | None = None, optimizer_options: Any = None,
            pass_rule: Callable[[ProfileResult], tuple[bool, str]] | None = None, tag: str = "",
            concentrate_weighting: np.ndarray | None = None) -> ProfileResult:
    """profile(problem, theta0, axes, ...) -> ProfileResult over the product grid of ``axes``.

    ``theta0`` (default the layout's starts) gives every parameter no axis touches. ``weighting`` is the
    fixed W of ``q_fixed`` (default the problem's current W); it is set on the problem with
    ``set_weighting`` for the profile and the previous W is restored at the end. ``modes`` picks the
    columns: "fixed" (``q_fixed`` and its blocks) and "cue" (``q_cue``).

    ``concentrate=True`` re-optimises every free parameter that no axis touches and that is not in
    ``hold``, minimising ``q_<objective>`` with ``optimizer.minimize`` (``optimizer_options``, default
    ``OptimizerOptions(max_iterations=200, record_cue=False)``); the point's ``convergence`` is the
    optimizer's class and its theta the optimum. ``concentrate_weighting`` (fixed objective only) is the W
    the concentrated parameters minimise, when it differs from the reported ``q_fixed`` weighting, e.g.
    a W that up-weights the moments that exactly identify the concentrated parameters.

    ``cold=True`` evaluates every point from the problem state at the start of the profile (via
    ``state_dict`` / ``load_state_dict``), so points do not depend on each other and may run in
    parallel. ``cold=False`` chains: each point commits its state and the next one warm-starts from
    it; a concentrated point starts its free parameters at the nearest completed point's optimum.

    ``run_dir`` checkpoints every point and resumes (module docstring). ``workers``, ``problem_factory``:
    module docstring. ``pass_rule(result) -> (passed, message)`` is applied at the end and stored in
    ``result.verdict`` and ``profile.txt``. ``tag`` enters the configuration hash; use it to tell apart
    two problems that the hash cannot see (e.g. with and without micro moments in one directory tree).
    """
    names = list(problem.layout.names)
    index = {n: k for k, n in enumerate(names)}
    theta0 = problem.layout.starts() if theta0 is None else np.array(theta0, dtype=float).ravel()
    if theta0.shape != (len(names),):
        raise ValueError(f"theta0 has shape {theta0.shape}, expected ({len(names)},)")
    axes = list(axes)
    if not axes:
        raise ValueError("a profile needs at least one axis")
    axis_names = [a.name for a in axes]
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(f"axis names must be unique: {axis_names}")
    touched: list[str] = []
    for a in axes:
        for pname in a.parameters():
            if pname not in index:
                raise KeyError(f"axis {a.name!r} sets unknown parameter {pname!r}; parameters are {names}")
            if pname in touched:
                raise ValueError(f"parameter {pname!r} is set by two axes")
            touched.append(pname)
    unknown_hold = [h for h in hold if h not in index]
    if unknown_hold:
        raise KeyError(f"hold names unknown parameters {unknown_hold}")
    modes = tuple(modes)
    if not modes or any(m not in ("fixed", "cue") for m in modes):
        raise ValueError(f"modes must be a non-empty subset of ('fixed', 'cue'), got {modes}")
    if objective not in ("fixed", "cue"):
        raise ValueError(f"objective must be 'fixed' or 'cue', got {objective!r}")

    W = np.array(weighting, dtype=float) if weighting is not None else _current_W(problem)
    if ("fixed" in modes or (concentrate and objective == "fixed")) and W is None:
        raise ValueError("q_fixed needs a weighting matrix: pass weighting= or give the problem a W attribute")
    fixed_names = touched + [h for h in hold if h not in touched]
    s = _Setup(names=names, index=index, theta0=theta0, axes=axes, fixed_names=fixed_names, W=W, modes=modes,
               concentrate=concentrate, objective=objective, cold=cold, economics=economics,
               optimizer_options=optimizer_options,
               W_concentrate=None if concentrate_weighting is None else np.array(concentrate_weighting, dtype=float))

    shape = tuple(len(a.values) for a in axes)
    grid = list(itertools.product(*[range(k) for k in shape]))
    config = {
        "format": PROFILE_FORMAT, "names": names, "theta0": [float(v).hex() for v in theta0],
        "axes": [a.describe() for a in axes], "modes": list(modes), "concentrate": bool(concentrate),
        "objective": objective if concentrate else None, "hold": list(hold), "cold": bool(cold),
        "W": None if W is None else hash_options([float(v).hex() for v in W.ravel()]),
        "economics": None if economics is None else getattr(economics, "__qualname__", repr(economics)),
        "optimizer_options": None if optimizer_options is None else repr(optimizer_options), "tag": tag,
    }
    store = _Store(None if run_dir is None else pathlib.Path(run_dir), config, new_run)
    done_by_index = store.load_points()
    resumed = len(done_by_index)

    n_workers = _spare_capacity(problem) if workers is None else int(workers)
    n_workers = max(1, min(n_workers, len(grid) - resumed if len(grid) > resumed else 1))
    if not cold:
        n_workers = 1
    if n_workers > 1 and problem_factory is None:
        if workers is not None:
            raise ValueError("workers > 1 needs problem_factory(worker, threads) to build independent problems")
        n_workers = 1

    columns: list[str] = []
    result = ProfileResult(axes=axes, names=names, points=[], columns=columns, concentrate=concentrate,
                           objective=objective, run_dir=store.path, resumed_points=resumed, workers=n_workers,
                           meta={"config_hash": store.hash, "config": config})

    def refresh() -> None:
        pts = sorted(done_by_index.values(), key=lambda p: p.index)
        cols: list[str] = []
        for c in ("q_fixed", "q_cue"):
            if any(c in p.values for p in pts):
                cols.append(c)
        for p in pts:
            for c in p.values:
                if c not in cols and c.startswith("q_fixed:"):
                    cols.append(c)
        for p in pts:
            for c in p.values:
                if c not in cols:
                    cols.append(c)
        result.points = pts
        columns[:] = cols
        store.write_outputs(result)

    def finish(p: ProfilePoint, state: dict[str, Any] | None) -> None:
        done_by_index[p.index] = p
        store.save_point(p, state)
        refresh()
        log.info("profile point %d/%d %s: %s %s (%.1f s)", p.index + 1, len(grid), p.coords, p.status,
                 {k: round(v, 6) for k, v in p.values.items() if k in ("q_fixed", "q_cue", "sd_xi")}, p.seconds)

    old_W = _current_W(problem)
    set_w = _hook(problem, "set_weighting")
    if W is not None and set_w is not None and (old_W is None or not np.array_equal(old_W, W)):
        set_w(W)
    try:
        todo = [(i, gi) for i, gi in enumerate(grid) if i not in done_by_index]
        if n_workers == 1:
            state_dict = _hook(problem, "state_dict")
            snapshot = state_dict() if (cold and state_dict is not None) else None
            if not cold and done_by_index and todo:
                last = max(done_by_index)
                st = store.load_state(last)
                load = _hook(problem, "load_state_dict")
                if st is not None and load is not None:
                    load(st)
            for i, gi in todo:
                start = theta0
                if concentrate and not cold:
                    near = _nearest_done(gi, {p.grid_index: p for p in done_by_index.values()})
                    if near is not None:
                        start = near.theta
                p = _evaluate_point(problem, s, i, gi, start, snapshot)
                finish(p, None if cold or state_dict is None else state_dict())
        else:
            threads = max(1, (os.cpu_count() or 1) // n_workers)
            assert problem_factory is not None
            log.info("profile: %d points on %d workers with %d threads each", len(todo), n_workers, threads)
            pool_problems: list[Any] = []
            snapshots: list[dict[str, Any] | None] = []
            for w in range(n_workers):
                pw = problem_factory(w, threads)
                fn = _hook(pw, "set_weighting")
                if W is not None and fn is not None:
                    fn(W)
                sd = _hook(pw, "state_dict")
                pool_problems.append(pw)
                snapshots.append(sd() if sd is not None else None)
            free = list(range(n_workers))

            def run(i: int, gi: tuple[int, ...], w: int) -> tuple[ProfilePoint, int]:
                return _evaluate_point(pool_problems[w], s, i, gi, theta0, snapshots[w]), w

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="profile") as ex:
                pending: set[concurrent.futures.Future[tuple[ProfilePoint, int]]] = set()
                queue = list(todo)
                while queue or pending:
                    while queue and free:
                        i, gi = queue.pop(0)
                        pending.add(ex.submit(run, i, gi, free.pop(0)))
                    finished, pending = concurrent.futures.wait(pending, return_when="FIRST_COMPLETED")
                    for f in finished:
                        p, w = f.result()
                        free.append(w)
                        finish(p, None)
            for pw in pool_problems:
                close = _hook(getattr(pw, "model", None), "close")
                if close is not None:
                    close()
    finally:
        if set_w is not None and old_W is not None and W is not None and not np.array_equal(old_W, W):
            set_w(old_W)

    refresh()
    if pass_rule is not None:
        result.verdict = pass_rule(result)
        store.write_outputs(result)
    return result


def ray_profile(problem: Problem, theta0: np.ndarray | None, direction: Mapping[str, float],
                lambdas: Sequence[float], grid: Mapping[str, Sequence[float]] | None = None, *,
                origin: float | Mapping[str, float] | None = None, ray_name: str = "lambda",
                **kwargs: Any) -> ProfileResult:
    """ray_profile(problem, theta0, direction, lambdas, grid=None, origin=None, ray_name="lambda", **kw).

    Points ``origin + lambda d`` on the parameters of ``direction`` (``origin`` None: their theta0
    values, so ``theta0 + lambda d``; ``origin=0.0``: ``lambda d``), crossed with ``grid`` (named
    parameters, iterated fastest). Keyword arguments as in ``profile``.
    """
    axes: list[Axis] = [RayAxis(direction, lambdas, name=ray_name, origin=origin)]
    axes += [ParamAxis(n, v) for n, v in (grid or {}).items()]
    return profile(problem, theta0, axes, **kwargs)


def grid_profile(problem: Problem, theta0: np.ndarray | None, grid: Mapping[str, Sequence[float]],
                 **kwargs: Any) -> ProfileResult:
    """grid_profile(problem, theta0, {name: values, ...}, **kw): product grid over one or more named
    parameters (the last name fastest). Keyword arguments as in ``profile``."""
    if not grid:
        raise ValueError("grid_profile needs at least one parameter")
    return profile(problem, theta0, [ParamAxis(n, v) for n, v in grid.items()], **kwargs)

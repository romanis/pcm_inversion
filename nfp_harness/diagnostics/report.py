"""Report JSON schema, validation and text rendering (DESIGN 5.10). Owned by WP6.

Every diagnostic block is stored in the report as a dict holding its numbers plus ``status``,
``messages`` and ``thresholds`` (``block_to_dict``). Message lines that are flags start with
``FLAG:`` and failures with ``FAIL:``; the text rendering prints them verbatim, so every line of
``report.txt`` comes from a field of ``report.json``.

Non-finite floats are not valid JSON: ``to_jsonable`` writes NaN as null and infinities as the
strings "inf" and "-inf".
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import pathlib
from typing import Any, Mapping

import numpy as np

from ..types import REPORT_SCHEMA_VERSION, ConvergenceClass, DiagnosticBlock, Report
from .degeneracies import DEGENERACIES

__all__ = ["REPORT_SCHEMA_VERSION", "SCHEMA", "validate", "render_text", "write_report", "to_jsonable",
           "block_to_dict"]

NOTE_DEGENERACIES = ("NORMALISATION_CAP_ACTIVE",)  # reported, but a stated normalisation, not a flag


def block_to_dict(block: DiagnosticBlock) -> dict[str, Any]:
    """The report form of a block: its values, then ``status``, ``messages`` and ``thresholds``."""
    d = dict(block.values)
    # a block whose values carry their own verdict under "status" (integration: STABLE / UNSTABLE,
    # as in DESIGN 5.10) keeps it there and stores the block status as "block_status"
    d["block_status" if "status" in d else "status"] = block.status
    d.update(messages=list(block.messages), thresholds=dict(block.thresholds))
    return d


def to_jsonable(x: Any) -> Any:
    """Convert numpy types, enums, dataclasses and non-finite floats to plain JSON values."""
    if isinstance(x, DiagnosticBlock):
        return to_jsonable(block_to_dict(x))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return to_jsonable(dataclasses.asdict(x))
    if isinstance(x, enum.Enum):
        return x.value
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return x


# ------------------------------------------------------------------------------------------------
# Schema (version REPORT_SCHEMA_VERSION). Type tags: "str", "int", "num" (number, null, "inf",
# "-inf"), "bool", "dict", "list", suffix "?" for nullable. A nested dict gives required sub-keys.
# ------------------------------------------------------------------------------------------------

_BLOCK = {"status": "status", "messages": "messages"}

SCHEMA: dict[str, Any] = {
    "schema_version": "int",
    "run_id": "str",
    "spec": {"name": "str", "hash": "str", "normalisations": "list"},
    "stage": "str",
    "created": "str",
    "code_commit": "str",
    "convergence": {"class": "class", "projected_gradient": "num", "pg_tol": "num", "gn_decrease": "num",
                    "gn_tol": "num", "accepted_steps": "int", "evaluations": "int", "wall_seconds": "num",
                    "kink_limited_steps": "int", "message": "str", **_BLOCK},
    "objective": {"q": "num", "parts": "dict", "weighting": "str", "n": "int", "k": "int", "p_free": "int",
                  "k_linear": "int"},
    "parameters": ("list", {"name": "str", "value": "num", "unit": "num", "lower": "num", "upper": "num",
                            "cap": "bool", "at_bound": "bool", "se": "num", "note": "str"}),
    "linear": ("list", {"name": "str", "value": "num", "se": "num"}),
    "identification": {"rc_natural": "num", "rc_jacobi": "num", "order_condition": "dict",
                       "directions": ("list", {"index": "int", "verdict": "verdict", "dq_per_unit": "num",
                                               "loadings": "dict", "named": "str?", "message": "str"}),
                       **_BLOCK},
    "collinearity": {"X1": "dict", "Z": "dict", "ZX1": "dict", "G_pairs": "list", **_BLOCK},
    "weak_identification": {"statistic": "num", "dof": "int", "critical_95": "num", "verdict": "str", **_BLOCK},
    "hansen_j": ("dict?", {"J": "num", "dof": "int", "p_value": "num", **_BLOCK}),
    "degeneracies": ("list", {"name": "degeneracy", "detected": "bool", "evidence": "dict", "message": "str"}),
    "economics": {**_BLOCK},
    "integration": ("dict?", {"status": "str", "block_status": "status", "dq": "num", "changes": "dict",
                              "messages": "messages"}),
    "multistart": ("dict?", {"n_runs": "int", "distinct_endpoints": "int", "runs": "list", "message": "str"}),
    "standard_errors": {"computed": "bool", "reason": "str?", **_BLOCK},
    "files": "dict",
}

_CLASSES = {c.value for c in ConvergenceClass}
_STATUS = {"ok", "flag", "fail", "not_run"}
_VERDICTS = {"UNITS", "IDENTIFICATION", "WEAK"}


def _check(value: Any, tag: Any, path: str) -> None:
    if isinstance(tag, dict):
        if not isinstance(value, dict):
            raise ValueError(f"report field {path!r}: expected an object, got {type(value).__name__}")
        for k, t in tag.items():
            if k not in value:
                raise ValueError(f"report field {path}.{k!r} is missing")
            _check(value[k], t, f"{path}.{k}")
        return
    if isinstance(tag, tuple):
        kind, item = tag
        if kind.endswith("?") and value is None:
            return
        base = kind.rstrip("?")
        if base == "dict":
            _check(value, item, path)
            return
        if not isinstance(value, list):
            raise ValueError(f"report field {path!r}: expected a list, got {type(value).__name__}")
        for i, v in enumerate(value):
            _check(v, item, f"{path}[{i}]")
        return
    nullable = tag.endswith("?")
    tag = tag.rstrip("?")
    if nullable and value is None:
        return
    ok = {
        "str": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "num": lambda v: v is None or v in ("inf", "-inf") or (isinstance(v, (int, float)) and not isinstance(v, bool)),
        "bool": lambda v: isinstance(v, bool),
        "dict": lambda v: isinstance(v, dict),
        "list": lambda v: isinstance(v, list),
        "class": lambda v: v in _CLASSES,
        "status": lambda v: v in _STATUS,
        "verdict": lambda v: v in _VERDICTS,
        "degeneracy": lambda v: v in DEGENERACIES,
        "messages": lambda v: isinstance(v, list) and all(isinstance(m, str) for m in v),
    }[tag](value)
    if not ok:
        raise ValueError(f"report field {path!r}: {value!r} is not a valid {tag}")


def validate(report: dict[str, Any]) -> None:
    """validate(report) raises ValueError naming the first field that breaks the schema of version
    ``REPORT_SCHEMA_VERSION``.

    ``report`` is the JSON form (``to_jsonable(Report.to_dict())``)."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"report field 'schema_version': {report.get('schema_version')!r} is not "
                         f"{REPORT_SCHEMA_VERSION}")
    for key, tag in SCHEMA.items():
        if key not in report:
            raise ValueError(f"report field {key!r} is missing")
        _check(report[key], tag, key)
    names = [d["name"] for d in report["degeneracies"]]
    if len(set(names)) != len(names):
        raise ValueError("report field 'degeneracies': a detector appears twice")
    json.dumps(report, allow_nan=False)  # raises ValueError on anything that is not strict JSON


# ------------------------------------------------------------------------------------------------
# Text rendering
# ------------------------------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, allow_nan=False, separators=(", ", ": "))
    return str(v)


_SKIP = {"status", "block_status", "messages", "thresholds", "message", "directions", "order_condition"}


def _numbers(d: Mapping[str, Any], indent: str = "  ") -> list[str]:
    lines = [f"{indent}{k}: {_fmt(v)}" for k, v in d.items() if k not in _SKIP]
    th = d.get("thresholds")
    if th:
        lines.append(f"{indent}thresholds: {_fmt(th)}")
    return lines


def _block_section(title: str, d: Mapping[str, Any] | None) -> list[str]:
    out = [f"== {title} =="]
    if d is None:
        out.append("  not run")
        return out
    out += list(d.get("messages", []))
    if d.get("message") and d.get("message") not in d.get("messages", []):
        out.append(d["message"])
    out += _numbers(d)
    return out


def render_text(report: Report | Mapping[str, Any]) -> str:
    """render_text(report) -> report.txt: headline line, then one section per block with message lines
    first and numbers after, flags marked ``FLAG:`` and failures ``FAIL:``. Nothing absent from the JSON."""
    r = to_jsonable(report.to_dict() if isinstance(report, Report) else report)
    body: list[str] = []
    spec = r.get("spec", {})
    body.append(f"Run {r.get('run_id')} | spec {spec.get('name')} ({spec.get('hash')}) | stage {r.get('stage')} | "
                f"created {r.get('created')} | commit {r.get('code_commit')}")
    for nm in spec.get("normalisations", []):
        body.append(f"Normalisation: {nm}")

    body += _block_section("Convergence", r.get("convergence"))

    obj = r.get("objective", {})
    body.append("== Objective ==")
    body += _numbers(obj)

    body.append("== Parameters ==")
    body.append(f"  {'name':<22}{'value':>13}{'se':>11}{'unit':>9}{'lower':>11}{'upper':>11}  flags / note")
    for p in r.get("parameters", []):
        marks = " ".join(x for x, on in (("cap", p.get("cap")), ("AT_BOUND", p.get("at_bound"))) if on)
        body.append(f"  {p['name']:<22}{_fmt(p.get('value')):>13}{_fmt(p.get('se')):>11}{_fmt(p.get('unit')):>9}"
                    f"{_fmt(p.get('lower')):>11}{_fmt(p.get('upper')):>11}  {marks} {p.get('note', '')}".rstrip())
    if r.get("linear"):
        body.append("== Linear parameters ==")
        for p in r["linear"]:
            body.append(f"  {p['name']:<22}{_fmt(p.get('value')):>13}{_fmt(p.get('se')):>11}")

    ident = r.get("identification", {})
    body.append("== Identification ==")
    oc = ident.get("order_condition") or {}
    body += list(oc.get("messages", []))
    body += list(ident.get("messages", []))
    body += _numbers({k: v for k, v in oc.items() if k not in _SKIP}, "  order_condition.")
    body += _numbers(ident)
    for dct in ident.get("directions", []):
        body.append(f"  direction {dct['index']}: {_fmt({k: v for k, v in dct.items() if k != 'message'})}")

    body += _block_section("Collinearity", r.get("collinearity"))
    body += _block_section("Weak identification", r.get("weak_identification"))
    body += _block_section("Overidentification", r.get("hansen_j"))

    body.append("== Degeneracies ==")
    detected = [d for d in r.get("degeneracies", []) if d.get("detected")]
    for d in detected:
        mark = "NOTE" if d["name"] in NOTE_DEGENERACIES else "FLAG"
        body.append(f"{mark}: {d['name']}: {d['message']}")
    clear = [d["name"] for d in r.get("degeneracies", []) if not d.get("detected")]
    if clear:
        body.append(f"Not detected: {', '.join(clear)}.")
    for d in r.get("degeneracies", []):
        body.append(f"  {d['name']}.evidence: {_fmt(d.get('evidence'))}")

    body += _block_section("Economics", r.get("economics"))
    body += _block_section("Integration stability", r.get("integration"))
    body += _block_section("Multi-start", r.get("multistart"))
    body += _block_section("Standard errors", r.get("standard_errors"))
    if r.get("files"):
        body.append("== Files ==")
        body += [f"  {k}: {v}" for k, v in r["files"].items()]

    n_flag = sum(line.startswith("FLAG:") for line in body)
    n_fail = sum(line.startswith("FAIL:") for line in body)
    conv = r.get("convergence", {})
    q = obj.get("q")
    n_par = (obj.get("p_free") or 0) + (obj.get("k_linear") or 0)
    head = (f"RESULT: {conv.get('class')} at q = {_fmt(q)} ({obj.get('weighting')}, {obj.get('k')} moments, "
            f"{n_par} parameters). {n_flag} flag{'s' if n_flag != 1 else ''}"
            f"{f', {n_fail} failure' + ('s' if n_fail != 1 else '') if n_fail else ''}.")
    return "\n".join([head, *body]) + "\n"


def _atomic_write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_report(report: Report, run_dir: str | pathlib.Path) -> None:
    """write_report(report, run_dir): validate, then write report.json and report.txt atomically."""
    d = to_jsonable(report.to_dict())
    validate(d)
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(run_dir / "report.json", json.dumps(d, indent=2, allow_nan=False) + "\n")
    _atomic_write(run_dir / "report.txt", render_text(d))

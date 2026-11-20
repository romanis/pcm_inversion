"""Run directory, atomic checkpoints and resume (DESIGN 4.7). Owned by WP5.

Layout of ``<base>/<spec_name>_<run_key[:10]>/``::

    run.json                    spec name, spec hash, run key, options, created
    checkpoint.npz/.json        state of the stage in progress (or the last finished one)
    checkpoint.prev.npz         previous array file, kept until the new JSON is in place
    stages/<stage>.npz/.json    final state of every finished stage (resume skips these)
    history.csv                 one row per trial evaluation, all stages
    log.txt, heartbeat          written by ``runtime``
    report.json, report.txt     written by the diagnostics at the end of each stage

A state is a nested dict of scalars, strings, lists and numpy arrays. Arrays go to the ``.npz``
(no pickle), everything else to the ``.json``, which references each array by key. Both files carry
the same random token, so a crash between the two ``os.replace`` calls is detected on load and the
matching pair (new JSON with new arrays, or old JSON with ``checkpoint.prev.npz``) is used.
"""

from __future__ import annotations

import csv
import datetime as _dt
import enum
import hashlib
import io
import json
import math
import os
import pathlib
import subprocess
import uuid
from typing import Any, Iterable, Sequence

import numpy as np

from .types import EvalStatus, HistoryRecord

__all__ = [
    "RunDirectory",
    "read_stage_file",
    "stage_parameter_names",
    "SpecMismatchError",
    "atomic_write",
    "hash_options",
    "code_commit",
    "history_to_state",
    "history_from_state",
    "HISTORY_FIELDS",
]

HISTORY_FIELDS = ("stage", "iteration", "accepted", "q", "q_fixed", "q_cue", "sd_xi", "rho", "Delta", "tau_cap",
                  "status", "seconds", "predicted_mass", "realised_mass", "sd_delta", "mean_alpha_buyers")
_FLOAT_FIELDS = ("q", "q_fixed", "q_cue", "sd_xi", "rho", "Delta", "tau_cap", "seconds", "predicted_mass",
                 "realised_mass", "sd_delta", "mean_alpha_buyers")
_ARRAY_TAG = "__ndarray__"


class SpecMismatchError(RuntimeError):
    """The run directory was created by a different spec; rerun with a new run to start over."""


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    """atomic_write(path, data): write ``<path>.tmp``, fsync, ``os.replace``, fsync the directory."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def hash_options(options: Any) -> str:
    """hash_options(options) -> sha256 hex of the canonical JSON of ``options`` (a run key or spec hash)."""
    blob = json.dumps(_jsonable(options), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


_COMMIT: str | None = None


def code_commit() -> str:
    """code_commit() -> git HEAD of the repository holding this package, or "unknown"."""
    global _COMMIT
    if _COMMIT is None:
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parent,
                                 capture_output=True, text=True, timeout=10, check=True)
            _COMMIT = out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            _COMMIT = "unknown"
    return _COMMIT


def _jsonable(obj: Any) -> Any:
    """Plain-JSON form of scalars and containers (arrays as lists); used for hashing only."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    return obj


def _float_json(x: float) -> Any:
    """JSON has no NaN/inf; encode them as strings."""
    if math.isfinite(x):
        return x
    return {"__float__": repr(x)}


def _split(obj: Any, key: str, arrays: dict[str, np.ndarray]) -> Any:
    """Move arrays out of a nested state into ``arrays``; return the JSON skeleton."""
    if isinstance(obj, np.ndarray):
        arrays[key] = obj
        return {_ARRAY_TAG: key}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"checkpoint state keys must be strings, got {k!r} at {key!r}")
            out[k] = _split(v, f"{key}/{k}" if key else k, arrays)
        return out
    if isinstance(obj, (list, tuple)):
        return {"__list__": [_split(v, f"{key}/{i}", arrays) for i, v in enumerate(obj)],
                "__tuple__": isinstance(obj, tuple)}
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, enum.Enum):
        obj = obj.value
    if isinstance(obj, float):
        return _float_json(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    raise TypeError(f"checkpoint state value of type {type(obj).__name__} at {key!r} is not serialisable")


def _join(obj: Any, arrays: Any) -> Any:
    if isinstance(obj, dict):
        if _ARRAY_TAG in obj:
            return np.array(arrays[obj[_ARRAY_TAG]])
        if "__float__" in obj:
            return float(obj["__float__"])
        if "__list__" in obj:
            items = [_join(v, arrays) for v in obj["__list__"]]
            return tuple(items) if obj.get("__tuple__") else items
        return {k: _join(v, arrays) for k, v in obj.items()}
    return obj


def history_to_state(records: Sequence[HistoryRecord]) -> dict[str, Any]:
    """history_to_state(records) -> dict of arrays and string lists (checkpoint form of the history)."""
    P = len(records[0].theta) if records else 0
    return {
        "stage": [r.stage for r in records],
        "status": [r.status.value for r in records],
        "iteration": np.array([r.iteration for r in records], dtype=np.int64),
        "accepted": np.array([r.accepted for r in records], dtype=bool),
        **{f: np.array([getattr(r, f) for r in records], dtype=float) for f in _FLOAT_FIELDS},
        "theta": (np.array([r.theta for r in records], dtype=float) if records else np.zeros((0, P))),
    }


def history_from_state(d: dict[str, Any]) -> list[HistoryRecord]:
    """history_from_state(d) -> records written by ``history_to_state``. Float fields a state written before
    they existed lacks (``sd_delta``, ``mean_alpha_buyers``) read as NaN."""
    out = []
    n = len(d["stage"])
    d = {**{f: np.full(n, np.nan) for f in _FLOAT_FIELDS if f not in d}, **d}
    for i in range(len(d["stage"])):
        out.append(HistoryRecord(
            iteration=int(d["iteration"][i]), accepted=bool(d["accepted"][i]),
            status=EvalStatus(d["status"][i]), theta=np.array(d["theta"][i], dtype=float), stage=d["stage"][i],
            **{f: float(d[f][i]) for f in _FLOAT_FIELDS},
        ))
    return out


def _reorder(row: list[str], old: list[str], new: list[str]) -> list[str]:
    """A history.csv row under header ``old`` rewritten under header ``new`` (missing columns empty)."""
    by_name = dict(zip(old, row))
    return [by_name.get(c, "") for c in new]


def _sib(stem: pathlib.Path, suffix: str) -> pathlib.Path:
    """``stem`` plus a suffix, without ``with_suffix`` (stage names may contain dots)."""
    return stem.with_name(stem.name + suffix)


def read_stage_file(path: str | pathlib.Path) -> dict[str, Any]:
    """read_stage_file(path) -> the state of one stored stage (``<run dir>/stages/<stage>.npz`` or its ``.json``), read
    without a RunDirectory and without the spec-hash check: for reading another run's stage (a warm start)."""
    p = pathlib.Path(path)
    stem = p.with_name(p.name[:-len(".npz")]) if p.name.endswith(".npz") else (
        p.with_name(p.name[:-len(".json")]) if p.name.endswith(".json") else p)
    doc = json.loads(_sib(stem, ".json").read_text())
    for candidate in (_sib(stem, ".npz"), _sib(stem, ".prev.npz")):
        if not candidate.exists():
            continue
        with np.load(candidate, allow_pickle=False) as z:
            if str(z["__token__"]) == doc["token"]:
                return _join(doc["state"], {k: z[k] for k in z.files if k != "__token__"})
    raise RuntimeError(f"stage {stem} has no matching array file (token {doc['token']})")


def stage_parameter_names(path: str | pathlib.Path) -> list[str] | None:
    """stage_parameter_names(path) -> the layout names of a stored stage (``<run dir>/stages/<stage>.npz``) in theta
    order, read from the ``theta_<name>`` columns of its run's ``history.csv`` header; None when the run directory has
    no history. For mapping another run's theta onto a layout by name."""
    hist = pathlib.Path(path).resolve().parent.parent / "history.csv"
    if not hist.exists():
        return None
    with open(hist, newline="") as fh:
        header = next(csv.reader(fh), [])
    names = [c[len("theta_"):] for c in header if c.startswith("theta_")]
    return names or None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunDirectory:
    """``<base>/<spec_name>_<run_key[:10]>/`` holding checkpoint.npz/json, history.csv, log.txt,
    heartbeat, report.json and report.txt.

    ``run_key`` hashes the CLI options that change results (nu, seed, stage list, weighting); build it
    with ``hash_options``. Opening an existing directory whose ``run.json`` carries another
    ``spec_hash`` raises ``SpecMismatchError`` unless ``new_run``. With ``new_run`` an existing
    directory is renamed to ``<name>.old-<UTC timestamp>`` (never deleted) and a fresh one is created.
    """

    path: pathlib.Path

    def __init__(self, base: str | pathlib.Path, spec_name: str, spec_hash: str, run_key: str,
                 new_run: bool = False, options: dict[str, Any] | None = None) -> None:
        """RunDirectory(base, spec_name, spec_hash, run_key, new_run=False, options=None)."""
        self.spec_name = spec_name
        self.spec_hash = spec_hash
        self.run_key = run_key
        self.path = pathlib.Path(base) / f"{spec_name}_{run_key[:10]}"
        self.parameter_names: list[str] | None = None
        meta_path = self.path / "run.json"
        if self.path.exists() and new_run:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(self.path, self.path.with_name(f"{self.path.name}.old-{stamp}"))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("spec_hash") != spec_hash:
                raise SpecMismatchError(
                    f"run directory {self.path} was created by spec hash {meta.get('spec_hash')!r}, the current "
                    f"spec hashes to {spec_hash!r}. The spec changed since this run started, so its checkpoint "
                    f"cannot be resumed. Rerun with --new-run to archive it and start over.")
        else:
            (self.path / "stages").mkdir(parents=True, exist_ok=True)
            meta = {"spec_name": spec_name, "spec_hash": spec_hash, "run_key": run_key, "created": _utc_now(),
                    "code_commit": code_commit(), "options": _jsonable(options or {})}
            atomic_write(meta_path, json.dumps(meta, indent=1, sort_keys=True).encode())
        (self.path / "stages").mkdir(parents=True, exist_ok=True)

    # ---- generic state files -----------------------------------------------------------------

    def _write_pair(self, stem: pathlib.Path, state: dict[str, Any]) -> None:
        arrays: dict[str, np.ndarray] = {}
        skeleton = _split(state, "", arrays)
        token = uuid.uuid4().hex
        buf = io.BytesIO()
        np.savez(buf, __token__=np.array(token), **arrays)
        doc = {"token": token, "spec_hash": self.spec_hash, "run_key": self.run_key, "written": _utc_now(),
               "code_commit": code_commit(), "state": skeleton}
        npz, js, prev = _sib(stem, ".npz"), _sib(stem, ".json"), _sib(stem, ".prev.npz")
        tmp_npz = npz.with_name(npz.name + ".tmp")
        with open(tmp_npz, "wb") as fh:
            fh.write(buf.getvalue())
            fh.flush()
            os.fsync(fh.fileno())
        if npz.exists():
            os.replace(npz, prev)
        os.replace(tmp_npz, npz)
        _fsync_dir(npz.parent)
        atomic_write(js, json.dumps(doc, indent=1).encode())

    def _read_pair(self, stem: pathlib.Path) -> dict[str, Any]:
        js = _sib(stem, ".json")
        doc = json.loads(js.read_text())
        if doc.get("spec_hash") != self.spec_hash:
            raise SpecMismatchError(f"{js} was written by spec hash {doc.get('spec_hash')!r}, current spec is "
                                    f"{self.spec_hash!r}; rerun with --new-run to start over.")
        for candidate in (_sib(stem, ".npz"), _sib(stem, ".prev.npz")):
            if not candidate.exists():
                continue
            with np.load(candidate, allow_pickle=False) as z:
                if str(z["__token__"]) == doc["token"]:
                    arrays = {k: z[k] for k in z.files if k != "__token__"}
                    return _join(doc["state"], arrays)
        raise RuntimeError(f"checkpoint {js} has no matching array file (token {doc['token']}); the run "
                           f"directory is inconsistent")

    # ---- current checkpoint ------------------------------------------------------------------

    def has_checkpoint(self) -> bool:
        """has_checkpoint() -> True if checkpoint.json exists with an array file (checkpoint.npz or .prev.npz)."""
        return (self.path / "checkpoint.json").exists() and (
            (self.path / "checkpoint.npz").exists() or (self.path / "checkpoint.prev.npz").exists())

    def save(self, state: dict[str, Any]) -> None:
        """save(state): atomic checkpoint.npz (arrays) then checkpoint.json (scalars, provenance)."""
        self._write_pair(self.path / "checkpoint", state)

    def load(self) -> dict[str, Any]:
        """load() -> state written by ``save``; checks spec_hash."""
        return self._read_pair(self.path / "checkpoint")

    # ---- finished stages ---------------------------------------------------------------------

    def _stage_stem(self, stage: str) -> pathlib.Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stage) or "stage"
        return self.path / "stages" / safe

    def stage_done(self, stage: str) -> bool:
        """stage_done(stage) -> True if ``save_stage(stage, ...)`` completed."""
        return _sib(self._stage_stem(stage), ".json").exists()

    def save_stage(self, stage: str, state: dict[str, Any]) -> None:
        """save_stage(stage, state): final state of a finished stage, read back by ``load_stage``."""
        self._write_pair(self._stage_stem(stage), state)

    def load_stage(self, stage: str) -> dict[str, Any]:
        """load_stage(stage) -> state written by ``save_stage``."""
        return self._read_pair(self._stage_stem(stage))

    # ---- history.csv -------------------------------------------------------------------------

    def _header(self, P: int) -> list[str]:
        names = self.parameter_names if self.parameter_names and len(self.parameter_names) == P else \
            [str(i) for i in range(P)]
        return list(HISTORY_FIELDS) + [f"theta_{n}" for n in names]

    @staticmethod
    def _row(record: HistoryRecord) -> list[Any]:
        row: list[Any] = []
        for f in HISTORY_FIELDS:
            v = getattr(record, f)
            row.append(v.value if isinstance(v, enum.Enum) else (int(v) if isinstance(v, (bool, np.bool_)) else v))
        return row + [repr(float(x)) for x in record.theta]

    def append_history(self, record: HistoryRecord) -> None:
        """append_history(record): one row of history.csv per trial (header written on first use)."""
        path = self.path / "history.csv"
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            self._upgrade_header(self._header(len(record.theta)))
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self._header(len(record.theta)))
            w.writerow(self._row(record))
            fh.flush()

    def rewrite_history(self, stage: str, records: Iterable[HistoryRecord]) -> None:
        """rewrite_history(stage, records): replace every row of ``stage`` by ``records`` (atomic).

        Used on resume, so trials lost with the killed process do not appear twice."""
        path = self.path / "history.csv"
        kept: list[list[str]] = []
        header: list[str] | None = None
        if path.exists() and path.stat().st_size:
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
            header, body = rows[0], rows[1:]
            kept = [r for r in body if r and r[0] != stage]
        records = list(records)
        if records and header is not None and header != self._header(len(records[0].theta)):
            new_header = self._header(len(records[0].theta))
            kept = [_reorder(r, header, new_header) for r in kept]
            header = new_header
        if header is None and records:
            header = self._header(len(records[0].theta))
        if header is None:
            return
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(kept)
        w.writerows(self._row(r) for r in records)
        atomic_write(path, buf.getvalue().encode())

    def _upgrade_header(self, header: list[str]) -> None:
        """Rewrite history.csv under ``header`` when the file was written with other columns (e.g. by a version
        without ``sd_delta``), carrying each old row's values over by column name, so rows stay aligned."""
        path = self.path / "history.csv"
        with open(path, newline="") as fh:
            first = next(csv.reader(fh), None)
        if first is None or first == header:
            return
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(_reorder(r, rows[0], header) for r in rows[1:] if r)
        atomic_write(path, buf.getvalue().encode())

    def read_history(self) -> list[dict[str, str]]:
        """read_history() -> rows of history.csv as dicts (strings)."""
        path = self.path / "history.csv"
        if not path.exists():
            return []
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

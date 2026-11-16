"""Thread environment, detach, heartbeat and logging (DESIGN 1.3, 4.7). Owned by WP5.

``detach`` is how a long run survives the agent shell that started it: the same command is
re-executed in a new session (``setsid``), with stdin closed and stdout/stderr appended to the run's
``log.txt``. The parent prints the child pid and exits. A killed run is resumed by running the same
command again (``optimizer.minimize`` finds the checkpoint in the run directory).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Sequence

__all__ = ["check_environment", "detach", "heartbeat", "status", "setup_logging", "DETACHED_ENV"]

DETACHED_ENV = "NFP_HARNESS_DETACHED"

_EXPECTED = {"OMP_WAIT_POLICY": "PASSIVE", "OMP_DYNAMIC": "FALSE"}


def check_environment() -> list[str]:
    """check_environment() -> warnings.

    Warns if OMP_WAIT_POLICY=PASSIVE, OMP_DYNAMIC=FALSE and OPENBLAS_NUM_THREADS were not set before
    numpy was imported (DESIGN 1.3). The process environment is what OpenMP and OpenBLAS read at load
    time, so a value missing here was missing for them too. Also warns on OMP_NUM_THREADS=1, which
    the requirements forbid for production runs.
    """
    out = []
    for key, want in _EXPECTED.items():
        got = os.environ.get(key)
        if got is None or got.upper() != want:
            out.append(f"{key} is {got!r}, expected {want!r}; set it in the script before importing numpy")
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        out.append("OPENBLAS_NUM_THREADS is not set; numpy may claim every core (set it before importing numpy)")
    if os.environ.get("OMP_NUM_THREADS") == "1":
        out.append("OMP_NUM_THREADS=1 serialises the kernel; production runs must not set it")
    for msg in out:
        logging.getLogger("nfp_harness.runtime").warning(msg)
    return out


def detach(log_path: str | pathlib.Path, argv: Sequence[str] | None = None, flag: str = "--detach",
           exit_parent: bool = True) -> int | None:
    """detach(log_path, argv=None, flag="--detach", exit_parent=True) -> child pid, or None in the child.

    Re-execute the same command (``sys.executable`` plus ``argv``, default ``sys.argv``, with ``flag``
    removed) in a new session via ``setsid(2)`` (``start_new_session``), with stdin from /dev/null and
    stdout/stderr appended to ``log_path``, so recycling the agent shell does not kill the run. The child
    gets ``NFP_HARNESS_DETACHED=1``; calling ``detach`` inside it is a no-op that returns None, so a
    script can call it unconditionally when ``--detach`` is given. The parent prints the pid and log
    path and exits with status 0 (``exit_parent=False`` returns the pid instead, for tests).
    """
    if os.environ.get(DETACHED_ENV) == "1":
        return None
    args = list(sys.argv if argv is None else argv)
    args = [a for a in args if a != flag]
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{DETACHED_ENV: "1"})
    with open(log_path, "ab") as log, open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen([sys.executable, *args], stdin=devnull, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=os.getcwd(), start_new_session=True, close_fds=True)
    print(f"detached: pid {proc.pid}, log {log_path}", flush=True)
    if exit_parent:
        sys.exit(0)
    return proc.pid


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def heartbeat(run_dir: str | pathlib.Path, iteration: int, **extra: Any) -> None:
    """heartbeat(run_dir, iteration, **extra): rewrite ``heartbeat`` with timestamp, pid and iteration."""
    path = pathlib.Path(run_dir) / "heartbeat"
    doc = {"time": _utc_now(), "unix": time.time(), "pid": os.getpid(), "iteration": int(iteration), **extra}
    tmp = path.with_name("heartbeat.tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child still answers kill(0); treat it as dead
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def status(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """status(run_dir) -> heartbeat contents and whether the pid is alive.

    Keys: ``exists`` (heartbeat present), ``alive``, ``age_seconds`` since the last heartbeat, plus the
    heartbeat fields (time, pid, iteration, ...), and ``checkpoint`` (stage, iteration, accepted steps,
    q) when checkpoint.json exists.
    """
    run_dir = pathlib.Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(run_dir), "exists": False, "alive": False}
    hb = run_dir / "heartbeat"
    if hb.exists():
        try:
            doc = json.loads(hb.read_text())
        except (OSError, ValueError):
            doc = {}
        out.update(doc)
        out["exists"] = True
        if "pid" in doc:
            out["alive"] = _alive(int(doc["pid"]))
        if "unix" in doc:
            out["age_seconds"] = time.time() - float(doc["unix"])
    ck = run_dir / "checkpoint.json"
    if ck.exists():
        try:
            state = json.loads(ck.read_text()).get("state", {})
            out["checkpoint"] = {k: state.get(k) for k in ("stage", "iteration", "accepted_steps", "q", "done")}
        except (OSError, ValueError):
            out["checkpoint"] = None
    return out


def setup_logging(log_path: str | pathlib.Path, level: int = logging.INFO) -> logging.Logger:
    """setup_logging(log_path) -> the ``nfp_harness`` logger, appending to ``log.txt`` in the run directory.

    Idempotent: a second call with the same path adds no second handler.
    """
    log_path = pathlib.Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfp_harness")
    logger.setLevel(level)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and pathlib.Path(h.baseFilename) == log_path:
            return logger
    h = logging.FileHandler(log_path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(h)
    return logger

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

__all__ = ["check_environment", "detach", "enable_crash_dumps", "heartbeat", "status", "setup_logging",
           "DETACHED_ENV"]

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
           exit_parent: bool = True, cwd: str | pathlib.Path | None = None) -> int | None:
    """detach(log_path, argv=None, flag="--detach", exit_parent=True) -> child pid, or None in the child.

    Re-execute the same command (``sys.executable`` plus ``argv``, default ``sys.argv``, with ``flag``
    removed) in a new session via ``setsid(2)`` (``start_new_session``), with stdin from /dev/null and
    stdout/stderr appended to ``log_path``, so recycling the agent shell does not kill the run. The child
    gets ``NFP_HARNESS_DETACHED=1``; calling ``detach`` inside it is a no-op that returns None, so a
    script can call it unconditionally when ``--detach`` is given. The parent prints the pid and log
    path and exits with status 0 (``exit_parent=False`` returns the pid instead, for tests).

    ``cwd`` sets the child's working directory, which is where the kernel writes a core dump
    (core_pattern ``core``, so ``core.<pid>``). A relative script path in ``argv[0]`` is made absolute
    so the child still finds it.
    """
    if os.environ.get(DETACHED_ENV) == "1":
        return None
    args = list(sys.argv if argv is None else argv)
    args = [a for a in args if a != flag]
    if cwd is not None and args and not os.path.isabs(args[0]) and os.path.exists(args[0]):
        args[0] = os.path.abspath(args[0])
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{DETACHED_ENV: "1"})
    with open(log_path, "ab") as log, open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen([sys.executable, *args], stdin=devnull, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=os.getcwd() if cwd is None else str(cwd), start_new_session=True, close_fds=True)
    print(f"detached: pid {proc.pid}, log {log_path}", flush=True)
    if exit_parent:
        sys.exit(0)
    return proc.pid


def enable_crash_dumps(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """enable_crash_dumps(run_dir) -> what was set up, also written to ``run_dir/crash_setup.json``.

    Makes a crash in the C++ kernel debuggable without slowing the run:
    - raises the soft core-size limit to the hard limit, so SIGSEGV, SIGBUS, SIGFPE and SIGABRT write
      a core file (where the kernel puts it is ``/proc/sys/kernel/core_pattern``, relative to the
      working directory when it has no slash);
    - enables ``faulthandler`` for all threads into ``run_dir/crash_trace.txt``, so the Python stacks
      at the crash are on disk next to the core (faulthandler re-raises the signal, so the core is
      still written);
    - copies every loaded ``nfp_harness/_native`` module into ``run_dir/native/`` with its sha256, so
      the core can be opened with the exact binary even after a rebuild. The modules carry full
      debug info (-O3 -g, not stripped).

    Open a core with: gdb <python> <core> -ex "set solib-search-path <run_dir>/native" -ex "thread apply all bt".
    """
    import faulthandler
    import hashlib
    import resource
    import shutil

    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))
    except (ValueError, OSError):
        pass
    trace = open(run_dir / "crash_trace.txt", "a")  # kept open for the process lifetime
    trace.write(f"--- pid {os.getpid()} started {_utc_now()}\n")
    trace.flush()
    faulthandler.enable(file=trace, all_threads=True)
    try:
        import nfp_harness._native  # noqa: F401  load the kernel now so its exact binary is snapshotted
    except ImportError:
        pass
    native = run_dir / "native"
    native.mkdir(exist_ok=True)
    modules = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if "_native" in f and f.endswith(".so"):
            digest = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
            dst = native / f"{digest[:12]}_{pathlib.Path(f).name}"
            if not dst.exists():
                shutil.copy2(f, dst)
            modules[name] = {"source": f, "sha256": digest, "copy": str(dst)}
    try:
        pattern = pathlib.Path("/proc/sys/kernel/core_pattern").read_text().strip()
        uses_pid = pathlib.Path("/proc/sys/kernel/core_uses_pid").read_text().strip() == "1"
    except OSError:
        pattern, uses_pid = "unknown", False
    core = pattern if "/" in pattern or pattern.startswith("|") else os.path.join(os.getcwd(), pattern)
    info = {"pid": os.getpid(), "core_limit": resource.getrlimit(resource.RLIMIT_CORE)[0],
            "core_pattern": pattern, "expected_core": core + (f".{os.getpid()}" if uses_pid else ""),
            "python": sys.executable, "faulthandler": str(run_dir / "crash_trace.txt"), "native": modules,
            "gdb": f"gdb {sys.executable} <core> -ex 'set solib-search-path {native}' -ex 'thread apply all bt'"}
    with open(run_dir / "crash_setup.json", "w") as fh:
        json.dump(info, fh, indent=1)
    if info["core_limit"] == 0:
        logging.getLogger("nfp_harness.runtime").warning("core dumps are disabled (hard limit 0)")
    return info


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

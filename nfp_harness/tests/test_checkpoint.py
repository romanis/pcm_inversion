"""Tests for checkpoint, resume and runtime (WP5; DESIGN 4.7, 4.8 test 8).

The kill-and-resume test runs this file as a child process (``python test_checkpoint.py child ...``):
a Rosenbrock toy with 0.2 s per evaluation, checkpointing into a RunDirectory, is killed with SIGKILL
after 5 accepted steps and then relaunched with the same command.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from nfp_harness import checkpoint as ck  # noqa: E402
from nfp_harness import runtime  # noqa: E402
from nfp_harness.checkpoint import RunDirectory, SpecMismatchError, hash_options  # noqa: E402
from nfp_harness.optimizer import OptimizerOptions, minimize  # noqa: E402
from nfp_harness.types import ConvergenceClass, EvalStatus, HistoryRecord  # noqa: E402

PY = sys.executable
SPEC_HASH = "toy-spec-1"
RUN_KEY = hash_options({"toy": "rosenbrock", "Delta_0": 0.3})


# ------------------------------------------------------------------------------------------------
# child process entry points
# ------------------------------------------------------------------------------------------------


def _child(base: str, sleep: float) -> None:
    from nfp_harness.tests.test_optimizer import rosenbrock_problem

    rd = RunDirectory(base, "toy", SPEC_HASH, RUN_KEY)
    runtime.setup_logging(rd.path / "log.txt")
    start = rd.load()["accepted_steps"] if rd.has_checkpoint() else 0
    print(f"START accepted_steps={start}", flush=True)
    res = minimize(rosenbrock_problem(sleep=sleep), np.eye(2), OptimizerOptions(stage="fit"), checkpoint=rd)
    out = {"theta": [float(x).hex() for x in res.theta], "class": res.convergence.value,
           "accepted": res.accepted_steps, "q_accepted": [float(r.q).hex() for r in res.history if r.accepted]}
    (pathlib.Path(base) / "result.json").write_text(json.dumps(out))
    print("DONE", flush=True)


def _detach_child(base: str) -> None:
    runtime.detach(pathlib.Path(base) / "log.txt")  # parent exits here; the child continues below
    pathlib.Path(base, "sid.json").write_text(json.dumps({"pid": os.getpid(), "sid": os.getsid(0),
                                                          "stdin_tty": os.isatty(0)}))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        _child(sys.argv[2], float(sys.argv[3]))
    elif sys.argv[1] == "detach":
        _detach_child(sys.argv[2])
    sys.exit(0)


# ------------------------------------------------------------------------------------------------
# 8. kill and resume
# ------------------------------------------------------------------------------------------------


def _cmd(base: pathlib.Path) -> list[str]:
    return [PY, str(pathlib.Path(__file__).resolve()), "child", str(base), "0.2"]


def _run_dir(base: pathlib.Path) -> pathlib.Path:
    return base / f"toy_{RUN_KEY[:10]}"


def test_kill_and_resume_bit_for_bit(tmp_path: pathlib.Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="2")
    ref = tmp_path / "uninterrupted"
    out = subprocess.run(_cmd(ref), env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    want = json.loads((ref / "result.json").read_text())
    assert want["class"] == ConvergenceClass.CONVERGED.value and want["accepted"] > 8

    base = tmp_path / "killed"
    proc = subprocess.Popen(_cmd(base), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ckpt = _run_dir(base) / "checkpoint.json"
    deadline = time.monotonic() + 120
    accepted = 0
    while time.monotonic() < deadline:
        if ckpt.exists():
            try:
                accepted = json.loads(ckpt.read_text())["state"]["accepted_steps"]
            except (ValueError, KeyError):
                accepted = 0
            if accepted >= 5:
                break
        time.sleep(0.02)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL
    assert accepted == 5, f"expected to kill right after accepted step 5, saw {accepted}"
    assert not (base / "result.json").exists()

    again = subprocess.run(_cmd(base), env=env, capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stderr
    assert "START accepted_steps=5" in again.stdout
    got = json.loads((base / "result.json").read_text())
    assert got["theta"] == want["theta"]  # bit for bit (float.hex)
    assert got["q_accepted"] == want["q_accepted"]
    assert got["accepted"] == want["accepted"] and got["class"] == want["class"]
    # history.csv holds each accepted trial once, even though trials after step 5 were lost and redone
    rows = RunDirectory(base, "toy", SPEC_HASH, RUN_KEY).read_history()
    assert sum(r["accepted"] == "1" for r in rows) == want["accepted"] + 1  # + the start point
    assert [int(r["iteration"]) for r in rows] == sorted(int(r["iteration"]) for r in rows)
    assert "resumed at iteration" in (_run_dir(base) / "log.txt").read_text()

    # a third launch finds the finished stage and returns it without new steps
    third = subprocess.run(_cmd(base), env=env, capture_output=True, text=True, timeout=300)
    assert third.returncode == 0, third.stderr
    assert json.loads((base / "result.json").read_text())["theta"] == want["theta"]


# ------------------------------------------------------------------------------------------------
# checkpoint files
# ------------------------------------------------------------------------------------------------


def _state(v: float) -> dict:
    return {"stage": "s", "theta": np.array([v, 2.0]), "nested": {"markets": [np.arange(3) * v, {"x": v}]},
            "nan": float("nan"), "none": None, "flag": True, "name": "x", "tup": (1, 2)}


def test_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    rd = RunDirectory(tmp_path, "toy", SPEC_HASH, RUN_KEY)
    assert not rd.has_checkpoint()
    rd.save(_state(1.5))
    assert rd.has_checkpoint()
    st = rd.load()
    np.testing.assert_array_equal(st["theta"], [1.5, 2.0])
    np.testing.assert_array_equal(st["nested"]["markets"][0], np.arange(3) * 1.5)
    assert st["nested"]["markets"][1] == {"x": 1.5} and np.isnan(st["nan"]) and st["none"] is None
    assert st["flag"] is True and st["name"] == "x" and st["tup"] == (1, 2)
    files = {p.name for p in rd.path.iterdir()}
    assert {"run.json", "checkpoint.npz", "checkpoint.json"} <= files
    assert not any(n.endswith(".tmp") for n in files)


def test_crash_between_npz_and_json_keeps_previous_checkpoint(tmp_path: pathlib.Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    rd = RunDirectory(tmp_path, "toy", SPEC_HASH, RUN_KEY)
    rd.save(_state(1.0))

    def boom(path: pathlib.Path, data: bytes) -> None:
        raise OSError("killed before the JSON was replaced")

    monkeypatch.setattr(ck, "atomic_write", boom)
    with pytest.raises(OSError):
        rd.save(_state(7.0))
    monkeypatch.undo()
    np.testing.assert_array_equal(rd.load()["theta"], [1.0, 2.0])  # old JSON + checkpoint.prev.npz
    rd.save(_state(3.0))
    np.testing.assert_array_equal(rd.load()["theta"], [3.0, 2.0])


def test_spec_hash_mismatch_refuses_unless_new_run(tmp_path: pathlib.Path) -> None:
    rd = RunDirectory(tmp_path, "toy", SPEC_HASH, RUN_KEY)
    rd.save(_state(1.0))
    with pytest.raises(SpecMismatchError, match="--new-run"):
        RunDirectory(tmp_path, "toy", "other-spec", RUN_KEY)
    fresh = RunDirectory(tmp_path, "toy", "other-spec", RUN_KEY, new_run=True)
    assert not fresh.has_checkpoint()
    archived = [p for p in tmp_path.iterdir() if p.name.startswith(rd.path.name + ".old-")]
    assert len(archived) == 1 and (archived[0] / "checkpoint.json").exists()  # archived, not deleted


def test_run_key_names_directory(tmp_path: pathlib.Path) -> None:
    k1, k2 = hash_options({"nu": 4, "seed": 1}), hash_options({"seed": 1, "nu": 4})
    assert k1 == k2 and k1 != hash_options({"nu": 16, "seed": 1})
    assert RunDirectory(tmp_path, "petrin_pcm", "h", k1).path.name == f"petrin_pcm_{k1[:10]}"


def test_history_roundtrip_and_rewrite(tmp_path: pathlib.Path) -> None:
    rd = RunDirectory(tmp_path, "toy", SPEC_HASH, RUN_KEY)
    rd.parameter_names = ["a", "b"]
    recs = [HistoryRecord(iteration=i, accepted=i % 2 == 0, q=1.0 / (i + 1), q_fixed=1.0, q_cue=float("nan"),
                          sd_xi=0.5, rho=0.9, Delta=0.3, tau_cap=1.0, status=EvalStatus.OK, seconds=0.1,
                          theta=np.array([i, -i], dtype=float), stage="s1") for i in range(4)]
    for r in recs:
        rd.append_history(r)
    back = ck.history_from_state(ck.history_to_state(recs))
    assert [r.q for r in back] == [r.q for r in recs] and back[1].accepted is False
    np.testing.assert_array_equal(back[3].theta, [3.0, -3.0])
    rd.rewrite_history("s1", recs[:2])
    rows = rd.read_history()
    assert len(rows) == 2 and "theta_a" in rows[0] and rows[1]["theta_b"] == repr(-1.0)


def test_finished_stage_is_skipped(tmp_path: pathlib.Path) -> None:
    from nfp_harness.tests.test_optimizer import rosenbrock_problem

    rd = RunDirectory(tmp_path, "toy", SPEC_HASH, RUN_KEY)
    first = minimize(rosenbrock_problem(), np.eye(2), OptimizerOptions(stage="fit"), checkpoint=rd)
    p = rosenbrock_problem()
    again = minimize(p, np.eye(2), OptimizerOptions(stage="fit"), checkpoint=rd)
    assert p.evals == 1  # one evaluation to rebuild the Evaluation, no steps
    np.testing.assert_array_equal(again.theta, first.theta)
    assert again.convergence is first.convergence and again.accepted_steps == first.accepted_steps


# ------------------------------------------------------------------------------------------------
# runtime
# ------------------------------------------------------------------------------------------------


def test_detach_runs_in_new_session(tmp_path: pathlib.Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop(runtime.DETACHED_ENV, None)
    out = subprocess.run([PY, str(pathlib.Path(__file__).resolve()), "detach", str(tmp_path), "--detach"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "detached: pid" in out.stdout
    target = tmp_path / "sid.json"
    deadline = time.monotonic() + 30
    while not target.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    doc = json.loads(target.read_text())
    assert doc["sid"] == doc["pid"]  # session leader: setsid took effect
    assert doc["sid"] != os.getsid(0) and doc["stdin_tty"] is False
    assert int(out.stdout.split("pid ")[1].split(",")[0]) == doc["pid"]


def test_heartbeat_and_status(tmp_path: pathlib.Path) -> None:
    assert runtime.status(tmp_path)["exists"] is False
    runtime.heartbeat(tmp_path, 7, stage="fit")
    st = runtime.status(tmp_path)
    assert st["exists"] and st["alive"] and st["iteration"] == 7 and st["pid"] == os.getpid()
    assert st["age_seconds"] < 60
    (tmp_path / "heartbeat").write_text(json.dumps({"pid": 2 ** 22 + 12345, "unix": time.time(), "iteration": 1}))
    assert runtime.status(tmp_path)["alive"] is False


def test_check_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert runtime.check_environment() == []
    monkeypatch.delenv("OMP_WAIT_POLICY")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    warnings = runtime.check_environment()
    assert any("OMP_WAIT_POLICY" in w for w in warnings) and any("OMP_NUM_THREADS=1" in w for w in warnings)

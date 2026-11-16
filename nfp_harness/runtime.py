"""Thread environment, detach, heartbeat and logging (DESIGN 1.3, 4.7). Owned by WP5."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

__all__ = ["check_environment", "detach", "heartbeat", "status", "setup_logging"]


def check_environment() -> list[str]:
    """check_environment() -> warnings.

    Warns if OMP_WAIT_POLICY=PASSIVE, OMP_DYNAMIC=FALSE and OPENBLAS_NUM_THREADS were not set before
    numpy was imported (DESIGN 1.3).
    """
    raise NotImplementedError("check_environment() -> list[str]: WP5, DESIGN 1.3")


def detach(log_path: str | pathlib.Path) -> None:
    """detach(log_path): re-execute the same command under ``setsid`` with stdin closed and
    stdout/stderr to ``log_path``, so recycling the agent shell does not kill the run."""
    raise NotImplementedError("detach(log_path): WP5, DESIGN 4.7")


def heartbeat(run_dir: str | pathlib.Path, iteration: int) -> None:
    """heartbeat(run_dir, iteration): rewrite ``heartbeat`` with timestamp, pid and iteration."""
    raise NotImplementedError("heartbeat(run_dir, iteration): WP5, DESIGN 4.7")


def status(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """status(run_dir) -> heartbeat contents and whether the pid is alive."""
    raise NotImplementedError("status(run_dir) -> dict: WP5, DESIGN 4.7")


def setup_logging(log_path: str | pathlib.Path) -> logging.Logger:
    """setup_logging(log_path) -> logger writing ``log.txt`` in the run directory."""
    raise NotImplementedError("setup_logging(log_path) -> logging.Logger: WP5, DESIGN 4.7")

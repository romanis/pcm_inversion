"""Run directory, atomic checkpoints and resume (DESIGN 4.7). Owned by WP5."""

from __future__ import annotations

import pathlib
from typing import Any

from .types import HistoryRecord

__all__ = ["RunDirectory", "atomic_write"]


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    """atomic_write(path, data): write ``<path>.tmp``, fsync, ``os.replace``."""
    raise NotImplementedError("atomic_write(path, data): WP5, DESIGN 4.7")


class RunDirectory:
    """``<base>/<spec_name>_<run_key[:10]>/`` holding checkpoint.npz/json, history.csv, log.txt,
    heartbeat, report.json and report.txt.

    ``run_key`` hashes the CLI options that change results (nu, seed, stage list, weighting). Resume
    refuses on a ``spec_hash`` mismatch unless a new run is requested.
    """

    path: pathlib.Path

    def __init__(self, base: str | pathlib.Path, spec_name: str, spec_hash: str, run_key: str,
                 new_run: bool = False) -> None:
        """RunDirectory(base, spec_name, spec_hash, run_key, new_run=False)."""
        raise NotImplementedError("RunDirectory(base, spec_name, spec_hash, run_key, new_run=False): WP5, DESIGN 4.7")

    def has_checkpoint(self) -> bool:
        """has_checkpoint() -> True if checkpoint.npz and checkpoint.json exist."""
        raise NotImplementedError("RunDirectory.has_checkpoint() -> bool: WP5, DESIGN 4.7")

    def save(self, state: dict[str, Any]) -> None:
        """save(state): atomic checkpoint.npz (arrays) then checkpoint.json (scalars, provenance)."""
        raise NotImplementedError("RunDirectory.save(state): WP5, DESIGN 4.7")

    def load(self) -> dict[str, Any]:
        """load() -> state written by ``save``; checks spec_hash."""
        raise NotImplementedError("RunDirectory.load() -> dict: WP5, DESIGN 4.7")

    def append_history(self, record: HistoryRecord) -> None:
        """append_history(record): one row of history.csv per trial."""
        raise NotImplementedError("RunDirectory.append_history(record): WP5, DESIGN 4.7")

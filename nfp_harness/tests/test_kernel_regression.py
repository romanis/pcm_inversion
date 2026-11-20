"""Regression oracle of the compiled kernel (GMY spec WP-G0): every existing ``pcm_kernel`` function reproduces the
baseline fixtures bit for bit, with 1 and 8 OpenMP threads.

The fixtures are written by ``nfp_harness/tests/fixtures/gmy_baseline.py`` into an untracked directory, read from
``$NFP_KERNEL_BASELINE`` or else ``$KIROCREW_SCRATCH/gmy/baseline``; the test is skipped when neither holds them.
Wall-clock fields (``seconds``, the timings inside ``stages``) are not compared.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("nfp_harness._native")

from nfp_harness.tests.fixtures import gmy_baseline  # noqa: E402


def _baseline_dir() -> pathlib.Path | None:
    env = os.environ.get("NFP_KERNEL_BASELINE")
    if env:
        return pathlib.Path(env)
    scratch = os.environ.get("KIROCREW_SCRATCH")
    return pathlib.Path(scratch) / "gmy" / "baseline" if scratch else None


BASE = _baseline_dir()
FILES = sorted(BASE.glob("*.npz")) if BASE is not None and BASE.is_dir() else []


def _same(a: object, b: object) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "US" or b.dtype.kind in "US":
        return str(a) == str(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


@pytest.mark.skipif(not FILES, reason="no kernel baseline fixtures (run nfp_harness/tests/fixtures/gmy_baseline.py)")
@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
@pytest.mark.parametrize("n_threads", [1, 8])
def test_kernel_outputs_are_bit_identical(path: pathlib.Path, n_threads: int) -> None:
    inp, ref, warm = gmy_baseline.load(path)
    got = gmy_baseline.outputs(inp, n_threads, np.asarray(inp["delta_cold"], dtype=float) if warm else None)
    assert set(got) == set(ref), sorted(set(got) ^ set(ref))
    bad = [k for k in sorted(ref) if not _same(got[k], ref[k])]
    assert not bad, f"{path.stem} at {n_threads} threads differs from the baseline in {bad}"

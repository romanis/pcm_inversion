"""Golden-master safety net of the refactor (step 0): each paper's numbers at one fixed point are bit-identical to the
goldens written on c96d4b8, at 1 and 8 kernel threads.

The cases and what they pin are in ``nfp_harness/tests/golden/harness.py``; the goldens are the tracked
``nfp_harness/tests/golden/<case>.npz`` (or ``$NFP_GOLDEN_DIR/<case>.npz``), rewritten by
``python -m nfp_harness.tests.golden.regenerate``. A missing golden or missing input data SKIPS the case with a
GoldenNotChecked warning (shown in the warnings summary), never passes it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nfp_harness.tests.golden import harness


class GoldenNotChecked(UserWarning):
    """A golden case was skipped: its file or its input data is missing."""


def _skip(msg: str) -> None:
    warnings.warn(GoldenNotChecked(msg))
    pytest.skip(msg)


PARAMS = [(name, t) for name, case in harness.CASES.items() for t in case.threads]


@pytest.mark.parametrize(("name", "threads"), PARAMS, ids=[f"{n}-{t}thr" for n, t in PARAMS])
def test_golden_outputs_are_bit_identical(name: str, threads: int) -> None:
    case = harness.CASES[name]
    path = harness.golden_path(name)
    if not path.exists():
        _skip(f"GOLDEN MISSING, {name} NOT CHECKED: {path} does not exist. Write it on the reference commit with "
              f"`python -m nfp_harness.tests.golden.regenerate {name}`.")
    missing = harness.missing_needs(case)
    if missing:
        _skip(f"GOLDEN {name} NOT CHECKED: missing inputs {missing}")
    ref, meta = harness.load(path)
    assert meta["sha256"] == harness.digest_all(ref), f"{path} is corrupt: its pins do not hash to its metadata"
    got, _ = harness.compute_isolated(name, threads, golden=path)
    bad = harness.differences(ref, got, dict(case.approx))
    assert not bad, (f"{name} at {threads} threads differs from {path.name} (written on {meta.get('commit')}) in "
                     f"{len(bad)} of {len(ref)} pins:\n  " + "\n  ".join(bad[:40]))


def test_comparison_sees_one_ulp_a_signed_zero_and_a_dtype() -> None:
    a = np.linspace(0.0, 1.0, 5)
    assert harness.differences({"x": a}, {"x": a.copy()}) == []
    assert harness.differences({"x": a}, {"x": np.nextafter(a, 2.0)})
    assert harness.differences({"x": np.zeros(2)}, {"x": -np.zeros(2)})
    assert harness.differences({"x": a}, {"x": a.astype(np.float32)})
    assert harness.differences({"x": np.array([np.nan])}, {"x": np.array([np.nan])}) == []
    assert harness.differences({"x": a}, {}) == ["missing x"]
    # a pin with a tolerance passes within it and fails beyond it, and never across dtypes
    assert harness.differences({"x": a + 1}, {"x": np.nextafter(a + 1, 3.0)}, {"x": 1e-14}) == []
    assert harness.differences({"x": a + 1}, {"x": a * 1.001 + 1}, {"x": 1e-14})
    assert harness.differences({"x": a}, {"x": a.astype(np.float32)}, {"x": 1.0})

"""Rewrite the golden files of ``nfp_harness/tests/golden`` (refactor step 0).

Run it only on the commit whose numbers every later refactor step must reproduce (the goldens of this directory were
written on c96d4b8, the head of feat/gmy-pcm-estimation); the commit and whether tracked files were modified are
stored in each file's metadata. Each case runs in its own interpreter (``harness.compute_isolated``) with its inputs
computed afresh, e.g. the warm-start delta from a cold solve.

    python -m nfp_harness.tests.golden.regenerate                    # every case, 8 kernel threads
    python -m nfp_harness.tests.golden.regenerate gmy_pcm nevo_pcm   # some cases
    python -m nfp_harness.tests.golden.regenerate --threads 16 --dir /tmp/goldens

The written cases are then checked by ``nfp_harness/tests/test_golden_eval.py`` (``NFP_GOLDEN_DIR`` points it at
another directory). A case whose data or modules are missing is reported and not written.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

from nfp_harness.tests.golden import harness


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="rewrite the refactor golden files")
    ap.add_argument("cases", nargs="*", help=f"cases (default: all of {sorted(harness.CASES)})")
    ap.add_argument("--threads", type=int, default=8, help=f"kernel threads, 1..{harness.MAX_THREADS}")
    ap.add_argument("--dir", type=pathlib.Path, default=None, help="output directory (default: $NFP_GOLDEN_DIR or "
                                                                    "this package directory)")
    a = ap.parse_args(argv)
    unknown = sorted(set(a.cases) - set(harness.CASES))
    if unknown:
        ap.error(f"unknown cases {unknown}; known: {sorted(harness.CASES)}")
    directory = a.dir or harness.golden_dir()
    state = harness.git_state()
    if state["dirty_tracked"]:
        print(f"WARNING: tracked files differ from {state['commit']}; the goldens will not describe that commit",
              file=sys.stderr)
    failed = []
    for name in a.cases or sorted(harness.CASES):
        case = harness.CASES[name]
        missing = harness.missing_needs(case)
        if missing:
            print(f"SKIP {name}: missing {missing}", file=sys.stderr)
            failed.append(name)
            continue
        t0 = time.time()
        pins, meta = harness.compute_isolated(name, a.threads)
        path = harness.golden_path(name, directory)
        harness.save(path, pins, dict(meta, note=case.note))
        print(f"wrote {path}: {len(pins)} pins, {path.stat().st_size / 1024:.0f} KiB, {time.time() - t0:.0f} s, "
              f"sha256 {harness.digest_all(pins)[:16]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

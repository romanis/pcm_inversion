"""Golden-master safety net of the refactor (step 0): every paper's numbers at one fixed point, pinned bit for bit.

``harness.py`` computes the pinned outputs of each case, ``regenerate.py`` rewrites the ``*.npz`` goldens next to it,
and ``nfp_harness/tests/test_golden_eval.py`` recomputes them and requires bit-identity.
"""

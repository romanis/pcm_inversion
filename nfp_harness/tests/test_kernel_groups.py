"""The compiled per-household sums (GMY spec WP-G5) against the evaluate_market oracle, through the binding and the
PCM model."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from nfp_harness.models.pcm import group_sums_oracle

pcm_kernel = pytest.importorskip("nfp_harness._native").pcm_kernel
if not hasattr(pcm_kernel, "group_sums"):
    pytest.skip("kernel built without group_sums", allow_module_level=True)

from nfp_harness.tests import test_group_micro as tg  # noqa: E402
from nfp_harness.tests.fixtures import gmy_baseline  # noqa: E402


def _same(a: dict, b: dict) -> None:
    assert np.array_equal(a["S"], b["S"])
    assert np.array_equal(np.stack(a["d_delta"]), np.stack(b["d_delta"]))
    assert np.array_equal(np.stack(a["d_theta"]), np.stack(b["d_theta"]))


@pytest.mark.parametrize("n_threads", [1, 8])
def test_kernel_equals_oracle_on_the_synthetic_model(n_threads: int) -> None:
    model, delta, data = tg.model_at()
    theta = tg.layout().starts()
    th, ph, sp = model.kernel_params(theta)
    for t, m in enumerate(model.markets):
        km = model.kernel_market(t)
        g = np.asarray(model.market_draws(t)["agent_of_draw"])
        U = np.column_stack([data["products"]["prices"][m.rows], data["products"]["x1"][m.rows]])
        kw = model.kernel_kwargs(theta)
        got = pcm_kernel.group_sums(delta[m.rows], km, th, ph, sp, g, int(g.max()) + 1, U, n_threads=n_threads, **kw)
        ref = group_sums_oracle(pcm_kernel, delta[m.rows], km, th, ph, sp, g, int(g.max()) + 1, U, True, 1, **kw)
        _same(got, ref)
        np.testing.assert_allclose(got["w"], ref["w"], rtol=1e-15)
    model.close()


def test_model_backends_agree() -> None:
    model, delta, data = tg.model_at()
    theta = tg.layout().starts()
    mom = tg.moment(model, data)
    model.group_sums_backend = "kernel"
    kern = mom.moments(tg.solution(model, delta, theta, True), True)
    model.group_sums_backend = "oracle"
    orc = mom.moments(tg.solution(model, delta, theta, True), True)
    for a, b in zip(kern, orc):
        assert np.array_equal(a, b)
    model.close()


def test_kernel_equals_oracle_on_gmy_1980() -> None:
    base = os.environ.get("NFP_KERNEL_BASELINE") or (
        str(pathlib.Path(os.environ["KIROCREW_SCRATCH"]) / "gmy" / "baseline") if "KIROCREW_SCRATCH" in os.environ
        else "")
    path = pathlib.Path(base) / "gmy_1980.npz"
    if not path.exists():
        pytest.skip("no GMY 1980 baseline fixture")
    inp, _, _ = gmy_baseline.load(path)
    km = gmy_baseline.market_of(inp)
    R = np.size(inp["weights"])
    g = np.concatenate([np.full(len(b), k) for k, b in enumerate(np.array_split(np.arange(R), 1000))])
    U = np.column_stack([inp["p"], inp["x"][:, 1], inp["x"][:, 4]])
    bkw = {inp["bound_key"]: inp["bound_value"]}
    args = (np.asarray(inp["delta_cold"], dtype=float), km, inp["theta"], inp["phi"], inp["sigma_p"], g, 1000, U)
    got = pcm_kernel.group_sums(*args, n_threads=8, **bkw)
    ref = group_sums_oracle(pcm_kernel, *args, True, 1, **bkw)
    _same(got, ref)


def test_binding_rejects_bad_groups() -> None:
    model, delta, data = tg.model_at()
    km = model.kernel_market(0)
    th, ph, sp = model.kernel_params(tg.layout().starts())
    rows = model.markets[0].rows
    g = np.asarray(model.market_draws(0)["agent_of_draw"]).copy()
    g[3] = g[-1]
    with pytest.raises(ValueError, match="non-decreasing"):
        pcm_kernel.group_sums(delta[rows], km, th, ph, sp, g, 16, np.ones((rows.size, 1)))
    with pytest.raises(ValueError, match="out of range"):
        pcm_kernel.group_sums(delta[rows], km, th, ph, sp, np.asarray(model.market_draws(0)["agent_of_draw"]), 3,
                              np.ones((rows.size, 1)))
    model.close()

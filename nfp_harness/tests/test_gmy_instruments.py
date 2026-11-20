"""Tests for the GMY instruments, the order condition and the problem without an aggregate block (GMY spec WP-G2)."""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import numpy as np
import pytest

from nfp_harness.diagnostics.identification import block_shares
from nfp_harness.linear import IdentificationError, LinearIV
from nfp_harness.moments import SarganIVMoments
from nfp_harness.params import ParameterLayout
from nfp_harness.problem import NFPGMMProblem
from nfp_harness.tests.test_problem import LogitRCModel, central_fd, make_data, make_layout, make_micro
from nfp_harness.types import KernelRequests, MarketIndex, ParameterSpec

ROOT = pathlib.Path(__file__).resolve().parents[2]


def gmy_module() -> Any:
    """gmy_replication/estimate_pcm.py under its own module name (petrin_replication has an estimate_pcm too)."""
    spec = importlib.util.spec_from_file_location("gmy_estimate_pcm", ROOT / "gmy_replication" / "estimate_pcm.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gmy() -> Any:
    from nfp_harness import gmy_data
    try:
        gmy_data.package_root()
    except FileNotFoundError:
        pytest.skip("GMY replication package not unpacked")
    mod = gmy_module()   # puts the vendored pyblp on sys.path
    pytest.importorskip("pyblp")
    return mod


@pytest.fixture(scope="module")
def raw(gmy: Any) -> tuple[Any, np.ndarray]:
    from nfp_harness import gmy_data
    p = gmy_data.load_products()
    X1, _ = gmy.GMYSpec.linear_design(p)
    return p, X1


def test_rank_added_by_each_instrument_set(gmy: Any, raw: tuple[Any, np.ndarray]) -> None:
    p, X1 = raw
    assert X1.shape == (9694, 107)
    rows = gmy.first_stage_table(p, X1)
    assert [r["rank_added"] for r in rows] == [1, 2, 3, 5, 4, 10, 6, 6, 11]
    assert [r["columns"] for r in rows] == [1, 2, 3, 5, 4, 18, 6, 6, 11]
    assert all(r["F_price"] > 10 for r in rows)


def test_default_z_and_linear_iv(gmy: Any, raw: tuple[Any, np.ndarray]) -> None:
    p, X1 = raw
    Z_ex, names = gmy.excluded_instruments(p, "gmy")
    assert names == list(gmy.GMY_EXCLUDED)
    Z = gmy.drop_collinear(np.column_stack([X1, Z_ex]), keep=X1.shape[1])
    assert Z.shape == (9694, 112)
    lin = LinearIV(X1, Z)
    assert lin.rank_ZX1 == 107 and lin.rank_Z == 112


class _Rows:
    """A micro-type moment stand-in with ``k`` rows and a fixed covariance (shapes only)."""

    is_aggregate = False
    is_micro = True

    def __init__(self, name: str, k: int) -> None:
        self.name, self.k = name, k
        self.dataset = type("D", (), {"name": name})()
        self.fixed_covariance = np.eye(k)

    def bind(self, markets: Any, n: int, P: int | None = None) -> None:
        pass

    def requests(self) -> KernelRequests:
        return KernelRequests()

    def covariance_requests(self) -> KernelRequests:
        return KernelRequests()


class _ShapeModel:
    def __init__(self, n: int) -> None:
        self.n_products = n
        self.markets = [MarketIndex(0, np.arange(n))]


def test_order_condition_of_the_gmy_problem(gmy: Any, raw: tuple[Any, np.ndarray]) -> None:
    p, X1 = raw
    Z = gmy.drop_collinear(np.column_stack([X1, gmy.excluded_instruments(p, "gmy")[0]]), keep=X1.shape[1])
    layout = ParameterLayout([ParameterSpec(f"t{k}", "taste", "x", f"nu{k}", 0.0, -1.0, 1.0, unit=1.0)
                              for k in range(22)])
    lin = LinearIV(X1, Z)
    model = _ShapeModel(X1.shape[0])
    prob = NFPGMMProblem(model, [SarganIVMoments(Z), _Rows("micro_gmy", 33), _Rows("second_choice", 41)], lin,
                         layout)
    info = prob.identification
    assert (info["rank_aggregate"], info["k_micro"], info["unknowns"], info["overidentifying"]) == (112, 74, 129, 57)
    with pytest.raises(IdentificationError):
        NFPGMMProblem(model, [SarganIVMoments(Z)], lin, layout)
    refused = NFPGMMProblem(model, [SarganIVMoments(Z)], lin, layout, evaluate_only=True)
    assert refused.identification["refused"] and refused.identification["overidentifying"] == -17


def test_problem_without_aggregate_block() -> None:
    d = make_data(0)
    model = LogitRCModel(d["products"], d["agents"], d["market_ids"], d["shares"])
    lin = LinearIV(d["X1"], d["Z"])
    prob = NFPGMMProblem(model, [make_micro()], lin, make_layout())
    assert prob.agg is None and prob.k == 3
    assert prob.identification["k_linear"] == 0 and prob.identification["rank_aggregate"] == 0
    theta = np.array([0.6, 0.2, 0.5])
    ev = prob.evaluate(theta)
    assert ev.ok and ev.g.shape == (3,)
    fd = central_fd(lambda th: prob.evaluate(th, need_jacobian=False).g, theta, h=1e-6)
    np.testing.assert_allclose(ev.G, fd, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(ev.aux["xi"], lin.fit(ev.aux["delta"]).xi, rtol=0, atol=1e-14)
    assert ev.aux["beta"].shape == (3,)
    W1 = prob.initial_weighting(theta)
    assert W1.shape == (3, 3)
    J, names = prob.full_jacobian(ev)
    assert J.shape == (3, 3) and names == list(prob.layout.names)
    S = prob.covariance(ev)
    assert S.shape == (3, 3)


def test_block_shares_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    G = rng.normal(size=(7, 3))
    W = np.diag(rng.uniform(0.5, 2.0, 7))
    sh = block_shares(G, W, {"a": slice(0, 4), "b": slice(4, 7)}, ["x", "y", "z"])
    for k, nm in enumerate("xyz"):
        assert abs(sh[nm]["a"] + sh[nm]["b"] - 1.0) < 1e-12
        np.testing.assert_allclose(sh[nm]["a"], G[:4, k] @ W[:4, :4] @ G[:4, k] / (G[:, k] @ W @ G[:, k]),
                                   rtol=1e-12)

"""Generic nested-fixed-point GMM problem (DESIGN 2.1-2.2). Owned by WP4."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import scipy.linalg

from .linear import IdentificationError, LinearIV, matrix_rank
from .moments import MICRO_OUTPUTS
from .params import ParameterLayout
from .types import EvalStatus, Evaluation, InnerModel, InnerSolution, KernelRequests, LinearFit, Moment, StepLimit

__all__ = ["NFPGMMProblem", "IdentificationError"]


class NFPGMMProblem:
    """Implements ``types.Problem`` for any model with an inner inversion.

    ``__init__`` collects ``moment.requests()`` from every moment once and refuses to build the
    problem (``IdentificationError``) when the moment conditions cannot identify the unknowns:
    ``rank(Z_aggregate) + k_micro < P_free + K_linear``, or ``rank(Z' X1) < K_linear`` (LEARNINGS 16).
    ``evaluate`` solves the inner model, fits the linear parameters, evaluates every moment and applies
    the chain rule ``D_t = -(ds_t/ddelta_t)^{-1} (ds_t/dtheta)``,
    ``G_b = d_theta_b + sum_t d_delta_b[:, rows_t] @ D_t``. It never raises for model failures; it
    returns a non-OK ``EvalStatus``.

    ``Evaluation.aux`` holds ``delta``, ``xi``, ``beta``, ``beta_names``, ``D`` (n, P), ``sd_xi``,
    ``q`` and ``q_blocks`` (with the current W), ``micro_predicted`` / ``micro_data`` per micro block,
    ``per_market``, and the raw ``solution`` (InnerSolution) and ``linear_fit`` (LinearFit).

    ``max_step`` exists on the instance only when the model has ``max_step``, so the optimizer's
    ``hasattr`` check sees the model's capability.
    """

    layout: ParameterLayout

    def __init__(self, model: InnerModel, moments: Sequence[Moment], linear: Any, layout: ParameterLayout,
                 W: np.ndarray | None = None) -> None:
        """NFPGMMProblem(model, moments, linear, layout, W=None).

        W defaults to the aggregate block of ``linear`` (``(Z'Z/n)^{-1}`` for ``LinearIV``) and the
        identity on micro rows; ``initial_weighting(theta0)`` gives pyblp's W1.
        """
        self.model = model
        self.moments = list(moments)
        self.linear = linear
        self.layout = layout
        self.n = int(model.n_products)
        self.P = layout.P
        if not self.moments:
            raise ValueError("no moments")
        names = [m.name for m in self.moments]
        if len(set(names)) != len(names):
            raise ValueError(f"moment names must be unique: {names}")

        # block layout
        self.blocks: dict[str, slice] = {}
        start = 0
        for m in self.moments:
            k_b = getattr(m, "k", None)
            if k_b is None:
                raise ValueError(f"moment {m.name!r} must expose its row count as attribute 'k'")
            self.blocks[m.name] = slice(start, start + int(k_b))
            start += int(k_b)
        self.k = start

        self.aggregate = [m for m in self.moments if getattr(m, "is_aggregate", False)]
        self.micro = [m for m in self.moments if getattr(m, "is_micro", False)]
        datasets = [m.dataset.name for m in self.micro]  # type: ignore[attr-defined]
        if len(set(datasets)) != len(datasets):
            raise ValueError(f"one micro moment object per dataset (cross-object covariances are taken as zero): "
                             f"{datasets}")
        if len(self.aggregate) != 1:
            raise ValueError(f"exactly one aggregate (IV) moment block is required, got {len(self.aggregate)}")
        self.agg = self.aggregate[0]
        self.agg_rows = self.blocks[self.agg.name]
        if isinstance(linear, LinearIV):
            Z = getattr(self.agg, "Z")
            if Z.shape != linear.Z.shape or not np.array_equal(Z, linear.Z):
                raise ValueError("the aggregate moments and LinearIV must use the same Z")
        for m in self.micro:
            m.bind(model.markets, self.n, self.P)  # type: ignore[attr-defined]

        requests = KernelRequests()
        for m in self.moments:
            requests = requests.merge(m.requests())
        self.requests = requests
        cov_requests = requests
        for m in self.micro:
            cov_requests = cov_requests.merge(m.covariance_requests())  # type: ignore[attr-defined]
        self.covariance_requests = cov_requests

        self.identification = self.check_identification()

        self.W: np.ndarray
        self.W_provenance = ""
        if W is None:
            W = np.eye(self.k)
            W[self.agg_rows, self.agg_rows] = getattr(linear, "W_aa", np.eye(self.agg_rows.stop - self.agg_rows.start))
            self.set_weighting(W)
            self.W_provenance = "default: aggregate block of the linear fit, identity on micro rows"
        else:
            self.set_weighting(W)
            self.W_provenance = "given"

        self.accepted_theta: np.ndarray | None = None
        if hasattr(model, "max_step"):
            self.max_step = self._max_step

    # ------------------------------------------------------------------ setup checks

    def check_identification(self) -> dict[str, Any]:
        """Count moments against unknowns and refuse (``IdentificationError``) when they cannot identify them.

        Effective moment count = rank of the aggregate instrument matrix + number of micro rows.
        Unknowns = free nonlinear parameters + concentrated linear coefficients. LinearIV itself
        refuses when rank(Z'X1) < K at construction.
        """
        Z = getattr(self.agg, "Z")
        rank_z = matrix_rank(Z)
        k_micro = sum(int(getattr(m, "k")) for m in self.moments if m is not self.agg)
        p_free = self.layout.n_free()
        k_linear = int(getattr(self.linear, "n_linear", 0))
        effective = rank_z + k_micro
        unknowns = p_free + k_linear
        info = {
            "k": self.k, "k_aggregate": int(Z.shape[1]), "rank_aggregate": rank_z, "k_micro": k_micro,
            "effective_moments": effective, "p_free": p_free, "k_linear": k_linear, "unknowns": unknowns,
            "overidentifying": effective - unknowns,
            "rank_ZX1": getattr(self.linear, "rank_ZX1", None),
        }
        if rank_z < Z.shape[1]:
            info["note"] = f"aggregate instruments have rank {rank_z} < {Z.shape[1]} columns"
        if effective < unknowns:
            raise IdentificationError(
                f"moment conditions cannot identify the unknowns: rank(Z) {rank_z} + {k_micro} micro moments = "
                f"{effective} < {p_free} free nonlinear + {k_linear} linear parameters = {unknowns}. "
                f"Use Z = [X1, excluded instruments] (LEARNINGS 16), add moments, or fix parameters.")
        return info

    # ------------------------------------------------------------------ evaluation

    def _failed(self, theta: np.ndarray, status: EvalStatus, t0: float, aux: dict[str, Any]) -> Evaluation:
        return Evaluation(theta=theta, g=np.full(self.k, np.nan), G=None, status=status, blocks=dict(self.blocks),
                          aux=aux, seconds=time.perf_counter() - t0)

    def objective(self, g: np.ndarray, W: np.ndarray | None = None) -> float:
        """q = n g' W g with the current W unless one is given."""
        W = self.W if W is None else W
        return float(self.n * g @ W @ g)

    def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation:
        """evaluate(theta, need_jacobian=True) -> Evaluation."""
        t0 = time.perf_counter()
        theta = np.array(theta, dtype=float).ravel()
        if theta.shape != (self.P,):
            raise ValueError(f"theta has shape {theta.shape}, expected ({self.P},)")
        if not self.layout.in_bounds(theta):
            lower, upper = self.layout.bounds()
            bad = [n for n, v, lo, hi in zip(self.layout.names, theta, lower, upper)
                   if not (np.isfinite(v) and lo <= v <= hi)]
            return self._failed(theta, EvalStatus.OUT_OF_DOMAIN, t0, {"note": f"outside hard bounds: {bad}"})

        sol = self.model.solve(theta, self.requests, need_jacobian)
        aux: dict[str, Any] = {"per_market": sol.per_market, "solution": sol}
        if "realised_mass" in sol.outputs:  # frontier guard of DESIGN 4.4, read by the optimizer
            aux["realised_mass"] = sol.outputs["realised_mass"]
        if sol.status is not EvalStatus.OK:
            return self._failed(theta, sol.status, t0, aux)

        delta = np.asarray(sol.delta, dtype=float)
        fit = self.linear.fit(delta)
        D_full: np.ndarray | None = None
        if need_jacobian:
            if sol.D is None:
                raise ValueError("model returned no D although need_jacobian=True")
            D_full = np.zeros((self.n, self.P))
            for mkt, D_t in zip(self.model.markets, sol.D):
                D_full[mkt.rows] = D_t

        g = np.empty(self.k)
        G = np.zeros((self.k, self.P)) if need_jacobian else None
        aux.update(delta=delta, xi=fit.xi, beta=fit.beta, beta_names=fit.names, linear_fit=fit, D=D_full,
                   sd_xi=float(np.std(fit.xi)), micro_predicted={}, micro_data={})
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                for m in self.moments:
                    rows = self.blocks[m.name]
                    blk = m.evaluate(sol, fit, need_jacobian)
                    g[rows] = blk.value
                    if blk.data is not None:
                        aux["micro_data"][m.name] = blk.data
                        aux["micro_predicted"][m.name] = blk.value + blk.data
                    if G is not None:
                        if blk.d_theta is not None:
                            G[rows] += blk.d_theta
                        if blk.d_delta is not None:
                            assert D_full is not None
                            G[rows] += blk.d_delta @ D_full
        except (FloatingPointError, ZeroDivisionError) as exc:
            aux["note"] = f"floating point error in moments: {exc}"
            return self._failed(theta, EvalStatus.NUMERICAL, t0, aux)

        if not np.all(np.isfinite(g)) or (G is not None and not np.all(np.isfinite(G))):
            aux["note"] = "non-finite values in g or G"
            return self._failed(theta, EvalStatus.NUMERICAL, t0, aux)

        aux["q"] = self.objective(g)
        aux["q_blocks"] = {name: float(self.n * g[s] @ self.W[s, s] @ g[s]) for name, s in self.blocks.items()}
        return Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK, blocks=dict(self.blocks), aux=aux,
                          seconds=time.perf_counter() - t0)

    # ------------------------------------------------------------------ covariance and weighting

    def _covariance_solution(self, ev: Evaluation) -> InnerSolution:
        """A solution at ev.theta that carries the product parts the micro covariances need.

        Re-solves the model (warm start, no Jacobian) when the evaluation did not request them. The
        extra solve writes the model's trial state at ev.theta, which the next evaluate overwrites.
        """
        sol: InnerSolution = ev.aux["solution"]
        table = sol.outputs.get(MICRO_OUTPUTS, {})
        needed = [p.name for p in self.covariance_requests.micro_parts]
        if all(name in table for name in needed):
            return sol
        sol2 = self.model.solve(ev.theta, self.covariance_requests, False)
        if sol2.status is not EvalStatus.OK:
            raise RuntimeError(f"inner solve for the micro covariance failed at theta: {sol2.status.value}")
        return sol2

    def covariance(self, ev: Evaluation) -> np.ndarray:
        """covariance(ev) -> S(theta) = blockdiag(S_aggregate, n V_micro), (k, k)."""
        if not ev.ok:
            raise ValueError(f"covariance needs an OK evaluation, got {ev.status.value}")
        fit: LinearFit = ev.aux["linear_fit"]
        sol_cov = self._covariance_solution(ev) if self.micro else ev.aux["solution"]
        parts = []
        for m in self.moments:
            if getattr(m, "is_micro", False):
                parts.append(self.n * m.covariance_block(sol_cov, fit))
            else:
                parts.append(m.covariance_block(ev.aux["solution"], fit))
        S = scipy.linalg.block_diag(*parts)
        return 0.5 * (S + S.T)

    def micro_covariance(self, ev: Evaluation) -> np.ndarray | None:
        """V_micro stacked over micro moment objects (block diagonal), or None without micro moments."""
        if not self.micro:
            return None
        sol_cov = self._covariance_solution(ev)
        return scipy.linalg.block_diag(*[m.covariance_block(sol_cov, ev.aux["linear_fit"]) for m in self.micro])

    def initial_weighting(self, theta0: np.ndarray) -> np.ndarray:
        """pyblp's W1 = blockdiag((Z'Z/n)^{-1}, (n V_micro(theta_0))^{-1}) in this problem's block order."""
        from .weighting import weight_from_covariance

        Z = getattr(self.agg, "Z")
        W = np.zeros((self.k, self.k))
        W[self.agg_rows, self.agg_rows] = weight_from_covariance(Z.T @ Z / self.n)
        if self.micro:
            ev = self.evaluate(theta0, need_jacobian=False)
            if not ev.ok:
                raise RuntimeError(f"cannot form W1: evaluation at theta0 failed ({ev.status.value})")
            sol_cov = self._covariance_solution(ev)
            for m in self.micro:
                rows = self.blocks[m.name]
                W[rows, rows] = weight_from_covariance(self.n * m.covariance_block(sol_cov, ev.aux["linear_fit"]))
        return W

    def set_weighting(self, W: np.ndarray) -> None:
        """set_weighting(W): store W and re-concentrate the linear parameters with its aggregate block.

        Refuses a W that couples aggregate and other rows: concentrating beta with the aggregate block
        alone is exact only for block-diagonal W (DESIGN 2.4). Evaluations made before the call are stale.
        """
        W = np.asarray(W, dtype=float)
        if W.shape != (self.k, self.k):
            raise ValueError(f"W has shape {W.shape}, expected {(self.k, self.k)}")
        scale = max(float(np.max(np.abs(W))), 1e-300)
        if np.max(np.abs(W - W.T)) > 1e-10 * scale:
            raise ValueError("W must be symmetric")
        a = self.agg_rows
        other = np.ones(self.k, dtype=bool)
        other[a] = False
        if other.any() and np.max(np.abs(W[a][:, other])) > 1e-12 * scale:
            raise ValueError("W couples aggregate and micro rows; concentrated beta needs block-diagonal W")
        self.W = 0.5 * (W + W.T)
        self.linear.set_weighting(self.W[a, a])
        self.W_provenance = "set_weighting"

    def full_jacobian(self, ev: Evaluation) -> tuple[np.ndarray, list[str]]:
        """Jacobian over (beta, theta) for inference (DESIGN 2.4): aggregate rows ``[-Z'X1/n, Z'D/n]``,
        micro rows ``[0, G_micro]``. Returns (k, K + P) and the column names."""
        if ev.G is None:
            raise ValueError("full_jacobian needs an evaluation with G")
        K = int(getattr(self.linear, "K", 0))
        J = np.zeros((self.k, K + self.P))
        J[:, K:] = ev.G
        if K:
            J[self.agg_rows, :K] = self.linear.full_jacobian_linear_block()
        return J, list(getattr(self.linear, "names", ())) + list(self.layout.names)

    # ------------------------------------------------------------------ CUE

    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]:
        """cue_terms(ev) -> (V: C x k, dV: C x k x P), the per-cluster terms of the CUE gradient.

        Aggregate columns hold ``sum_{j in c} z_j xi_j`` and its theta derivative with M held at the
        current weighting; micro columns are zero (the micro block of S is held fixed, see
        ``cue_fixed_covariance``). ``g_aggregate = V.sum(0)[aggregate] / n``.
        """
        if not ev.ok or ev.G is None or ev.aux.get("D") is None:
            raise ValueError("cue_terms needs an OK evaluation with a Jacobian")
        fit: LinearFit = ev.aux["linear_fit"]
        assert fit.M is not None
        dxi = fit.M @ ev.aux["D"]
        Va, dVa = self.agg.cluster_terms(fit.xi, dxi)  # type: ignore[attr-defined]
        C = Va.shape[0]
        V = np.zeros((C, self.k))
        dV = np.zeros((C, self.k, self.P))
        V[:, self.agg_rows] = Va
        dV[:, self.agg_rows, :] = dVa
        return V, dV

    def cue_fixed_covariance(self, ev: Evaluation) -> np.ndarray:
        """The part of S CUE holds fixed: ``blockdiag(0, n V_micro(ev.theta))`` in this problem's row order."""
        S = np.zeros((self.k, self.k))
        if self.micro:
            sol_cov = self._covariance_solution(ev)
            for m in self.micro:
                rows = self.blocks[m.name]
                S[rows, rows] = self.n * m.covariance_block(sol_cov, ev.aux["linear_fit"])
        return S

    # ------------------------------------------------------------------ optional hooks

    def _max_step(self, ev: Evaluation, dtheta: np.ndarray) -> StepLimit:
        """max_step(ev, dtheta) -> StepLimit, delegated to the model."""
        return self.model.max_step(ev.aux["solution"], np.asarray(dtheta, dtype=float))

    def commit(self, theta: np.ndarray) -> None:
        """commit(theta): theta accepted, promote the model's trial state."""
        if hasattr(self.model, "commit"):
            self.model.commit(theta)
        self.accepted_theta = np.array(theta, dtype=float)

    def restore(self, theta: np.ndarray) -> None:
        """restore(theta): trial rejected, back to the model's accepted state."""
        if hasattr(self.model, "restore"):
            self.model.restore(theta)

    def state_dict(self) -> dict[str, Any]:
        """state_dict() -> warm starts etc. for checkpoints."""
        d: dict[str, Any] = {
            "W": self.W.copy(),
            "W_provenance": self.W_provenance,
            "accepted_theta": None if self.accepted_theta is None else self.accepted_theta.copy(),
        }
        if hasattr(self.model, "state_dict"):
            d["model"] = self.model.state_dict()  # type: ignore[attr-defined]
        return d

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """load_state_dict(d): restore what ``state_dict`` wrote."""
        if d.get("W") is not None:
            self.set_weighting(np.asarray(d["W"]))
        self.W_provenance = d.get("W_provenance", self.W_provenance)
        theta = d.get("accepted_theta")
        self.accepted_theta = None if theta is None else np.asarray(theta, dtype=float)
        if "model" in d and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(d["model"])  # type: ignore[attr-defined]

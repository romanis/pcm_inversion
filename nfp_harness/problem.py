"""Generic nested-fixed-point GMM problem (DESIGN 2.1-2.2). Owned by WP4."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import scipy.linalg

from .linear import IdentificationError, LinearIV, matrix_rank
from .moments import MICRO_OUTPUTS, IVMoments, SarganIVMoments, SupplyDomainError
from .params import ParameterLayout
from .types import EvalStatus, Evaluation, InnerModel, InnerSolution, KernelRequests, LinearFit, Moment, StepLimit
from .weighting import cluster_sums

__all__ = ["NFPGMMProblem", "IdentificationError"]


def _supply_summary(info: dict[str, Any]) -> dict[str, Any]:
    """The scalar diagnostics of a supply evaluation (``SupplyMoments.log_costs`` info) for ``aux``."""
    return {k: v for k, v in info.items() if np.isscalar(v)}


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

    Supply moments (at most one block with ``is_supply``, e.g. ``moments.SupplyMoments``) add the cost
    coefficients gamma as concentrated linear parameters. With W block diagonal, beta is concentrated with
    the aggregate block (``linear``) and gamma with the supply block. When W couples the aggregate and supply
    rows (step 2 with the cross terms of S), (beta, gamma) are concentrated jointly:
    ``b = (X'W_R X)^{-1} X'W_R m`` over the aggregate and supply rows R, with ``m = [Z'delta/n; Z_s' log mc/n]``
    and ``X = blockdiag(Z'X1/n, Z_s'w/n)``, so ``g_R = m - X b`` and ``G_R = dm - X (X'W_R X)^{-1} X'W_R dm``
    (exact for any such W; it needs plain ``IVMoments``). Then ``aux`` also holds ``gamma``, ``omega``,
    ``linear_coefficients`` = [beta, gamma], ``linear_names_all`` and ``supply`` (markup diagnostics). A
    supply block that cannot be evaluated (singular ownership Jacobian, mc <= 0) gives status NUMERICAL with
    the reason in ``aux["note"]``.

    ``max_step`` exists on the instance only when the model has ``max_step`` (and ``max_step=True``), so
    the optimizer's ``hasattr`` check sees the model's capability.
    """

    layout: ParameterLayout

    def __init__(self, model: InnerModel, moments: Sequence[Moment], linear: Any, layout: ParameterLayout,
                 W: np.ndarray | None = None, max_step: bool = True, evaluate_only: bool = False) -> None:
        """NFPGMMProblem(model, moments, linear, layout, W=None, max_step=True, evaluate_only=False).

        W defaults to the aggregate block of ``linear`` (``(Z'Z/n)^{-1}`` for ``LinearIV``) and the
        identity on micro rows; ``initial_weighting(theta0)`` gives pyblp's W1. ``max_step=False`` hides
        the model's ``max_step`` from the optimizer, so steps are controlled by the trust region alone.
        ``evaluate_only=True`` builds a problem that fails the order condition (for evaluation, never estimation):
        ``identification`` then holds the count with ``refused`` and the reason instead of raising.
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
        self.calibration = [m for m in self.moments if getattr(m, "is_calibration", False)]
        datasets = [m.dataset.name for m in self.micro]  # type: ignore[attr-defined]
        if len(set(datasets)) != len(datasets):
            raise ValueError(f"one micro moment object per dataset (cross-object covariances are taken as zero): "
                             f"{datasets}")
        if len(self.aggregate) > 1:
            raise ValueError(f"at most one aggregate (IV) moment block, got {len(self.aggregate)}")
        # Without an aggregate block (GMY's nonlinear stage: micro and second-choice moments only) beta is not a GMM
        # unknown: ``linear`` still fits it at every delta (its own default weighting) for xi and the reports.
        self.agg: Any = self.aggregate[0] if self.aggregate else None
        self.agg_rows = self.blocks[self.agg.name] if self.agg is not None else slice(0, 0)
        supply = [m for m in self.moments if getattr(m, "is_supply", False)]
        if len(supply) > 1:
            raise ValueError(f"at most one supply moment block, got {len(supply)}")
        self.supply: Any = supply[0] if supply else None
        self.supply_rows: slice | None = self.blocks[self.supply.name] if self.supply is not None else None
        # the step-2 W keeps the aggregate-supply cross terms of S only where they can be concentrated jointly
        # (plain IVMoments); with SarganIVMoments W2 is block diagonal across the two blocks
        self.coupled_blocks: tuple[tuple[str, ...], ...] = (
            ((self.agg.name, self.supply.name),) if self.supply is not None and self.agg is not None
            and type(self.agg) is IVMoments else ())
        self._joint: dict[str, Any] | None = None
        if self.supply is not None:
            try:
                units = np.asarray(layout.units(), dtype=float)
            except ValueError:
                units = None
            self.supply.bind(model, self.P, units=units)
            if self.supply.n != self.n:
                raise ValueError(f"supply block has {self.supply.n} rows, model has {self.n} products")
        if isinstance(linear, LinearIV) and self.agg is not None:
            Z = getattr(self.agg, "Z")
            if Z.shape != linear.Z.shape or not np.array_equal(Z, linear.Z):
                raise ValueError("the aggregate moments and LinearIV must use the same Z")
        for m in self.moments:
            if m in self.micro:
                m.bind(model.markets, self.n, self.P)  # type: ignore[attr-defined]
            if hasattr(m, "bind_agents"):   # household-level micro moments (GroupMicroMoments)
                agents, draws = getattr(model, "agents", None), getattr(model, "market_draws", None)
                if agents is None or draws is None:
                    raise ValueError(f"moment {m.name!r} needs a model with an agent table and market_draws")
                m.bind_agents(agents, [np.asarray(draws(t)["agent_rows"]) for t in range(len(model.markets))])

        requests = KernelRequests()
        for m in self.moments:
            requests = requests.merge(m.requests())
        self.requests = requests
        cov_requests = requests
        for m in self.micro:
            cov_requests = cov_requests.merge(m.covariance_requests())  # type: ignore[attr-defined]
        self.covariance_requests = cov_requests

        self.evaluate_only = bool(evaluate_only)
        self.identification = self.check_identification(raise_error=not self.evaluate_only)

        self.W: np.ndarray
        self.W_provenance = ""
        if W is None:
            W = np.eye(self.k)
            if self.agg is not None:
                W[self.agg_rows, self.agg_rows] = getattr(linear, "W_aa",
                                                          np.eye(self.agg_rows.stop - self.agg_rows.start))
            if self.supply is not None:
                W[self.supply_rows, self.supply_rows] = self.supply.W_ss
            self.set_weighting(W)
            self.W_provenance = "default: aggregate block of the linear fit, identity on micro rows"
        else:
            self.set_weighting(W)
            self.W_provenance = "given"

        self.accepted_theta: np.ndarray | None = None
        if max_step and hasattr(model, "max_step"):
            self.max_step = self._max_step

    # ------------------------------------------------------------------ setup checks

    def check_identification(self, raise_error: bool = True) -> dict[str, Any]:
        """Count moments against unknowns and refuse (``IdentificationError``) when they cannot identify them.

        Effective moment count = rank of the aggregate instrument matrix + number of micro rows.
        Unknowns = free nonlinear parameters + concentrated linear coefficients. LinearIV itself
        refuses when rank(Z'X1) < K at construction. Without an aggregate block rank(Z) counts 0 and beta is no
        unknown of this problem (it is fitted outside the GMM, as GMY's linear stage).
        """
        Z = getattr(self.agg, "Z") if self.agg is not None else np.zeros((self.n, 0))
        rank_z = matrix_rank(Z)
        k_micro = sum(int(getattr(m, "k")) for m in self.moments if m is not self.agg and m is not self.supply)
        p_free = self.layout.n_free()
        k_linear = int(getattr(self.linear, "n_linear", 0)) if self.agg is not None else 0
        k_supply = 0
        if self.supply is not None:
            k_supply = matrix_rank(self.supply.Z)
            k_linear += int(self.supply.K)
        effective = rank_z + k_micro + k_supply
        unknowns = p_free + k_linear
        info = {
            "k": self.k, "k_aggregate": int(Z.shape[1]), "rank_aggregate": rank_z, "k_micro": k_micro,
            "rank_supply": k_supply,
            "effective_moments": effective, "p_free": p_free, "k_linear": k_linear, "unknowns": unknowns,
            "overidentifying": effective - unknowns,
            "rank_ZX1": getattr(self.linear, "rank_ZX1", None),
        }
        if rank_z < Z.shape[1]:
            info["note"] = f"aggregate instruments have rank {rank_z} < {Z.shape[1]} columns"
        if effective < unknowns:
            exc = IdentificationError(
                f"moment conditions cannot identify the unknowns: rank(Z) {rank_z} + {k_micro} micro moments"
                + (f" + rank(Z_s) {k_supply}" if self.supply is not None else "")
                + f" = {effective} < {p_free} free nonlinear + {k_linear} linear parameters = {unknowns}. "
                f"Use Z = [X1, excluded instruments] (LEARNINGS 16), add moments, or fix parameters.")
            if raise_error:
                raise exc
            info.update(refused=True, note=str(exc))
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
        D_full: np.ndarray | None = None
        if need_jacobian:
            if sol.D is None:
                raise ValueError("model returned no D although need_jacobian=True")
            D_full = np.zeros((self.n, self.P))
            for mkt, D_t in zip(self.model.markets, sol.D):
                D_full[mkt.rows] = D_t

        sup = self.supply
        y_s = dy_s = None
        if sup is not None:
            try:
                y_s, dy_s, s_info = sup.log_costs(sol, need_jacobian)
            except SupplyDomainError as exc:
                aux["note"] = str(exc)
                aux["supply"] = _supply_summary(exc.info)
                return self._failed(theta, EvalStatus.NUMERICAL, t0, aux)
            aux["supply"] = _supply_summary(s_info)

        joint = None
        if self._joint is not None:
            assert y_s is not None
            joint = self._joint_fit(delta, D_full, y_s, dy_s)
            fit = joint["fit"]
        else:
            fit = self.linear.fit(delta)

        g = np.empty(self.k)
        G = np.zeros((self.k, self.P)) if need_jacobian else None
        aux.update(delta=delta, xi=fit.xi, beta=fit.beta, beta_names=fit.names, linear_fit=fit, D=D_full,
                   sd_xi=float(np.std(fit.xi)), micro_predicted={}, micro_data={})
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                for m in self.moments:
                    rows = self.blocks[m.name]
                    if joint is not None and (m is self.agg or m is sup):
                        sel = joint["rows_of"][m.name]
                        g[rows] = joint["g"][sel]
                        if G is not None:
                            G[rows] = joint["G"][sel]
                        continue
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
                if sup is not None:
                    assert y_s is not None
                    gamma, omega = (joint["gamma"], joint["omega"]) if joint is not None else sup.concentrate(y_s)
                    aux.update(gamma=gamma, omega=omega, log_mc=y_s,
                               linear_coefficients=np.r_[fit.beta, gamma],
                               linear_names_all=tuple(fit.names) + tuple(sup.names))
        except (FloatingPointError, ZeroDivisionError) as exc:
            aux["note"] = f"floating point error in moments: {exc}"
            return self._failed(theta, EvalStatus.NUMERICAL, t0, aux)

        if not np.all(np.isfinite(g)) or (G is not None and not np.all(np.isfinite(G))):
            aux["note"] = "non-finite values in g or G"
            return self._failed(theta, EvalStatus.NUMERICAL, t0, aux)

        for m in self.calibration:
            summary = getattr(m, "summary", None)
            if callable(summary):
                aux[m.name] = summary(sol)
        for m in self.moments:
            extra = getattr(m, "aux_summary", None)
            if callable(extra):
                aux[m.name] = extra(sol)
        aux["q"] = self.objective(g)
        aux["q_blocks"] = {name: float(self.n * g[s] @ self.W[s, s] @ g[s]) for name, s in self.blocks.items()}
        return Evaluation(theta=theta, g=g, G=G, status=EvalStatus.OK, blocks=dict(self.blocks), aux=aux,
                          seconds=time.perf_counter() - t0)

    def _joint_fit(self, delta: np.ndarray, D_full: np.ndarray | None, y: np.ndarray, dy: np.ndarray | None
                   ) -> dict[str, Any]:
        """Joint concentration of (beta, gamma) when W couples the aggregate and supply rows (class docstring)."""
        j = self._joint
        assert j is not None and self.supply is not None
        sup, n = self.supply, self.n
        Z = getattr(self.agg, "Z")
        K = int(self.linear.K)
        m = np.r_[Z.T @ delta / n, sup.Z.T @ y / n]
        b = j["Pj"] @ m
        beta, gamma = b[:K], b[K:]
        g = m - j["X"] @ b
        G = None
        if D_full is not None:
            assert dy is not None
            dm = np.vstack([Z.T @ D_full / n, sup.Z.T @ dy / n])
            G = dm - j["X"] @ (j["Pj"] @ dm)
        xi = delta - self.linear.X1 @ beta
        fit = LinearFit(names=tuple(self.linear.names), beta=beta, xi=xi, M=None)
        return {"fit": fit, "gamma": gamma, "omega": y - sup.w @ gamma, "g": g, "G": G, "rows_of": j["rows_of"]}

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
            elif m is self.supply:
                parts.append(m.covariance_from(ev.aux["omega"]))
            else:
                parts.append(m.covariance_block(ev.aux["solution"], fit))
        S = scipy.linalg.block_diag(*parts)
        if self.supply is not None:
            cross = self.supply_cross_covariance(fit.xi, ev.aux["omega"])
            S[self.agg_rows, self.supply_rows] = cross
            S[self.supply_rows, self.agg_rows] = cross.T
        return 0.5 * (S + S.T)

    def supply_cross_covariance(self, xi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Clustered ``(1/n) sum_c (sum_{j in c} z_j xi_j)(sum_{j in c} z_sj omega_j)'`` (k_a, k_s).

        Both blocks must use the same clusters (else the cross term is not defined and this raises). With
        Sargan aggregate moments the aggregate side is divided by ``sqrt(xi'xi/n)`` as its moments are.
        """
        sup = self.supply
        assert sup is not None
        ids_a, ids_s = getattr(self.agg, "clustering_ids", None), sup.clustering_ids
        if (ids_a is None) != (ids_s is None) or (ids_a is not None and not np.array_equal(ids_a, ids_s)):
            raise ValueError("aggregate and supply moments use different clusters; their cross covariance is "
                             "not defined")
        Z = getattr(self.agg, "Z")
        Va = cluster_sums(Z * xi[:, None], ids_a)
        Vs = cluster_sums(sup.Z * omega[:, None], ids_s)
        cross = Va.T @ Vs / self.n
        if isinstance(self.agg, SarganIVMoments):
            cross = cross / float(np.sqrt(xi @ xi / self.n))
        return cross

    def micro_covariance(self, ev: Evaluation) -> np.ndarray | None:
        """V_micro stacked over micro moment objects (block diagonal), or None without micro moments."""
        if not self.micro:
            return None
        sol_cov = self._covariance_solution(ev)
        return scipy.linalg.block_diag(*[m.covariance_block(sol_cov, ev.aux["linear_fit"]) for m in self.micro])

    def initial_weighting(self, theta0: np.ndarray) -> np.ndarray:
        """pyblp's W1 = blockdiag((Z'Z/n)^{-1}, (Z_s'Z_s/n)^{-1}, (n V_micro(theta_0))^{-1}) in this problem's
        block order (the supply block only with supply moments)."""
        from .weighting import weight_from_covariance

        W = np.zeros((self.k, self.k))
        if self.agg is not None:
            Z = getattr(self.agg, "Z")
            W[self.agg_rows, self.agg_rows] = weight_from_covariance(Z.T @ Z / self.n)
        if self.supply is not None:
            Zs = self.supply.Z
            W[self.supply_rows, self.supply_rows] = weight_from_covariance(Zs.T @ Zs / self.n)
        for m in self.calibration:   # a calibration row's covariance is a declared constant
            rows = self.blocks[m.name]
            W[rows, rows] = weight_from_covariance(np.atleast_2d(m.covariance_block(None, None)))  # type: ignore
        if self.micro and all(getattr(m, "fixed_covariance", None) is not None for m in self.micro):
            for m in self.micro:   # a fixed V_micro needs no evaluation at theta0
                rows = self.blocks[m.name]
                W[rows, rows] = weight_from_covariance(self.n * m.fixed_covariance)  # type: ignore[attr-defined]
        elif self.micro:
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

        Refuses a W that couples aggregate (or supply) rows with micro rows: concentrating the linear
        parameters on those rows alone is exact only then (DESIGN 2.4). A W that couples the aggregate and
        supply rows switches to the joint concentration of (beta, gamma) (class docstring). Evaluations made
        before the call are stale.
        """
        W = np.asarray(W, dtype=float)
        if W.shape != (self.k, self.k):
            raise ValueError(f"W has shape {W.shape}, expected {(self.k, self.k)}")
        scale = max(float(np.max(np.abs(W))), 1e-300)
        if np.max(np.abs(W - W.T)) > 1e-10 * scale:
            raise ValueError("W must be symmetric")
        a = self.agg_rows
        lin = np.zeros(self.k, dtype=bool)
        lin[a] = True
        if self.supply_rows is not None:
            lin[self.supply_rows] = True
        if (~lin).any() and lin.any() and np.max(np.abs(W[lin][:, ~lin])) > 1e-12 * scale:
            raise ValueError("W couples aggregate (or supply) and micro rows; concentrated linear parameters need "
                             "block-diagonal W between them")
        self.W = 0.5 * (W + W.T)
        if self.agg is not None:
            self.linear.set_weighting(self.W[a, a])
        self._joint = None
        if self.supply is not None:
            s = self.supply_rows
            assert s is not None
            self.supply.set_weighting(self.W[s, s])
            if self.agg is not None and np.max(np.abs(self.W[a, s])) > 1e-12 * scale:
                self._joint = self._joint_projection()
        self.W_provenance = "set_weighting"

    def _joint_projection(self) -> dict[str, Any]:
        """``X``, ``Pj = (X'W_R X)^{-1} X'W_R`` and the row map of the joint (beta, gamma) concentration."""
        if type(self.agg) is not IVMoments or not isinstance(self.linear, LinearIV):
            raise ValueError("a W coupling aggregate and supply rows needs plain IVMoments and LinearIV")
        sup = self.supply
        a, s = self.agg_rows, self.supply_rows
        assert sup is not None and s is not None
        idx = np.r_[np.arange(a.start, a.stop), np.arange(s.start, s.stop)]
        ka, K = a.stop - a.start, int(self.linear.K)
        X = np.zeros((idx.size, K + sup.K))
        X[:ka, :K] = self.linear.ZtX1 / self.n
        X[ka:, K:] = sup.ZtW
        W_R = self.W[np.ix_(idx, idx)]
        B = X.T @ W_R
        Pj = np.linalg.solve(B @ X, B)
        return {"X": X, "Pj": Pj, "rows_of": {self.agg.name: slice(0, ka), sup.name: slice(ka, idx.size)}}

    def full_jacobian(self, ev: Evaluation) -> tuple[np.ndarray, list[str]]:
        """Jacobian over (beta, theta) for inference (DESIGN 2.4): aggregate rows ``[-Z'X1/n, Z'D/n]``,
        micro rows ``[0, G_micro]``; with supply moments the cost coefficients follow beta and the supply rows
        are ``[0, -Z_s'w/n, G_supply]``. Returns (k, K [+ K_s] + P) and the column names."""
        if ev.G is None:
            raise ValueError("full_jacobian needs an evaluation with G")
        K = int(getattr(self.linear, "K", 0)) if self.agg is not None else 0
        Ks = int(self.supply.K) if self.supply is not None else 0
        J = np.zeros((self.k, K + Ks + self.P))
        J[:, K + Ks:] = ev.G
        if K:
            J[self.agg_rows, :K] = self.linear.full_jacobian_linear_block()
        names = list(getattr(self.linear, "names", ())) if self.agg is not None else []
        if self.supply is not None:
            J[self.supply_rows, K:K + Ks] = -self.supply.ZtW
            names += list(self.supply.names)
        return J, names + list(self.layout.names)

    # ------------------------------------------------------------------ CUE

    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]:
        """cue_terms(ev) -> (V: C x k, dV: C x k x P), the per-cluster terms of the CUE gradient.

        Aggregate columns hold ``sum_{j in c} z_j xi_j`` and its theta derivative with M held at the
        current weighting; micro columns are zero (the micro block of S is held fixed, see
        ``cue_fixed_covariance``). ``g_aggregate = V.sum(0)[aggregate] / n``.
        """
        if self.supply is not None:
            raise NotImplementedError("CUE with supply moments is not implemented")
        if self.agg is None:
            raise NotImplementedError("CUE needs an aggregate moment block")
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

    def history_extras(self, ev: Evaluation, accepted: bool) -> dict[str, float]:
        """history_extras(ev, accepted) -> {"sd_delta", "mean_alpha_buyers"} for the optimizer's history: the sd
        of delta at every evaluated point and, at accepted points of a PCM model (one with ``kernel_market``),
        the mean price sensitivity of predicted buyers (``pcm_post.spread_in_buyer_dollars``, LEARNINGS 25)."""
        out: dict[str, float] = {}
        delta = ev.aux.get("delta") if ev.aux else None
        if delta is None:
            return out
        out["sd_delta"] = float(np.std(delta))
        if accepted and hasattr(self.model, "kernel_market") and hasattr(self.model, "kernel_params"):
            from .models.pcm_post import spread_in_buyer_dollars

            out["mean_alpha_buyers"] = float(spread_in_buyer_dollars(self.model, ev.theta, delta)["mean_alpha_buyers"])
        return out

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

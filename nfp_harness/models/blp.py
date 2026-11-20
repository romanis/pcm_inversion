"""Random-coefficients logit (BLP) inner model on the harness's ``InnerModel`` interface (DESIGN 3b).

    u_ijt = delta_jt + sum_k C_ik x_jtk + eps_ijt,  eps type-1 extreme value,  u_i0t = 0
    C_ik  = sum_{(k, d) in pattern} pi_kd D_id + sum_{k in unobs} sigma_k nu_ik

theta is ``(pi in pattern order, sigma in unobs order)``. Each agent i of market t (weight w_i, demographics D_i, taste
draws nu_i) is one simulated consumer; market shares are s_t = P_t w_t with P_t (J_t, H_t) the logit choice
probabilities. ``solve`` inverts every market by the BLP contraction accelerated by SQUAREM (Varadhan and Roland 2008),
in parallel over markets, from the accepted delta (``commit`` / ``restore``, a warm start ``warm_from_delta``), and
returns the outputs moments ask for: ``shares``, ``probabilities`` (P_t per market), logit second-choice matrices for
``SecondChoiceMoments`` and household sums for ``GroupMicroMoments`` (both without derivatives).

The model has no analytic derivatives yet: ``solve(..., need_jacobian=True)`` raises, so a problem on it is evaluated,
or fitted by finite differences, not by the Gauss-Newton optimizer. It knows nothing about moments, weighting or papers.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

from ..second_choice import logit_second_choice_matrix
from ..types import EvalStatus, InnerSolution, KernelRequests, MarketIndex

__all__ = ["MixedLogitModel"]


class MixedLogitModel:
    """MixedLogitModel(markets, X, D, weights, nu, pattern, unobs, shares, threads=1, tol=1e-13, max_iterations=5000,
    product_columns=None).

    Per market t (in ``markets`` order): ``X[t]`` (J_t, K) the characteristics with a coefficient, ``D[t]`` (H_t, Dm)
    the agents' demographics, ``weights[t]`` (H_t,) their weights, ``nu[t]`` (H_t, len(unobs)) their taste draws,
    ``shares[t]`` (J_t,) the data shares. ``pattern`` lists (X column, D column) pairs with a pi, ``unobs`` the X columns
    with a sigma. ``product_columns`` ({name: (n,) array in product-table order}) are the columns group sums may ask
    for; ``agents`` and ``agent_rows`` (the agent-table rows of each market's agents) are what household-level micro
    moments bind to. The contraction starts at the logit inversion log s_j - log s_0 until a delta is accepted."""

    def __init__(self, markets: Sequence[MarketIndex], X: Sequence[np.ndarray], D: Sequence[np.ndarray],
                 weights: Sequence[np.ndarray], nu: Sequence[np.ndarray], pattern: Sequence[tuple[int, int]],
                 unobs: Sequence[int], shares: Sequence[np.ndarray], threads: int = 1, tol: float = 1e-13,
                 max_iterations: int = 5000, product_columns: Mapping[str, np.ndarray] | None = None,
                 agents: Any = None, agent_rows: Sequence[np.ndarray] | None = None) -> None:
        self.markets = list(markets)
        self.n_products = int(sum(np.asarray(m.rows).size for m in self.markets))
        self.X, self.D, self.w, self.nu = list(X), list(D), list(weights), list(nu)
        self.H = [int(np.asarray(w).size) for w in self.w]
        self.pattern = [(int(c), int(d)) for c, d in pattern]
        self.unobs = [int(c) for c in unobs]
        self.K = self.X[0].shape[1] if self.X else 0
        self.tol, self.max_iterations = float(tol), int(max_iterations)
        self.log_s = [np.log(np.asarray(s, dtype=float)) for s in shares]
        s0 = [np.exp(ls).sum() for ls in self.log_s]
        self.delta0 = [ls - np.log(1 - s) for ls, s in zip(self.log_s, s0)]   # the logit inversion
        self.delta = [d.copy() for d in self.delta0]                           # accepted (warm start)
        self._last: list[np.ndarray] | None = None
        self.product_columns = dict(product_columns or {})
        self.agents = agents                     # the agent table and each market's rows of it, for household-level
        self.agent_rows = None if agent_rows is None else [np.asarray(r) for r in agent_rows]   # micro moments
        self.pool = ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(self.markets))))
        self.evaluations = 0

    @property
    def P(self) -> int:
        return len(self.pattern) + len(self.unobs)

    # ------------------------------------------------------------------ one market

    def coefficients(self, theta: np.ndarray, t: int) -> np.ndarray:
        """(K, H_t) coefficient on each characteristic for each agent of market t."""
        C = np.zeros((self.K, self.H[t]))
        for k, (c, d) in enumerate(self.pattern):
            C[c] += theta[k] * self.D[t][:, d]
        for m, c in enumerate(self.unobs):
            C[c] += theta[len(self.pattern) + m] * self.nu[t][:, m]
        return C

    @staticmethod
    def probabilities_at(delta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """(J, H) logit choice probabilities at mean utilities ``delta`` and deviations ``mu`` (J, H)."""
        u = delta[:, None] + mu
        m = np.maximum(u.max(axis=0), 0.0)
        e = np.exp(u - m)
        return e / (np.exp(-m) + e.sum(axis=0))

    def solve_market(self, theta: np.ndarray, t: int, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """BLP contraction with SQUAREM from ``delta``; (delta, P (J, H), iterations). Raises when it does not reach
        ``tol`` in ``max_iterations``."""
        mu = self.X[t] @ self.coefficients(theta, t)
        w, ls = self.w[t], self.log_s[t]

        def g(d: np.ndarray) -> np.ndarray:
            return d + ls - np.log(self.probabilities_at(d, mu) @ w)

        x = delta.copy()
        for it in range(1, self.max_iterations + 1):
            x1 = g(x)
            r = x1 - x
            if np.max(np.abs(r)) < self.tol:
                x = x1
                break
            x2 = g(x1)
            v = x2 - 2 * x1 + x
            nv = float(v @ v)
            step = -np.sqrt(float(r @ r) / nv) if nv > 0 else -1.0
            step = min(step, -1.0)
            xn = x - 2 * step * r + step * step * v
            x = g(xn) if np.all(np.isfinite(xn)) else x2
        else:
            raise RuntimeError(f"contraction did not converge in market {self.markets[t].market_id}")
        return x, self.probabilities_at(x, mu), it

    # ------------------------------------------------------------------ all markets

    def solve_all(self, theta: np.ndarray, warm: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """(delta per market, P per market) at theta, from the accepted delta (``warm``) or the logit inversion."""
        theta = np.asarray(theta, dtype=float)
        start = self.delta if warm else self.delta0
        res = list(self.pool.map(lambda t: self.solve_market(theta, t, start[t]), range(len(self.markets))))
        self.evaluations += 1
        self._last = [r[0] for r in res]
        return [r[0] for r in res], [r[1] for r in res]

    def solve(self, theta: np.ndarray, requests: KernelRequests, need_jacobian: bool) -> InnerSolution:
        """``InnerModel.solve``: every market inverted from the accepted delta, with ``shares``, ``probabilities`` and
        the requested ``second_choice`` and ``group_sums`` outputs (module docstring)."""
        if need_jacobian:
            raise NotImplementedError("MixedLogitModel has no derivatives: evaluate with need_jacobian=False or fit by "
                                      "finite differences")
        t0 = time.time()
        deltas, Ps = self.solve_all(theta)
        delta = np.empty(self.n_products)
        shares = np.empty(self.n_products)
        for m, d, P, w in zip(self.markets, deltas, Ps, self.w):
            delta[np.asarray(m.rows)] = d
            shares[np.asarray(m.rows)] = P @ w
        outputs: dict[str, Any] = {"shares": shares, "probabilities": Ps}
        if requests.second_choice:
            want = {str(x) for x in requests.second_choice_markets}
            outputs["second_choice"] = [
                {"C": logit_second_choice_matrix(P, w), "s": P @ w} if str(m.market_id) in want else None
                for m, P, w in zip(self.markets, Ps, self.w)]
        if requests.group_sums:
            cols = list(requests.group_sums)
            missing = [c for c in cols if c not in self.product_columns]
            if missing:
                raise KeyError(f"group sums of columns the model was not given: {missing}")
            out = []
            for m, P, w in zip(self.markets, Ps, self.w):
                x = np.column_stack([np.ones(P.shape[0])] + [np.asarray(self.product_columns[c],
                                                                        dtype=float)[np.asarray(m.rows)] for c in cols])
                out.append({"columns": cols, "S": (x.T @ P).T * w[:, None], "d_delta": None, "d_theta": None, "w": w})
            outputs["group_sums"] = out
        seconds = time.time() - t0
        per = [{"market_id": m.market_id, "status": "ok", "seconds": seconds} for m in self.markets]
        return InnerSolution(delta=delta, D=None, status=EvalStatus.OK, per_market=per, outputs=outputs)

    def market_draws(self, t: int) -> dict[str, np.ndarray]:
        """The agent-table rows of market t's agents (one draw per agent), for ``GroupMicroMoments.bind_agents``."""
        if self.agent_rows is None:
            raise ValueError("MixedLogitModel was built without agent_rows")
        return {"agent_rows": self.agent_rows[t], "agent_of_draw": np.arange(self.H[t])}

    def commit(self, theta: np.ndarray) -> None:
        """Accept the last solution's delta as the warm start of later solves."""
        if self._last is not None:
            self.delta = [d.copy() for d in self._last]

    def restore(self, theta: np.ndarray) -> None:
        self._last = None

    def warm_from_delta(self, delta: np.ndarray) -> None:
        """Accept ``delta`` ((n,) in product-table order) as the warm start."""
        self.delta = [np.asarray(delta, dtype=float)[np.asarray(m.rows)].copy() for m in self.markets]

    def close(self) -> None:
        self.pool.shutdown(wait=False)

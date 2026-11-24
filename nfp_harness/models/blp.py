"""Random-coefficients logit (BLP) inner model on the harness's ``InnerModel`` interface (DESIGN 3b).

    u_ijt = delta_jt + sum_k C_ik x_jtk + eps_ijt,  eps type-1 extreme value,  u_i0t = 0
    C_ik  = sum_{(k, d) in pattern} pi_kd D_id + sum_{k in unobs} sigma_k nu_ik

theta is ``(pi in pattern order, sigma in unobs order)``, then, with an exponential law (``exp_column``), the
coefficients of ``exp_terms``: agent h's coefficient on X column ``exp_column`` gains ``-exp(e_h)`` with ``e_h =
offset_h + sum_e theta_e z_he`` (``exp_offset`` per market, z_he a demographic or a taste draw), a lognormal price
sensitivity such as the PCM's ``log alpha_h = log c_h + phi' d_h + sigma_p nu_h``. Its basis is ``b_he = -exp(e_h)
z_he`` on that column, so every derivative below holds with that basis. Each agent i of market t (weight w_i, demographics D_i, taste
draws nu_i) is one simulated consumer; market shares are s_t = P_t w_t with P_t (J_t, H_t) the logit choice
probabilities. ``solve`` inverts every market by the BLP contraction accelerated by SQUAREM (Varadhan and Roland 2008),
in parallel over markets, from the accepted delta (``commit`` / ``restore``, a warm start ``warm_from_delta``), and
returns the outputs moments ask for: ``shares``, ``probabilities`` (P_t per market), logit second-choice matrices for
``SecondChoiceMoments`` and household sums for ``GroupMicroMoments``.

Derivatives (``need_jacobian=True``). Theta parameter p moves the utility of product j for agent h by
``dmu_jhp = x_j,c(p) b_hp``, with c(p) its X column and b_hp its basis (D_hd for a pi, nu_hm for a sigma). With
``w_h P_jh`` the purchase mass:

- ``ds/ddelta = diag(s) - (P w) P'`` and ``ds_j/dtheta_p = sum_h w_h P_jh b_hp (x_jc(p) - xbar_hc(p))``, xbar_h = P_h' X;
- ``D_t = d delta / d theta = -(ds/ddelta)^{-1} ds/dtheta`` by the implicit function theorem, per market;
- ``ds_dp`` (``KernelRequests(price_derivatives=True)``): ``ds_j/dp_k = sum_h w_h P_jh a_h (1[j = k] - P_kh)`` with
  ``a_h = price_coefficient + C_h,price`` the agent's price coefficient (``price_column`` must be set; the linear
  coefficient is the caller's, the model does not know beta);
- group sums: ``dS_hc / ddelta_m = w_h P_mh (U_mc - Ubar_hc)`` and ``dS_hc / dtheta_p = w_h b_hp (sum_j P_jh U_jc
  x_jc(p) - Ubar_hc xbar_hc(p))`` with U = [1, product columns], Ubar_h = P_h' U;
- second-choice adjoint: ``sum_jk A_cjk dN_jk - sum_j B_cj ds_j`` with ``N_jk = sum_h w_h P_jh P_kh / (R_h - P_jh)``
  (k != j, R_h = sum_l P_lh), the contraction's gradient in P_lh pulled back through the logit derivatives
  (``second_choice_adjoint``).

Products are in each market's ``rows`` order. The model knows nothing about moments, weighting or papers. Micro parts
(``RatioMicroMoment``) are not served.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import numpy as np

from ..second_choice import inside_mass_without, logit_second_choice_matrix
from ..types import EvalStatus, InnerSolution, KernelRequests, MarketIndex

__all__ = ["MixedLogitModel"]


class MixedLogitModel:
    """MixedLogitModel(markets, X, D, weights, nu, pattern, unobs, shares, threads=1, tol=1e-13, max_iterations=5000,
    product_columns=None, agents=None, agent_rows=None, price_column=None, price_coefficient=0.0, exp_column=None,
    exp_terms=(), exp_offset=None).

    Per market t (in ``markets`` order): ``X[t]`` (J_t, K) the characteristics with a coefficient, ``D[t]`` (H_t, Dm)
    the agents' demographics, ``weights[t]`` (H_t,) their weights, ``nu[t]`` (H_t, len(unobs)) their taste draws,
    ``shares[t]`` (J_t,) the data shares. ``pattern`` lists (X column, D column) pairs with a pi, ``unobs`` the X columns
    with a sigma. ``product_columns`` ({name: (n,) array in product-table order}) are the columns group sums may ask
    for; ``agents`` and ``agent_rows`` (the agent-table rows of each market's agents) are what household-level micro
    moments bind to. ``price_column`` (an X column) and ``price_coefficient`` (the mean linear price coefficient in
    delta) define ``ds_dp``; with ``price_column`` the model also carries what supply moments bind to
    (``spec.share_column``, ``products``, ``kernel_market(t).p``). ``exp_column`` (an X column), ``exp_terms``
    (``("D", d)`` or ``("nu", m)`` per exponential coefficient) and ``exp_offset`` (per market (H_t,), default 0)
    define the exponential law (module docstring). The contraction starts at the logit inversion log s_j - log s_0
    until a delta is accepted."""

    def __init__(self, markets: Sequence[MarketIndex], X: Sequence[np.ndarray], D: Sequence[np.ndarray],
                 weights: Sequence[np.ndarray], nu: Sequence[np.ndarray], pattern: Sequence[tuple[int, int]],
                 unobs: Sequence[int], shares: Sequence[np.ndarray], threads: int = 1, tol: float = 1e-13,
                 max_iterations: int = 5000, product_columns: Mapping[str, np.ndarray] | None = None,
                 agents: Any = None, agent_rows: Sequence[np.ndarray] | None = None, price_column: int | None = None,
                 price_coefficient: float = 0.0, exp_column: int | None = None,
                 exp_terms: Sequence[tuple[str, int]] = (), exp_offset: Sequence[np.ndarray] | None = None) -> None:
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
        self.price_column = None if price_column is None else int(price_column)
        self.price_coefficient = float(price_coefficient)
        self.exp_column = None if exp_column is None else int(exp_column)
        self.exp_terms = [(str(k), int(i)) for k, i in exp_terms]
        if self.exp_terms and self.exp_column is None:
            raise ValueError("exp_terms need exp_column")
        if any(k not in ("D", "nu") for k, _ in self.exp_terms):
            raise ValueError(f"exp_terms kinds must be 'D' or 'nu', got {self.exp_terms}")
        self.exp_offset = ([np.zeros(h) for h in self.H] if exp_offset is None
                           else [np.asarray(o, dtype=float) for o in exp_offset])
        if self.price_column is not None:   # what supply moments bind to (SupplyMoments.bind, log_costs)
            from types import SimpleNamespace

            s = np.empty(self.n_products)
            for mk, ls in zip(self.markets, self.log_s):
                s[np.asarray(mk.rows)] = np.exp(ls)
            self.spec = SimpleNamespace(share_column="shares")
            self.products = {"shares": s}
            self._prices = [SimpleNamespace(p=np.asarray(x[:, self.price_column], dtype=float)) for x in self.X]
        self.pool = ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(self.markets))))
        self.evaluations = 0

    @property
    def P(self) -> int:
        return len(self.pattern) + len(self.unobs) + len(self.exp_terms)

    @property
    def theta_columns(self) -> np.ndarray:
        """(P,) the X column each theta parameter multiplies."""
        return np.array([c for c, _ in self.pattern] + self.unobs + [self.exp_column] * len(self.exp_terms), dtype=int)

    def kernel_market(self, t: int) -> Any:
        """Market t's prices as ``.p`` (``price_column`` of X), what supply moments read."""
        return self._prices[t]

    def _exp_z(self, t: int) -> np.ndarray:
        """(H_t, E) the variables of the exponential law's coefficients."""
        cols = [self.D[t][:, i] if k == "D" else self.nu[t][:, i] for k, i in self.exp_terms]
        return np.column_stack(cols) if cols else np.zeros((self.H[t], 0))

    def exp_coefficient(self, theta: np.ndarray, t: int) -> np.ndarray:
        """(H_t,) ``-exp(e_h)``, each agent's exponential coefficient on ``exp_column`` (zeros without the law)."""
        if not self.exp_terms:
            return np.zeros(self.H[t])
        th = np.asarray(theta, dtype=float)[len(self.pattern) + len(self.unobs):]
        return -np.exp(self.exp_offset[t] + self._exp_z(t) @ th)

    # ------------------------------------------------------------------ one market

    def coefficients(self, theta: np.ndarray, t: int) -> np.ndarray:
        """(K, H_t) coefficient on each characteristic for each agent of market t."""
        C = np.zeros((self.K, self.H[t]))
        for k, (c, d) in enumerate(self.pattern):
            C[c] += theta[k] * self.D[t][:, d]
        for m, c in enumerate(self.unobs):
            C[c] += theta[len(self.pattern) + m] * self.nu[t][:, m]
        if self.exp_terms:
            assert self.exp_column is not None
            C[self.exp_column] += self.exp_coefficient(theta, t)
        return C

    def basis(self, t: int, theta: np.ndarray | None = None) -> np.ndarray:
        """(H_t, P) b_hp: d C_h,c(p) / d theta_p for each agent of market t (``theta`` is needed by the exponential
        law's columns, ``-exp(e_h) z_he``)."""
        cols = [self.D[t][:, d] for _, d in self.pattern] + [self.nu[t][:, m] for m in range(len(self.unobs))]
        if self.exp_terms:
            if theta is None:
                raise ValueError("the exponential law's basis depends on theta")
            cols += list((self.exp_coefficient(theta, t)[:, None] * self._exp_z(t)).T)
        return np.column_stack(cols) if cols else np.zeros((self.H[t], 0))

    @staticmethod
    def probabilities_at(delta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """(J, H) logit choice probabilities at mean utilities ``delta`` and deviations ``mu`` (J, H)."""
        u = delta[:, None] + mu
        m = np.maximum(u.max(axis=0), 0.0)
        e = np.exp(u - m)
        return e / (np.exp(-m) + e.sum(axis=0))

    def probabilities(self, theta: np.ndarray, t: int, delta: np.ndarray) -> np.ndarray:
        """(J_t, H_t) choice probabilities of market t at (theta, delta), no inversion."""
        return self.probabilities_at(np.asarray(delta, dtype=float), self.X[t] @ self.coefficients(theta, t))

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

    # ------------------------------------------------------------------ derivatives of one market

    def share_derivatives(self, theta: np.ndarray, t: int, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(ds/ddelta (J, J), ds/dtheta (J, P)) of market t at choice probabilities P (module docstring)."""
        w, B, Xc = self.w[t], self.basis(t, theta), self.X[t][:, self.theta_columns]
        Pw = P * w
        ds_dd = np.diag(Pw.sum(axis=1)) - Pw @ P.T
        xbar = P.T @ Xc                                  # (H, P): xbar_h,c(p)
        ds_dt = Xc * (Pw @ B) - Pw @ (B * xbar)
        return ds_dd, ds_dt

    def ddelta_dtheta(self, theta: np.ndarray, t: int, P: np.ndarray) -> np.ndarray:
        """(J, P) D_t = -(ds/ddelta)^{-1} ds/dtheta (implicit function theorem)."""
        ds_dd, ds_dt = self.share_derivatives(theta, t, P)
        return -np.linalg.solve(ds_dd, ds_dt)

    def price_coefficients(self, theta: np.ndarray, t: int) -> np.ndarray:
        """(H_t,) each agent's price coefficient ``price_coefficient + C_h,price``."""
        if self.price_column is None:
            raise ValueError("MixedLogitModel needs price_column for price coefficients")
        return self.price_coefficient + self.coefficients(np.asarray(theta, dtype=float), t)[self.price_column]

    def price_derivatives(self, theta: np.ndarray, t: int, P: np.ndarray) -> np.ndarray:
        """(J, J) ds_j/dp_k of market t at P with the agent price coefficients ``price_coefficient + C_h,price``."""
        if self.price_column is None:
            raise ValueError("MixedLogitModel needs price_column for ds_dp")
        a = self.price_coefficients(theta, t)
        Pwa = P * (self.w[t] * a)
        return np.diag(Pwa.sum(axis=1)) - Pwa @ P.T

    def group_sums(self, theta: np.ndarray, t: int, P: np.ndarray, cols: Sequence[str], derivatives: bool,
                   delta: np.ndarray | None = None) -> dict[str, Any]:
        """Per-household sums of market t (contract of ``moments.GroupMicroMoments``; one household per agent).

        With ``delta`` (market t's mean utilities, as ``solve`` passes them) the entry also carries ``M`` (H, C+1),
        each household's E[U | purchase] from its inside-good logit Q_jh = exp(u_jh) / sum_k exp(u_kh) in log space,
        and, with ``derivatives``, ``M_d_delta`` (H, C+1, J) and ``M_d_theta`` (H, C+1, P). A logit household always
        buys with positive probability, but P_h = sum_j P_jh underflows for a very price-sensitive one (the lognormal
        law), and S / P_h then loses every digit and is dropped once P_h is 0; M does not."""
        missing = [c for c in cols if c not in self.product_columns]
        if missing:
            raise KeyError(f"group sums of columns the model was not given: {missing}")
        w, rows = self.w[t], np.asarray(self.markets[t].rows)
        U = np.column_stack([np.ones(P.shape[0])] + [np.asarray(self.product_columns[c], dtype=float)[rows]
                                                     for c in cols])
        out: dict[str, Any] = {"columns": list(cols), "S": (U.T @ P).T * w[:, None], "d_delta": None, "d_theta": None,
                               "w": w}
        if derivatives:
            B, Xc = self.basis(t, theta), self.X[t][:, self.theta_columns]
            Ubar, xbar = P.T @ U, P.T @ Xc                                     # (H, C+1), (H, P)
            out["d_delta"] = (w[:, None, None] * P.T[:, None, :]) * (U.T[None, :, :] - Ubar[:, :, None])
            UX = np.einsum("jh,jc,jp->hcp", P, U, Xc)
            out["d_theta"] = (w[:, None] * B)[:, None, :] * (UX - Ubar[:, :, None] * xbar[:, None, :])
        if delta is not None:
            u = np.asarray(delta, dtype=float)[:, None] + self.X[t] @ self.coefficients(theta, t)
            Q = np.exp(u - u.max(axis=0))
            Q /= Q.sum(axis=0)
            M = Q.T @ U                                                        # (H, C+1)
            out["M"] = M
            if derivatives:
                xq = Q.T @ Xc
                out["M_d_delta"] = Q.T[:, None, :] * (U.T[None, :, :] - M[:, :, None])
                out["M_d_theta"] = B[:, None, :] * (np.einsum("jh,jc,jp->hcp", Q, U, Xc)
                                                    - M[:, :, None] * xq[:, None, :])
        return out

    def second_choice_adjoint(self, theta: np.ndarray, t: int, P: np.ndarray, A: np.ndarray, B: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
        """(out_delta (K, J), out_theta (K, P)): ``sum_jk A[c, j, k] dN_jk - sum_j B[c, j] ds_j`` in delta_t and theta
        (module docstring; the contract of ``moments.SecondChoiceMoments``)."""
        w = self.w[t]
        J = P.shape[0]
        rest = inside_mass_without(P)                                          # r_jh = R_h - P_jh
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_r = np.where(rest > 0, 1.0 / rest, 0.0)
        q = P * inv_r                                                          # P_jh / r_jh
        Bt, Xc = self.basis(t, theta), self.X[t][:, self.theta_columns]
        A = np.asarray(A, dtype=float).copy()
        A[:, np.arange(J), np.arange(J)] = 0.0
        out_d = np.empty((A.shape[0], J))
        out_t = np.empty((A.shape[0], Bt.shape[1]))
        for c in range(A.shape[0]):
            a = A[c] @ P                                                       # (J, H): sum_k A_jk P_kh
            t3 = np.sum(P * a * inv_r ** 2, axis=0)                            # (H,)
            g = w[None, :] * (a * inv_r + A[c].T @ q - t3[None, :] + P * a * inv_r ** 2
                              - np.asarray(B, dtype=float)[c][:, None])        # d / dP_lh
            dev = P * (g - np.sum(g * P, axis=0)[None, :])                     # P_lh (g_lh - gbar_h)
            out_d[c] = dev.sum(axis=1)
            out_t[c] = np.einsum("lh,lp,hp->p", dev, Xc, Bt)
        return out_d, out_t

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
        """``InnerModel.solve``: every market inverted from the accepted delta, with ``shares``, ``probabilities``,
        the requested ``second_choice`` (and with ``need_jacobian`` its ``second_choice_adjoint``), ``group_sums`` and
        ``ds_dp`` outputs, and D per market with ``need_jacobian`` (module docstring)."""
        if requests.micro_parts:
            raise NotImplementedError("MixedLogitModel does not serve micro parts")
        theta = np.asarray(theta, dtype=float)
        t0 = time.time()
        deltas, Ps = self.solve_all(theta)
        delta = np.empty(self.n_products)
        shares = np.empty(self.n_products)
        for m, d, P, w in zip(self.markets, deltas, Ps, self.w):
            delta[np.asarray(m.rows)] = d
            shares[np.asarray(m.rows)] = P @ w
        outputs: dict[str, Any] = {"shares": shares, "probabilities": Ps}
        T = range(len(self.markets))
        if requests.second_choice:
            want = {str(x) for x in requests.second_choice_markets}
            outputs["second_choice"] = [
                {"C": logit_second_choice_matrix(P, w), "s": P @ w} if str(m.market_id) in want else None
                for m, P, w in zip(self.markets, Ps, self.w)]
            if need_jacobian:
                outputs["second_choice_adjoint"] = (
                    lambda t, A, B, theta=theta.copy(): self.second_choice_adjoint(theta, t, Ps[t], A, B))
        if requests.group_sums:
            cols = list(requests.group_sums)
            outputs["group_sums"] = list(self.pool.map(
                lambda t: self.group_sums(theta, t, Ps[t], cols, need_jacobian, deltas[t]), T))
        if requests.price_derivatives:
            outputs["ds_dp"] = [self.price_derivatives(theta, t, Ps[t]) for t in T]
        outputs["theta"] = theta.copy()
        D = None
        status = EvalStatus.OK
        if need_jacobian:
            try:
                D = list(self.pool.map(lambda t: self.ddelta_dtheta(theta, t, Ps[t]), T))
            except np.linalg.LinAlgError:
                status = EvalStatus.SINGULAR_INNER_JACOBIAN
        seconds = time.time() - t0
        per = [{"market_id": m.market_id, "status": status.value, "seconds": seconds} for m in self.markets]
        return InnerSolution(delta=delta, D=D, status=status, per_market=per, outputs=outputs)

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

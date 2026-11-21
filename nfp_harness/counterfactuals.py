"""Counterfactuals of an estimated demand system, the same engine for every inner model (DESIGN 6.5).

The border: a scenario sees one market through a ``MarketDemand`` adapter, at fixed theta and fixed demand
unobservables, with only prices (and the product set) changing. An adapter answers ``shares(p, keep)``, ``ds_dp(p)``
and ``surplus(p, keep)``; everything else (implied costs, Nash prices, mergers, cost shocks, removal diversion,
Lerner tables) is model-agnostic and lives here. Two adapters ship:

- ``PCMMarketDemand``: the pure characteristics model through the kernel (``pcm_post``): shares and ds/dp by
  ``evaluate_market``, consumer surplus by the frontier formula (``consumer_surplus``), prices re-sorted with ties
  spread at every evaluation. Its Nash prices are the bounded best responses of ``pcm_post.best_response_box``
  (LEARNINGS 29): ds/dp jumps where a draw's frontier changes, so no FOC iteration is used.
- ``LogitMarketDemand``: a ``MixedLogitModel`` market, the mean utility moving with price by the linear price
  coefficient when price is in delta; consumer surplus is the log-sum ``sum_h w_h log(1 + sum_j exp u_hj) / |a_h|``
  with ``a_h`` agent h's price coefficient. Its Nash prices solve the first-order conditions by the zeta map of
  Morrow and Skerlos (2011), ``p = c + Lambda^{-1} ((Omega * Gamma)(p - c) - s)`` with ds/dp = diag(Lambda) - Gamma.

Units: prices, costs and surplus are in the price units of the data; surplus is per unit of market mass (per household
when the agent weights sum to one), so a dollar total is households x the change. A paper script chooses markets,
firms and products and calls ``implied_costs``, ``merger``, ``cost_shock``, ``removal_diversion`` and
``lerner_table`` (``ownership_change`` for a cartel or brand reassignment); ``run_scenarios`` runs a list of them on one market.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .models import pcm_post as pp

__all__ = ["MarketDemand", "PCMMarketDemand", "LogitMarketDemand", "implied_costs", "nash_prices", "baseline_prices",
           "ownership_change", "merger", "cost_shock", "removal_diversion", "lerner_table", "run_scenarios"]


class MarketDemand(Protocol):
    """One market of a fitted demand system: ``p0`` (J,) observed prices, ``firm_ids`` (J,), ``rows`` (J,) product
    rows, ``shares(p, keep=None)``, ``ds_dp(p)`` and ``surplus(p, keep=None)`` (``keep``: positions of the products
    present, default all), and ``nash_method`` (``"foc"`` or ``"best_response"``)."""

    p0: np.ndarray
    firm_ids: np.ndarray
    rows: np.ndarray
    nash_method: str

    def shares(self, p: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray | None: ...

    def ds_dp(self, p: np.ndarray) -> np.ndarray: ...

    def surplus(self, p: np.ndarray, keep: np.ndarray | None = None) -> float: ...


# --------------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------------


class PCMMarketDemand:
    """PCMMarketDemand(model, theta, delta, t, firm_ids, min_gap=0.001, n_threads=None): market ``t`` of a
    ``PCMModel`` at theta and the (n,) row-order ``delta``, products in the market's price order. A price vector is
    re-sorted (ties spread to ``min_gap``) before every kernel call and the answers are put back in this order, so
    any price vector is allowed; a non-positive price is refused (None shares)."""

    nash_method = "best_response"

    def __init__(self, model: Any, theta: np.ndarray, delta: np.ndarray, t: int, firm_ids: np.ndarray,
                 min_gap: float = 0.001, n_threads: int | None = None) -> None:
        self.th, self.ph, self.sp = model.kernel_params(theta)
        self.bound = pp._bound(model, theta)
        self.nt = pp._threads(model, n_threads)
        self.rows = np.asarray(model.markets[t].rows)
        self.f = pp._fields(model.kernel_market(t))
        self.p0 = self.f["p"].astype(float)
        self.d = np.asarray(delta, dtype=float)[self.rows]
        self.firm_ids = np.asarray(firm_ids)[self.rows]
        self.min_gap = float(min_gap)
        self.weights = self.f["weights"]

    def _market(self, p: np.ndarray, keep: np.ndarray | None) -> tuple[np.ndarray, Any, np.ndarray] | None:
        idx = np.arange(self.p0.size) if keep is None else np.asarray(keep, dtype=int)
        order, ps = pp._sorted_prices(np.asarray(p, dtype=float)[idx], self.min_gap)
        if ps.size and ps[0] <= 0:
            return None
        sel = idx[order]
        return order, pp._market_like(self.f, sel, ps), sel

    def shares(self, p: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray | None:
        m = self._market(p, keep)
        if m is None:
            return None
        order, km, sel = m
        s, _ = pp._eval(km, self.d[sel], self.th, self.ph, self.sp, self.nt, price_derivatives=False,
                        bound=self.bound)
        out = np.empty_like(s)
        out[order] = s
        return out

    def ds_dp(self, p: np.ndarray) -> np.ndarray:
        m = self._market(p, None)
        if m is None:
            raise ValueError("non-positive price")
        order, km, sel = m
        _, J = pp._eval(km, self.d[sel], self.th, self.ph, self.sp, self.nt, bound=self.bound)
        assert J is not None
        out = np.empty_like(J)
        out[np.ix_(order, order)] = J
        return out

    def surplus(self, p: np.ndarray, keep: np.ndarray | None = None) -> float:
        m = self._market(p, keep)
        if m is None:
            return float("nan")
        _, km, sel = m
        return float(pp._cs(km, self.d[sel], self.th, self.ph, self.sp, self.nt, self.bound)["total"])


class LogitMarketDemand:
    """LogitMarketDemand(model, theta, delta, t, firm_ids, price_coefficient=None): market ``t`` of a
    ``MixedLogitModel`` (its ``price_column`` set) at theta and the (n,) row-order ``delta``. ``price_coefficient``
    is the linear price coefficient inside delta (default the model's ``price_coefficient``): delta moves by it times
    the price change."""

    nash_method = "foc"

    def __init__(self, model: Any, theta: np.ndarray, delta: np.ndarray, t: int, firm_ids: np.ndarray,
                 price_coefficient: float | None = None) -> None:
        if model.price_column is None:
            raise ValueError("LogitMarketDemand needs a MixedLogitModel with price_column")
        self.model, self.t = model, int(t)
        self.theta = np.asarray(theta, dtype=float)
        self.rows = np.asarray(model.markets[t].rows)
        self.c = int(model.price_column)
        self.X0 = np.asarray(model.X[t], dtype=float)
        self.p0 = self.X0[:, self.c].copy()
        self.d0 = np.asarray(delta, dtype=float)[self.rows]
        self.alpha = float(model.price_coefficient if price_coefficient is None else price_coefficient)
        self.C = model.coefficients(self.theta, t)           # (K, H)
        self.a = self.alpha + self.C[self.c]                  # (H,) price coefficients
        self.w = np.asarray(model.w[t], dtype=float)
        self.firm_ids = np.asarray(firm_ids)[self.rows]
        self.weights = self.w

    def _u(self, p: np.ndarray, keep: np.ndarray | None) -> np.ndarray:
        X = self.X0.copy()
        X[:, self.c] = p
        u = (self.d0 + self.alpha * (p - self.p0))[:, None] + X @ self.C
        return u if keep is None else u[np.asarray(keep, dtype=int)]

    def probabilities(self, p: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
        u = self._u(np.asarray(p, dtype=float), keep)
        m = np.maximum(u.max(axis=0), 0.0)
        e = np.exp(u - m)
        return e / (np.exp(-m) + e.sum(axis=0))

    def shares(self, p: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray | None:
        return self.probabilities(p, keep) @ self.w

    def ds_dp(self, p: np.ndarray) -> np.ndarray:
        lam, gam, _ = self.zeta_parts(p)
        return np.diag(lam) - gam

    def zeta_parts(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Lambda (J,), Gamma (J, J), s) with ds/dp = diag(Lambda) - Gamma, Lambda_j = sum_h w_h a_h P_jh and
        Gamma_jk = sum_h w_h a_h P_jh P_kh (Morrow and Skerlos 2011), for the zeta-map Nash solver."""
        P = self.probabilities(p)
        Pwa = P * (self.w * self.a)
        return Pwa.sum(axis=1), Pwa @ P.T, P @ self.w

    def surplus(self, p: np.ndarray, keep: np.ndarray | None = None) -> float:
        u = self._u(np.asarray(p, dtype=float), keep)
        m = np.maximum(u.max(axis=0), 0.0)
        ls = m + np.log(np.exp(-m) + np.exp(u - m).sum(axis=0))
        return float(np.sum(self.w * ls / np.abs(self.a)))


# --------------------------------------------------------------------------------------------
# equilibrium
# --------------------------------------------------------------------------------------------


def _ownership(owner: np.ndarray) -> np.ndarray:
    owner = np.asarray(owner).ravel()
    return (owner[:, None] == owner[None, :]).astype(float)


def implied_costs(demand: MarketDemand, rcond_min: float = 1e-12) -> dict[str, Any]:
    """Marginal costs that make the observed prices a Nash equilibrium under the observed ownership:
    ``c = p0 - markup``, ``markup = -(Omega * ds/dp')^{-1} s`` (``pcm_post.nash_markups``)."""
    s = demand.shares(demand.p0)
    assert s is not None
    nash = pp.nash_markups(s, demand.ds_dp(demand.p0), demand.firm_ids, rcond_min)
    return {"costs": demand.p0 - nash["markup"], "markup": nash["markup"], "status": nash["status"],
            "rcond": nash["rcond"], "shares": s}


def _foc_prices(demand: MarketDemand, costs: np.ndarray, owner: np.ndarray, p_start: np.ndarray,
                tol: float = 1e-10, max_iterations: int = 2000, damping: float = 1.0) -> dict[str, Any]:
    """Damped fixed point p <- (1 - d) p + d (c + markup(p)), the damping halved whenever the FOC residual grows."""
    Om = _ownership(owner)
    p = np.asarray(p_start, dtype=float).copy()
    c = np.asarray(costs, dtype=float)
    res_prev = math.inf
    d = float(damping)
    for it in range(1, max_iterations + 1):
        s = demand.shares(p)
        if s is None:
            return {"status": "failed", "prices": p, "iterations": it}
        A = Om * demand.ds_dp(p).T
        try:
            mk = -np.linalg.solve(A, s)
        except np.linalg.LinAlgError:
            return {"status": "singular", "prices": p, "iterations": it}
        target = c + mk
        res = float(np.max(np.abs(target - p) / np.maximum(np.abs(p), 1e-12)))
        if res < tol:
            return {"status": "converged", "prices": target, "iterations": it, "foc_residual": res}
        if res > res_prev:
            d = max(d / 2.0, 1e-3)
        res_prev = res
        p = (1.0 - d) * p + d * target
        p = np.maximum(p, 1e-6 * np.abs(demand.p0))
    return {"status": "not_converged", "prices": p, "iterations": max_iterations, "foc_residual": res_prev}


def _zeta_prices(demand: Any, costs: np.ndarray, owner: np.ndarray, p_start: np.ndarray, tol: float = 1e-10,
                 max_iterations: int = 5000) -> dict[str, Any]:
    """Morrow and Skerlos (2011) zeta map p <- c + Lambda^{-1} ((Omega * Gamma)(p - c) - s) for a logit-type demand
    (``zeta_parts``); its fixed points are the FOC solutions and it converges where p = c + markup(p) cycles."""
    Om = _ownership(owner)
    c = np.asarray(costs, dtype=float)
    p = np.asarray(p_start, dtype=float).copy()
    res = math.inf
    for it in range(1, max_iterations + 1):
        lam, gam, s = demand.zeta_parts(p)
        target = c + ((Om * gam) @ (p - c) - s) / lam
        res = float(np.max(np.abs(target - p) / np.maximum(np.abs(p), 1e-12)))
        p = target
        if not np.all(np.isfinite(p)):
            return {"status": "failed", "prices": p, "iterations": it, "method": "zeta"}
        if res < tol:
            lam, gam, s = demand.zeta_parts(p)
            foc = s + (Om * (np.diag(lam) - gam).T) @ (p - c)
            return {"status": "converged", "prices": p, "iterations": it, "method": "zeta",
                    "foc_residual": float(np.max(np.abs(foc)) / max(float(np.max(s)), 1e-300))}
    return {"status": "not_converged", "prices": p, "iterations": max_iterations, "method": "zeta", "step": res}


def nash_prices(demand: MarketDemand, costs: np.ndarray, owner: np.ndarray, p_start: np.ndarray | None = None,
                **kw: Any) -> dict[str, Any]:
    """Nash prices at ``costs`` under ``owner`` (J,) firm labels: for ``nash_method`` "foc" the zeta map when the
    adapter has ``zeta_parts`` (else the damped FOC fixed point), for "best_response" the bounded best responses of
    ``pcm_post.best_response_box`` (``kw``: box, max_sweeps, golden_tol, deviation). Returns ``status``, ``prices``
    and the solver's diagnostics."""
    p0 = demand.p0 if p_start is None else np.asarray(p_start, dtype=float)
    if demand.nash_method == "foc":
        if hasattr(demand, "zeta_parts"):
            return _zeta_prices(demand, costs, owner, p0, **kw)
        return _foc_prices(demand, costs, owner, p0, **kw)
    br = pp.best_response_box(lambda p: demand.shares(p), p0, costs, np.asarray(owner), **kw)
    return {k: v for k, v in br.items() if k != "shares"}


def baseline_prices(demand: MarketDemand, costs: np.ndarray, **kw: Any) -> dict[str, Any]:
    """Nash prices at ``costs`` under the observed ownership, the baseline every scenario is compared with; cached on
    the adapter per (costs, solver options), so scenarios on one market solve it once."""
    key = (np.asarray(costs, dtype=float).tobytes(), repr(sorted(kw.items())))
    cache = getattr(demand, "_baseline_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(demand, "_baseline_cache", cache)
        except AttributeError:   # an adapter that refuses attributes: no cache
            pass
    if key not in cache:
        cache[key] = nash_prices(demand, costs, demand.firm_ids, **kw)
    return cache[key]


def _summary(demand: MarketDemand, p_new: np.ndarray, base_s: np.ndarray, base_cs: float) -> dict[str, Any]:
    s = demand.shares(p_new)
    s = np.full(p_new.size, np.nan) if s is None else s
    dp = (p_new - demand.p0) / demand.p0
    return {"prices": p_new, "shares": s, "price_change_share_weighted": float(np.sum(dp * base_s) / np.sum(base_s)),
            "inside_share": float(np.nansum(s)), "inside_share_change": float(np.nansum(s) - base_s.sum()),
            "cs": demand.surplus(p_new), "cs_change": demand.surplus(p_new) - base_cs}


def ownership_change(demand: MarketDemand, costs: np.ndarray, owner: np.ndarray, label: str = "", **kw: Any
                     ) -> dict[str, Any]:
    """New Nash prices with costs fixed when the products' owners become ``owner`` (J,) (a merger, a cartel of
    several firms, a reassignment of brands). Reports the share-weighted price change (all products, those whose
    owner's portfolio changed and the rest), the change in inside share and in consumer surplus, each against the
    solver's own pre-change equilibrium at the same costs (``baseline``; equal to p0 under FOC pricing)."""
    owner = np.asarray(owner).astype(object)
    old = demand.firm_ids.astype(object)
    # a product is affected when the set of products its owner prices changes
    changed = np.array([not np.array_equal(old == old[j], owner == owner[j]) for j in range(old.size)])
    base_s = demand.shares(demand.p0)
    assert base_s is not None
    base_cs = demand.surplus(demand.p0)
    pre = baseline_prices(demand, costs, **kw)
    eq = nash_prices(demand, costs, owner, p_start=pre["prices"], **kw)
    out = {"label": label, "status": eq["status"], "baseline_status": pre["status"], "affected": int(changed.sum()),
           **_summary(demand, eq["prices"], base_s, base_cs),
           "baseline": _summary(demand, pre["prices"], base_s, base_cs)}
    dp = (eq["prices"] - pre["prices"]) / pre["prices"]

    def wavg(m: np.ndarray) -> float:
        return float(np.sum(dp[m] * base_s[m]) / np.sum(base_s[m])) if m.any() else float("nan")

    out["price_change_vs_baseline"] = {"affected": wavg(changed), "others": wavg(~changed),
                                       "all": wavg(np.ones(dp.size, dtype=bool))}
    out["cs_change_vs_baseline"] = out["cs"] - out["baseline"]["cs"]
    out["solver"] = {k: v for k, v in eq.items() if k not in ("prices",)}
    return out


def merger(demand: MarketDemand, costs: np.ndarray, a: Any, b: Any, **kw: Any) -> dict[str, Any]:
    """Firm ``b``'s products move to firm ``a`` (``ownership_change``)."""
    owner = demand.firm_ids.astype(object).copy()
    if not (owner == a).any() or not (owner == b).any():
        raise ValueError(f"merger: firms {a!r} and {b!r} must both sell in the market")
    owner[owner == b] = a
    out = ownership_change(demand, costs, owner, label=f"{a} + {b}", **kw)
    out["firms"] = [str(a), str(b)]
    return out


def cost_shock(demand: MarketDemand, costs: np.ndarray, mask: np.ndarray, rate: float = 0.10, **kw: Any
               ) -> dict[str, Any]:
    """Costs of the products in ``mask`` (J,) times 1 + ``rate``; new Nash prices. Pass-through: the shocked
    products' share-weighted dp / dc, and the rivals' share-weighted relative price change."""
    mask = np.asarray(mask, dtype=bool)
    c1 = np.asarray(costs, dtype=float).copy()
    c1[mask] *= 1.0 + rate
    base_s = demand.shares(demand.p0)
    assert base_s is not None
    base_cs = demand.surplus(demand.p0)
    pre = baseline_prices(demand, costs, **kw)
    eq = nash_prices(demand, c1, demand.firm_ids, p_start=pre["prices"], **kw)
    dp = eq["prices"] - pre["prices"]
    dc = c1 - np.asarray(costs, dtype=float)
    w_own = base_s[mask]
    rival = ~mask
    out = {"rate": rate, "status": eq["status"], "products": int(mask.sum()),
           **_summary(demand, eq["prices"], base_s, base_cs),
           "pass_through_own": float(np.sum(dp[mask] / dc[mask] * w_own) / np.sum(w_own)) if mask.any() else None,
           "rival_price_change": float(np.sum(dp[rival] / pre["prices"][rival] * base_s[rival])
                                       / max(np.sum(base_s[rival]), 1e-300)),
           "shocked_price_change": float(np.sum(dp[mask] / pre["prices"][mask] * w_own) / np.sum(w_own))
           if mask.any() else None}
    out["cs_change_vs_baseline"] = out["cs"] - demand.surplus(pre["prices"])
    out["solver"] = {k: v for k, v in eq.items() if k not in ("prices",)}
    return out


def removal_diversion(demand: MarketDemand, products: Sequence[int]) -> list[dict[str, Any]]:
    """For each product position j in ``products``: remove it at fixed prices and report the diversion
    ``(s_k' - s_k) / s_j`` to every other product (``to``, J entries, NaN at j) and to the outside good
    (``outside``), which sum to one, and the CS change of the removal."""
    p0 = demand.p0
    s0 = demand.shares(p0)
    assert s0 is not None
    cs0 = demand.surplus(p0)
    J = p0.size
    out = []
    for j in products:
        keep = np.setdiff1d(np.arange(J), [int(j)])
        s1 = demand.shares(p0, keep)
        assert s1 is not None
        to = np.full(J, np.nan)
        to[keep] = (s1 - s0[keep]) / s0[j]
        out.append({"product": int(j), "row": int(demand.rows[j]), "share": float(s0[j]), "to": to,
                    "outside": float(1.0 - np.nansum(to)), "cs_change": demand.surplus(p0, keep) - cs0})
    return out


def lerner_table(p: np.ndarray, costs: np.ndarray, s: np.ndarray, labels: Mapping[str, np.ndarray]
                 ) -> dict[str, Any]:
    """Lerner index (p - c) / p: median and sales-weighted mean overall and by each labelling in ``labels`` (e.g.
    ``{"year": ..., "segment": ...}``, (n,) arrays aligned with p)."""
    p, c, s = (np.asarray(v, dtype=float) for v in (p, costs, s))
    L = (p - c) / p
    ok = np.isfinite(L)

    def stats(m: np.ndarray) -> dict[str, float]:
        m = m & ok
        return {"median": float(np.median(L[m])) if m.any() else float("nan"),
                "sales_weighted_mean": float(np.sum(L[m] * s[m]) / np.sum(s[m])) if m.any() else float("nan"),
                "products": int(m.sum())}

    out: dict[str, Any] = {"all": stats(np.ones(L.size, dtype=bool))}
    for name, lab in labels.items():
        lab = np.asarray(lab)
        out[name] = {str(v): stats(lab == v) for v in sorted(set(lab.tolist()), key=str)}
    return out


def run_scenarios(demand: MarketDemand, scenarios: Sequence[tuple[str, Callable[..., Any], Mapping[str, Any]]]
                  ) -> dict[str, Any]:
    """Implied costs of ``demand``, then each (name, function, kwargs) scenario (``merger``, ``cost_shock`` with
    costs passed for it; ``removal_diversion`` without). A scenario that raises is reported with its error."""
    ic = implied_costs(demand)
    out: dict[str, Any] = {"implied_costs": ic}
    for name, fn, kw in scenarios:
        try:
            out[name] = fn(demand, **kw) if fn is removal_diversion else fn(demand, ic["costs"], **kw)
        except Exception as exc:   # one failed scenario must not lose the others
            out[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return out

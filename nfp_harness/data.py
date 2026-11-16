"""Model-agnostic data helpers: tie breaking, QMC draws, agent crossing (DESIGN 6.1). Owned by WP4."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

__all__ = ["break_price_ties", "sobol_normal_draws", "cross_agents"]


def break_price_ties(prices: np.ndarray, market_ids: np.ndarray, min_gap: float) -> np.ndarray:
    """break_price_ties(prices, market_ids, min_gap) -> prices spread to at least ``min_gap`` within
    each market, preserving order (LEARNINGS 3).

    Within a market, products are taken in stable price order (ties keep their row order) and each
    price is raised to at least the previous one plus ``min_gap``. Rows keep their positions; only
    values change. Same rule as ``break_price_ties`` in the local ``petrin_replication/pcm_petrin.py``.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    market_ids = np.asarray(market_ids).ravel()
    if prices.shape != market_ids.shape:
        raise ValueError(f"{prices.size} prices for {market_ids.size} market ids")
    if not min_gap >= 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    out = prices.copy()
    for t in np.unique(market_ids):
        rows = np.flatnonzero(market_ids == t)
        order = rows[np.argsort(prices[rows], kind="stable")]
        for prev, cur in zip(order[:-1], order[1:]):
            if out[cur] < out[prev] + min_gap:
                out[cur] = out[prev] + min_gap
    return out


def sobol_normal_draws(n: int, dim: int, seed: int, require_power_of_two: bool = True) -> np.ndarray:
    """sobol_normal_draws(n, dim, seed) -> (n, dim) scrambled Sobol standard normal draws.

    n is a power of two (owner decision 6: 4,096 development, 16,384 reported, 32,768 doubling check).
    ``require_power_of_two=False`` allows other n (scipy then warns about balance properties).
    """
    if n < 1 or dim < 1:
        raise ValueError(f"n and dim must be positive, got n={n}, dim={dim}")
    m = int(round(np.log2(n)))
    power = 2**m == n
    if require_power_of_two and not power:
        raise ValueError(f"n = {n} is not a power of two (DESIGN 9, owner decision 6)")
    engine = qmc.Sobol(dim, scramble=True, seed=seed)
    u = engine.random_base2(m) if power else engine.random(n)
    return norm.ppf(u)


def cross_agents(agent_weights: np.ndarray, agent_demographics: np.ndarray, nu: np.ndarray,
                 nu_per_agent: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cross_agents(agent_weights, agent_demographics, nu, nu_per_agent) -> (weights, demographics, nu).

    Crosses each agent with taste draws. With ``nu_per_agent`` given, ``nu`` has
    ``A * nu_per_agent`` rows and agent a gets rows ``a * nu_per_agent ... (a + 1) * nu_per_agent - 1``
    with weight ``w_agent / nu_per_agent``. With ``nu_per_agent=None``, the R rows of ``nu`` are split
    into A consecutive blocks whose sizes differ by at most one (``np.array_split``; this allows R a
    power of two with A = 1,000 agents), and each draw of agent a has weight ``w_a / count_a``. In
    both cases every agent keeps its total weight. Returns the per-draw weights (R,),
    demographics (R, d) and nu (R, dim).
    """
    w = np.asarray(agent_weights, dtype=float).ravel()
    demo = np.asarray(agent_demographics, dtype=float)
    if demo.ndim == 1:
        demo = demo[:, None]
    nu = np.asarray(nu, dtype=float)
    if nu.ndim == 1:
        nu = nu[:, None]
    A = w.size
    if demo.shape[0] != A:
        raise ValueError(f"{demo.shape[0]} demographic rows for {A} agents")
    R = nu.shape[0]
    if nu_per_agent is not None:
        if R != A * nu_per_agent:
            raise ValueError(f"nu has {R} rows, expected {A} agents x {nu_per_agent} = {A * nu_per_agent}")
        counts = np.full(A, int(nu_per_agent))
    else:
        if R < A:
            raise ValueError(f"{R} draws for {A} agents: every agent needs at least one draw")
        counts = np.array([len(b) for b in np.array_split(np.arange(R), A)])
    agent_of_draw = np.repeat(np.arange(A), counts)
    return (w / counts)[agent_of_draw], demo[agent_of_draw], nu

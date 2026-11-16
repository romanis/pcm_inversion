"""Model-agnostic data helpers: tie breaking, QMC draws, agent crossing (DESIGN 6.1). Owned by WP4."""

from __future__ import annotations

import numpy as np

__all__ = ["break_price_ties", "sobol_normal_draws", "cross_agents"]


def break_price_ties(prices: np.ndarray, market_ids: np.ndarray, min_gap: float) -> np.ndarray:
    """break_price_ties(prices, market_ids, min_gap) -> prices spread to at least ``min_gap`` within
    each market, preserving order (LEARNINGS 3)."""
    raise NotImplementedError("break_price_ties(prices, market_ids, min_gap) -> np.ndarray: WP4, DESIGN 6.1")


def sobol_normal_draws(n: int, dim: int, seed: int) -> np.ndarray:
    """sobol_normal_draws(n, dim, seed) -> (n, dim) scrambled Sobol standard normal draws.

    n is a power of two (owner decision 6: 4,096 development, 16,384 reported, 32,768 doubling check).
    """
    raise NotImplementedError("sobol_normal_draws(n, dim, seed) -> np.ndarray: WP4, DESIGN 6.1")


def cross_agents(agent_weights: np.ndarray, agent_demographics: np.ndarray, nu: np.ndarray,
                 nu_per_agent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cross_agents(agent_weights, agent_demographics, nu, nu_per_agent) -> (weights, demographics, nu).

    Crosses each agent with ``nu_per_agent`` taste draws; weight ``w_agent / nu_per_agent``.
    """
    raise NotImplementedError("cross_agents(agent_weights, agent_demographics, nu, nu_per_agent): WP4, DESIGN 6.1")

"""Ray and grid profiles of the objective (DESIGN 6.4 stage 1). Owned by WP9a."""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .types import Problem

__all__ = ["grid_profile", "ray_profile"]


def grid_profile(problem: Problem, theta0: np.ndarray, grid: Mapping[str, Sequence[float]], *,
                 workers: int = 4, threads_per_worker: int = 48, cold: bool = True,
                 csv_path: str | pathlib.Path | None = None,
                 pass_rule: Callable[[Any], tuple[bool, str]] | None = None) -> Any:
    """grid_profile(problem, theta0, grid, workers=4, threads_per_worker=48, cold=True, csv_path=None,
    pass_rule=None) -> table of q and its parts on the product grid of named parameters.

    Parameters not in ``grid`` stay at ``theta0``. Grid points run in parallel across workers, each
    inverted cold or warm. ``pass_rule(table) -> (passed, verdict)`` is applied when given.
    """
    raise NotImplementedError("grid_profile(problem, theta0, grid, ...): WP9a, DESIGN 6.4")


def ray_profile(problem: Problem, theta0: np.ndarray, ray: Mapping[str, float], lambdas: Sequence[float],
                grid: Mapping[str, Sequence[float]], **kwargs: Any) -> Any:
    """ray_profile(problem, theta0, ray, lambdas, grid, **kwargs) -> table.

    Rays ``theta[ray names] = lambda * ray`` crossed with ``grid`` (e.g. sigma = lambda x SIG_REF
    crossed with sigma_p values); ``kwargs`` as in ``grid_profile``.
    """
    raise NotImplementedError("ray_profile(problem, theta0, ray, lambdas, grid, **kwargs): WP9a, DESIGN 6.4")

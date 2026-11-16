"""Parameter layout, natural units and bounds (DESIGN 2.5). Owned by WP4.

``ParameterSpec`` is defined in ``nfp_harness.types`` and re-exported here.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .types import ParameterSpec

__all__ = ["ParameterSpec", "ParameterLayout"]


class ParameterLayout:
    """Ordered nonlinear parameters of a spec.

    Attributes: ``specs: list[ParameterSpec]``, ``names: list[str]``, ``P: int``.

    The optimizer works in identity coordinates divided by the natural unit (no log transforms):
    ``u = (theta - start) / unit``. Automatic natural units, where ``spec.unit`` is None (prices and
    delta in $1000):

    - taste spread on column k: ``unit = 1 / sd(x_k)`` (dummies set ``unit = 1`` in the spec);
    - demographic interaction on column k with basis b: ``unit = 1 / (sd(x_k) sd(b))``;
    - scale parameter phi_q on basis b: ``unit = 1 / sd(b)``;
    - sigma_p: ``unit = 0.1``.
    """

    specs: list[ParameterSpec]
    names: list[str]
    P: int

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        """ParameterLayout(specs). Names must be unique."""
        raise NotImplementedError("ParameterLayout.__init__(specs): WP4, DESIGN 2.5")

    def units(self, data: Any) -> np.ndarray:
        """units(data) -> (P,) natural units; the automatic rule applied where ``unit`` is None."""
        raise NotImplementedError("ParameterLayout.units(data) -> np.ndarray: WP4, DESIGN 2.5")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """bounds() -> (lower, upper), each (P,), in natural parameterisation."""
        raise NotImplementedError("ParameterLayout.bounds() -> (lower, upper): WP4, DESIGN 2.5")

    def free_mask(self) -> np.ndarray:
        """free_mask() -> (P,) bool, False for parameters fixed at start."""
        raise NotImplementedError("ParameterLayout.free_mask() -> np.ndarray: WP4, DESIGN 2.5")

    def to_scaled(self, theta: np.ndarray) -> np.ndarray:
        """to_scaled(theta) -> u = (theta - start) / unit."""
        raise NotImplementedError("ParameterLayout.to_scaled(theta) -> u: WP4, DESIGN 2.5")

    def from_scaled(self, u: np.ndarray) -> np.ndarray:
        """from_scaled(u) -> theta = start + unit * u."""
        raise NotImplementedError("ParameterLayout.from_scaled(u) -> theta: WP4, DESIGN 2.5")

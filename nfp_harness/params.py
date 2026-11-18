"""Parameter layout, natural units and bounds (DESIGN 2.5). Owned by WP4.

``ParameterSpec`` is defined in ``nfp_harness.types`` and re-exported here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

from .types import ParameterSpec

__all__ = ["ParameterSpec", "ParameterLayout", "SIGMA_P_UNIT"]

#: Natural unit of the log-alpha spread sigma_p (DESIGN 2.5).
SIGMA_P_UNIT = 0.1

# A column whose standard deviation is below this cannot define an automatic unit.
_SD_FLOOR = 1e-12


def _lookup(data: Any, name: str) -> np.ndarray | None:
    """Column ``name`` of ``data`` as a float array, or None if ``data`` does not hold it.

    ``data`` is any mapping-like object indexed by column name (a dict of arrays, a pandas
    DataFrame, a ``collections.ChainMap`` over a product table and an agent table).
    """
    if data is None:
        return None
    try:
        if name not in data:
            return None
        column = data[name]
    except TypeError:
        return None
    return np.asarray(column, dtype=float).ravel()


def _sd(data: Any, name: str, what: str, param: str) -> float:
    column = _lookup(data, name)
    if column is None:
        raise ValueError(f"{param}: automatic unit needs {what} column {name!r}, which the data do not hold; "
                         f"set 'unit' in the spec")
    sd = float(np.std(column))
    if not sd > _SD_FLOOR:
        raise ValueError(f"{param}: {what} column {name!r} has sd {sd:.3g}, so the automatic unit is undefined; "
                         f"set 'unit' in the spec")
    return sd


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

    A taste parameter is a spread when its basis column is absent from the data (draws are
    standard normal by construction) and an interaction when the data hold the basis column.
    ``units(data)`` resolves the automatic rule once and caches the result; ``to_scaled`` and
    ``from_scaled`` use the cached units.
    """

    specs: list[ParameterSpec]
    names: list[str]
    P: int

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        """ParameterLayout(specs). Names must be unique."""
        self.specs = list(specs)
        self.names = [s.name for s in self.specs]
        self.P = len(self.specs)
        duplicates = sorted({n for n in self.names if self.names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate parameter names: {duplicates}")
        self._units: np.ndarray | None = None
        if all(s.unit is not None for s in self.specs):
            self._units = np.array([float(s.unit) for s in self.specs])  # type: ignore[arg-type]

    # ---------------------------------------------------------------- lookups

    def index(self, name: str) -> int:
        """Position of parameter ``name`` in theta."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no parameter named {name!r}; known: {self.names}") from None

    def __getitem__(self, name: str) -> ParameterSpec:
        return self.specs[self.index(name)]

    def __len__(self) -> int:
        return self.P

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.specs)

    def starts(self) -> np.ndarray:
        """(P,) start values."""
        return np.array([s.start for s in self.specs], dtype=float)

    def cap_mask(self) -> np.ndarray:
        """(P,) bool, True where the upper bound is a stated normalisation (``cap=True``)."""
        return np.array([s.cap for s in self.specs], dtype=bool)

    def n_free(self) -> int:
        return int(self.free_mask().sum())

    # ---------------------------------------------------------------- units and bounds

    def units(self, data: Any = None) -> np.ndarray:
        """units(data) -> (P,) natural units; the automatic rule applied where ``unit`` is None.

        ``data`` maps column names (product characteristics and agent basis columns) to arrays.
        The result is cached, so later calls may pass ``data=None``.
        """
        if data is None:
            if self._units is None:
                missing = [s.name for s in self.specs if s.unit is None]
                raise ValueError(f"automatic units not resolved for {missing}: call units(data) first")
            return self._units.copy()
        units = np.empty(self.P)
        for k, s in enumerate(self.specs):
            units[k] = s.unit if s.unit is not None else self._automatic_unit(s, data)
        self._units = units
        return units.copy()

    @staticmethod
    def _automatic_unit(s: ParameterSpec, data: Any) -> float:
        if s.role == "sigma_p":
            return SIGMA_P_UNIT
        if s.role == "scale":
            if s.basis is None:
                raise ValueError(f"{s.name}: scale parameter without a basis column needs an explicit unit")
            return 1.0 / _sd(data, s.basis, "scale basis", s.name)
        if s.role == "taste":
            if s.column is None:
                raise ValueError(f"{s.name}: taste parameter without a column needs an explicit unit")
            sd_x = _sd(data, s.column, "characteristic", s.name)
            if s.basis is not None and _lookup(data, s.basis) is not None:
                return 1.0 / (sd_x * _sd(data, s.basis, "taste basis", s.name))
            return 1.0 / sd_x
        raise ValueError(f"{s.name}: no automatic unit for role {s.role!r}; set 'unit' in the spec")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """bounds() -> (lower, upper), each (P,), in natural parameterisation."""
        lower = np.array([s.lower for s in self.specs], dtype=float)
        upper = np.array([s.upper for s in self.specs], dtype=float)
        return lower, upper

    def in_bounds(self, theta: np.ndarray) -> bool:
        """True if every component of theta is finite and within its hard bounds."""
        theta = np.asarray(theta, dtype=float)
        lower, upper = self.bounds()
        return bool(theta.shape == (self.P,) and np.all(np.isfinite(theta))
                    and np.all(theta >= lower) and np.all(theta <= upper))

    def free_mask(self) -> np.ndarray:
        """free_mask() -> (P,) bool, False for parameters fixed at start."""
        return np.array([s.free for s in self.specs], dtype=bool)

    # ---------------------------------------------------------------- coordinates

    def to_scaled(self, theta: np.ndarray) -> np.ndarray:
        """to_scaled(theta) -> u = (theta - start) / unit."""
        theta = np.asarray(theta, dtype=float)
        return (theta - self.starts()) / self.units()

    def from_scaled(self, u: np.ndarray) -> np.ndarray:
        """from_scaled(u) -> theta = start + unit * u."""
        u = np.asarray(u, dtype=float)
        return self.starts() + self.units() * u

    # ---------------------------------------------------------------- variants

    def replace(self, changes: Mapping[str, Mapping[str, Any]] | None = None, **by_name: Mapping[str, Any]
                ) -> ParameterLayout:
        """A new layout with some specs changed, e.g. ``replace(sigma_p={"free": False, "start": 2.0})``.

        Used by pre-fits and profiles to fix parameters or move starts. Resolved units carry over
        for parameters whose spec keeps ``unit=None``.
        """
        all_changes = dict(changes or {})
        all_changes.update(by_name)
        unknown = set(all_changes) - set(self.names)
        if unknown:
            raise KeyError(f"no parameters named {sorted(unknown)}")
        specs = [dataclasses.replace(s, **all_changes[s.name]) if s.name in all_changes else s for s in self.specs]
        new = ParameterLayout(specs)
        if new._units is None and self._units is not None:
            units = self._units.copy()
            for k, s in enumerate(specs):
                if s.unit is not None:
                    units[k] = s.unit
            new._units = units
        return new

    def with_starts(self, theta: np.ndarray) -> ParameterLayout:
        """A new layout whose starts are theta. Theta must lie within the bounds."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.P,):
            raise ValueError(f"theta has shape {theta.shape}, expected ({self.P},)")
        return self.replace({n: {"start": float(v)} for n, v in zip(self.names, theta)})

    def describe(self) -> list[dict[str, Any]]:
        """One dict per parameter for reports (name, start, bounds, unit, cap, free, note)."""
        units = self._units
        out = []
        for k, s in enumerate(self.specs):
            out.append({
                "name": s.name, "role": s.role, "column": s.column, "basis": s.basis, "start": s.start,
                "lower": s.lower, "upper": s.upper, "cap": s.cap, "free": s.free, "note": s.note,
                "unit": None if units is None else float(units[k]),
            })
        return out

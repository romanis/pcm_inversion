"""Micro moments over household sets, as tables and as a direct numpy evaluator (model-agnostic).

A micro-moment table has one row per moment: ``name``, ``survey`` (a 0/1 agent column: the household is in that
survey's sample), ``char`` (the data label of the characteristic), ``kind`` ("diff": E[x | set a] - E[x | set b];
"mean": E[x | set a], with ``b`` equal to ``a``; "ratio": P(buy | a) / P(buy | b)), ``demographic`` (the agent column
defining the sets), ``a`` and ``b`` (its levels), ``value`` and ``variance``. A household set is the households with
``demographic == level`` and ``survey == 1``; its 0/1 agent column is named by ``set_column``. ``char_map`` maps a data
label to the product column of the model counterpart.

The model counterpart of E[x | H] is the household-weighted mean over the households in H (pooled over the markets,
each market's weights summing to 1) of that household's expected x given a purchase: ``MicroMomentEvaluator``. This is
not pyblp's purchase-weighted E[x | H, inside]. ``moments.GroupMicroMoments`` computes the same rows from the kernel's
per-household sums with derivatives; the evaluator here is its direct reference.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["MicroMoments", "set_column", "add_set_columns", "group_moment_rows", "MicroMomentEvaluator",
           "model_micro_moments"]


@dataclasses.dataclass(frozen=True, eq=False)
class MicroMoments:
    """A micro-moment table (columns in the module docstring). ``weight`` is diag(1 / variance)."""

    table: pd.DataFrame

    @property
    def values(self) -> np.ndarray:
        return self.table.value.to_numpy(float)

    @property
    def variances(self) -> np.ndarray:
        return self.table.variance.to_numpy(float)

    @property
    def weight(self) -> np.ndarray:
        return np.diag(1.0 / self.variances)


def set_column(survey: str, demographic: str, level: Any) -> str:
    """Name of the 0/1 agent column of a household set: ``<survey>_<demographic>_<level>``."""
    return f"{survey}_{demographic}_{int(level)}"


def add_set_columns(agents: pd.DataFrame, surveys: Sequence[str],
                    demographics: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    """``agents`` with one 0/1 column per (survey, demographic, level), surveys outermost, then demographics in mapping
    order, then levels: demographic = level and the household in the survey (``agents[survey] == 1``)."""
    a = agents.copy()
    for s in surveys:
        for col, levels in demographics.items():
            for v in levels:
                a[set_column(s, col, v)] = ((a[col] == v) & (a[s] == 1)).astype(float)
    return a


def group_moment_rows(micro: MicroMoments, char_map: Mapping[str, str]) -> list[tuple[str, str | None, str, str]]:
    """(kind, product column or None, set column a, set column b) for each row of ``micro``, in order: the arguments
    of ``moments.GroupMomentRow``."""
    out = []
    for r in micro.table.itertuples():
        char = char_map[r.char] if r.kind != "ratio" else None
        out.append((r.kind, char, set_column(r.survey, r.demographic, r.a), set_column(r.survey, r.demographic, r.b)))
    return out


class MicroMomentEvaluator:
    """MicroMomentEvaluator(products, agents, micro, char_map)(probabilities) -> (k,) model counterparts.

    ``probabilities[t]`` is the (J_t, H_t) purchase probability matrix of the t-th market in sorted market order, rows
    in ``products`` order within the market, columns in ``agents`` order within the market. For a "diff" row each
    household's expected x given a purchase, sum_j x_j P_jh / sum_j P_jh, is averaged with the household weights over
    the households of set a and of set b, pooled across markets (a "mean" row: set a only). For a "ratio" row the
    household's purchase probability sum_j P_jh is averaged the same way.
    """

    def __init__(self, products: pd.DataFrame, agents: pd.DataFrame, micro: MicroMoments,
                 char_map: Mapping[str, str]) -> None:
        self.years = np.unique(products.market_ids.to_numpy())
        self.micro = micro
        self.J = [int((products.market_ids == y).sum()) for y in self.years]
        self.H = [int((agents.market_ids == y).sum()) for y in self.years]
        chars = sorted(set(char_map.values()))
        self.chars = chars
        self.x = [products.loc[products.market_ids == y, chars].to_numpy(float) for y in self.years]
        # per moment row and side: one (H_t,) weight vector per market, zero outside the set
        self.sides: list[tuple[np.ndarray, np.ndarray]] = []
        w_t = [agents.loc[agents.market_ids == y] for y in self.years]
        for row in micro.table.itertuples():
            pair = []
            for level in (row.a, row.b):
                vec = [np.where((ag[row.demographic].to_numpy() == level) & (ag[row.survey].to_numpy() == 1),
                                ag.weights.to_numpy(float), 0.0) for ag in w_t]
                pair.append(np.concatenate(vec))
            self.sides.append((pair[0], pair[1]))
        self.kind = micro.table.kind.to_numpy()
        self.col = [chars.index(char_map[c]) if k != "ratio" else -1
                    for c, k in zip(micro.table.char, micro.table.kind)]

    def __call__(self, probabilities: Sequence[np.ndarray]) -> np.ndarray:
        if len(probabilities) != self.years.size:
            raise ValueError(f"{len(probabilities)} probability matrices for {self.years.size} markets")
        inside, cond = [], []
        for t, P in enumerate(probabilities):
            P = np.asarray(P, dtype=float)
            if P.shape != (self.J[t], self.H[t]):
                raise ValueError(f"{self.years[t]}: probabilities {P.shape}, expected ({self.J[t]}, {self.H[t]})")
            s = P.sum(axis=0)
            inside.append(s)
            cond.append((self.x[t].T @ P) / s)       # (C, H_t)
        S = np.concatenate(inside)
        X = np.concatenate(cond, axis=1)
        out = np.empty(len(self.sides))
        for k, (wa, wb) in enumerate(self.sides):
            v = S if self.kind[k] == "ratio" else X[self.col[k]]
            a, b = (wa @ v) / wa.sum(), (wb @ v) / wb.sum()
            out[k] = a / b if self.kind[k] == "ratio" else a if self.kind[k] == "mean" else a - b
        return out


def model_micro_moments(probabilities: Sequence[np.ndarray], products: pd.DataFrame, agents: pd.DataFrame,
                        micro: MicroMoments, char_map: Mapping[str, str]) -> np.ndarray:
    """model_micro_moments(probabilities, products, agents, micro, char_map) -> (k,): ``MicroMomentEvaluator`` in one
    call."""
    return MicroMomentEvaluator(products, agents, micro, char_map)(probabilities)

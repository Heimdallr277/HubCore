"""Patient Rule Induction Method (PRIM) for scenario discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Box:
    limits: dict[str, tuple[float, float]]
    restricted: list[str]
    coverage: float
    density: float
    mass: float
    n_points: int

    def describe(self, labels: dict[str, str] | None = None, fmt: str = "{:.3g}") -> list[str]:
        labels = labels or {}
        out = []
        for key in self.restricted:
            lo, hi = self.limits[key]
            out.append(f"{labels.get(key, key)} in [{fmt.format(lo)}, {fmt.format(hi)}]")
        return out


def prim(X: pd.DataFrame, y: np.ndarray, alpha: float = 0.05, min_mass: float = 0.05,
         max_steps: int = 200) -> list[Box]:
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=bool)
    if X.empty or not X.columns.is_unique or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("PRIM needs finite observations with unique input columns")
    if not 0 < alpha < 0.5 or not 0 < min_mass <= 1 or y.ndim != 1:
        raise ValueError("Check the peel fraction, minimum mass and one-dimensional target")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    total_pos = y.sum()
    N = len(y)
    cols = list(X.columns)
    values = X.to_numpy(dtype=float)
    full_lo = values.min(axis=0)
    full_hi = values.max(axis=0)
    lo = full_lo.copy()
    hi = full_hi.copy()
    inside = np.ones(N, dtype=bool)

    def make_box() -> Box:
        n_in = int(inside.sum())
        pos_in = int(y[inside].sum())
        restricted = [cols[j] for j in range(len(cols))
                      if lo[j] > full_lo[j] + 1e-12 or hi[j] < full_hi[j] - 1e-12]
        return Box(
            limits={c: (float(lo[j]), float(hi[j])) for j, c in enumerate(cols)},
            restricted=restricted,
            coverage=pos_in / total_pos if total_pos else 0.0,
            density=pos_in / n_in if n_in else 0.0,
            mass=n_in / N,
            n_points=n_in,
        )

    trajectory = [make_box()]
    if total_pos == 0 or total_pos == N:
        return trajectory
    for _ in range(max_steps):
        if inside.sum() / N <= min_mass:
            break
        best = None
        pts = values[inside]
        ys = y[inside]
        for j in range(len(cols)):
            col = pts[:, j]
            q_lo = np.quantile(col, alpha)
            q_hi = np.quantile(col, 1 - alpha)
            for side, keep in (("lo", col >= q_lo), ("hi", col <= q_hi)):
                if keep.all() or keep.sum() == 0:
                    # Ties: peel at least one point if possible
                    continue
                if keep.sum() / N < min_mass:
                    continue
                dens = ys[keep].mean()
                score = (dens, keep.sum())
                if best is None or score > best[0]:
                    best = (score, j, side, q_lo if side == "lo" else q_hi)
        if best is None:
            break
        _, j, side, q = best
        if side == "lo":
            lo[j] = q
            inside &= values[:, j] >= q
        else:
            hi[j] = q
            inside &= values[:, j] <= q
        trajectory.append(make_box())
    return trajectory


def box_stats(X: pd.DataFrame, y: np.ndarray, limits: dict[str, tuple[float, float]],
              restricted: list[str]) -> tuple[float, float, float, int]:
    """(coverage, density, mass, n) of the box defined by ``restricted`` limits."""
    y = np.asarray(y, dtype=bool)
    inside = np.ones(len(X), dtype=bool)
    for c in restricted:
        lo, hi = limits[c]
        inside &= (X[c].to_numpy() >= lo) & (X[c].to_numpy() <= hi)
    n_in = int(inside.sum())
    pos = int(y[inside].sum())
    total = int(y.sum())
    return (pos / total if total else 0.0, pos / n_in if n_in else 0.0, n_in / len(X), n_in)


def prune(box: Box, X: pd.DataFrame, y: np.ndarray, max_density_loss: float = 0.03) -> Box:
    """Drop restrictions that barely matter, for interpretability.
    """
    X = X.reset_index(drop=True)
    restricted = list(box.restricted)
    target = box.density - max_density_loss
    while restricted:
        trials = []
        for c in restricted:
            rest = [d for d in restricted if d != c]
            cov, dens, mass, n_in = box_stats(X, y, box.limits, rest)
            trials.append((dens, c))
        dens, c = max(trials)
        if dens >= target:
            restricted.remove(c)
        else:
            break
    full = {c: (float(X[c].min()), float(X[c].max())) for c in X.columns}
    limits = {c: (box.limits[c] if c in restricted else full[c]) for c in X.columns}
    cov, dens, mass, n_in = box_stats(X, y, limits, restricted)
    return Box(limits=limits, restricted=restricted, coverage=cov, density=dens, mass=mass, n_points=n_in)


def quasi_p_values(box: Box, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """One-sided binomial quasi-p-value of each restriction (Bryant & Lempert, 2010).
    """
    from scipy.stats import binom

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=bool)
    out = {}
    for d in box.restricted:
        rest = [c for c in box.restricted if c != d]
        inside = np.ones(len(X), dtype=bool)
        for c in rest:
            lo, hi = box.limits[c]
            inside &= (X[c].to_numpy() >= lo) & (X[c].to_numpy() <= hi)
        n_wo, k_wo = int(inside.sum()), int(y[inside].sum())
        lo, hi = box.limits[d]
        inside &= (X[d].to_numpy() >= lo) & (X[d].to_numpy() <= hi)
        n_in, k_in = int(inside.sum()), int(y[inside].sum())
        p0 = k_wo / n_wo if n_wo else 0.0
        out[d] = float(binom.sf(k_in - 1, n_in, p0)) if n_in else 1.0
    return out


def prune_significant(box: Box, X: pd.DataFrame, y: np.ndarray, alpha: float = 0.05) -> tuple[Box, dict]:
    """Iteratively drop the least significant restriction while its quasi-p > alpha."""
    X = X.reset_index(drop=True)
    current = box
    while current.restricted:
        pv = quasi_p_values(current, X, y)
        worst = max(pv, key=pv.get)
        if pv[worst] <= alpha:
            return current, pv
        rest = [c for c in current.restricted if c != worst]
        full = {c: (float(X[c].min()), float(X[c].max())) for c in X.columns}
        limits = {c: (current.limits[c] if c in rest else full[c]) for c in X.columns}
        cov, dens, mass, n_in = box_stats(X, y, limits, rest)
        current = Box(limits=limits, restricted=rest, coverage=cov, density=dens, mass=mass, n_points=n_in)
    return current, {}


def explain(box: Box, X: pd.DataFrame, y: np.ndarray, alpha: float = 0.05,
            max_density_loss: float = 0.05) -> tuple[Box, dict]:
    """Interpretable box: density-preserving pruning, then significance pruning."""
    b = prune(box, X, y, max_density_loss=max_density_loss)
    return prune_significant(b, X, y, alpha=alpha)


def suggest(trajectory: list[Box], min_density: float = 0.0) -> int:
    """Index of a balanced box: maximizes the F1 score of coverage and density."""
    best, idx = -1.0, 0
    for i, b in enumerate(trajectory):
        if b.density < min_density:
            continue
        f1 = 0.0 if b.coverage + b.density == 0 else 2 * b.coverage * b.density / (b.coverage + b.density)
        if f1 > best:
            best, idx = f1, i
    return idx


def open_limits(box: Box, training: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Use open-ended bounds on unpeeled sides when scoring independent evaluations."""
    return {k: (lo if lo > training[k].min() + 1e-12 else -np.inf,
                hi if hi < training[k].max() - 1e-12 else np.inf)
            for k, (lo, hi) in box.limits.items()}

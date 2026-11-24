"""Cooperative cost games: Shapley value, excesses, least-core, repair, partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import linprog

from .costs import member_matrix


class GameError(RuntimeError):
    pass


def checked_game(cost: np.ndarray, n: int) -> np.ndarray:
    c = np.asarray(cost, dtype=float)
    if not 2 <= n <= 8 or c.shape != (1 << n,) or not np.isfinite(c).all() or c[0] != 0:
        raise GameError("Supply finite costs for every coalition of 2-8 firms, with c(empty) = 0")
    return c


def numerical_tol(cost: np.ndarray) -> float:
    return 1e-7 + 1e-10 * max(1.0, float(np.max(np.abs(cost))))


def popcount(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64)
    out = np.zeros_like(masks)
    m = masks.copy()
    while np.any(m):
        out += m & 1
        m >>= 1
    return out


def shapley(cost: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley value of a cost game (efficient: sums to cost[N])."""
    cost = checked_game(cost, n)
    masks = np.arange(1 << n)
    sizes = popcount(masks)
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) if s < n else 0.0
                  for s in range(n + 1)])
    phi = np.zeros(n)
    for i in range(n):
        without = masks[(masks >> i & 1) == 0]
        phi[i] = np.sum(w[sizes[without]] * (cost[without | (1 << i)] - cost[without]))
    return phi


def excess(x: np.ndarray, cost: np.ndarray, n: int) -> np.ndarray:
    """Excess of every coalition (length 2**n; entry 0 is 0)."""
    M = member_matrix(n)
    e = M @ np.asarray(x, dtype=float) - np.asarray(cost, dtype=float)
    e[0] = 0.0
    return e


@dataclass
class Stability:
    max_excess: float            # EUR/yr, max over proper coalitions
    worst_mask: int
    blocking: list[tuple[int, float]]   # (mask, excess) above the threshold, sorted desc
    ir_violations: list[int]     # firms paying more than stand-alone
    stable: bool

    @property
    def n_blocking(self) -> int:
        return len(self.blocking)


def stability(x: np.ndarray, cost: np.ndarray, n: int, rel_tol: float = 0.01,
              abs_tol: float = 500.0) -> Stability:
    """Stability report for allocation x.
    """
    cost = checked_game(cost, n)
    x = np.asarray(x, dtype=float)
    tol = numerical_tol(cost)
    if x.shape != (n,) or not np.isfinite(x).all() or abs(x.sum() - cost[-1]) > tol:
        raise GameError("The allocation must be finite and sum to the hub cost")
    if not np.isfinite([abs_tol, rel_tol]).all() or min(abs_tol, rel_tol) < 0:
        raise GameError("Materiality thresholds must be finite and non-negative")
    e = excess(x, cost, n)
    grand = (1 << n) - 1
    proper = np.arange(1, grand)
    thr = np.maximum(abs_tol, rel_tol * np.abs(np.asarray(cost)[proper]))
    ep = e[proper]
    j = int(np.argmax(ep))
    block_idx = np.flatnonzero(ep > thr + tol)
    blocking = sorted(((int(proper[k]), float(ep[k])) for k in block_idx), key=lambda t: -t[1])
    singles = [1 << i for i in range(n)]
    ir = [i for i, s in enumerate(singles) if e[s] > max(abs_tol, rel_tol * abs(cost[s])) + tol]
    return Stability(max_excess=float(ep[j]), worst_mask=int(proper[j]), blocking=blocking,
                     ir_violations=ir, stable=len(blocking) == 0)


def _core_constraints(cost: np.ndarray, n: int):
    grand = (1 << n) - 1
    cost = checked_game(cost, n)
    proper = np.arange(1, grand)
    M = member_matrix(n)[proper]
    return M, np.asarray(cost, dtype=float)[proper], float(cost[grand])


def least_core_value(cost: np.ndarray, n: int) -> float:
    """eps* = min_x max_{S proper} (x(S) - c(S)) subject to x(N) = c(N)."""
    M, cS, cN = _core_constraints(cost, n)
    # variables: x (n), eps (1); minimize eps
    c = np.r_[np.zeros(n), 1.0]
    A_ub = np.hstack([M, -np.ones((M.shape[0], 1))])
    A_eq = np.r_[np.ones(n), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=cS, A_eq=A_eq, b_eq=[cN],
                  bounds=[(None, None)] * (n + 1), method="highs")
    if res.status != 0:
        raise GameError(f"least-core LP failed: {res.message}")
    return float(res.x[-1])


def closest_in_least_core(cost: np.ndarray, n: int, target: np.ndarray,
                          eps: float | None = None, slack: float = 1e-12) -> tuple[np.ndarray, float]:

    if eps is None:
        eps = max(least_core_value(cost, n), 0.0)
    M, cS, cN = _core_constraints(cost, n)
    target = np.asarray(target, dtype=float)
    tol = slack * max(1.0, abs(cN))
    # variables: x (n), d (n) ; min sum d ; d >= x - t ; d >= t - x
    c = np.r_[np.zeros(n), np.ones(n)]
    eye = np.eye(n)
    A_ub = np.vstack([
        np.hstack([M, np.zeros((M.shape[0], n))]),       # x(S) <= c(S) + eps
        np.hstack([eye, -eye]),                               # x - d <= t
        np.hstack([-eye, -eye]),                              # -x - d <= -t
    ])
    b_ub = np.r_[cS + eps + tol, target, -target]
    A_eq = np.r_[np.ones(n), np.zeros(n)][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[cN],
                  bounds=[(None, None)] * n + [(0, None)] * n, method="highs")
    if res.status != 0:
        raise GameError(f"repair LP failed: {res.message}")
    return res.x[:n], float(eps)


def optimal_partition(cost: np.ndarray, n: int) -> tuple[list[int], float]:
    """Cheapest way to split the firms into disjoint hubs (dynamic programming, O(3^n))."""
    cost = checked_game(cost, n)
    full = 1 << n
    best = np.full(full, np.inf)
    choice = np.zeros(full, dtype=np.int64)
    best[0] = 0.0
    for mask in range(1, full):
        low = mask & -mask
        rest = mask ^ low
        sub = rest
        while True:
            s = sub | low
            val = cost[s] + best[mask ^ s]
            if val < best[mask]:
                best[mask] = val
                choice[mask] = s
            if sub == 0:
                break
            sub = (sub - 1) & rest
    parts, m = [], full - 1
    while m:
        parts.append(int(choice[m]))
        m ^= choice[m]
    return sorted(parts, key=lambda s: -popcount(np.array([s]))[0]), float(best[full - 1])


def members(mask: int, n: int) -> list[int]:
    return [i for i in range(n) if mask >> i & 1]


def subgame(cost: np.ndarray, n: int, keep: list[int]) -> np.ndarray:
    """Cost vector of the game restricted to the firms in ``keep``.
    """
    k = len(keep)
    sub = np.zeros(1 << k)
    for smask in range(1, 1 << k):
        full = 0
        for j in range(k):
            if smask >> j & 1:
                full |= 1 << keep[j]
        sub[smask] = cost[full]
    return sub

"""Continuous LP costs for a cost game with repeated, independently cyclic days.

Within a month, days with identical hourly demands have identical feasible
schedules for any fixed contract, monthly peak and battery. One copy weighted
by its number of occurrences retains the full annual objective.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from . import profiles
from .costs import Assumptions
from .lp_reference import SolverError
from .park import Park


@lru_cache(maxsize=24)
def _constraints(months: tuple[int, ...], duration: float, eta: float):
    # Design variables are contract, 12 monthly peaks and battery capacity.
    # Hourly variables are charge, discharge, state of charge and shortfall.
    n = 24 * len(months)
    identity = sparse.eye(n, format="csr")
    zero = sparse.csr_matrix((n, n))
    blank = sparse.csr_matrix((n, 14))
    hours = np.arange(n)
    peak = sparse.coo_matrix((-np.ones(n), (hours, 1 + np.repeat(months, 24))), shape=(n, 14)).tocsr()
    storage = sparse.coo_matrix((-np.ones(n), (hours, np.full(n, 13))), shape=(n, 14)).tocsr()

    # The previous hour of the first hour is the last hour of that same day.
    change = np.eye(24) - np.roll(np.eye(24), 1, axis=0)
    cycles = sparse.kron(sparse.eye(len(months)), sparse.csr_matrix(change), format="csr")
    equalities = sparse.hstack([blank, -eta * identity, identity, cycles, zero], format="csr")

    link = sparse.csr_matrix(np.c_[-np.ones(12), np.eye(12), np.zeros(12)])
    peak_below_contract = sparse.hstack([link, sparse.csr_matrix((12, 4 * n))])
    limits = sparse.vstack([
        sparse.hstack([peak, identity, -identity, zero, -identity]),       # imports <= monthly peak
        sparse.hstack([blank, -identity, identity, zero, identity]),       # imports >= 0
        sparse.hstack([storage / duration, identity, zero, zero, zero]),  # charging power
        sparse.hstack([storage / duration, zero, identity, zero, zero]),  # discharge power
        sparse.hstack([storage, zero, zero, identity, zero]),             # energy capacity
        peak_below_contract,
    ], format="csr")
    no_storage = sparse.vstack([
        sparse.hstack([peak[:, :13], -identity]),
        sparse.hstack([link[:, :13], sparse.csr_matrix((12, n))]),
    ], format="csr")
    return equalities, limits, no_storage


def _operating_cost(days: np.ndarray, months: np.ndarray, counts: np.ndarray, cap: float,
                    a: Assumptions) -> float:
    load = days.ravel()
    n = len(load)
    equalities, limits, no_storage = _constraints(tuple(months), a.battery_duration_h, a.battery_rte)
    penalties = a.unserved_cost * np.repeat(counts, 24)
    capacity_prices = np.r_[12 * a.tariff_contract, np.full(12, a.tariff_peak)]
    bounds_without = np.column_stack((np.zeros(13 + n), np.r_[cap, np.full(12, np.inf), load]))
    without = linprog(
        np.r_[capacity_prices, penalties], A_ub=no_storage, b_ub=np.r_[-load, np.zeros(12)],
        bounds=bounds_without, method="highs",
    )
    if not without.success:
        raise SolverError(f"Storage-free reference LP failed: {without.message}")

    bounds_with = np.column_stack((np.zeros(14 + 4 * n), np.r_[cap, np.full(13 + 3 * n, np.inf), load]))
    with_storage = linprog(
        np.r_[capacity_prices, a.crf * a.battery_cost_kwh, np.zeros(3 * n), penalties],
        A_eq=equalities, b_eq=np.zeros(n), A_ub=limits, b_ub=np.r_[-load, load, np.zeros(3 * n + 12)],
        bounds=bounds_with, method="highs",
    )
    if not with_storage.success:
        raise SolverError(f"Storage reference LP failed: {with_storage.message}")
    return float(min(without.fun, with_storage.fun + a.crf * a.battery_fixed))


def coalition_cost(load: np.ndarray, cap: float, a: Assumptions,
                   month_of_day: np.ndarray | None = None) -> float:
    """Optimal annualized cost before group overhead. Time steps have a one-hour duration."""
    a.validate()
    demand = np.asarray(load, dtype=float)
    if demand.ndim != 1 or not demand.size or demand.size % 24 or not np.isfinite(demand).all():
        raise ValueError("Supply whole days of finite hourly loads")
    if demand.min() < 0 or not np.isfinite(cap) or cap < 0:
        raise ValueError("Loads and the capacity ceiling must be non-negative and finite")
    days = demand.reshape(-1, 24)
    months = profiles.month_of_day() if month_of_day is None else np.asarray(month_of_day)
    if (months.shape != (len(days),) or not np.issubdtype(months.dtype, np.integer)
            or np.any((months < 0) | (months > 11))):
        raise ValueError("Supply a month index from 0 to 11 for each day")
    unique, counts = np.unique(np.column_stack((months, days)), axis=0, return_counts=True)
    return _operating_cost(unique[:, 1:], unique[:, 0].astype(int), counts, cap, a)


def price_game(park: Park, a: Assumptions, growth: np.ndarray | None = None) -> np.ndarray:
    """Reference cost for every nonempty member subset, including its own overhead."""
    loads = park.loads(growth)
    contracts = park.contracts()
    costs = np.zeros(1 << park.n)
    for mask in range(1, len(costs)):
        idx = [j for j in range(park.n) if (mask >> j) & 1]
        cap = float(contracts[idx].sum())
        overhead = 0.0
        if len(idx) > 1:
            cap *= a.group_capacity_factor
            overhead = a.hub_fixed + a.hub_per_member * len(idx)
        costs[mask] = coalition_cost(loads[idx].sum(axis=0), cap, a) + overhead
    return costs

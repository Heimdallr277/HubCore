"""Exact linear-programming reference model for one coalition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from . import profiles
from .costs import Assumptions


class SolverError(RuntimeError):
    pass


@dataclass
class LPResult:
    cost: float                  # annual cost excluding hub overhead
    contract_kw: float
    battery_kwh: float
    monthly_peak_kw: np.ndarray
    unserved_kwh: float
    imports: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    unserved: np.ndarray


def solve(load: np.ndarray, cap: float, a: Assumptions, *,
          battery: str | float = "optimise", contract: float | None = None,
          month_of_day: np.ndarray | None = None, linked_midnights: bool = False) -> LPResult:
    """Solve the reference LP for one coalition load profile.
    """
    a.validate()
    X = np.asarray(load, dtype=float)
    if X.ndim != 1 or not X.size or X.size % 24 or not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("Provide whole days of finite non-negative hourly demand")
    if not np.isfinite(cap) or cap < 0 or (contract is not None and not 0 <= contract <= cap):
        raise ValueError("Fixed contract must lie between zero and a finite capacity right")
    if battery == "optimise":
        r0 = _solve(load, cap, a, 0.0, contract, month_of_day, linked_midnights)
        r1 = _solve(load, cap, a, None, contract, month_of_day, linked_midnights)
        r1_cost = r1.cost + (a.crf * a.battery_fixed if r1.battery_kwh > 1e-6 else 0.0)
        if r1_cost < r0.cost:
            r1.cost = r1_cost
            return r1
        return r0
    if battery == "none":
        return _solve(load, cap, a, 0.0, contract, month_of_day, linked_midnights)
    e = float(battery)
    if not np.isfinite(e) or e < 0:
        raise ValueError("Battery size must be finite and non-negative")
    res = _solve(load, cap, a, e, contract, month_of_day, linked_midnights)
    if e > 0:
        res.cost += a.crf * a.battery_fixed
    return res


def _solve(load, cap, a: Assumptions, e_fixed, contract, month_of_day, linked_midnights) -> LPResult:
    X = np.asarray(load, dtype=float)
    T = X.size
    if T % 24:
        raise ValueError("load length must be a whole number of days")
    D = T // 24
    mod = profiles.month_of_day() if month_of_day is None else np.asarray(month_of_day)
    if mod.shape != (D,) or not np.issubdtype(mod.dtype, np.integer) or np.any((mod < 0) | (mod > 11)):
        raise ValueError("month_of_day needs one integer month (0-11) per day")
    month_t = np.repeat(mod, 24)
    eta = a.battery_rte
    inv_h = 1.0 / a.battery_duration_h

    iP, iE = 0, 1
    iM = 2
    o_imp = 14
    o_ch = o_imp + T
    o_dis = o_ch + T
    o_soc = o_dis + T
    o_s0 = o_soc + T
    o_u = o_s0 + D
    nvar = o_u + T
    t = np.arange(T)
    day = t // 24
    first = (t % 24) == 0
    last = (t % 24) == 23

    c = np.zeros(nvar)
    c[iP] = 12.0 * a.tariff_contract
    c[iM:iM + 12] = a.tariff_peak
    c[iE] = a.crf * a.battery_cost_kwh
    c[o_u:o_u + T] = a.unserved_cost

    # --- equalities -------------------------------------------------------
    rows, cols, vals = [], [], []

    def add(r, cidx, v):
        rows.append(np.broadcast_to(r, np.shape(cidx)).ravel())
        cols.append(np.ravel(cidx))
        vals.append(np.broadcast_to(v, np.shape(cidx)).ravel().astype(float))

    # balance rows 0..T-1
    add(t, o_imp + t, 1.0)
    add(t, o_ch + t, -1.0)
    add(t, o_dis + t, 1.0)
    add(t, o_u + t, 1.0)
    # dynamics rows T..2T-1
    r = T + t
    add(r, o_soc + t, 1.0)
    add(r, o_ch + t, -eta)
    add(r, o_dis + t, 1.0)
    nf = ~first
    add(r[nf], o_soc + t[nf] - 1, -1.0)
    add(r[first], o_s0 + day[first], -1.0)
    # cyclic rows 2T..2T+D-1
    rd = 2 * T + np.arange(D)
    add(rd, o_soc + t[last], 1.0)
    add(rd, o_s0 + np.arange(D), -1.0)
    extra = D - 1 if linked_midnights else 0
    if extra:
        link = 2 * T + D + np.arange(extra)
        add(link, o_s0 + np.arange(1, D), 1.0)
        add(link, np.full(extra, o_s0), -1.0)
    A_eq = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(2 * T + D + extra, nvar))
    b_eq = np.concatenate([X, np.zeros(T + D + extra)])

    # --- inequalities (<= 0) --------------------------------------------------
    rows, cols, vals = [], [], []
    k = 0
    add(k + t, o_imp + t, 1.0); add(k + t, np.full(T, iP), -1.0); k += T          # imp <= P
    add(k + t, o_imp + t, 1.0); add(k + t, iM + month_t, -1.0); k += T            # imp <= M_m
    add(k + t, o_soc + t, 1.0); add(k + t, np.full(T, iE), -1.0); k += T          # soc <= E
    dd = np.arange(D)
    add(k + dd, o_s0 + dd, 1.0); add(k + dd, np.full(D, iE), -1.0); k += D        # soc0 <= E
    add(k + t, o_ch + t, 1.0); add(k + t, np.full(T, iE), -inv_h); k += T         # ch <= E/h
    add(k + t, o_dis + t, 1.0); add(k + t, np.full(T, iE), -inv_h); k += T        # dis <= E/h
    A_ub = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(k, nvar))
    b_ub = np.zeros(k)

    bounds = [(0, None)] * nvar
    if contract is None:
        bounds[iP] = (0, float(cap))
    else:
        bounds[iP] = (float(contract), float(contract))
    bounds[iE] = (0, None) if e_fixed is None else (float(e_fixed), float(e_fixed))
    for i in range(T):
        bounds[o_u + i] = (0, float(X[i]))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise SolverError(f"LP not solved: {res.message}")
    x = res.x
    return LPResult(
        cost=float(res.fun),
        contract_kw=float(x[iP]),
        battery_kwh=float(x[iE]),
        monthly_peak_kw=x[iM:iM + 12].copy(),
        unserved_kwh=float(x[o_u:o_u + T].sum()),
        imports=x[o_imp:o_imp + T].copy(),
        charge=x[o_ch:o_ch + T].copy(),
        discharge=x[o_dis:o_dis + T].copy(),
        soc=x[o_soc:o_soc + T].copy(),
        unserved=x[o_u:o_u + T].copy(),
    )

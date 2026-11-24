"""Coalition cost model (fast evaluator).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from . import profiles

MONTHS = 12
BATTERY_EXTENSION_FACTOR = 1.5
MAX_BATTERY_EXTENSIONS = 12     # geometric steps per pass
BATTERY_EXTENSION_PASSES = 3    # extend-then-refine passes


@dataclass(frozen=True)
class Assumptions:
    """Economic and technical assumptions (one future)."""

    tariff_contract: float = 2.29      # EUR per kW contracted per month (Liander 2026, MS)
    tariff_peak: float = 3.57          # EUR per kW of monthly peak per month (Liander 2026, MS)
    battery_cost_kwh: float = 250.0    # EUR per kWh installed
    battery_fixed: float = 20_000.0    # EUR per installation (integration, permits)
    battery_duration_h: float = 2.0    # power = energy / duration
    battery_rte: float = 0.90          # round-trip efficiency
    battery_life_y: int = 15
    discount_rate: float = 0.06
    unserved_cost: float = 0.40        # EUR per kWh of demand above the contract
    group_capacity_factor: float = 1.0  # group capacity = factor x sum of member contracts
    hub_fixed: float = 15_000.0        # EUR per year, coordination / EMS / legal
    hub_per_member: float = 1_000.0    # EUR per member per year
    battery_hours_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    battery_refine_rounds: int = 3
    contract_grid_points: int = 30
    max_shortfall_days: int = 45

    @property
    def crf(self) -> float:
        """Capital recovery factor."""
        r, n = self.discount_rate, self.battery_life_y
        if r == 0:
            return 1.0 / n
        return float(r / -np.expm1(-n * np.log1p(r)))

    def updated(self, **changes) -> "Assumptions":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        checks = {
            "tariff_contract": self.tariff_contract >= 0,
            "tariff_peak": self.tariff_peak >= 0,
            "battery_cost_kwh": self.battery_cost_kwh >= 0,
            "battery_fixed": self.battery_fixed >= 0,
            "battery_duration_h": self.battery_duration_h > 0,
            "battery_rte": 0 < self.battery_rte <= 1,
            "battery_life_y": self.battery_life_y >= 1,
            "discount_rate": 0 <= self.discount_rate < 1,
            "unserved_cost": self.unserved_cost >= 0,
            "group_capacity_factor": 0 < self.group_capacity_factor <= 1.0,
            "hub_fixed": self.hub_fixed >= 0,
            "hub_per_member": self.hub_per_member >= 0,
            "battery_hours_grid": 0.0 in self.battery_hours_grid
            and all(h >= 0 for h in self.battery_hours_grid),
            "contract_grid_points": self.contract_grid_points >= 5,
            "battery_refine_rounds": 0 <= self.battery_refine_rounds <= 6,
            "max_shortfall_days": 1 <= self.max_shortfall_days <= profiles.DAYS,
        }
        bad = [k for k, ok in checks.items() if not ok or not np.isfinite(getattr(self, k)).all()]
        if bad:
            raise ValueError(f"Invalid assumptions: {', '.join(bad)}")


ASSUMPTION_FIELDS = [f.name for f in fields(Assumptions)]


@dataclass
class CoalitionCosts:
    """Costs for all coalitions, indexed by bitmask (index 0 = empty set)."""

    n: int
    cost: np.ndarray
    contract_kw: np.ndarray
    battery_kwh: np.ndarray
    capacity_right_kw: np.ndarray
    raw_peak_kw: np.ndarray
    unserved_kwh: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)
    # True where the chosen battery is the largest size the search evaluated (a boundary
    # optimum: a larger battery might be cheaper). Check such coalitions with the LP.
    battery_at_search_limit: np.ndarray | None = None

    @property
    def grand(self) -> int:
        return (1 << self.n) - 1

    def standalone(self) -> np.ndarray:
        return np.array([self.cost[1 << i] for i in range(self.n)])

    def savings(self, mask: int | None = None) -> float:
        mask = self.grand if mask is None else mask
        members = [i for i in range(self.n) if mask >> i & 1]
        return float(sum(self.cost[1 << i] for i in members) - self.cost[mask])


def member_matrix(n: int) -> np.ndarray:
    """Binary membership matrix (2^n, n); row k = members of bitmask k."""
    masks = np.arange(1 << n)
    return ((masks[:, None] >> np.arange(n)) & 1).astype(float)


def _month_starts() -> np.ndarray:
    month = profiles.month_of_day()
    return np.flatnonzero(np.r_[True, month[1:] != month[:-1]])


def _cyclic_delivery(diff: np.ndarray, energy: np.ndarray, power: np.ndarray, eta: float) -> np.ndarray:
    """Maximum delivered kWh below a constant import ceiling on independent cyclic days.
    """
    demand = np.minimum(np.maximum(diff, 0.), power[:, None])
    charge = eta * np.minimum(np.maximum(-diff, 0.), power[:, None])
    net = charge - demand
    soc = np.where(net.sum(axis=1) >= 0., energy, 0.)
    for delta in net.T:
        soc = np.minimum(energy, np.maximum(0., soc + delta))
    unserved = np.zeros(len(energy))
    for delta in net.T:
        soc = soc + delta
        unserved += np.maximum(-soc, 0.)
        soc = np.minimum(energy, np.maximum(0., soc))
    return demand.sum(axis=1) - unserved


def _levels_flat(Xd: np.ndarray, energy: np.ndarray, power: np.ndarray, eta: float,
                 iterations: int) -> np.ndarray:
    """Binary search for the lowest feasible shaving level of individual days.
    """
    lo = np.maximum(Xd.mean(axis=1), Xd.max(axis=1) - power)   # valid lower bounds
    hi = Xd.max(axis=1)
    lo = np.minimum(lo, hi)
    E = energy
    B = power
    B2 = power[:, None]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        diff = Xd - mid[:, None]
        ex = np.maximum(diff, 0.0)
        need = ex.sum(axis=1)
        recharge = eta * np.minimum(np.maximum(-diff, 0.0), B2).sum(axis=1)
        attainable = (ex.max(axis=1) <= B + 1e-9) & (recharge + 1e-9 >= need)
        feasible = (need <= E + 1e-9) & attainable
        # Only multiple discharge windows can use more than E in one day.
        revisit = (need > E + 1e-9) & attainable
        if revisit.any():
            idx = np.flatnonzero(revisit)
            positive = diff[idx] > 0
            idx = idx[(positive & ~np.roll(positive, 1, axis=1)).sum(axis=1) > 1]
            if idx.size:
                delivered = _cyclic_delivery(diff[idx], E[idx], B[idx], eta)
                feasible[idx] = delivered + 1e-9 >= need[idx]
        hi = np.where(feasible, mid, hi)
        lo = np.where(feasible, lo, mid)
    return hi


def daily_levels(X3: np.ndarray, energy: np.ndarray, power: np.ndarray, eta: float,
                 iterations: int = 20, top_k: int | None = None,
                 month_starts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lowest daily load level the battery can shave to (conservative, per day).
    """
    m, D, _ = X3.shape
    peak = X3.max(axis=2)
    lb = np.maximum(X3.mean(axis=2), peak - power[:, None])         # lower bound on L_d
    if month_starts is None:
        month_starts = _month_starts()
    month_id = np.repeat(np.arange(len(month_starts)),
                         np.diff(np.r_[month_starts, D]))
    month_lb = np.maximum.reduceat(lb, month_starts, axis=1)[:, month_id]   # (m, D)
    candidate = peak >= month_lb - 1e-9
    if top_k is not None and top_k >= D:
        candidate[:] = True
    elif top_k is not None:
        kth_lb = -np.partition(-lb, top_k - 1, axis=1)[:, top_k - 1]
        candidate |= peak >= kth_lb[:, None] - 1e-9
    L = peak.copy()
    rows, days = np.nonzero(candidate)
    if rows.size:
        L[rows, days] = _levels_flat(X3[rows, days], energy[rows], power[rows], eta, iterations)
    return L, candidate


def _unserved(Xd: np.ndarray, P: np.ndarray, energy: np.ndarray, power: np.ndarray,
              eta: float) -> np.ndarray:
    """Unserved energy (kWh) for candidate contracts.
    """
    diff = Xd[:, None, :, :] - P[:, :, None, None]           # (m, C, K, 24)
    ex = np.maximum(diff, 0.0)
    need = ex.sum(axis=3)                                       # (m, C, K)
    E = energy[:, None, None]
    if np.all(energy <= 0):
        return need.sum(axis=2)
    B4 = power[:, None, None, None]
    power_lim = np.minimum(ex, B4).sum(axis=3)
    recharge = eta * np.minimum(np.maximum(-diff, 0.0), B4).sum(axis=3)
    shaved = np.minimum(np.minimum(E, power_lim), recharge)
    revisit = (np.minimum(power_lim, recharge) > E + 1e-9) & (E > 0)
    if revisit.any():
        row, contract, day = np.nonzero(revisit)
        delta = diff[row, contract, day]
        positive = delta > 0
        multi = (positive & ~np.roll(positive, 1, axis=1)).sum(axis=1) > 1
        if multi.any():
            r, c, d = row[multi], contract[multi], day[multi]
            shaved[r, c, d] = _cyclic_delivery(delta[multi], energy[r], power[r], eta)
    return np.maximum(need - shaved, 0.0).sum(axis=2)


def no_battery_design(loads: np.ndarray, caps: np.ndarray, a: Assumptions,
                      month_starts: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact storage-free solution, including economic curtailment in each month.
    """
    rows = len(loads)
    cu = a.unserved_cost
    ends = np.r_[month_starts[1:], profiles.DAYS] * 24
    months = [loads[:, start * 24:end] for start, end in zip(month_starts, ends)]
    peaks = np.zeros((rows, len(months)))
    contracts = np.zeros(rows)
    if cu > 0:
        sorted_months = [np.sort(month, axis=1) for month in months]
        for m, values in enumerate(sorted_months):
            rank = int(min(a.tariff_peak / cu, values.shape[1]))
            if rank < values.shape[1]:
                peaks[:, m] = values[:, -(rank + 1)]
        annual = np.sort(loads, axis=1)
        for j, cap in enumerate(caps):
            left, right = -1, annual.shape[1] - 1
            while right - left > 1:
                mid = (left + right) // 2
                trial = annual[j, mid]
                counts = [len(v[j]) - np.searchsorted(v[j], trial, side="right") for v in sorted_months]
                value = np.maximum(cu * np.asarray(counts) - a.tariff_peak, 0).sum()
                if value > 12 * a.tariff_contract:
                    left = mid
                else:
                    right = mid
            # Zero is an extra breakpoint for a strictly positive minimum load.
            value_at_zero = sum(max(cu * np.count_nonzero(v[j]) - a.tariff_peak, 0) for v in months)
            if value_at_zero > 12 * a.tariff_contract:
                contracts[j] = min(cap, annual[j, right])
    billed = np.minimum(contracts[:, None], peaks)
    unserved = sum(np.maximum(month - billed[:, [m]], 0).sum(axis=1) for m, month in enumerate(months))
    detail = {"contract_kw": contracts, "battery_kwh": np.zeros(rows), "unserved_kwh": unserved,
              "c_contract": 12 * a.tariff_contract * contracts, "c_peak": a.tariff_peak * billed.sum(axis=1),
              "c_battery": np.zeros(rows), "c_unserved": cu * unserved}
    return detail["c_contract"] + detail["c_peak"] + detail["c_unserved"], detail


def coalition_costs(loads: np.ndarray, contracts: np.ndarray, a: Assumptions,
                    masks: np.ndarray | None = None) -> CoalitionCosts:
    """Evaluate c(S) for all coalitions (or the given bitmasks).
    """
    a.validate()
    loads = np.asarray(loads, dtype=float)
    if loads.ndim != 2 or loads.shape[1] != profiles.HOURS or not 1 <= loads.shape[0] <= 8:
        raise ValueError("loads must contain 1-8 rows of 8760 hourly values")
    if not np.all(np.isfinite(loads)) or np.any(loads < 0):
        raise ValueError("Hourly loads must be finite and non-negative")
    n = loads.shape[0]
    contracts = np.asarray(contracts, dtype=float)
    if contracts.shape != (n,) or not np.isfinite(contracts).all() or np.any(contracts < 0):
        raise ValueError("Provide one finite, non-negative contract per firm")
    full = 1 << n
    if masks is None:
        masks = np.arange(1, full)
    masks = np.asarray(masks)
    if (masks.ndim != 1 or not masks.size or not np.issubdtype(masks.dtype, np.integer)
            or np.any((masks <= 0) | (masks >= full)) or len(np.unique(masks)) != len(masks)):
        raise ValueError("Provide distinct nonempty coalition masks for this park")
    # The contract-by-day work arrays grow with the number of masks. Bound peak memory on eight-firm parks.
    if len(masks) > 64:
        chunks = np.array_split(masks, (len(masks) + 63) // 64)
        batches = [coalition_costs(loads, contracts, a, batch) for batch in chunks]
        first = batches[0]
        for batch in batches[1:]:
            selected = np.isfinite(batch.cost) & (np.arange(full) != 0)
            outputs = ("cost", "contract_kw", "battery_kwh", "capacity_right_kw",
                       "raw_peak_kw", "unserved_kwh")
            for name in outputs:
                getattr(first, name)[selected] = getattr(batch, name)[selected]
            first.battery_at_search_limit[selected] = batch.battery_at_search_limit[selected]
            for name, values in batch.components.items():
                first.components[name][selected] = values[selected]
        return first
    Mfull = member_matrix(n)
    M = Mfull[masks]                                    # (m, n)
    size = M.sum(axis=1)
    X = M @ loads                                       # (m, 8760)
    m = X.shape[0]
    X3 = X.reshape(m, profiles.DAYS, 24)
    cap = M @ np.asarray(contracts, dtype=float)
    cap = np.where(size >= 2, a.group_capacity_factor * cap, cap)
    raw_peak = X.max(axis=1)
    mean_load = X.mean(axis=1)
    month_starts = _month_starts()
    K = min(a.max_shortfall_days, profiles.DAYS)
    eta = a.battery_rte
    overhead = np.where(size >= 2, a.hub_fixed + a.hub_per_member * size, 0.0)

    hi_all = np.minimum(cap, raw_peak)                  # never contract above the raw peak
    keys = ("contract_kw", "battery_kwh", "unserved_kwh",
            "c_contract", "c_peak", "c_battery", "c_unserved")

    def evaluate(energy: np.ndarray,
                 sel: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Optimal contract and cost for given battery sizes of all coalitions, or of rows ``sel``.
        """
        X3s = X3 if sel is None else X3[sel]
        hi = hi_all if sel is None else hi_all[sel]
        ms = X3s.shape[0]
        power = energy / a.battery_duration_h
        if np.all(energy <= 0):
            L = X3s.max(axis=2)
        else:
            L, _ = daily_levels(X3s, energy, power, eta, top_k=K + 1, month_starts=month_starts)
            L = np.where(energy[:, None] > 0, L, X3s.max(axis=2))
        Mm0 = np.maximum.reduceat(L, month_starts, axis=1)             # (ms, 12)
        order = np.argsort(-L, axis=1)
        topk = order[:, :K]
        L_sorted = np.take_along_axis(L, order, axis=1)
        floor = L_sorted[:, K] if K < profiles.DAYS else np.zeros(ms)
        lo = np.maximum(floor, 0.0)
        degenerate = lo >= hi
        lo_eff = np.where(degenerate, hi, lo)
        grid = lo_eff[:, None] + (hi - lo_eff)[:, None] * np.linspace(0, 1, a.contract_grid_points)[None, :]
        extra = np.clip(Mm0, lo_eff[:, None], hi[:, None])
        P = np.concatenate([grid, extra, hi[:, None]], axis=1)          # (ms, C)
        Xtop = X3s[np.arange(ms)[:, None], topk]                        # (ms, K, 24)
        U = np.zeros_like(P)
        active = ~degenerate
        if active.any():
            U[active] = _unserved(Xtop[active], P[active], energy[active], power[active], eta)
        # Rows where even the capacity right is below the (K+1)-th largest day
        # level: evaluate the cap itself with all days (no truncation).
        if degenerate.any():
            U_full = _unserved(X3s[degenerate], hi[degenerate][:, None],
                               energy[degenerate], power[degenerate], eta)
            U[degenerate] = U_full[:, :1]
            P[degenerate] = hi[degenerate][:, None]
        c_contract = 12.0 * a.tariff_contract * P
        c_peak = a.tariff_peak * np.minimum(P[:, :, None], Mm0[:, None, :]).sum(axis=2)
        c_unserved = a.unserved_cost * U
        total = c_contract + c_peak + c_unserved
        j = np.argmin(total, axis=1)
        idx = np.arange(ms)
        c_batt = np.where(energy > 0, a.crf * (a.battery_cost_kwh * energy + a.battery_fixed), 0.0)
        detail = {"contract_kw": P[idx, j], "battery_kwh": energy, "unserved_kwh": U[idx, j],
                  "c_contract": c_contract[idx, j], "c_peak": c_peak[idx, j],
                  "c_battery": c_batt, "c_unserved": c_unserved[idx, j]}
        return total[idx, j] + c_batt, detail

    best, out = no_battery_design(X, cap, a, month_starts)
    best_hours = np.zeros(m)
    largest_tried = np.zeros(m)                          # largest battery size evaluated, per row

    def consider(hours: np.ndarray, sel: np.ndarray | None = None) -> None:
        nonlocal best, best_hours
        if sel is None:
            cand, detail = evaluate(hours * mean_load)
            better = cand < best - 1e-9
            best = np.where(better, cand, best)
            best_hours = np.where(better, hours, best_hours)
            for key in keys:
                out[key] = np.where(better, detail[key], out[key])
            np.maximum(largest_tried, hours, out=largest_tried)
            return
        cand, detail = evaluate(hours * mean_load[sel], sel)
        better = cand < best[sel] - 1e-9
        rows_better = sel[better]
        best[rows_better] = cand[better]
        best_hours[rows_better] = hours[better]
        for key in keys:
            out[key][rows_better] = detail[key][better]
        largest_tried[sel] = np.maximum(largest_tried[sel], hours)

    def at_limit() -> np.ndarray:
        return (best_hours > 0) & (best_hours >= largest_tried * (1 - 1e-12))

    def refine(sel: np.ndarray | None = None) -> None:
        for r in range(a.battery_refine_rounds):          # local refinement
            step = 0.33 / (2 ** r)
            centre = best_hours.copy() if sel is None else best_hours[sel].copy()
            if not np.any(centre > 0):
                break
            for factor in (1.0 - step * 0.75, 1.0 + step):
                consider(centre * factor, sel)

    for h in a.battery_hours_grid:                       # coarse grid
        if h > 0:
            consider(np.full(m, float(h)))
    refine()

    for _ in range(BATTERY_EXTENSION_PASSES):
        grow = np.flatnonzero(at_limit())
        if not grow.size:
            break
        sel = grow
        for _ in range(MAX_BATTERY_EXTENSIONS):
            consider(largest_tried[sel] * BATTERY_EXTENSION_FACTOR, sel)
            sel = sel[at_limit()[sel]]
            if not sel.size:
                break
        refine(grow)
    search_limit = at_limit()

    def scatter(vals: np.ndarray) -> np.ndarray:
        arr = np.full(full, np.nan)
        arr[0] = 0.0
        arr[masks] = vals
        return arr

    total_cost = best + overhead
    comps = {k: scatter(out[k]) for k in ("c_contract", "c_peak", "c_battery", "c_unserved")}
    comps["c_overhead"] = scatter(overhead)
    limit_flags = np.zeros(full, dtype=bool)
    limit_flags[masks] = search_limit
    return CoalitionCosts(
        n=n,
        cost=scatter(total_cost),
        contract_kw=scatter(out["contract_kw"]),
        battery_kwh=scatter(out["battery_kwh"]),
        capacity_right_kw=scatter(cap),
        raw_peak_kw=scatter(raw_peak),
        unserved_kwh=scatter(out["unserved_kwh"]),
        components=comps,
        battery_at_search_limit=limit_flags,
    )


def park_costs(park, a: Assumptions, growth: np.ndarray | None = None) -> CoalitionCosts:
    """Convenience wrapper: costs of all coalitions of a :class:`~hubcore.park.Park`."""
    return coalition_costs(park.loads(growth), park.contracts(), a)

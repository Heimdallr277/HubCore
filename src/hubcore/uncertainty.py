"""Deep-uncertainty space and Latin hypercube sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .costs import Assumptions
from .park import Park


@dataclass(frozen=True)
class Uncertainty:
    key: str            # Assumptions field name, or "growth:<firm index>"
    label: str
    low: float
    high: float
    base: float
    unit: str
    status: str         # Sourced / Sourced range / Illustrative / Scenario
    note: str = ""

    @property
    def is_growth(self) -> bool:
        return self.key.startswith("growth:")

    @property
    def firm_index(self) -> int:
        return int(self.key.split(":")[1])


ASSUMPTION_UNCERTAINTIES: tuple[Uncertainty, ...] = (
    Uncertainty("tariff_contract", "Contracted-capacity tariff", 1.5, 4.0, 2.29, "EUR/kW/month",
                "Sourced base", "Liander 2026, MS category; range is an illustrative stress test"),
    Uncertainty("tariff_peak", "Monthly-peak tariff", 2.0, 6.5, 3.57, "EUR/kW/month",
                "Sourced base", "Liander 2026, MS category; range is an illustrative stress test"),
    Uncertainty("battery_cost_kwh", "Battery installed cost", 150.0, 400.0, 250.0, "EUR/kWh",
                "Illustrative", "Illustrative unit cost plus a separate fixed integration allowance"),
    Uncertainty("unserved_cost", "Cost of demand above contract", 0.15, 1.0, 0.40, "EUR/kWh",
                "Illustrative", "Proxy for generator rental or deferred electrification"),
    Uncertainty("group_capacity_factor", "Assumed group-capacity factor", 0.8, 1.0, 1.0, "x sum of contracts",
                "Illustrative", "Grid operator sets group capacity to the group's current and future use"),
    Uncertainty("hub_fixed", "Hub fixed overhead", 5_000.0, 40_000.0, 15_000.0, "EUR/yr",
                "Illustrative", "Coordination, energy management system, legal entity"),
    Uncertainty("hub_per_member", "Hub overhead per member", 500.0, 5_000.0, 1_000.0, "EUR/yr",
                "Illustrative", "Metering and EMS integration per firm"),
)


def default_space(park: Park, base: Assumptions | None = None) -> list[Uncertainty]:
    """Economic uncertainties plus one growth multiplier per firm with a plan."""
    base = base or Assumptions()
    space = [replace(u, base=getattr(base, u.key)) for u in ASSUMPTION_UNCERTAINTIES]
    for i, f in enumerate(park.firms):
        if f.addon != "none" and f.addon_mwh > 0:
            space.append(Uncertainty(f"growth:{i}", f"{f.name}: electrification realized",
                                     0.0, 2.0, 1.0, "x plan", "Scenario",
                                     "0 = plan abandoned, 1 = as planned, 2 = twice the plan"))
    return space


def sample(space: list[Uncertainty], n: int, seed: int = 42) -> pd.DataFrame:
    """Latin hypercube sample of ``n`` futures (columns = uncertainty keys)."""
    if n < 2:
        raise ValueError("Need at least 2 futures")
    sampler = qmc.LatinHypercube(d=len(space), seed=seed)
    u = sampler.random(n)
    lows = np.array([s.low for s in space])
    highs = np.array([s.high for s in space])
    values = qmc.scale(u, lows, highs)
    df = pd.DataFrame(values, columns=[s.key for s in space])
    df.index.name = "future"
    return df


def nominal(space: list[Uncertainty]) -> pd.Series:
    return pd.Series({s.key: s.base for s in space})


def to_future(row: pd.Series | dict, base: Assumptions, n_firms: int) -> tuple[Assumptions, np.ndarray]:
    """Translate one sampled row into (assumptions, growth multipliers)."""
    growth = np.ones(n_firms)
    changes = {}
    for key, val in dict(row).items():
        if key.startswith("growth:"):
            growth[int(key.split(":")[1])] = float(val)
        else:
            changes[key] = float(val)
    return base.updated(**changes), growth

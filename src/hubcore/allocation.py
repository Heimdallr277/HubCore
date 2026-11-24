"""Cost-allocation rules for the grand coalition.

Each rule returns cost shares x (EUR/yr, length n) that sum to c(N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import games


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    family: str          # "practical key" or "game-theoretic"
    description: str


RULES: dict[str, Rule] = {r.key: r for r in [
    Rule("capacity", "Capacity key", "practical key",
         "Hub cost split in proportion to each firm's existing contracted capacity (kW)."),
    Rule("consumption", "Consumption key", "practical key",
         "Hub cost split in proportion to annual electricity use (kWh)."),
    Rule("prorata", "Pro-rata savings", "practical key",
         "Every firm saves the same percentage of its stand-alone cost."),
    Rule("equal", "Equal savings", "practical key",
         "Total savings divided equally between firms (EUR per firm)."),
    Rule("shapley", "Shapley value", "game-theoretic",
         "Each firm pays its average marginal cost over all orders in which the hub could form."),
    Rule("leastcore", "Least-core (nearest to Shapley)", "game-theoretic",
         "The split that minimizes the strongest incentive of any group to leave, "
         "chosen as close as possible to the Shapley value."),
]}


def allocate(rule: str, cost: np.ndarray, n: int, contracts: np.ndarray,
             annual_kwh: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=float)
    grand = (1 << n) - 1
    cN = cost[grand]
    standalone = np.array([cost[1 << i] for i in range(n)])
    if rule == "capacity":
        w = np.asarray(contracts, dtype=float)
        return cN * w / w.sum()
    if rule == "consumption":
        w = np.asarray(annual_kwh, dtype=float)
        return cN * w / w.sum()
    if rule == "prorata":
        if standalone.sum() == 0:
            return np.full(n, cN / n)
        return cN * standalone / standalone.sum()
    if rule == "equal":
        return standalone - (standalone.sum() - cN) / n
    if rule == "shapley":
        return games.shapley(cost, n)
    if rule == "leastcore":
        eps = games.least_core_value(cost, n)
        x, _ = games.closest_in_least_core(cost, n, games.shapley(cost, n), eps=eps)
        return x
    raise KeyError(f"Unknown allocation rule '{rule}'")


def all_rules(cost, n, contracts, annual_kwh) -> dict[str, np.ndarray]:
    return {k: allocate(k, cost, n, contracts, annual_kwh) for k in RULES}

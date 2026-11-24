"""The featured decision: six-firm example, seed 42, future 26.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hubcore import allocation, games

FEATURED = {"future": 26, "n": 200, "seed": 42, "member": "Cold store",
            "rule": "leastcore", "clause": "adaptive"}

SCREENING = "Fast screening"
CONTINUOUS = "Continuous LP"
SCREENING_NOTE = ("Precomputed for all 200 futures. Battery sizes come from a finite grid "
                  "with local refinement, so each coalition cost is a feasible upper bound.")
CONTINUOUS_NOTE = ("All 63 coalitions repriced by a HiGHS linear program with continuous "
                   "contract and battery sizes, the same loads, daily storage cycle and future.")


@dataclass
class Verdict:
    method: str
    hub_cost: float
    standalone: np.ndarray        # per member
    allocated: np.ndarray         # adaptive least-core cost shares, per member
    excess: np.ndarray            # allocated - standalone, per member
    cutoff: np.ndarray            # materiality cutoff per member
    eps: float                    # signed least-core value of the game
    stable: bool                  # passes the materiality test (all subgroups)
    material: list[tuple[str, float, float]]   # (group, excess, cutoff) material objections

    @property
    def passes(self) -> bool:
        return self.stable


def verdict(method: str, cost: np.ndarray, names: list[str], contracts: np.ndarray,
            kwh: np.ndarray, rel_tol: float, abs_tol: float, rule: str = "leastcore") -> Verdict:
    n = len(names)
    x = allocation.allocate(rule, cost, n, contracts, kwh)
    report = games.stability(x, cost, n, rel_tol, abs_tol)
    e = games.excess(x, cost, n)
    singles = np.array([1 << i for i in range(n)])
    standalone = cost[singles]
    material = []
    for mask, value in report.blocking:
        group = " + ".join(names[i] for i in games.members(mask, n))
        material.append((group, float(value), float(max(abs_tol, rel_tol * abs(cost[mask])))))
    return Verdict(method=method, hub_cost=float(cost[-1]), standalone=standalone, allocated=x,
                   excess=e[singles], cutoff=np.maximum(abs_tol, rel_tol * np.abs(standalone)),
                   eps=float(games.least_core_value(cost, n)), stable=report.stable, material=material)


def attribution(screening: np.ndarray, continuous: np.ndarray, member: int, names: list[str],
                contracts: np.ndarray, kwh: np.ndarray, rel_tol: float, abs_tol: float) -> dict:
    """How much of the change comes from repricing one member's standalone option alone."""
    swap = screening.copy()
    swap[1 << member] = continuous[1 << member]
    only = verdict("screening with one repriced standalone cost", swap, names, contracts, kwh, rel_tol, abs_tol)
    return {
        "standalone_screening": float(screening[1 << member]),
        "standalone_continuous": float(continuous[1 << member]),
        "standalone_change": float(continuous[1 << member] - screening[1 << member]),
        "eps_only_member_repriced": only.eps,
        "passes_only_member_repriced": only.stable,
    }


def ties(v: Verdict, names: list[str], tol: float = 0.01) -> list[str]:
    """Members whose excess equals the largest member excess (least-core equalization)."""
    top = float(v.excess.max())
    return [names[i] for i in range(len(names)) if abs(v.excess[i] - top) <= tol]

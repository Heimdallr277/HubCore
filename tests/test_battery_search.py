"""Battery-size search of the fast evaluator: boundary optima and the upper-bound property.
"""

import numpy as np
import pytest

from hubcore import annual_reference, costs, lp_reference, profiles
from hubcore.costs import Assumptions, coalition_costs, member_matrix
from hubcore.park import EXAMPLE_PARK, SMALL_PARK, Firm, Park

# An office with workplace EV charging whose existing right (488 kW) is far below its peak.
BINDING = Park("binding right", (Firm("Gym", "evening", 2438, "process", 262),
                                 Firm("Office", "office", 1953, "ev_workplace_day", 536, 488.0)))
OFFICE = 2   # bitmask of the office alone


def _lp(park, cc, mask, a):
    load = member_matrix(park.n)[mask] @ park.loads()
    return annual_reference.coalition_cost(load, cc.capacity_right_kw[mask], a)


@pytest.mark.parametrize("a", [Assumptions(), Assumptions(unserved_cost=1.0, battery_cost_kwh=150.0)],
                         ids=["default", "slider-limits"])
def test_binding_right_gets_a_tight_upper_bound(a):
    cc = coalition_costs(BINDING.loads(), BINDING.contracts(), a)
    fast = cc.cost[OFFICE]
    lp = _lp(BINDING, cc, OFFICE, a)
    assert fast >= lp * (1 - 1e-9)                    # still a feasible upper bound
    assert (fast - lp) / lp < 0.04                    # was 5.7% (default) and 48.6% (slider limits)
    assert cc.battery_kwh[OFFICE] > 13.4 * BINDING.loads()[1].mean()   # beyond the old search range
    assert not cc.battery_at_search_limit.any()


def test_without_extension_the_boundary_is_flagged(monkeypatch):
    monkeypatch.setattr(costs, "BATTERY_EXTENSION_PASSES", 0)
    a = Assumptions()
    cc = coalition_costs(BINDING.loads(), BINDING.contracts(), a)
    assert cc.battery_at_search_limit[OFFICE]
    # 8 h grid end, then three winning upward refinements (+33%, +16.5%, +8.25%)
    hours = 8 * 1.33 * 1.165 * 1.0825
    assert cc.battery_kwh[OFFICE] == pytest.approx(hours * BINDING.loads()[1].mean(), rel=1e-12)
    lp = _lp(BINDING, cc, OFFICE, a)
    assert (cc.cost[OFFICE] - lp) / lp > 0.05


def test_presets_never_reach_the_search_limit():
    """No preset coalition has a boundary optimum, so the extension leaves the bundled results unchanged."""
    for park in (EXAMPLE_PARK, SMALL_PARK):
        cc = coalition_costs(park.loads(), park.contracts(), Assumptions())
        assert not cc.battery_at_search_limit.any()


def test_rows_are_evaluated_independently():
    """Selected masks give exactly the values they have in the full evaluation (row-wise search)."""
    loads, contracts, a = EXAMPLE_PARK.loads(), EXAMPLE_PARK.contracts(), Assumptions()
    full = coalition_costs(loads, contracts, a)
    part = coalition_costs(loads, contracts, a, np.array([3, 21, 38, 63]))
    for name in ("cost", "contract_kw", "battery_kwh", "unserved_kwh"):
        rows = [3, 21, 38, 63]
        np.testing.assert_array_equal(getattr(part, name)[rows], getattr(full, name)[rows])
    assert part.battery_at_search_limit.dtype == bool and part.battery_at_search_limit.shape == (64,)


@pytest.mark.slow
def test_random_parks_upper_bound_and_storage_free_exactness():
    """Seeded random parks within the app's input ranges, including tightened contracts."""
    rng = np.random.default_rng(20261122)
    archetypes, addons = list(profiles.ARCHETYPES), list(profiles.ADDONS)
    gaps = []
    for _ in range(8):
        firms = []
        for i in range(int(rng.integers(2, 4))):
            addon = addons[int(rng.integers(len(addons)))]
            kind = archetypes[int(rng.integers(len(archetypes)))]
            firm = Firm(f"F{i}", kind, float(rng.uniform(100, 2500)), addon,
                        float(rng.uniform(0, 800)) if addon != "none" else 0.0)
            if rng.random() < 0.5:
                firm = Firm(firm.name, firm.archetype, firm.annual_mwh, firm.addon, firm.addon_mwh,
                            float(round(firm.contract() * rng.uniform(0.3, 1.0))))
            firms.append(firm)
        park = Park("random", tuple(firms))
        a = Assumptions(tariff_contract=float(rng.uniform(1.5, 4.0)),
                        tariff_peak=float(rng.uniform(2.0, 6.5)),
                        battery_cost_kwh=float(rng.uniform(150, 400)),
                        unserved_cost=float(rng.uniform(0.15, 1.0)),
                        group_capacity_factor=float(rng.uniform(0.8, 1.0)))
        loads = park.loads(rng.uniform(0, 2, park.n))
        cc = coalition_costs(loads, park.contracts(), a)
        storage_free = coalition_costs(loads, park.contracts(), a.updated(battery_hours_grid=(0.0,)))
        M = member_matrix(park.n)
        for mask in range(1, 1 << park.n):
            load = M[mask] @ loads
            fast = cc.cost[mask] - cc.components["c_overhead"][mask]
            lp = annual_reference.coalition_cost(load, cc.capacity_right_kw[mask], a)
            assert fast >= lp * (1 - 1e-9) - 1e-6
            gaps.append((fast - lp) / lp)
            exact = lp_reference.solve(load, cc.capacity_right_kw[mask], a, battery="none").cost
            no_battery = storage_free.cost[mask] - storage_free.components["c_overhead"][mask]
            assert no_battery == pytest.approx(exact, rel=1e-9, abs=1e-6)
    assert max(gaps) < 0.06

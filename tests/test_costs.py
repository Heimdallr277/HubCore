"""Verification of the coalition cost model."""

import numpy as np
import pytest

from hubcore import games, lp_reference, profiles
from hubcore.costs import Assumptions, _month_starts, coalition_costs, daily_levels, member_matrix
from hubcore.park import EXAMPLE_PARK, SMALL_PARK, Firm, Park, ParkError, validate

# These two theorem tests assume all demand is served. Set the penalty above the
# annual contract and peak price of preserving any individual peak-hour kWh.
PURE = dict(unserved_cost=10_000., hub_fixed=0.0, hub_per_member=0.0,
            battery_hours_grid=(0.0,), battery_refine_rounds=0)


def closed_form_capacity_cost(load, a):
    month = np.repeat(profiles.month_of_day(), 24)
    monthly = [load[month == m].max() for m in range(12)]
    return 12 * a.tariff_contract * load.max() + a.tariff_peak * sum(monthly)


def test_no_battery_no_cap_matches_closed_form():
    a = Assumptions(**PURE)
    loads = EXAMPLE_PARK.loads()
    big = np.full(EXAMPLE_PARK.n, 1e6)            # capacity rights never bind
    cc = coalition_costs(loads, big, a)
    M = member_matrix(EXAMPLE_PARK.n)
    for mask in (1, 2, 5, 22, cc.grand):
        expected = closed_form_capacity_cost(M[mask] @ loads, a)
        assert cc.cost[mask] == pytest.approx(expected, rel=1e-9)


def test_coincident_peak_allocation_is_in_core():
    """Theorem: in a pure peak-capacity game (cost = sum of max-of-additive
    terms), charging each firm its load at the grand coalition's peak hours
    is a core allocation."""
    a = Assumptions(**PURE)
    park = EXAMPLE_PARK
    loads = park.loads()
    cc = coalition_costs(loads, np.full(park.n, 1e6), a)
    total = loads.sum(axis=0)
    month = np.repeat(profiles.month_of_day(), 24)
    t_star = int(np.argmax(total))
    x = 12 * a.tariff_contract * loads[:, t_star]
    for m in range(12):
        idx = np.flatnonzero(month == m)
        tm = idx[np.argmax(total[idx])]
        x = x + a.tariff_peak * loads[:, tm]
    assert x.sum() == pytest.approx(cc.cost[cc.grand], rel=1e-9)
    assert games.excess(x, cc.cost, park.n)[1:].max() <= 1e-6 * cc.cost[cc.grand]


def test_pooling_is_subadditive_without_overhead():
    a = Assumptions(hub_fixed=0.0, hub_per_member=0.0)
    cc = coalition_costs(EXAMPLE_PARK.loads(), EXAMPLE_PARK.contracts(), a)
    n = EXAMPLE_PARK.n
    rng = np.random.default_rng(0)
    for _ in range(40):
        s, t = rng.integers(1, 1 << n, size=2)
        t = t & ~s
        if t == 0:
            continue
        # tolerance: grid search over contract/battery sizes is approximate
        assert cc.cost[s | t] <= cc.cost[s] + cc.cost[t] + 0.005 * (cc.cost[s] + cc.cost[t])


def test_costs_increase_with_tariff():
    loads, contracts = SMALL_PARK.loads(), SMALL_PARK.contracts()
    low = coalition_costs(loads, contracts, Assumptions(tariff_contract=2.0)).cost
    high = coalition_costs(loads, contracts, Assumptions(tariff_contract=3.0)).cost
    assert np.all(high[1:] >= low[1:] - 1e-6)


def test_daily_level_pruning_is_exact_where_it_matters():
    loads = EXAMPLE_PARK.loads()
    M = member_matrix(EXAMPLE_PARK.n)
    X = (M[[3, 21, 63]] @ loads).reshape(3, profiles.DAYS, 24)
    energy = 1.5 * X.mean(axis=(1, 2))
    power = energy / 2.0
    starts = _month_starts()
    pruned, _ = daily_levels(X, energy, power, 0.9, top_k=46, month_starts=starts)
    full, _ = daily_levels(X, energy, power, 0.9, top_k=profiles.DAYS, month_starts=starts)
    np.testing.assert_allclose(np.maximum.reduceat(pruned, starts, axis=1),
                               np.maximum.reduceat(full, starts, axis=1), rtol=1e-9)
    np.testing.assert_allclose(-np.sort(-pruned, axis=1)[:, :46], -np.sort(-full, axis=1)[:, :46], rtol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("mask", [1, 2, 8, 3, 38, 63])
def test_fast_evaluator_against_exact_lp(mask):
    """The fast evaluator is a conservative bound of the exact LP.
    """
    a = Assumptions()
    park = EXAMPLE_PARK
    loads = park.loads()
    cc = coalition_costs(loads, park.contracts(), a)
    X = member_matrix(park.n)[mask] @ loads
    fast = cc.cost[mask] - cc.components["c_overhead"][mask]
    same_e = lp_reference.solve(X, cc.capacity_right_kw[mask], a, battery=cc.battery_kwh[mask])
    best = lp_reference.solve(X, cc.capacity_right_kw[mask], a)
    assert fast >= same_e.cost * (1 - 1e-6)
    assert (fast - same_e.cost) / same_e.cost <= 0.02
    assert (fast - best.cost) / best.cost <= 0.03


def test_invalid_inputs_raise_readable_errors():
    with pytest.raises(ValueError, match="battery_rte"):
        Assumptions(battery_rte=1.5).validate()
    with pytest.raises(ParkError, match="between"):
        validate(Park("one", (Firm("A", "office", 100),)))
    with pytest.raises(ParkError, match="unique"):
        validate(Park("dup", (Firm("A", "office", 100), Firm("A", "office", 100))))
    with pytest.raises(ParkError, match="annual use"):
        validate(Park("neg", (Firm("A", "office", 0), Firm("B", "office", 100))))


def test_profile_library_integrates_to_one_mwh():
    lib = profiles.library()
    np.testing.assert_allclose(lib.sum().to_numpy(), 1000.0, rtol=1e-4)
    assert len(profiles.month_of_day()) == profiles.DAYS

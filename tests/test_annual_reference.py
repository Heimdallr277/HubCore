"""Equivalence of the independently formulated weighted-day LP and the chronological benchmark."""
import numpy as np
import pytest

from hubcore import annual_reference, lp_reference
from hubcore.costs import Assumptions


@pytest.mark.parametrize("price, fixed, cap", [(0.0, 250.0, 10), (0.45, 0.0, 17), (9.0, 1.0, 12)])
def test_occurrences_preserve_the_annual_objective(price, fixed, cap):
    a = Assumptions(tariff_contract=0.02, tariff_peak=0.6, unserved_cost=price,
                    battery_cost_kwh=0.1, battery_fixed=fixed)
    first = np.tile([3.0, 4.0, 34.0], 8)
    second = np.roll(first, 2)
    load = np.concatenate([first, second, second, first, first, second, first])
    months = np.array([0, 0, 0, 1, 1, 1, 1])
    full = lp_reference.solve(load, cap, a, month_of_day=months)
    assert annual_reference.coalition_cost(load, cap, a, months) == pytest.approx(full.cost, abs=1e-7)


def test_linked_and_independent_days_are_not_interchangeable():
    a = Assumptions(tariff_contract=0.0, tariff_peak=0.0, unserved_cost=10., battery_fixed=0.,
                    battery_cost_kwh=0.001, battery_rte=1.)
    demand = np.r_[np.full(12, 20.), np.zeros(24), np.full(12, 20.)]
    months = np.zeros(2, dtype=int)
    compact = annual_reference.coalition_cost(demand, 10., a, months)
    full = lp_reference.solve(demand, 10., a, month_of_day=months)
    assert compact == pytest.approx(full.cost, abs=1e-7)
    assert compact < lp_reference.solve(demand, 10., a, month_of_day=months, linked_midnights=True).cost


def test_custom_future_and_fixed_signing_use_their_respective_settings():
    from hubcore import robustness, uncertainty
    from hubcore.park import SMALL_PARK
    from ui import state

    park = SMALL_PARK
    base = Assumptions(hub_fixed=39_000.0)
    ens = robustness.run_ensemble(park, base, uncertainty.default_space(park, base), 2, 91)
    keep = (0, 2)
    future, signed = state.reference_future(ens, keep, 0, fixed=True)
    a, growth = uncertainty.to_future(ens.samples.iloc[0], base, park.n)
    for cost, settings, g in [(future, a, growth), (signed, base, np.ones(park.n))]:
        load = park.loads(g)[list(keep)].sum(axis=0)
        cap = park.contracts()[list(keep)].sum() * settings.group_capacity_factor
        independent = lp_reference.solve(load, cap, settings)
        assert cost[-1] == pytest.approx(
            independent.cost + settings.hub_fixed + len(keep) * settings.hub_per_member, abs=1e-5)

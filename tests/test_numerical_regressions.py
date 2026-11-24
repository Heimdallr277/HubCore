"""Analytic, dispatch and allocation checks for known numerical edge cases."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from hubcore import allocation, games, lp_reference, prim, robustness, uncertainty
from hubcore.costs import Assumptions, _cyclic_delivery, _unserved, coalition_costs
from hubcore.park import SMALL_PARK


def daily_flow_lp(load, ceiling, energy, power, eta):
    """Independent 72-variable daily LP. Positive headroom and excess are disjoint."""
    eye = np.eye(24)
    delta = eye - np.roll(eye, 1, axis=0)
    eq = np.hstack([-eta * eye, eye, delta])
    cmax = np.minimum(np.maximum(ceiling - load, 0), power)
    dmax = np.minimum(np.maximum(load - ceiling, 0), power)
    caps = np.r_[cmax, dmax, np.full(24, energy)]
    result = linprog(
        np.r_[np.zeros(24), -np.ones(24), np.zeros(24)],
        A_eq=eq,
        b_eq=np.zeros(24),
        bounds=list(zip(np.zeros(72), caps)),
        method="highs",
    )
    assert result.success
    return -result.fun


def test_multiple_cycles_and_daily_flow_oracle():
    rng = np.random.default_rng(804)
    loads = np.vstack([np.tile([0.0, 20.0], 12), rng.uniform(0, 30, (24, 24))])
    energy = np.r_[5.0, rng.uniform(1, 50, 24)]
    power = energy / 2
    ceiling = 10.0
    expected = [daily_flow_lp(x, ceiling, e, b, 0.9) for x, e, b in zip(loads, energy, power)]
    actual = _cyclic_delivery(loads - ceiling, energy, power, 0.9)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert actual[0] > 5 * energy[0]
    shortfall = _unserved(loads[:, None, :], np.full((len(loads), 1), ceiling), energy, power, 0.9)[:, 0]
    np.testing.assert_allclose(shortfall, np.maximum(loads - ceiling, 0).sum(axis=1) - expected, atol=1e-8)


@pytest.mark.parametrize("penalty", [0.0, 0.40, 100.0])
def test_no_storage_economic_peak_shaving_against_annual_lp(penalty):
    x = np.full(8760, 100.0)
    x[: 24 * 31] = 200.0
    x[24 * 34] = 190.0
    a = Assumptions(unserved_cost=penalty, battery_hours_grid=(0.0,))
    c = coalition_costs(x[None, :], np.array([200.0]), a)
    direct = lp_reference.solve(x, 200.0, a, battery="none")
    assert c.cost[1] == pytest.approx(direct.cost, abs=1e-6)
    if penalty == 0.4:
        assert c.cost[1] == pytest.approx(10173.0)


def test_fixed_contract_bounds_and_input_diagnostics():
    with pytest.raises(ValueError, match="capacity right"):
        lp_reference.solve(
            np.full(24, 10.0), 1.0, Assumptions(), battery="none", contract=10.0, month_of_day=[0]
        )
    with pytest.raises(ValueError, match="month"):
        lp_reference.solve(np.ones(24), 1.0, Assumptions(), battery="none", month_of_day=[12])
    with pytest.raises(ValueError, match="finite"):
        coalition_costs(np.full((2, 8760), np.nan), [10, 10], Assumptions())
    with pytest.raises(games.GameError, match="finite"):
        games.optimal_partition(np.array([0, 1, np.inf, 3.0]), 2)
    with pytest.raises(games.GameError, match="sum"):
        games.stability(np.zeros(2), np.array([0, 10, 12, 15]), 2)


def test_reference_schedule_balance_peaks_and_annual_cost():
    x = np.tile([5.0, 25.0, 10.0, 40.0, 8.0, 15.0], 8)
    a = Assumptions(tariff_peak=0.6, unserved_cost=10.0)
    r = lp_reference.solve(x, 22.0, a, battery=12.0, contract=22.0, month_of_day=[0, 1])
    soc = r.soc.reshape(2, 24)
    np.testing.assert_allclose(r.imports + r.unserved + r.discharge - r.charge, x, atol=1e-8)
    np.testing.assert_allclose(
        soc - np.roll(soc, 1, axis=1), (a.battery_rte * r.charge - r.discharge).reshape(2, 24), atol=1e-8
    )
    np.testing.assert_allclose(r.monthly_peak_kw[:2], r.imports.reshape(2, 24).max(axis=1), atol=1e-7)
    assert r.imports.max() <= 22.000001 and soc.min() >= -1e-7 and soc.max() <= 12.000001
    assert max(r.charge.max(), r.discharge.max()) <= 6.000001
    expected = (
        12 * a.tariff_contract * 22
        + a.tariff_peak * r.monthly_peak_kw.sum()
        + a.unserved_cost * r.unserved.sum()
    )
    assert r.cost == pytest.approx(expected + a.crf * (a.battery_cost_kwh * 12 + a.battery_fixed))


def test_signed_leastcore_and_proposal_repair_have_different_targets():
    c = np.array([0.0, 1, 2, 2, 4, 4, 4, 4])
    phi = games.shapley(c, 3)
    x = allocation.allocate("leastcore", c, 3, np.ones(3), np.ones(3))
    repair, _ = games.closest_in_least_core(c, 3, phi)
    assert games.least_core_value(c, 3) == pytest.approx(-0.5)
    assert games.excess(x, c, 3)[1:-1].max() == pytest.approx(-0.5)
    np.testing.assert_allclose(repair, phi)
    assert games.excess(repair, c, 3)[1:-1].max() == pytest.approx(-1 / 3)


def test_savings_shapley_independent_permutation_definition():
    rng = np.random.default_rng(95)
    n = 4
    c = np.r_[0.0, rng.uniform(20, 90, 15)]
    single = c[1 << np.arange(n)]
    v = np.array([sum(single[i] for i in range(n) if (m >> i) & 1) - cost for m, cost in enumerate(c)])
    values = []
    for order in itertools.permutations(range(n)):
        marg = np.zeros(n)
        m = 0
        for i in order:
            nxt = m | 1 << i
            marg[i] = v[nxt] - v[m]
            m = nxt
        values.append(marg)
    np.testing.assert_allclose(single - games.shapley(c, n), np.mean(values, axis=0))


def test_signing_base_lhs_strata_and_signed_regret():
    p = SMALL_PARK
    a = Assumptions(hub_fixed=39_000.0, battery_hours_grid=(0.0,))
    space = uncertainty.default_space(p, a)
    samples = uncertainty.sample(space, 11, seed=390)
    for u in space:
        idx = np.floor((samples[u.key] - u.low) * 11 / (u.high - u.low))
        np.testing.assert_array_equal(np.sort(idx), np.arange(11))
    ens = robustness.run_ensemble(p, a, space, 2, 123)
    np.testing.assert_allclose(ens.nominal_costs, coalition_costs(p.loads(), p.contracts(), a).cost)
    df = robustness.evaluate_policies(ens)
    for k, c in enumerate(ens.costs):
        signed = games.least_core_value(c, p.n)
        row = df[(df.future == k) & (df.rule == "prorata") & (df.clause == "fixed")].iloc[0]
        allocation0 = allocation.allocate("prorata", ens.nominal_costs, p.n, p.contracts(), ens.nominal_kwh)
        x = allocation0 * c[-1] / ens.nominal_costs[-1]
        expected = games.excess(x, c, p.n)[1:-1].max() - signed
        assert row.regret == pytest.approx(expected, abs=1e-6)


def test_prim_scores_held_out_planted_rule_and_constant_features():
    rng = np.random.default_rng(238)

    def data(n):
        X = pd.DataFrame(rng.random((n, 4)), columns=list("abcd"))
        return X, ((X.a > 0.64) & (X.c < 0.51)).to_numpy()

    train, y = data(1600)
    valid, y2 = data(1100)
    traj = prim.prim(train, y)
    box, _ = prim.explain(traj[prim.suggest(traj)], train, y)
    assert set(box.restricted) == {"a", "c"}
    bounds = prim.open_limits(box, train)
    inside = np.ones(len(valid), dtype=bool)
    for k in box.restricted:
        inside &= valid[k].between(*bounds[k]).to_numpy()
    cov, den, mass, count = prim.box_stats(valid, y2, bounds, box.restricted)
    assert (cov, den, mass, count) == pytest.approx(
        (y2[inside].sum() / y2.sum(), y2[inside].mean(), inside.mean(), inside.sum())
    )
    assert cov > 0.85 and den > 0.85
    assert len(prim.prim(pd.DataFrame({"flat": np.ones(20)}), np.arange(20) % 2 == 0)) == 1


def test_shared_midnight_state_changes_value_on_opposite_peak_days():
    # Each day independently needs a different initial SOC to cover its peak.
    x = np.r_[np.tile(20.0, 12), np.zeros(12), np.zeros(12), np.tile(20.0, 12)]
    a = Assumptions(tariff_peak=0.0, unserved_cost=1.0, battery_rte=1.0, battery_duration_h=2.0)
    args = dict(battery=120.0, contract=10.0, month_of_day=np.zeros(2, dtype=int))
    independent = lp_reference.solve(x, 10.0, a, **args)
    linked = lp_reference.solve(x, 10.0, a, linked_midnights=True, **args)
    assert independent.unserved_kwh == pytest.approx(0.0)
    assert linked.unserved_kwh == pytest.approx(120.0)
    assert linked.cost - independent.cost == pytest.approx(120.0)
    np.testing.assert_allclose(
        linked.soc - np.roll(linked.soc, 1), linked.charge - linked.discharge, atol=1e-8
    )


def test_broken_preset_file_is_visible(tmp_path):
    path = tmp_path / "damaged.npz"
    path.write_bytes(b"incomplete")
    a = Assumptions()
    with pytest.raises(robustness.EnsembleFileError, match="damaged.npz"):
        robustness.load(path, SMALL_PARK, a, uncertainty.default_space(SMALL_PARK), 200, 42)

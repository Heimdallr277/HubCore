import numpy as np
import pandas as pd
import pytest

from hubcore import robustness as R
from hubcore import uncertainty as U
from hubcore.costs import Assumptions
from hubcore.park import SMALL_PARK


def test_sampling_is_reproducible_and_within_bounds():
    space = U.default_space(SMALL_PARK)
    s1, s2 = U.sample(space, 50, seed=3), U.sample(space, 50, seed=3)
    pd.testing.assert_frame_equal(s1, s2)
    for u in space:
        assert s1[u.key].between(u.low, u.high).all()
    assert not s1.equals(U.sample(space, 50, seed=4))


def test_ensemble_is_deterministic():
    space = U.default_space(SMALL_PARK)
    a = Assumptions()
    e1 = R.run_ensemble(SMALL_PARK, a, space, n_futures=3, seed=11)
    e2 = R.run_ensemble(SMALL_PARK, a, space, n_futures=3, seed=11)
    np.testing.assert_array_equal(e1.costs, e2.costs)


def test_policies_invariants():
    space = U.default_space(SMALL_PARK)
    a = Assumptions()
    ens = R.run_ensemble(SMALL_PARK, a, space, n_futures=4, seed=5)
    df = R.evaluate_policies(ens)
    assert len(df) == 4 * 6 * 2
    lc = df[(df.rule == "leastcore") & (df.clause == "adaptive")]
    assert (lc.regret < 1e-6 * lc.hub_cost.abs().max() + 1e-6).all()
    summary = R.summarise(df)
    assert {"stable_share", "all_better_off_share", "mean_regret"} <= set(summary.columns)
    assert summary.stable_share.between(0, 1).all()


def test_fixed_equals_adaptive_in_the_nominal_future():
    space = U.default_space(SMALL_PARK)
    a = Assumptions()
    ens = R.run_ensemble(SMALL_PARK, a, space, n_futures=2, seed=5)
    # overwrite futures with the nominal one
    ens.costs[:] = ens.nominal_costs
    ens.kwh[:] = ens.nominal_kwh
    df = R.evaluate_policies(ens)
    fixed = df[df.clause == "fixed"].set_index(["future", "rule"]).max_excess
    adaptive = df[df.clause == "adaptive"].set_index(["future", "rule"]).max_excess
    np.testing.assert_allclose(fixed.sort_index(), adaptive.sort_index(), atol=1e-6)


def test_saved_ensemble_round_trip(tmp_path):
    space = U.default_space(SMALL_PARK)
    a = Assumptions()
    ens = R.run_ensemble(SMALL_PARK, a, space, n_futures=2, seed=9)
    path = tmp_path / "e.npz"
    R.save(ens, path)
    back = R.load(path, SMALL_PARK, a, space, 2, 9)
    assert back is not None
    np.testing.assert_array_equal(back.costs, ens.costs)
    assert R.load(path, SMALL_PARK, a.updated(tariff_peak=9.0), space, 2, 9) is None


def test_policy_allocation_matches_the_ensemble_evaluation():
    """The single policy implementation used by the app reproduces evaluate_policies exactly."""
    from hubcore import games

    space = U.default_space(SMALL_PARK)
    ens = R.run_ensemble(SMALL_PARK, Assumptions(), space, n_futures=3, seed=21)
    df = R.evaluate_policies(ens).set_index(["future", "rule", "clause"])
    contracts = SMALL_PARK.contracts()
    for k in range(ens.n_futures):
        for rule in ("capacity", "prorata", "shapley", "leastcore"):
            for clause in R.CLAUSES:
                x = R.policy_allocation(rule, clause, ens.costs[k], contracts, ens.kwh[k],
                                        ens.nominal_costs, ens.nominal_kwh)
                assert x.sum() == pytest.approx(ens.costs[k][-1])
                st = games.stability(x, ens.costs[k], SMALL_PARK.n)
                assert st.max_excess == df.loc[(k, rule, clause), "max_excess"]
    with pytest.raises(KeyError, match="clause"):
        R.policy_allocation("shapley", "monthly", ens.costs[0], contracts, ens.kwh[0],
                            ens.nominal_costs, ens.nominal_kwh)


def test_live_ensembles_record_battery_search_limits():
    ens = R.run_ensemble(SMALL_PARK, Assumptions(), U.default_space(SMALL_PARK), n_futures=2, seed=3)
    assert ens.battery_limit_groups.shape == (2,) and (ens.battery_limit_groups >= 0).all()

"""Rebuild the seed-42 case evidence and score its selected PRIM rules on fresh futures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from precompute import path_for  # noqa: E402

from hubcore import allocation, games, prim  # noqa: E402
from hubcore import robustness as R  # noqa: E402
from hubcore import uncertainty as U  # noqa: E402
from hubcore.costs import Assumptions, park_costs  # noqa: E402
from hubcore.park import EXAMPLE_PARK as PARK  # noqa: E402


def result(ens, holdout, members):
    df = R.evaluate_policies(ens, members=members)
    summary = R.summarise(df)
    subset = df[(df.rule == "leastcore") & (df.clause == "adaptive")]
    failure = ~subset.stable.to_numpy()
    keys = [u.key for u in ens.space if not u.is_growth or u.firm_index in members]
    X = ens.samples[keys]
    trajectory = prim.prim(X, failure)
    box, _ = prim.explain(trajectory[prim.suggest(trajectory)], X, failure)
    test = R.evaluate_policies(holdout, members=members, rules=["leastcore"])
    test_fail = ~test[test.clause == "adaptive"].stable.to_numpy()

    limits = prim.open_limits(box, X)
    coverage, density, mass, count = prim.box_stats(holdout.samples[keys], test_fail, limits, box.restricted)
    return dict(
        members=[PARK.names[i] for i in members],
        viable=int(subset.hub_viable.sum()),
        exact_core_exists=int(subset.core_exists.sum()),
        policies=summary.to_dict("records"),
        adaptive_material_pass=int(subset.stable.sum()),
        train_failures=int(failure.sum()),
        box={k: [None if not np.isfinite(bound) else bound for bound in limits[k]] for k in box.restricted},
        training=dict(coverage=box.coverage, density=box.density, mass=box.mass, inside=box.n_points),
        holdout=dict(
            seed=holdout.seed,
            n=len(test_fail),
            total_failures=int(test_fail.sum()),
            coverage=coverage,
            density=density,
            mass=mass,
            inside=count,
        ),
    )


def main():
    a = Assumptions()
    space = U.default_space(PARK)
    ens = R.load(path_for(PARK.name), PARK, a, space, 200, 42)
    if ens is None:
        raise SystemExit("Regenerate the preset ensembles with scripts/precompute.py first.")
    holdout = R.run_ensemble(PARK, a, space, 100, 73, workers=3)
    cc = park_costs(PARK, a)
    alloc = allocation.all_rules(cc.cost, PARK.n, PARK.contracts(), PARK.loads().sum(axis=1))
    report = {
        "ensemble": {"seed": 42, "n": 200, "fingerprint": R.fingerprint(PARK, a, space, 200, 42)},
        "nominal": {
            "hub": cc.cost[-1],
            "standalone": cc.standalone().sum(),
            "saving": cc.savings(),
            "right_kw": cc.capacity_right_kw[-1],
            "contract_kw": cc.contract_kw[-1],
            "battery_kwh": cc.battery_kwh[-1],
            "unserved_kwh": cc.unserved_kwh[-1],
            "cost_component_changes": {
                k: v[-1] - sum(v[1 << i] for i in range(PARK.n)) for k, v in cc.components.items()
            },
            "eps_star": games.least_core_value(cc.cost, PARK.n),
            "rules": {k: games.stability(x, cc.cost, PARK.n).max_excess for k, x in alloc.items()},
        },
        "six_firms": result(ens, holdout, list(range(PARK.n))),
        "without_cold_store": result(ens, holdout, [0, 1, 3, 4, 5]),
    }
    (ROOT / "data/case_results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Compare all nominal coalitions and selected uncertain futures with hourly HiGHS LPs."""

from __future__ import annotations

import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from hubcore import allocation, games, lp_reference, uncertainty  # noqa: E402
from hubcore.costs import Assumptions, coalition_costs, member_matrix  # noqa: E402
from hubcore.park import EXAMPLE_PARK, SMALL_PARK  # noqa: E402


def compare(job):
    meta, x, cap, energy, fast, a = job
    same = lp_reference.solve(x, cap, a, battery=energy)
    best = lp_reference.solve(x, cap, a)
    err = float(np.max(np.abs(best.imports - best.charge + best.discharge + best.unserved - x)))
    soc = best.soc.reshape(-1, 24)
    soc_error = np.abs(
        soc - np.roll(soc, 1, axis=1) - (a.battery_rte * best.charge - best.discharge).reshape(-1, 24)
    )
    if (
        err > 1e-5
        or soc_error.max() > 1e-5
        or best.imports.max() > cap + 1e-5
        or best.cost > same.cost + 1e-4
        or same.cost > fast + 1e-4
    ):
        raise AssertionError(f"Schedule or cost bounds failed on {meta}")
    return {
        **meta,
        "fast_eur": fast,
        "lp_same_battery_eur": same.cost,
        "lp_optimal_eur": best.cost,
        "gap_same_battery_pct": 100 * (fast / same.cost - 1),
        "gap_optimal_pct": 100 * (fast / best.cost - 1),
        "hourly_balance_error_kw": err,
        "daily_soc_error_kwh": float(soc_error.max()),
    }


def main():
    report = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
        }
    }
    jobs, games_data = [], {}
    for park in [EXAMPLE_PARK, SMALL_PARK]:
        sample = uncertainty.sample(uncertainty.default_space(park), 200, 42)
        for sid in [-1, 0, 41, 99, 199] if park == EXAMPLE_PARK else [-1]:
            a, g = (
                (Assumptions(), None)
                if sid < 0
                else uncertainty.to_future(sample.iloc[sid], Assumptions(), park.n)
            )
            loads = park.loads(g)
            masks = np.arange(1, 1 << park.n) if sid < 0 else [1, 2, 38, 63]
            t = time.perf_counter()
            cc = coalition_costs(loads, park.contracts(), a, masks)
            if sid < 0:
                games_data[park.name] = (cc, park)
                if park == EXAMPLE_PARK:
                    report["fast_evaluator_seconds_all_63_groups"] = round(time.perf_counter() - t, 3)
            for m in masks:
                X = member_matrix(park.n)[m] @ loads
                meta = {
                    "park": park.name,
                    "future": int(sid),
                    "mask": int(m),
                    "group": " + ".join(park.names[i] for i in games.members(int(m), park.n)),
                }
                jobs.append(
                    (
                        meta,
                        X,
                        cc.capacity_right_kw[m],
                        cc.battery_kwh[m],
                        cc.cost[m] - cc.components["c_overhead"][m],
                        a,
                    )
                )
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(compare, jobs))
    report["lp_comparison"] = rows
    for name, subset in [
        ("all", rows),
        ("nominal", [r for r in rows if r["future"] == -1]),
        ("stress", [r for r in rows if r["future"] != -1]),
    ]:
        report[name] = {
            "count": len(subset),
            **{
                key: max(r[key] for r in subset)
                for key in [
                    "gap_same_battery_pct",
                    "gap_optimal_pct",
                    "hourly_balance_error_kw",
                    "daily_soc_error_kwh",
                ]
            },
        }
    report["nominal_game_checks"] = {}
    for name, (cc, park) in games_data.items():
        exact = cc.components["c_overhead"].copy()
        for r in rows:
            if r["park"] == name and r["future"] == -1:
                exact[r["mask"]] += r["lp_optimal_eur"]
        entry = {}
        for kind, c in [("surrogate", cc.cost), ("lp", exact)]:
            x = allocation.all_rules(c, park.n, park.contracts(), park.loads().sum(axis=1))
            entry[kind] = {
                "hub_cost": c[-1],
                "hub_saving": sum(c[1 << i] for i in range(park.n)) - c[-1],
                "eps_star": games.least_core_value(c, park.n),
                "partition": games.optimal_partition(c, park.n)[0],
                "passes_materiality": {k: games.stability(v, c, park.n).stable for k, v in x.items()},
                "max_excess_by_rule": {k: games.stability(v, c, park.n).max_excess for k, v in x.items()},
            }
        cap = cc.capacity_right_kw[-1]
        load = park.loads().sum(axis=0)
        kwargs = dict(battery=cc.battery_kwh[-1], contract=cc.contract_kw[-1])
        independent = lp_reference.solve(load, cap, Assumptions(), **kwargs)
        linked = lp_reference.solve(load, cap, Assumptions(), linked_midnights=True, **kwargs)
        entry["fixed_design_chronology"] = {
            "daily_independent_eur": independent.cost,
            "linked_midnights_eur": linked.cost,
            "linked_minus_independent_eur": linked.cost - independent.cost,
        }
        report["nominal_game_checks"][name] = entry
    (ROOT / "data/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "lp_comparison"}, indent=2))


if __name__ == "__main__":
    main()

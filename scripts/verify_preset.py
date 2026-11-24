"""Reprice the six-firm preset and holdout with continuous, full-year coalition LPs."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from case_study import result as case_result  # noqa: E402
from precompute import path_for  # noqa: E402

from hubcore import allocation, annual_reference, games, prim  # noqa: E402
from hubcore import robustness as R  # noqa: E402
from hubcore import uncertainty as U  # noqa: E402
from hubcore.costs import Assumptions  # noqa: E402
from hubcore.park import EXAMPLE_PARK as PARK  # noqa: E402
from hubcore.park import SMALL_PARK  # noqa: E402

BASE = Assumptions()
SPACE = U.default_space(PARK, BASE)
TRAIN = (200, 42)
HOLDOUT = (100, 73)
DATA = ROOT / "data"
ARRAYS = DATA / "preset_lp_costs.npz"
REPORT = DATA / "preset_lp_checks.json"
MEMBERS = {"six_firms": list(range(6)), "without_cold_store": [0, 1, 3, 4, 5]}


def reprice(row):
    assumptions, growth = U.to_future(row, BASE, PARK.n)
    return annual_reference.price_game(PARK, assumptions, growth)


def validate_against_hourly_lp():
    verification = json.loads((DATA / "verification.json").read_text(encoding="utf-8"))
    errors = []
    parks = {p.name: p for p in [PARK, SMALL_PARK]}
    for row in verification["lp_comparison"]:
        park = parks[row["park"]]
        mask = row["mask"]
        idx = games.members(mask, park.n)
        a, growth = BASE, np.ones(park.n)
        if row["future"] >= 0:
            samples = U.sample(U.default_space(park, BASE), *TRAIN)
            a, growth = U.to_future(samples.iloc[row["future"]], BASE, park.n)
        cap = float(park.contracts()[idx].sum()) * (a.group_capacity_factor if len(idx) > 1 else 1.0)
        bill = annual_reference.coalition_cost(park.loads(growth)[idx].sum(axis=0), cap, a)
        errors.append(abs(bill - row["lp_optimal_eur"]))
    if max(errors) > 1e-4:
        raise AssertionError("Weighted-day reference disagrees with the full chronological LP")
    return {"count": len(errors), "max_absolute_difference_eur": max(errors)}


def cost_gaps(fast, exact):
    size = np.array([m.bit_count() for m in range(1, 1 << PARK.n)])
    fixed = fast.samples["hub_fixed"].to_numpy()[:, None]
    per_member = fast.samples["hub_per_member"].to_numpy()[:, None]
    overhead = (fixed + per_member * size) * (size > 1)
    difference = fast.costs[:, 1:] - exact.costs[:, 1:]
    if difference.min() < -1e-4:
        raise AssertionError("Screening cost is below the continuous optimum")
    gaps = 100 * difference / (exact.costs[:, 1:] - overhead)
    future, col = (int(i) for i in np.unravel_index(np.argmax(gaps), gaps.shape))
    return {
        "coalition_futures": int(gaps.size),
        "median_pct": float(np.median(gaps)), "p95_pct": float(np.quantile(gaps, .95)),
        "max_pct": float(gaps.max()), "above_2pct": int((gaps > 2).sum()),
        "min_fast_minus_reference_eur": float(difference.min()),
        "worst": {"seed": fast.seed, "future": future, "mask": col + 1,
                  "members": [PARK.names[i] for i in games.members(col + 1, PARK.n)],
                  "screening_before_overhead_eur": float(fast.costs[future, col+1] - overhead[future, col]),
                  "reference_before_overhead_eur": float(exact.costs[future, col+1] - overhead[future, col])},
    }


def decision_flags(ensemble, members):
    flags = R.evaluate_policies(ensemble, rules=["leastcore"], members=members)
    return flags[flags.clause == "adaptive"].set_index("future")


def membership_results(fast, exact, held_exact, members, fast_case):
    fast_flags = decision_flags(fast, members)
    exact_flags = decision_flags(exact, members)
    changed = {}
    for key in ["hub_viable", "core_exists", "stable"]:
        before, after = fast_flags[key], exact_flags[key]
        changed[key] = {"screening_pass_reference_fail": before.index[before & ~after].tolist(),
                        "screening_fail_reference_pass": before.index[~before & after].tolist()}
    ref_case = case_result(exact, held_exact, members)
    ref_case["best_fixed_pass"] = round(len(exact.costs) * max(
        row["stable_share"] for row in ref_case["policies"] if row["clause"] == "fixed"))
    del ref_case["policies"]

    limits = {key: (-np.inf if lo is None else lo, np.inf if hi is None else hi)
              for key, (lo, hi) in fast_case["box"].items()}
    hold_failed = ~decision_flags(held_exact, members).stable.to_numpy()
    cov, den, mass, inside = prim.box_stats(held_exact.samples, hold_failed, limits, list(limits))
    boundaries = {}
    margins = []
    for k, cost in enumerate(fast.costs):
        c = games.subgame(cost, PARK.n, members)
        x = allocation.allocate("leastcore", c, len(members), PARK.contracts()[members], fast.kwh[k, members])
        threshold = np.maximum(500.0, .01 * np.abs(c[1:-1]))
        margins.append(float(np.max(games.excess(x, c, len(members))[1:-1] - threshold)))
    for metric, series in [("core_exists", fast_flags.eps_star.to_numpy()), ("stable", np.asarray(margins))]:
        for passing in [True, False]:
            ids = np.flatnonzero(fast_flags[metric].to_numpy() == passing)
            if not ids.size:
                continue
            k = int(ids[np.argmin(abs(series[ids]))])
            boundaries[f"{metric}_{'pass' if passing else 'fail'}"] = {
                "future": k, "screening_margin_eur": float(series[k]),
                "reference_pass": bool(exact_flags[metric].loc[k]),
            }
    return {"reference": ref_case, "changed_future_ids": changed, "nearest_screening_cases": boundaries,
            "screening_rule_on_lp_holdout": {"seed": held_exact.seed, "n": len(hold_failed),
            "failures": int(hold_failed.sum()), "inside": inside, "failures_inside": int(round(den * inside)),
            "coverage": cov, "density": den, "mass": mass}}


def fingerprint(ensemble):
    return R.fingerprint(PARK, BASE, SPACE, ensemble.n_futures, ensemble.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Processes used to solve independent games")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    fast = R.load(path_for(PARK.name, *TRAIN), PARK, BASE, SPACE, *TRAIN)
    if fast is None:
        raise SystemExit("Run scripts/precompute.py to regenerate the seed-42 screening costs first.")
    print("Checking the reduced model against 86 unaggregated hourly LPs…", flush=True)
    comparisons = validate_against_hourly_lp()
    print("Rebuilding the 100-future screening holdout…", flush=True)
    held_fast = R.run_ensemble(PARK, BASE, SPACE, *HOLDOUT, workers=args.workers)
    with ProcessPoolExecutor(args.workers) as pool:
        def exact_game(ensemble, nominal):
            jobs = ensemble.samples.to_dict("records")
            costs = []
            for i, values in enumerate(pool.map(reprice, jobs), 1):
                costs.append(values)
                if i % 50 == 0:
                    print(f"Continuous LP, seed {ensemble.seed}: {i}/{len(jobs)} games", flush=True)
            return replace(ensemble, costs=np.asarray(costs), nominal_costs=nominal,
                           battery_limit_groups=None)

        nominal = annual_reference.price_game(PARK, BASE)
        exact = exact_game(fast, nominal)
        held_exact = exact_game(held_fast, nominal)

    signed_case = json.loads((DATA / "case_results.json").read_text(encoding="utf-8"))
    if signed_case["ensemble"]["fingerprint"] != fingerprint(fast):
        raise ValueError("Regenerate the matching case study before verifying its reported rules")
    report = {
        "scope": "Voorbeeldpark, all 63 nonempty groups. Percentage gaps exclude coordination overhead.",
        "training_design": {"n": fast.n_futures, "seed": fast.seed, "fingerprint": fingerprint(fast)},
        "holdout_design": {"n": held_fast.n_futures, "seed": held_fast.seed,
                           "fingerprint": fingerprint(held_fast)},
        "full_hourly_crosscheck": comparisons,
        "training_gaps": cost_gaps(fast, exact), "holdout_gaps": cost_gaps(held_fast, held_exact),
        "memberships": {key: membership_results(fast, exact, held_exact, idx, signed_case[key])
                        for key, idx in MEMBERS.items()},
    }
    # Write only completed files. The inputs, scope and pricing source are retained for every array.
    arrays_tmp = ARRAYS.with_suffix(".tmp.npz")
    np.savez_compressed(arrays_tmp, nominal=nominal, training=exact.costs, holdout=held_exact.costs,
                        training_design=fingerprint(fast), holdout_design=fingerprint(held_fast))
    arrays_tmp.replace(ARRAYS)
    report_tmp = REPORT.with_suffix(".tmp.json")
    report_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report_tmp.replace(REPORT)
    print(f"Wrote {ARRAYS.name} and {REPORT.name}")


if __name__ == "__main__":
    main()

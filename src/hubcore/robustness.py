"""Robustness analysis of cost-sharing deals across an ensemble of futures.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from . import allocation, games, profiles
from .costs import Assumptions, coalition_costs
from .park import Park
from .uncertainty import Uncertainty, sample, to_future

CLAUSES = {"fixed": "Fixed key (set at signing)", "adaptive": "Recomputed each year"}
CODE_VERSION = "0.2.3"  # Screening cost model version used in the experiment fingerprint


@dataclass
class Ensemble:
    park: Park
    base: Assumptions
    space: list[Uncertainty]
    samples: pd.DataFrame
    costs: np.ndarray            # (N, 2**n)
    kwh: np.ndarray              # (N, n)
    nominal_costs: np.ndarray    # (2**n,)
    nominal_kwh: np.ndarray      # (n,)
    seed: int
    battery_limit_groups: np.ndarray | None = None

    @property
    def n_futures(self) -> int:
        return self.costs.shape[0]


def _future_costs(args) -> tuple[np.ndarray, np.ndarray, int]:
    loads_base, loads_addon, contracts, a, growth = args
    loads = loads_base + growth[:, None] * loads_addon
    cc = coalition_costs(loads, contracts, a)
    return cc.cost, loads.sum(axis=1), int(cc.battery_at_search_limit.sum())


def preset_filename(park_name: str, n_futures: int, seed: int) -> str:
    """File name of a bundled ensemble, e.g. ``voorbeeldpark__example_N200_seed42.npz``."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in park_name).strip("_")
    return f"{slug}_N{n_futures}_seed{seed}.npz"


def signing_shares(rule: str, signing_cost: np.ndarray, contracts: np.ndarray,
                   signing_kwh: np.ndarray) -> np.ndarray:
    """Cost-sharing percentages (as fractions) frozen at signing for a fixed-key clause."""
    n = len(contracts)
    if signing_cost[(1 << n) - 1] <= 0:
        raise ValueError("A positive signing cost is needed for percentage-based fixed keys")
    return allocation.allocate(rule, signing_cost, n, contracts, signing_kwh) / signing_cost[(1 << n) - 1]


def policy_allocation(rule: str, clause: str, cost: np.ndarray, contracts: np.ndarray, kwh: np.ndarray,
                      signing_cost: np.ndarray, signing_kwh: np.ndarray) -> np.ndarray:
    """Cost shares (EUR/yr) of one realized future under a (rule, clause) policy.
    """
    if clause == "fixed":
        return signing_shares(rule, signing_cost, contracts, signing_kwh) * cost[-1]
    if clause == "adaptive":
        return allocation.allocate(rule, cost, len(contracts), contracts, kwh)
    raise KeyError(f"Unknown contract clause '{clause}'")


def fingerprint(park: Park, base: Assumptions, space: list[Uncertainty], n_futures: int, seed: int) -> str:
    payload = json.dumps({
        "version": CODE_VERSION, "profiles": profiles.data_hash(),
        "park": [asdict(f) for f in park.firms],
        "base": base.as_dict(),
        "space": [(s.key, s.low, s.high, s.base) for s in space],
        "n": n_futures, "seed": seed,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_ensemble(park: Park, base: Assumptions, space: list[Uncertainty], n_futures: int = 200,
                 seed: int = 42, progress: Callable[[float], None] | None = None,
                 workers: int = 1) -> Ensemble:
    """Evaluate all coalition costs in ``n_futures`` Latin-hypercube futures."""
    samples = sample(space, n_futures, seed)
    base_loads = np.vstack([f.base_load() for f in park.firms])
    addon_loads = np.vstack([f.addon_load() for f in park.firms])
    contracts = park.contracts()
    jobs = []
    for _, row in samples.iterrows():
        a, growth = to_future(row, base, park.n)
        jobs.append((base_loads, addon_loads, contracts, a, growth))
    # Fixed shares are signed at the actual Build settings. Economic ranges describe future designs.
    growth_base = {u.key: u.base for u in space if u.is_growth}
    a0, g0 = to_future(growth_base, base, park.n)
    nom_cost, nom_kwh, _ = _future_costs((base_loads, addon_loads, contracts, a0, g0))

    costs = np.zeros((n_futures, 1 << park.n))
    kwh = np.zeros((n_futures, park.n))
    limit = np.zeros(n_futures, dtype=int)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (c, e, b) in enumerate(ex.map(_future_costs, jobs, chunksize=4)):
                costs[k], kwh[k], limit[k] = c, e, b
                if progress:
                    progress((k + 1) / n_futures)
    else:
        for k, job in enumerate(jobs):
            costs[k], kwh[k], limit[k] = _future_costs(job)
            if progress:
                progress((k + 1) / n_futures)
    return Ensemble(park, base, list(space), samples, costs, kwh, nom_cost, nom_kwh, seed, limit)


def save(ens: Ensemble, path: Path) -> None:
    np.savez_compressed(
        path, costs=ens.costs, kwh=ens.kwh, nominal_costs=ens.nominal_costs,
        nominal_kwh=ens.nominal_kwh, samples=ens.samples.to_numpy(),
        columns=np.array([str(c) for c in ens.samples.columns], dtype=str), seed=ens.seed,
        fingerprint=fingerprint(ens.park, ens.base, ens.space, ens.n_futures, ens.seed),
    )


class EnsembleFileError(ValueError):
    """An existing preset file is unreadable or incomplete."""


def load(path: Path, park: Park, base: Assumptions, space: list[Uncertainty],
         n_futures: int, seed: int) -> Ensemble | None:
    """A different design is a cache miss. A broken file is a diagnostic."""
    if not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) != fingerprint(park, base, space, n_futures, seed):
                return None
            keys = [u.key for u in space]
            if list(z["columns"]) != keys or int(z["seed"]) != seed:
                raise ValueError("inconsistent design metadata")
            shapes = {"samples": (n_futures, len(keys)), "costs": (n_futures, 1 << park.n),
                      "kwh": (n_futures, park.n), "nominal_costs": (1 << park.n,), "nominal_kwh": (park.n,)}
            arrays = {k: z[k] for k in shapes}
            valid = all(arrays[k].shape == shape and np.isfinite(arrays[k]).all()
                        for k, shape in shapes.items())
            if not valid:
                raise ValueError("missing or invalid numerical values")
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
        raise EnsembleFileError(f"Cannot use {Path(path).name}: {exc}") from exc
    samples = pd.DataFrame(arrays["samples"], columns=keys)
    samples.index.name = "future"
    return Ensemble(park, base, list(space), samples, arrays["costs"], arrays["kwh"],
                    arrays["nominal_costs"], arrays["nominal_kwh"], seed)


def evaluate_policies(ens: Ensemble, rules: list[str] | None = None,
                      rel_tol: float = 0.01, abs_tol: float = 500.0,
                      members: list[int] | None = None) -> pd.DataFrame:
    """Stability of every (rule, clause) policy in every future (long table).
    """
    park = ens.park
    keep = list(range(park.n)) if members is None else sorted(members)
    if len(set(keep)) != len(keep) or any(i < 0 or i >= park.n for i in keep):
        raise ValueError("Hub members must be distinct indices in the park")
    if len(keep) < 2:
        raise ValueError("A hub needs at least two members")
    n = len(keep)
    grand = (1 << n) - 1
    contracts = park.contracts()[keep]
    rules = rules or list(allocation.RULES)
    nom = games.subgame(ens.nominal_costs, park.n, keep)
    nom_kwh = ens.nominal_kwh[keep]
    nominal_shares = {r: signing_shares(r, nom, contracts, nom_kwh) for r in rules}
    records = []
    for k in range(ens.n_futures):
        c = games.subgame(ens.costs[k], park.n, keep)
        kwh = ens.kwh[k][keep]
        standalone = np.array([c[1 << i] for i in range(n)])
        eps = games.least_core_value(c, n)
        viable = c[grand] < standalone.sum()
        core_tol = games.numerical_tol(c)
        cache = {}
        for r in rules:
            for clause in CLAUSES:
                if clause == "fixed":
                    x = nominal_shares[r] * c[grand]          # = policy_allocation(r, "fixed", ...)
                else:
                    if r not in cache:
                        cache[r] = policy_allocation(r, clause, c, contracts, kwh, nom, nom_kwh)
                    x = cache[r]
                st = games.stability(x, c, n, rel_tol=rel_tol, abs_tol=abs_tol)
                worse = np.zeros(n, dtype=bool)
                worse[st.ir_violations] = True
                worse_full = int(sum(1 << keep[i] for i in np.flatnonzero(worse)))
                worst_full = int(sum(1 << keep[i] for i in games.members(st.worst_mask, n)))
                records.append({
                    "future": k, "rule": r, "clause": clause,
                    "stable": st.stable, "n_blocking": st.n_blocking,
                    "max_excess": st.max_excess, "worst_mask": worst_full,
                    "regret": max(st.max_excess - eps, 0.0),
                    "in_core": st.max_excess <= core_tol,
                    "firms_worse_off": int(worse.sum()),
                    "worse_off_mask": worse_full,
                    "all_better_off": not worse.any(),
                    "hub_viable": bool(viable),
                    "core_exists": bool(eps <= core_tol),
                    "eps_star": eps,
                    "hub_cost": float(c[grand]),
                    "hub_savings": float(standalone.sum() - c[grand]),
                })
    return pd.DataFrame.from_records(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Robustness metrics per policy."""
    g = df.groupby(["rule", "clause"], sort=False)
    out = pd.DataFrame({
        "stable_share": g["stable"].mean(),
        "exact_core_share": g["in_core"].mean(),
        "all_better_off_share": g["all_better_off"].mean(),
        "mean_blocking_groups": g["n_blocking"].mean(),
        "median_max_excess": g["max_excess"].median(),
        "p90_max_excess": g["max_excess"].quantile(0.9),
        "mean_regret": g["regret"].mean(),
        "hub_viable_share": g["hub_viable"].mean(),
        "core_exists_share": g["core_exists"].mean(),
    }).reset_index()
    out["rule_label"] = out["rule"].map(lambda r: allocation.RULES[r].label)
    out["clause_label"] = out["clause"].map(CLAUSES)
    return out


def firm_exit_frequency(df: pd.DataFrame, n: int, rule: str, clause: str) -> np.ndarray:
    """Share of futures in which each firm is worse off than alone under a policy."""
    sub = df[(df.rule == rule) & (df.clause == clause)]
    masks = sub["worse_off_mask"].to_numpy()
    return np.array([np.mean((masks >> i) & 1) for i in range(n)])


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 1)))

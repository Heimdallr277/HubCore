"""Session state and cached computations for the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import astuple, fields
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import streamlit as st

from hubcore import allocation, annual_reference, games, lp_reference, profiles
from hubcore import robustness as R
from hubcore import uncertainty as U
from hubcore.costs import Assumptions, CoalitionCosts, coalition_costs
from hubcore.park import EXAMPLE_PARK, PRESETS, Firm, Park
from ui import featured

ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED = ROOT / "data" / "precomputed"
N_DEFAULT = 200
SEED_DEFAULT = 42

# Failures of the optional LP checks are reported in the page instead of as a traceback.
SOLVER_ERRORS = (lp_reference.SolverError, games.GameError, ValueError)

ARCH_LABEL = {k: a.label for k, a in profiles.ARCHETYPES.items()}
ARCH_KEY = {v: k for k, v in ARCH_LABEL.items()}
ADDON_LABEL = dict(profiles.ADDONS)
ADDON_KEY = {v: k for k, v in ADDON_LABEL.items()}


# ---------------------------------------------------------------- state ----
def init() -> None:
    ss = st.session_state
    ss.setdefault("preset", EXAMPLE_PARK.name)
    ss.setdefault("park", EXAMPLE_PARK)
    ss.setdefault("assumptions", Assumptions())
    ss.setdefault("ensembles", {})
    ss.setdefault("policy_tables", {})
    ss.setdefault("rel_tol", 0.01)
    ss.setdefault("abs_tol", 500.0)


def park() -> Park:
    return st.session_state["park"]


def assumptions() -> Assumptions:
    return st.session_state["assumptions"]


def is_preset_unchanged() -> bool:
    p = park()
    return p.name in PRESETS and PRESETS[p.name] == p and assumptions() == Assumptions()


# -------------------------------------------------------- serialization ----
def park_key(p: Park) -> tuple:
    return (p.name, tuple(astuple(f) for f in p.firms))


def park_from_key(key: tuple) -> Park:
    name, firms = key
    base = PRESETS.get(name)
    return Park(name=name, firms=tuple(Firm(*f) for f in firms),
                description=base.description if base else "")


def a_key(a: Assumptions) -> tuple:
    return tuple((f.name, getattr(a, f.name)) for f in fields(Assumptions))


def a_from_key(key: tuple) -> Assumptions:
    return Assumptions(**dict(key))


def park_to_frame(p: Park) -> pd.DataFrame:
    return pd.DataFrame({
        "Firm": [f.name for f in p.firms],
        "Type": [ARCH_LABEL[f.archetype] for f in p.firms],
        "Annual use (MWh)": [float(f.annual_mwh) for f in p.firms],
        "Electrification plan": [ADDON_LABEL[f.addon] for f in p.firms],
        "Planned extra use (MWh/yr)": [float(f.addon_mwh) for f in p.firms],
        "Existing contract (kW)": [f.contract() for f in p.firms],
    })


def frame_to_park(df: pd.DataFrame, name: str) -> Park:
    firms = []
    for _, r in df.dropna(how="all").iterrows():
        if pd.isna(r.get("Firm")) or pd.isna(r.get("Annual use (MWh)")):
            raise ValueError("Each firm needs a name and annual use")
        if r.get("Type") not in ARCH_KEY or r.get("Electrification plan") not in ADDON_KEY:
            raise ValueError("Choose a firm type and an electrification plan for each row")
        addon = ADDON_KEY[r["Electrification plan"]]
        raw_extra = r.get("Planned extra use (MWh/yr)")
        extra = 0.0 if pd.isna(raw_extra) else float(raw_extra)
        contract = r.get("Existing contract (kW)")
        contract = None if contract is None or (isinstance(contract, float) and np.isnan(contract)) else float(contract)
        firm = Firm(
            name=str(r.get("Firm") or "").strip(),
            archetype=ARCH_KEY[r["Type"]],
            annual_mwh=float(r.get("Annual use (MWh)") or 0.0),
            addon=addon,
            addon_mwh=extra if addon != "none" else 0.0,
            contract_kw=None,
        )
        if contract is not None and firm.annual_mwh > 0 and contract != firm.contract():
            firm = Firm(firm.name, firm.archetype, firm.annual_mwh, firm.addon, firm.addon_mwh, contract)
        firms.append(firm)
    return Park(name=name, firms=tuple(firms))


# ------------------------------------------------------------- caching ----
@st.cache_data(show_spinner=False, max_entries=64)
def _costs(pk: tuple, ak: tuple) -> CoalitionCosts:
    p, a = park_from_key(pk), a_from_key(ak)
    return coalition_costs(p.loads(), p.contracts(), a)


def costs(p: Park | None = None, a: Assumptions | None = None) -> CoalitionCosts:
    return _costs(park_key(p or park()), a_key(a or assumptions()))


@st.cache_data(show_spinner=False, max_entries=64)
def _loads(pk: tuple) -> np.ndarray:
    return park_from_key(pk).loads()


def loads(p: Park | None = None) -> np.ndarray:
    return _loads(park_key(p or park()))


@st.cache_data(show_spinner=False, max_entries=64)
def _allocations(pk: tuple, ak: tuple) -> dict[str, np.ndarray]:
    p = park_from_key(pk)
    cc = _costs(pk, ak)
    kwh = _loads(pk).sum(axis=1)
    return allocation.all_rules(cc.cost, p.n, p.contracts(), kwh)


def allocations() -> dict[str, np.ndarray]:
    return _allocations(park_key(park()), a_key(assumptions()))


@st.cache_data(show_spinner=False, max_entries=64)
def _game_facts(pk: tuple, ak: tuple) -> dict:
    p = park_from_key(pk)
    cc = _costs(pk, ak)
    eps = games.least_core_value(cc.cost, p.n)
    parts, total = games.optimal_partition(cc.cost, p.n)
    return {"eps": eps, "partition": parts, "partition_cost": total}


def game_facts() -> dict:
    return _game_facts(park_key(park()), a_key(assumptions()))


def groups_at_battery_limit(cc: CoalitionCosts, names: list[str]) -> list[str]:
    """Groups whose screening battery is the largest size searched (largest groups first)."""
    flags = cc.battery_at_search_limit
    if flags is None:
        return []
    masks = sorted(np.flatnonzero(flags), key=lambda m: (-int(m).bit_count(), int(m)))
    return [" + ".join(names[i] for i in games.members(int(m), cc.n)) for m in masks]


# ------------------------------------------------------------ ensembles ----
def precomputed_path(p: Park, n: int, seed: int) -> Path:
    return PRECOMPUTED / R.preset_filename(p.name, n, seed)


def ensemble_id(p: Park, a: Assumptions, n: int, seed: int) -> str:
    return R.fingerprint(p, a, U.default_space(p, a), n, seed)


def get_ensemble(n: int = N_DEFAULT, seed: int = SEED_DEFAULT) -> R.Ensemble | None:
    p, a = park(), assumptions()
    fid = ensemble_id(p, a, n, seed)
    store = st.session_state["ensembles"]
    if fid in store:
        return store[fid]
    try:
        ens = R.load(precomputed_path(p, n, seed), p, a, U.default_space(p, a), n, seed)
    except R.EnsembleFileError as exc:
        st.warning(f"{exc}. Run the selected stress test to regenerate results for this session.")
        return None
    if ens is not None:
        store[fid] = ens
    return ens


def run_ensemble(n: int, seed: int, progress) -> R.Ensemble:
    p, a = park(), assumptions()
    ens = R.run_ensemble(p, a, U.default_space(p, a), n, seed, progress=progress, workers=1)
    st.session_state["ensembles"][ensemble_id(p, a, n, seed)] = ens
    return ens


def policy_table(ens: R.Ensemble, members: tuple[int, ...]) -> pd.DataFrame:
    rel, abs_ = st.session_state["rel_tol"], st.session_state["abs_tol"]
    key = (R.fingerprint(ens.park, ens.base, ens.space, ens.n_futures, ens.seed), members, rel, abs_)
    store = st.session_state["policy_tables"]
    if key not in store:
        store[key] = R.evaluate_policies(ens, rel_tol=rel, abs_tol=abs_, members=list(members))
    return store[key]


@st.cache_data(show_spinner="Solving the hourly reference LP…", max_entries=8)
def dispatch(pk: tuple, ak: tuple, cap: float, contract: float, energy: float):
    return lp_reference.solve(_loads(pk).sum(axis=0), cap, a_from_key(ak), battery=energy, contract=contract,
                              linked_midnights=True)

@st.cache_data(show_spinner=False)
def preset_reference_summary() -> dict:
    try:
        data = json.loads((ROOT / "data" / "preset_lp_checks.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"training_design", "training_gaps", "holdout_gaps", "memberships"} <= data.keys():
            raise ValueError("Missing reference benchmark fields")
        return data
    except (OSError, ValueError) as exc:
        st.warning(f"The bundled full-ensemble LP summary is unavailable ({exc}). Individual futures can still be repriced.")
        return {}


@st.cache_data(show_spinner=False, max_entries=1)
def _bundled_reference() -> tuple[str, np.ndarray, np.ndarray]:
    with np.load(ROOT / "data" / "preset_lp_costs.npz", allow_pickle=False) as z:
        return str(z["training_design"]), z["training"], z["nominal"]


@st.cache_data(show_spinner=False, max_entries=24)
def _reference_game(pk, ak, members, growth):
    full = park_from_key(pk)
    group = Park(name=full.name, firms=tuple(full.firms[i] for i in members))
    return annual_reference.price_game(group, a_from_key(ak), np.asarray(growth))


def bundled_reference(ens, future: int) -> np.ndarray | None:
    """Bundled continuous-LP prices of every coalition in ``future``, or None (no solving)."""
    design = R.fingerprint(ens.park, ens.base, ens.space, ens.n_futures, ens.seed)
    if design != preset_reference_summary().get("training_design", {}).get("fingerprint"):
        return None
    try:
        stored_design, costs, _ = _bundled_reference()
    except (OSError, ValueError, BadZipFile, KeyError):
        return None
    count = 1 << ens.park.n
    if stored_design != design or costs.shape != (ens.n_futures, count) or not np.isfinite(costs[future]).all():
        return None
    return costs[future]


def featured_applies() -> bool:
    """The featured decision is defined for the unchanged six-firm preset only."""
    return is_preset_unchanged() and park().name == EXAMPLE_PARK.name


def featured_case(with_lp: bool = True) -> dict | None:
    """Screening and continuous-LP verdicts for the featured future (None if not applicable)."""
    if not featured_applies():
        return None
    case = featured.FEATURED
    ens = get_ensemble(case["n"], case["seed"])
    if ens is None:
        return None
    p = ens.park
    k = case["future"]
    rel, abs_ = st.session_state["rel_tol"], st.session_state["abs_tol"]
    args = (p.names, p.contracts(), ens.kwh[k], rel, abs_)
    out = {"future": k, "member": case["member"], "member_index": p.names.index(case["member"]),
           "names": p.names, "rel": rel, "abs": abs_,
           "inputs": ens.samples.iloc[k].rename(index={u.key: f"{u.label} ({u.unit})" for u in ens.space}),
           "screening_costs": ens.costs[k],
           "screening": featured.verdict(featured.SCREENING, ens.costs[k], *args)}
    lp = bundled_reference(ens, k)
    out["bundled"] = lp is not None
    if lp is None and with_lp:
        lp, _ = reference_future(ens, tuple(range(p.n)), k, fixed=False)
    if lp is not None:
        out["continuous_costs"] = lp
        out["continuous"] = featured.verdict(featured.CONTINUOUS, lp, *args)
        out["attribution"] = featured.attribution(ens.costs[k], lp, out["member_index"], *args)
    return out


def reference_future(ens, members: tuple[int, ...], future: int, fixed: bool):
    """Reprice the selected game. Fixed percentages use a separately repriced signing game."""
    design = R.fingerprint(ens.park, ens.base, ens.space, ens.n_futures, ens.seed)
    if design == preset_reference_summary().get("training_design", {}).get("fingerprint"):
        try:
            stored_design, costs, nominal = _bundled_reference()
            count = 1 << ens.park.n
            if (stored_design != design or costs.shape != (ens.n_futures, count) or nominal.shape != (count,)
                    or not np.isfinite(costs[future]).all() or not np.isfinite(nominal).all()):
                raise ValueError("Reference arrays do not match the selected design")
            current = games.subgame(costs[future], ens.park.n, list(members))
            signing = games.subgame(nominal, ens.park.n, list(members)) if fixed else None
            return current, signing
        except (OSError, ValueError, BadZipFile, KeyError) as exc:
            st.warning(f"Bundled LP prices are unavailable ({exc}). Repricing the selected game now.")
    a, growth = U.to_future(ens.samples.iloc[future], ens.base, ens.park.n)
    pk = park_key(ens.park)
    active = tuple(growth[i] for i in members)
    current = _reference_game(pk, a_key(a), members, active)
    signing = None
    if fixed:
        nominal_growth = {u.key: u.base for u in ens.space if u.is_growth}
        a0, g0 = U.to_future(nominal_growth, ens.base, ens.park.n)
        signing = _reference_game(pk, a_key(a0), members, tuple(g0[i] for i in members))
    return current, signing

"""Reference load-profile library.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 8760
DAYS = 365
DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_FILE = DATA_DIR / "profiles_2025.csv.gz"


@lru_cache(maxsize=1)
def data_hash() -> str:
    # Use the profile contents; gzip headers can vary by platform.
    import gzip

    with gzip.open(PROFILE_FILE, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Archetype:
    """A firm type with a base load shape and evidence status."""

    key: str
    label: str
    shape: str
    description: str
    source: str


ARCHETYPES: dict[str, Archetype] = {
    a.key: a
    for a in [
        Archetype("office", "Office / services", "g1",
                  "Weekday 08-18h demand, low at night and weekends.",
                  "BDEW G1 (proxy)"),
        Archetype("evening", "Leisure / evening business", "g2",
                  "Demand peaks in the evening (e.g. fitness center, restaurant).",
                  "BDEW G2 (proxy)"),
        Archetype("continuous", "Continuous operation", "g3",
                  "Near-flat 24/7 demand (e.g. cold storage, data room).",
                  "BDEW G3 (proxy)"),
        Archetype("retail", "Retail / trade", "g4",
                  "Shop-hours demand including Saturdays.",
                  "BDEW G4 (proxy)"),
        Archetype("bakery", "Food production (early shift)", "g5",
                  "Early-morning production peak (bakery-type).",
                  "BDEW G5 (proxy)"),
        Archetype("weekend", "Weekend business", "g6",
                  "Demand concentrated on weekends.",
                  "BDEW G6 (proxy)"),
        Archetype("manufacturing", "Manufacturing (two shifts)", "industry_2shift",
                  "Flat 06-22h on weekdays, low otherwise.",
                  "demandlib industrial step profile (stylized)"),
        Archetype("logistics", "Logistics / warehouse", "industry_2shift",
                  "Warehouse operations 06-22h on weekdays.",
                  "demandlib industrial step profile (stylized)"),
        Archetype("general", "General commerce", "g0",
                  "Average commercial customer.",
                  "BDEW G0 (proxy)"),
    ]
}

ADDONS: dict[str, str] = {
    "none": "None",
    "ev_depot_night": "EV fleet, night charging",
    "ev_workplace_day": "Workplace EV charging",
    "process": "Process electrification",
}


@lru_cache(maxsize=1)
def library() -> pd.DataFrame:
    """Return the profile library (8760 x shapes); cached."""
    if not PROFILE_FILE.exists():  # pragma: no cover - guarded by tests
        raise FileNotFoundError(
            f"Profile library not found at {PROFILE_FILE}. "
            "Run `python scripts/make_profiles.py` (requires demandlib)."
        )
    df = pd.read_csv(PROFILE_FILE, index_col=0, parse_dates=True)
    if df.shape[0] != HOURS:
        raise ValueError(f"Profile library must have {HOURS} rows, got {df.shape[0]}")
    return df


def shape(name: str) -> np.ndarray:
    """Hourly shape (kW per MWh/yr) as a float array of length 8760."""
    lib = library()
    if name not in lib.columns:
        raise KeyError(f"Unknown profile shape '{name}'. Available: {list(lib.columns)}")
    return lib[name].to_numpy(dtype=float)


@lru_cache(maxsize=1)
def month_of_day() -> np.ndarray:
    """Month index (0-11) for each of the 365 days."""
    idx = library().index
    return (idx.month.to_numpy()[::24] - 1).astype(int)


@lru_cache(maxsize=1)
def timestamps() -> pd.DatetimeIndex:
    return library().index

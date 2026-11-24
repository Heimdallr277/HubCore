"""Generate the hourly reference load-profile library used by HubCore.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2025
OUT = Path(__file__).resolve().parents[1] / "src" / "hubcore" / "data" / "profiles_2025.csv.gz"

# Selected non-working dates in the synthetic 2025 calendar, including King's Day on Saturday.
# Easter Sunday and Whit Sunday already follow the Sunday pattern. Good
# Friday and Liberation Day (May 5, 2025) are deliberately not included: many firms work on
# them. Changing this list changes the profiles, their digest and every bundled result.
NL_HOLIDAYS_2025 = {
    dt.date(2025, 1, 1): "Nieuwjaarsdag",
    dt.date(2025, 4, 21): "Tweede Paasdag",
    dt.date(2025, 4, 26): "Koningsdag",
    dt.date(2025, 5, 29): "Hemelvaartsdag",
    dt.date(2025, 6, 9): "Tweede Pinksterdag",
    dt.date(2025, 12, 25): "Eerste Kerstdag",
    dt.date(2025, 12, 26): "Tweede Kerstdag",
}


def _normalise(series: pd.Series) -> np.ndarray:
    """Scale an hourly kW series so that it integrates to 1 MWh (1000 kWh)."""
    values = np.asarray(series, dtype=float)
    return values * (1000.0 / values.sum())


def bdew_shapes() -> pd.DataFrame:
    import demandlib.bdew as bdew

    slp = bdew.ElecSlp(YEAR, holidays=NL_HOLIDAYS_2025)
    keys = ["g0", "g1", "g2", "g3", "g4", "g5", "g6"]
    quarter = slp.get_scaled_power_profiles({k: 1.0 for k in keys})
    hourly = quarter.resample("h").mean()
    assert len(hourly) == 8760, len(hourly)
    return pd.DataFrame({k: _normalise(hourly[k]) for k in keys}, index=hourly.index)


def industry_shape(index: pd.DatetimeIndex) -> np.ndarray:
    from demandlib import particular_profiles as pp

    idx15 = pd.date_range(f"{YEAR}-01-01", periods=8760 * 4, freq="15min")
    ilp = pp.IndustrialLoadProfile(idx15, holidays=NL_HOLIDAYS_2025)
    profile = ilp.simple_profile(
        1.0,
        am=dt.time(6, 0, 0),
        pm=dt.time(22, 0, 0),
        profile_factors={
            "week": {"day": 1.0, "night": 0.25},
            "weekend": {"day": 0.2, "night": 0.15},
            "holiday": {"day": 0.2, "night": 0.15},
        },
    )
    hourly = profile.resample("h").mean()
    hourly.index = index
    return _normalise(hourly)


def ev_depot_night(index: pd.DatetimeIndex) -> np.ndarray:
    """Depot charging of an electric truck/van fleet.

    Assumption: constant charging power 21:00-05:00 in nights that follow a
    working day (Mon-Fri evenings), no charging on weekends/holidays.
    """
    hours = index.hour
    weekday = index.dayofweek  # Monday = 0
    is_holiday = np.isin(index.date, list(NL_HOLIDAYS_2025))
    evening = (hours >= 21) & (weekday <= 4) & ~is_holiday
    # early-morning hours belong to the previous evening's session
    prev = index - pd.Timedelta(hours=8)
    prev_holiday = np.isin(prev.date, list(NL_HOLIDAYS_2025))
    morning = (hours < 5) & (prev.dayofweek <= 4) & ~prev_holiday & (prev.hour >= 16)
    shape = (evening | morning).astype(float)
    return _normalise(pd.Series(shape))


def ev_workplace_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Workplace charging of employee/visitor cars.

    Assumption: weekdays (excluding holidays) 08:00-17:00 with a morning
    arrival peak: relative weights 1.0 at 08-11h, 0.6 at 11-14h, 0.3 at 14-17h.
    """
    hours = index.hour
    weekday = index.dayofweek
    is_holiday = np.isin(index.date, list(NL_HOLIDAYS_2025))
    working = (weekday <= 4) & ~is_holiday
    w = np.zeros(len(index))
    w[(hours >= 8) & (hours < 11)] = 1.0
    w[(hours >= 11) & (hours < 14)] = 0.6
    w[(hours >= 14) & (hours < 17)] = 0.3
    return _normalise(pd.Series(w * working))


def main() -> None:
    df = bdew_shapes()
    df["industry_2shift"] = industry_shape(df.index)
    df["ev_depot_night"] = ev_depot_night(df.index)
    df["ev_workplace_day"] = ev_workplace_day(df.index)
    df.index.name = "timestamp"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(OUT, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    meta = {
        "year": YEAR,
        "unit": "kW per 1 MWh/year",
        "columns": list(df.columns),
        "peak_to_mean": {c: round(float(df[c].max() / df[c].mean()), 3) for c in df.columns},
        "holidays": {str(k): v for k, v in NL_HOLIDAYS_2025.items()},
    }
    (OUT.parent / "profiles_2025.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT} {df.shape}")
    print(json.dumps(meta["peak_to_mean"], indent=2))


if __name__ == "__main__":
    main()

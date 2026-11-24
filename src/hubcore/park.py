"""Firms, parks and their hourly loads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import profiles

MAX_FIRMS = 8
MIN_FIRMS = 2
DEFAULT_HEADROOM = 1.10  # existing contract = 110% of base-load peak (assumption)


class ParkError(ValueError):
    """Raised for invalid park definitions (shown to users as a message)."""


@dataclass(frozen=True)
class Firm:
    """One firm on the business park."""

    name: str
    archetype: str
    annual_mwh: float
    addon: str = "none"
    addon_mwh: float = 0.0
    contract_kw: float | None = None

    def base_load(self) -> np.ndarray:
        arch = profiles.ARCHETYPES[self.archetype]
        return self.annual_mwh * profiles.shape(arch.shape)

    def addon_load(self) -> np.ndarray:
        if self.addon == "none" or self.addon_mwh <= 0:
            return np.zeros(profiles.HOURS)
        if self.addon == "process":
            arch = profiles.ARCHETYPES[self.archetype]
            return self.addon_mwh * profiles.shape(arch.shape)
        return self.addon_mwh * profiles.shape(self.addon)

    def contract(self) -> float:
        if self.contract_kw is not None:
            return float(self.contract_kw)
        peak = float(self.base_load().max())
        return float(math.ceil(DEFAULT_HEADROOM * peak / 10.0) * 10.0)


@dataclass(frozen=True)
class Park:
    name: str
    firms: tuple[Firm, ...]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n(self) -> int:
        return len(self.firms)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.firms]

    def contracts(self) -> np.ndarray:
        return np.array([f.contract() for f in self.firms])

    def loads(self, growth: np.ndarray | None = None) -> np.ndarray:
        """Hourly load matrix (n x 8760) in kW.
        """
        g = np.ones(self.n) if growth is None else np.asarray(growth, dtype=float)
        if g.shape != (self.n,) or not np.isfinite(g).all() or (g < 0).any():
            raise ParkError(f"Non-negative finite growth vector must have length {self.n}")
        return np.vstack([f.base_load() + gi * f.addon_load() for f, gi in zip(self.firms, g)])

    def with_firm(self, firm: Firm) -> "Park":
        return replace(self, firms=self.firms + (firm,))

    def without(self, index: int) -> "Park":
        return replace(self, firms=tuple(f for i, f in enumerate(self.firms) if i != index))


def validate(park: Park) -> None:
    """Raise :class:`ParkError` with a readable message if the park is invalid."""
    if not (MIN_FIRMS <= park.n <= MAX_FIRMS):
        raise ParkError(f"A hub needs between {MIN_FIRMS} and {MAX_FIRMS} firms (got {park.n}).")
    names = [f.name.strip() for f in park.firms]
    if any(not n for n in names):
        raise ParkError("Every firm needs a name.")
    if len(set(names)) != len(names):
        raise ParkError("Firm names must be unique.")
    for f in park.firms:
        if f.archetype not in profiles.ARCHETYPES:
            raise ParkError(f"{f.name}: unknown firm type '{f.archetype}'.")
        if f.addon not in profiles.ADDONS:
            raise ParkError(f"{f.name}: unknown electrification plan '{f.addon}'.")
        if not (1 <= f.annual_mwh <= 50_000):
            raise ParkError(f"{f.name}: annual use must be between 1 and 50,000 MWh.")
        if not (0 <= f.addon_mwh <= 50_000):
            raise ParkError(f"{f.name}: planned extra use must be between 0 and 50,000 MWh.")
        if f.contract_kw is not None and not (1 <= f.contract_kw <= 20_000):
            raise ParkError(f"{f.name}: contracted capacity must be between 1 and 20,000 kW.")


# ---------------------------------------------------------------------------
# Presets (fictional firms; numbers are illustrative)
# ---------------------------------------------------------------------------

EXAMPLE_PARK = Park(
    name="Voorbeeldpark (example)",
    description=(
        "Six fictional firms on a congested business park. Several have electrification "
        "plans that do not fit within their current grid contracts."
    ),
    firms=(
        Firm("Office HQ", "office", 700, "ev_workplace_day", 150),
        Firm("Logistics DC", "logistics", 800, "ev_depot_night", 500),
        Firm("Cold store", "continuous", 1400, "process", 200),
        Firm("Bakery", "bakery", 450, "process", 250),
        Firm("Fitness center", "evening", 250, "none", 0),
        Firm("Metalworks", "manufacturing", 1600, "process", 500),
    ),
    tags=("preset",),
)

SMALL_PARK = Park(
    name="Three-firm park (teaching example)",
    description="A day-time office, a night-charging fleet operator and an evening business.",
    firms=(
        Firm("Office", "office", 600, "none", 0),
        Firm("Fleet depot", "logistics", 300, "ev_depot_night", 500),
        Firm("Sports hall", "evening", 300, "none", 0),
    ),
    tags=("preset",),
)

PRESETS: dict[str, Park] = {p.name: p for p in (EXAMPLE_PARK, SMALL_PARK)}

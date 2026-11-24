"""Precompute robustness ensembles for the preset parks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hubcore import robustness as R  # noqa: E402
from hubcore import uncertainty as U  # noqa: E402
from hubcore.costs import Assumptions  # noqa: E402
from hubcore.park import PRESETS  # noqa: E402

OUT = ROOT / "data" / "precomputed"
N_FUTURES = 200
SEED = 42


def path_for(park_name: str, n: int = N_FUTURES, seed: int = SEED) -> Path:
    return OUT / R.preset_filename(park_name, n, seed)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = Assumptions()
    for name, park in PRESETS.items():
        space = U.default_space(park)
        t0 = time.time()
        ens = R.run_ensemble(park, base, space, N_FUTURES, SEED, workers=R.default_workers())
        R.save(ens, path_for(name))
        print(f"{name}: {N_FUTURES} futures in {time.time() - t0:.0f}s -> {path_for(name).name}")


if __name__ == "__main__":
    main()

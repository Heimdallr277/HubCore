import numpy as np
import pandas as pd

from hubcore import prim


def test_prim_recovers_planted_box():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.uniform(0, 1, size=(1000, 5)), columns=list("abcde"))
    y = ((X.a > 0.7) & (X.b < 0.4)).to_numpy()
    traj = prim.prim(X, y, alpha=0.05, min_mass=0.03)
    box, pvals = prim.explain(traj[prim.suggest(traj)], X, y)
    assert all(v < 0.05 for v in pvals.values())
    assert set(box.restricted) == {"a", "b"}
    assert box.coverage >= 0.9 and box.density >= 0.9
    lo_a, _ = box.limits["a"]
    _, hi_b = box.limits["b"]
    assert abs(lo_a - 0.7) < 0.05 and abs(hi_b - 0.4) < 0.05


def test_prim_trajectory_is_monotone_in_mass():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.uniform(size=(300, 3)), columns=["x", "y", "z"])
    y = (X.x > 0.5).to_numpy()
    traj = prim.prim(X, y)
    masses = [b.mass for b in traj]
    assert all(m1 >= m2 for m1, m2 in zip(masses, masses[1:]))
    assert traj[0].coverage == 1.0

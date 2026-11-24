"""Verification of the cooperative-game layer against known results."""


import numpy as np
import pytest

from hubcore import games


def random_game(n, rng):
    cost = np.zeros(1 << n)
    cost[1:] = rng.uniform(1, 10, size=(1 << n) - 1)
    return cost


def airport_game(b):
    n = len(b)
    cost = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        cost[mask] = max(b[i] for i in range(n) if mask >> i & 1)
    return cost


def test_shapley_is_efficient():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 7):
        c = random_game(n, rng)
        assert games.shapley(c, n).sum() == pytest.approx(c[-1])


def test_shapley_symmetry_and_dummy():
    # players 0 and 1 symmetric; player 2 is a dummy with stand-alone cost 4
    n = 3
    base = {0: 0, 1: 5, 2: 5, 3: 7}           # costs over players {0,1}
    c = np.zeros(8)
    for mask in range(1, 8):
        c[mask] = base[mask & 3] + (4 if mask & 4 else 0)
    phi = games.shapley(c, n)
    assert phi[0] == pytest.approx(phi[1])
    assert phi[2] == pytest.approx(4.0)


def test_shapley_additivity():
    rng = np.random.default_rng(1)
    c1, c2 = random_game(4, rng), random_game(4, rng)
    np.testing.assert_allclose(games.shapley(c1 + c2, 4), games.shapley(c1, 4) + games.shapley(c2, 4))


def test_airport_game_closed_form_and_core():
    b = [2.0, 3.0, 7.0, 10.0]                  # sorted runway requirements
    n = len(b)
    c = airport_game(b)
    expected, prev, acc = [], 0.0, 0.0
    for k, bk in enumerate(b):                 # Littlechild-Owen formula
        acc += (bk - prev) / (n - k)
        expected.append(acc)
        prev = bk
    np.testing.assert_allclose(games.shapley(c, n), expected)
    assert games.least_core_value(c, n) <= 1e-9          # concave game: core non-empty
    assert games.stability(games.shapley(c, n), c, n, rel_tol=0, abs_tol=1e-7).stable


def test_least_core_of_empty_core_game():
    # three players, any pair can be served for 1, everyone together for 2
    c = np.array([0, 1, 1, 1, 1, 1, 1, 2], dtype=float)
    eps = games.least_core_value(c, 3)
    assert eps == pytest.approx(1 / 3, abs=1e-9)
    x, used = games.closest_in_least_core(c, 3, np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(x, [2 / 3] * 3, atol=1e-5)


def test_repair_returns_core_point_and_keeps_core_proposals():
    c = airport_game([1.0, 4.0, 6.0])
    phi = games.shapley(c, 3)
    x, _ = games.closest_in_least_core(c, 3, phi)
    np.testing.assert_allclose(x, phi, atol=1e-6)           # already in core: unchanged
    bad = np.array([4.0, 1.0, 1.0])                          # player 0 overcharged
    x, _ = games.closest_in_least_core(c, 3, bad)
    assert x.sum() == pytest.approx(c[-1])
    assert games.excess(x, c, 3)[1:-1].max() <= 1e-6


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def test_optimal_partition_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 5
    c = random_game(n, rng)
    best = min(sum(c[sum(1 << i for i in block)] for block in p) for p in _partitions(list(range(n))))
    parts, total = games.optimal_partition(c, n)
    assert total == pytest.approx(best)
    assert sorted(i for s in parts for i in games.members(s, n)) == list(range(n))


def test_subgame_mapping():
    rng = np.random.default_rng(4)
    c = random_game(4, rng)
    sub = games.subgame(c, 4, [1, 3])
    assert sub[1] == c[1 << 1] and sub[2] == c[1 << 3] and sub[3] == c[(1 << 1) | (1 << 3)]


def test_stability_threshold():
    c = airport_game([2.0, 3.0])
    x = np.array([2.6, 0.4])                    # player 0 pays 0.6 more than alone
    assert not games.stability(x, c, 2, rel_tol=0, abs_tol=0.1).stable
    assert games.stability(x, c, 2, rel_tol=0, abs_tol=1.0).stable

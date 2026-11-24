import numpy as np
import pytest
from scipy import stats

from outliers.scoring import count_z, empirical_z, pct_z


def test_poisson_tail_matches_known_value():
    # P(X >= 5 | Poisson(1)) = 0.00366 -> z ~ 2.68
    z = count_z(5, 1.0, 0.0, "count", 0, 8)
    assert z == pytest.approx(stats.norm.isf(stats.poisson(1).sf(4)), rel=1e-6)
    assert z == pytest.approx(2.68, abs=0.01)


def test_count_z_sign_and_monotonic():
    zs = [count_z(x, 10, 0.1, "count", 0, 8) for x in range(0, 40, 5)]
    assert zs[0] < 0 < zs[-1]
    assert all(a <= b for a, b in zip(zs, zs[1:]))


def test_count_z_near_mean_is_small():
    assert abs(count_z(10, 10, 0.1, "count", 0, 8)) < 0.5


def test_overdispersion_lowers_surprise():
    assert count_z(30, 10, 0.3, "count", 0, 8) < count_z(30, 10, 0.0, "count", 0, 8)


def test_normal_uses_floor():
    # Tiny mean: floor of 1.0 caps the z at (x - mu) / 1.0.
    assert count_z(3, 0.1, 0.0, "normal", 1.0, 8) == pytest.approx(2.9)


def test_z_is_capped():
    assert count_z(100, 1, 0.0, "count", 0, 8) == 8.0


def test_pct_z_directions():
    assert pct_z(14, 15, 0.45, 150, "count", 8) > 2.5
    assert pct_z(2, 18, 0.48, 150, "count", 8) < -2.5
    assert abs(pct_z(7, 15, 0.47, 150, "normal", 8)) < 0.5


def test_empirical_z():
    values = np.sort(np.arange(1000, dtype=float))
    z_hi, p_hi = empirical_z(999, values, "high", 8)
    assert p_hi == pytest.approx(2 / 1001)
    assert z_hi > 2.5
    z_lo, _ = empirical_z(0, values, "both", 8)
    assert z_lo < -2.5
    assert empirical_z(0, values, "high", 8)[0] == 0.0

import pandas as pd
import pytest

from outliers.baselines import blend_rate, recency_weights

BASE = dict(method="bayesian", seasons_back=2, season_decay=[1.0, 0.5], recency_half_life_games=0,
            min_games_own=8, exp_decay_half_life_games=0, fixed_weights=[0.6, 0.3, 0.1])


def games(rows):
    """rows: (season_idx, minutes, points)"""
    return pd.DataFrame(rows, columns=["SEASON_IDX", "MIN", "PTS"])


# Last season: 1000 min, 500 pts (0.5/min). Two seasons ago: 1000 min, 300 pts (0.3/min).
# This season: 100 min, 100 pts (1.0/min).
HIST = games([(2, 1000, 300), (1, 1000, 500), (0, 100, 100)])


def test_bayesian_known_answer():
    # Prior seasons weighted 1.0 / 0.5: (500 + 150) / (1000 + 500) = 0.4333, shrunk toward
    # fill 0.4 with k=400: (650 + 160) / 1900 = 0.42632. Then blended with this season at k=400.
    prior = (650 + 400 * 0.4) / (1500 + 400)
    expected = (400 * prior + 100) / (400 + 100)
    assert blend_rate(HIST, "PTS", "MIN", 400, 0.4, 150, BASE) == pytest.approx(expected)


def test_bayesian_current_season_takes_over():
    long_season = games([(1, 1000, 500), (0, 3000, 3000)])
    rate = blend_rate(long_season, "PTS", "MIN", 400, 0.5, 150, BASE)
    assert rate > 0.9


def test_bayesian_rookie_uses_fill():
    rookie = games([(0, 30, 30)])
    rate = blend_rate(rookie, "PTS", "MIN", 400, 0.3, 150, BASE)
    assert rate == pytest.approx((400 * 0.3 + 30) / 430)


def test_no_fill_requires_min_games():
    rookie = games([(0, 30, 30)] * 3)
    assert blend_rate(rookie, "PTS", "MIN", 400, None, 150, BASE) is None


def test_season_only_ignores_prior_seasons():
    cfg = dict(BASE, method="season_only")
    assert blend_rate(HIST, "PTS", "MIN", 400, 0.4, 0, cfg) == pytest.approx(1.0)


def test_fixed_weights():
    cfg = dict(BASE, method="fixed_weights")
    own = (0.6 * 1.0 + 0.3 * 0.5 + 0.1 * 0.3) / 1.0
    assert blend_rate(HIST, "PTS", "MIN", 400, 0.4, 0, cfg) == pytest.approx(own)


def test_exp_decay_without_half_life_is_pooled_mean():
    cfg = dict(BASE, method="exp_decay")
    assert blend_rate(HIST, "PTS", "MIN", 400, 0.4, 0, cfg) == pytest.approx(900 / 2100)


def test_seasons_back_limits_window():
    cfg = dict(BASE, method="exp_decay", seasons_back=1)
    assert blend_rate(HIST, "PTS", "MIN", 400, 0.4, 0, cfg) == pytest.approx(600 / 1100)


def test_recency_weights():
    w = recency_weights(3, 1)
    assert list(w) == [0.25, 0.5, 1.0]
    assert list(recency_weights(3, 0)) == [1, 1, 1]

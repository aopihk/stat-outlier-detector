"""Synthetic league data so tests run offline and deterministically."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from outliers.config import load_config

SEASONS = [("2024-25", date(2024, 10, 22)), ("2025-26", date(2025, 10, 21)), ("2026-27", date(2026, 10, 20))]
SCORED_NIGHT = date(2026, 11, 23)


def _player_rows(rng, pid, team, mpg, rates, start, n_games, season):
    rows = []
    for g in range(n_games):
        mins = max(1, int(rng.normal(mpg, 3)))
        fga = rng.poisson(rates["FGA"] * mins)
        fgm = rng.binomial(fga, 0.47)
        fg3a = rng.poisson(rates["FG3A"] * mins)
        fg3m = rng.binomial(fg3a, 0.36)
        fta = rng.poisson(rates["FTA"] * mins)
        ftm = rng.binomial(fta, 0.78)
        oreb = rng.poisson(rates["OREB"] * mins)
        dreb = rng.poisson(rates["DREB"] * mins)
        rows.append(dict(
            PLAYER_ID=pid, PLAYER_NAME=f"Player {pid}", TEAM_ID=team, TEAM_ABBREVIATION=f"T{team}",
            GAME_ID=f"{season}-{team}-{g}", GAME_DATE=(start + timedelta(days=2 * g)).isoformat(),
            MATCHUP=f"T{team} vs. T{team + 1}", WL="W", MIN=mins,
            FGM=fgm, FGA=fga, FG3M=min(fg3m, fgm), FG3A=fg3a, FTM=ftm, FTA=fta,
            OREB=oreb, DREB=dreb, REB=oreb + dreb, AST=rng.poisson(rates["AST"] * mins),
            STL=rng.poisson(0.03 * mins), BLK=rng.poisson(0.02 * mins), TOV=rng.poisson(0.05 * mins),
            PF=rng.poisson(0.07 * mins), PTS=2 * fgm + min(fg3m, fgm) + ftm, PLUS_MINUS=0,
            SEASON=season, SEASON_TYPE="Regular Season",
        ))
    return rows


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def league():
    """40 players over 3 seasons; on SCORED_NIGHT player 7 (a ~10 PPG role player) scores 40+."""
    rng = np.random.default_rng(7)
    rates = dict(FGA=0.35, FG3A=0.12, FTA=0.1, OREB=0.03, DREB=0.12, AST=0.08)
    rows = []
    for pid in range(1, 41):
        mpg = [12, 20, 28, 34][pid % 4]
        team = pid % 10
        for season, start in SEASONS:
            n = 70 if season != "2026-27" else 15
            rows += _player_rows(rng, pid, team, mpg, rates, start, n, season)
    df = pd.DataFrame(rows)
    df = df[df["GAME_DATE"] < SCORED_NIGHT.isoformat()]
    night = df[df["SEASON"] == "2026-27"].groupby("PLAYER_ID").tail(1).copy()
    night["GAME_DATE"] = SCORED_NIGHT.isoformat()
    night["GAME_ID"] = night["GAME_ID"] + "-night"
    star = night["PLAYER_ID"] == 7
    night.loc[star, ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PTS", "MIN"]] = [16, 24, 5, 8, 6, 7, 43, 30]
    players = pd.DataFrame({"PLAYER_ID": range(1, 41), "POSITION": ["G", "F", "C", "G-F"] * 10,
                            "FROM_YEAR": [2020] * 40})
    return pd.concat([df, night], ignore_index=True), players

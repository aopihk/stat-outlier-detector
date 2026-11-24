"""Data provider interface, so the stats source can be swapped (e.g. an ESPN fallback)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

# Columns every provider must return from game_logs(). Player logs add
# PLAYER_ID / PLAYER_NAME; team logs don't have them.
LOG_COLUMNS = [
    "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]
PLAYER_COLUMNS = ["PLAYER_ID", "PLAYER_NAME"] + LOG_COLUMNS


class ProviderError(RuntimeError):
    pass


@dataclass
class GameStatus:
    game_id: str
    home: str
    away: str
    final: bool
    status_text: str


class DataProvider(ABC):
    @abstractmethod
    def game_logs(self, season: str, season_type: str, level: str) -> pd.DataFrame:
        """All games of a season. level is "P" (player) or "T" (team)."""

    @abstractmethod
    def scoreboard(self, d: date) -> list[GameStatus]:
        """Games scheduled on a date with their status."""

    @abstractmethod
    def player_index(self, season: str) -> pd.DataFrame:
        """Columns: PLAYER_ID, POSITION, FROM_YEAR."""

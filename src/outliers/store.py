"""SQLite cache of game logs and player info."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TABLES = {"P": "player_games", "T": "team_games"}


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS fetches (
                level TEXT, season TEXT, season_type TEXT, fetched_at TEXT, n_rows INTEGER,
                PRIMARY KEY (level, season, season_type))"""
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _has_table(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def fetched_at(self, level: str, season: str, season_type: str) -> str | None:
        row = self.conn.execute(
            "SELECT fetched_at FROM fetches WHERE level=? AND season=? AND season_type=?",
            (level, season, season_type),
        ).fetchone()
        return row[0] if row else None

    def replace_logs(self, level: str, season: str, season_type: str, df: pd.DataFrame) -> None:
        """Replace one season's logs. Re-fetching picks up stat corrections."""
        table = TABLES[level]
        df = df.copy()
        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type
        with self.conn:
            if self._has_table(table):
                self.conn.execute(
                    f"DELETE FROM {table} WHERE SEASON=? AND SEASON_TYPE=?", (season, season_type)
                )
            if len(df):
                df.to_sql(table, self.conn, if_exists="append", index=False)
            self.conn.execute(
                "INSERT OR REPLACE INTO fetches VALUES (?, ?, ?, ?, ?)",
                (level, season, season_type, datetime.now(timezone.utc).isoformat(), len(df)),
            )

    def load_logs(self, level: str, seasons: list[str]) -> pd.DataFrame:
        table = TABLES[level]
        if not self._has_table(table):
            return pd.DataFrame()
        marks = ",".join("?" * len(seasons))
        return pd.read_sql(f"SELECT * FROM {table} WHERE SEASON IN ({marks})", self.conn, params=seasons)

    def replace_players(self, season: str, df: pd.DataFrame) -> None:
        df = df.copy()
        df["SEASON"] = season
        with self.conn:
            if self._has_table("players"):
                self.conn.execute("DELETE FROM players WHERE SEASON=?", (season,))
            df.to_sql("players", self.conn, if_exists="append", index=False)
            self.conn.execute(
                "INSERT OR REPLACE INTO fetches VALUES ('I', ?, '', ?, ?)",
                (season, datetime.now(timezone.utc).isoformat(), len(df)),
            )

    def load_players(self) -> pd.DataFrame:
        """Latest known POSITION / FROM_YEAR per player across stored seasons."""
        if not self._has_table("players"):
            return pd.DataFrame(columns=["PLAYER_ID", "POSITION", "FROM_YEAR"])
        df = pd.read_sql("SELECT * FROM players ORDER BY SEASON", self.conn)
        return df.drop_duplicates("PLAYER_ID", keep="last")[["PLAYER_ID", "POSITION", "FROM_YEAR"]]

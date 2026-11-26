"""stats.nba.com via the nba_api package."""

import logging
import queue
import threading
import time
from datetime import date

import pandas as pd

from .base import LOG_COLUMNS, PLAYER_COLUMNS, DataProvider, GameStatus, ProviderError

log = logging.getLogger(__name__)


def run_with_deadline(fn, deadline: float):
    """Run fn() with a hard wall-clock limit.

    requests' timeout only bounds the wait between bytes, so a stalled or trickling
    stats.nba.com response can hang for many minutes. The call runs in a daemon thread
    that is abandoned (not killed) if it overruns, so it can't keep the process alive.
    """
    out: queue.Queue = queue.Queue(maxsize=1)

    def target():
        try:
            out.put((True, fn()))
        except BaseException as e:
            out.put((False, e))

    threading.Thread(target=target, daemon=True).start()
    try:
        ok, value = out.get(timeout=deadline)
    except queue.Empty:
        raise TimeoutError(f"no complete response within {deadline:g}s") from None
    if ok:
        return value
    raise value


class NbaApiProvider(DataProvider):
    def __init__(self, timeout: int = 30, retries: int = 4, sleep: float = 0.8, deadline: float = 90):
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.deadline = deadline

    def _call(self, what: str, fn):
        last = None
        for attempt in range(self.retries):
            time.sleep(self.sleep)
            try:
                return run_with_deadline(fn, self.deadline)
            except Exception as e:  # nba_api surfaces requests/JSON errors directly
                last = e
                wait = 2 ** attempt * 2
                log.warning("%s failed (attempt %d/%d): %s; retrying in %ds",
                            what, attempt + 1, self.retries, e, wait)
                time.sleep(wait)
        raise ProviderError(f"{what} failed after {self.retries} attempts: {last}")

    def game_logs(self, season: str, season_type: str, level: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguegamelog

        def fetch():
            return leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation=level,
                timeout=self.timeout,
            ).get_data_frames()[0]

        df = self._call(f"LeagueGameLog {season} {season_type} {level}", fetch)
        cols = PLAYER_COLUMNS if level == "P" else LOG_COLUMNS
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df[cols].copy()
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
        return df

    def scoreboard(self, d: date) -> list[GameStatus]:
        from nba_api.stats.endpoints import scoreboardv3

        def fetch():
            return scoreboardv3.ScoreboardV3(
                game_date=d.isoformat(), league_id="00", timeout=self.timeout
            ).get_dict()

        data = self._call(f"Scoreboard {d}", fetch)
        games = data.get("scoreboard", {}).get("games", [])
        return [
            GameStatus(
                game_id=str(g["gameId"]),
                home=g["homeTeam"]["teamTricode"],
                away=g["awayTeam"]["teamTricode"],
                final=g.get("gameStatus") == 3,
                status_text=g.get("gameStatusText", ""),
            )
            for g in games
        ]

    def player_index(self, season: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import playerindex

        def fetch():
            return playerindex.PlayerIndex(season=season, timeout=self.timeout).get_data_frames()[0]

        df = self._call(f"PlayerIndex {season}", fetch)
        out = pd.DataFrame({
            "PLAYER_ID": df["PERSON_ID"].astype(int),
            "POSITION": df["POSITION"].fillna(""),
            "FROM_YEAR": pd.to_numeric(df["FROM_YEAR"], errors="coerce"),
        })
        return out

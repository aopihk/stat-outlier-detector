"""Keep the local store up to date for a target date."""

import logging
from datetime import date

from .providers import DataProvider
from .seasons import REGULAR, season_for, season_types_for, season_window
from .store import Store

log = logging.getLogger(__name__)


def ensure_data(store: Store, provider: DataProvider, d: date, cfg: dict, refresh: bool = True) -> None:
    """Fetch whatever is missing for scoring date `d`.

    Prior seasons are fetched once. The target date's season is re-fetched when
    `refresh` is set (the daily run), which also picks up stat corrections.
    """
    current = season_for(d)
    for season in season_window(d, cfg["baseline"]["seasons_back"]):
        types = season_types_for(d) if season == current else [REGULAR]
        for season_type in types:
            for level in ("P", "T"):
                have = store.fetched_at(level, season, season_type)
                if have and not (refresh and season == current):
                    continue
                log.info("Fetching %s %s %s logs", season, season_type, "player" if level == "P" else "team")
                store.replace_logs(level, season, season_type, provider.game_logs(season, season_type, level))
        if not store.fetched_at("I", season, "") or (refresh and season == current):
            log.info("Fetching %s player index", season)
            store.replace_players(season, provider.player_index(season))

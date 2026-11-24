"""Season and date helpers. NBA seasons are labelled like "2026-27"."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

REGULAR = "Regular Season"
PLAYIN = "PlayIn"
PLAYOFFS = "Playoffs"


def season_start_year(d: date) -> int:
    return d.year if d.month >= 8 else d.year - 1


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_for(d: date) -> str:
    return season_label(season_start_year(d))


def prior_seasons(d: date, n: int) -> list[str]:
    start = season_start_year(d)
    return [season_label(start - i) for i in range(1, n + 1)]


def season_window(d: date, seasons_back: int) -> list[str]:
    """Current season first, then prior seasons, most recent first."""
    return [season_for(d)] + prior_seasons(d, seasons_back)


def season_types_for(d: date) -> list[str]:
    if 4 <= d.month <= 7:
        return [REGULAR, PLAYIN, PLAYOFFS]
    return [REGULAR]


def last_night(tz: str) -> date:
    return (datetime.now(ZoneInfo(tz)) - timedelta(days=1)).date()


def parse_date(s: str) -> date:
    return date.fromisoformat(s)

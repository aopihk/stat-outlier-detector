from .base import DataProvider, GameStatus, ProviderError

__all__ = ["DataProvider", "GameStatus", "ProviderError", "get_provider"]


def get_provider(cfg: dict) -> DataProvider:
    from .nba_api_provider import NbaApiProvider

    data = cfg["data"]
    return NbaApiProvider(
        timeout=data["timeout_seconds"],
        retries=data["retries"],
        sleep=data["request_sleep_seconds"],
    )

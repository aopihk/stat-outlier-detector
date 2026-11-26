import pytest

from conftest import SCORED_NIGHT
from outliers.config import with_overrides
from outliers.detect import detect_night, milestones
from outliers.ranking import select


@pytest.mark.parametrize("method", ["bayesian", "exp_decay", "fixed_weights", "season_only"])
@pytest.mark.parametrize("distribution", ["count", "normal"])
def test_big_night_ranks_first(cfg, league, method, distribution):
    logs, players = league
    c = with_overrides(cfg, [f'baseline.method="{method}"', f'baseline.distribution="{distribution}"'])
    result = detect_night(SCORED_NIGHT, logs, logs.iloc[0:0], players, c)
    sel = select(result.candidates, c)
    assert not sel.quiet
    assert sel.items[0].entity == "P:7"
    assert sel.items[0].kind == "personal"


def test_one_entry_per_player(cfg, league):
    logs, players = league
    result = detect_night(SCORED_NIGHT, logs, logs.iloc[0:0], players, cfg)
    sel = select(result.candidates, with_overrides(cfg, ["general.min_score=0.5"]))
    entities = [c.entity for c in sel.items]
    assert len(entities) == len(set(entities))


def test_no_lookahead(cfg, league):
    """Adding a huge game *after* the scored night must not change the scored night."""
    logs, players = league
    before = detect_night(SCORED_NIGHT, logs, logs.iloc[0:0], players, cfg)
    future = logs[logs["PLAYER_ID"] == 7].tail(1).copy()
    future["GAME_DATE"], future["GAME_ID"], future["PTS"] = "2026-12-01", "future", 90
    import pandas as pd
    after = detect_night(SCORED_NIGHT, pd.concat([logs, future]), logs.iloc[0:0], players, cfg)
    assert [round(c.z, 6) for c in before.candidates] == [round(c.z, 6) for c in after.candidates]


def test_milestones(cfg):
    row = dict(PTS=41, REB=12, AST=11, STL=1, BLK=0, FG3M=3)
    names = [n for n, _ in milestones(row, cfg["milestones"])]
    assert "40-point triple-double (41/12/11)" in names
    row = dict(PTS=8, REB=7, AST=6, STL=5, BLK=5, FG3M=0)
    assert any(n.startswith("5x5") for n, _ in milestones(row, cfg["milestones"]))


def test_empty_night(cfg, league):
    logs, players = league
    from datetime import date
    result = detect_night(date(2026, 7, 4), logs, logs.iloc[0:0], players, cfg)
    assert result.candidates == [] and result.game_ids == []


def test_context_lists_each_stat_once(cfg):
    from outliers.detect import Candidate
    def cand(kind, stat, z, reason):
        return Candidate(kind, "P:1", "A", "T", "T vs U", stat, 4, None, z, reason, "")
    cands = [cand("personal", "STL", 3.2, "4 STL (a 1-in-1,500 game for this player)"), cand("league", "LEAGUE:STL", 2.1, "4 STL (top 2%)"),
             cand("league", "LEAGUE:BLK", 2.3, "4 BLK (top 1%)")]
    item = select(cands, cfg).items[0]
    assert item.reason == "4 STL (a 1-in-1,500 game for this player)"
    assert item.context == ["4 BLK (top 1%)"]


def test_rarity_wording():
    from outliers.detect import odds, rarity
    assert odds(2.5) == "1-in-160"
    assert odds(3.0) == "1-in-740"
    assert odds(4.0) == "1-in-32,000"
    assert odds(-3.0) == "1-in-740"
    assert odds(8.0) == "rarer than 1-in-a-million"
    assert rarity(8.0, "player") == "a rarer-than-1-in-a-million game for this player"
    assert rarity(-2.7, "player") == "a 1-in-290 low for this player"
    assert rarity(3.0, "team") == "a 1-in-740 game for this team"


def _history(cur_pts, prior_pts=()):
    import pandas as pd
    rows = [(1, p, 10, 25) for p in prior_pts] + [(0, p, 10, 25) for p in cur_pts]
    return pd.DataFrame(rows, columns=["SEASON_IDX", "PTS", "FG3A", "FG3M"])


def test_stat_note_season_high_and_average():
    from outliers.detect import stat_note
    games = _history([11, 12, 0, 6, 4, 10, 13, 14], prior_pts=[30] * 10)
    assert stat_note(games, "PTS", 27, False, False, 3) == "season high, averaging 8.8 this season"


def test_stat_note_only_claims_career_high_inside_window():
    from outliers.detect import stat_note
    games = _history([10] * 6, prior_pts=[20] * 10)
    assert stat_note(games, "PTS", 31, False, True, 3).startswith("career high")
    assert stat_note(games, "PTS", 31, False, False, 3).startswith("highest in 3 seasons")


def test_stat_note_lows_and_early_season():
    from outliers.detect import stat_note
    assert stat_note(_history([20, 22, 25, 19, 30]), "PTS", 4, True, False, 3) == \
        "season low, averaging 23.2 this season"
    # Two games in: no season claims; fall back to last season's average.
    early = _history([20, 22], prior_pts=[18] * 12)
    assert stat_note(early, "PTS", 21, False, False, 3) == "averaged 18.0 last season"
    assert stat_note(early, "PTS", 35, False, False, 3) == "highest in 3 seasons, averaged 18.0 last season"
    import pandas as pd
    assert stat_note(pd.DataFrame(columns=["SEASON_IDX", "PTS"]), "PTS", 30, False, False, 3) == ""

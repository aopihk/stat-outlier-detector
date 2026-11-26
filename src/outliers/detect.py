"""Generate outlier candidates for one night of games."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .baselines import (
    PCT_PARTS,
    PlayerBaseline,
    build_league_context,
    build_team_context,
    team_rate,
    team_value,
)
from scipy import stats

from .scoring import count_z, empirical_z, pct_z
from .seasons import REGULAR, season_start_year, season_window

LABELS = {"PTS": "PTS", "REB": "REB", "OREB": "OREB", "AST": "AST", "STL": "STL",
          "BLK": "BLK", "TOV": "TOV", "FG3M": "3PM", "REB_MARGIN": "REB margin"}
PCT_LABELS = {"FG_PCT": "FG", "FG3_PCT": "3PT", "FT_PCT": "FT"}
NUMERIC = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
           "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]


@dataclass
class Candidate:
    kind: str           # personal | league | team | under
    entity: str         # "P:<id>" or "T:<id>"
    name: str
    team: str
    matchup: str
    stat: str
    value: float
    expected: float | None
    z: float
    reason: str
    line: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    context: list[str] = field(default_factory=list)
    brief: str = ""     # shorter reason used when this shows up as context on another item

    def __post_init__(self):
        self.brief = self.brief or self.reason

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class NightResult:
    date: date
    candidates: list[Candidate]
    game_ids: list[str]
    n_players: int
    notes: list[str] = field(default_factory=list)


def _prepare(df: pd.DataFrame, d: date, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split logs into (scored night, history before it) and add helper columns."""
    if df.empty:
        return df, df
    df = df[df["SEASON"].isin(seasons)].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["ONE"] = 1.0
    df["SEASON_IDX"] = df["SEASON"].map({s: i for i, s in enumerate(seasons)})
    ds = d.isoformat()
    night = df[df["GAME_DATE"] == ds]
    hist = df[(df["GAME_DATE"] < ds) & (df["SEASON_TYPE"] == REGULAR)]
    return night, hist.sort_values("GAME_DATE")


def add_team_columns(team: pd.DataFrame) -> pd.DataFrame:
    if team.empty:
        return team
    team = team.copy()
    for col in NUMERIC:
        team[col] = pd.to_numeric(team[col], errors="coerce").fillna(0)
    team["POSS"] = (team["FGA"] + 0.44 * team["FTA"] - team["OREB"] + team["TOV"]).clip(lower=60)
    opp = team[["GAME_ID", "TEAM_ID", "REB", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_ID", "REB": "OPP_REB", "PTS": "OPP_PTS"})
    team = team.merge(opp, on="GAME_ID")
    team = team[team["TEAM_ID"] != team["OPP_ID"]].copy()
    team["REB_MARGIN"] = team["REB"] - team["OPP_REB"]
    return team


def player_line(r) -> str:
    parts = [f"{int(r.PTS)} pts", f"{int(r.REB)} reb", f"{int(r.AST)} ast"]
    if r.STL >= 3:
        parts.append(f"{int(r.STL)} stl")
    if r.BLK >= 3:
        parts.append(f"{int(r.BLK)} blk")
    shooting = f"{int(r.FGM)}-{int(r.FGA)} FG"
    if r.FG3A:
        shooting += f", {int(r.FG3M)}-{int(r.FG3A)} 3PT"
    return f"{', '.join(parts)} · {shooting} · {int(r.MIN)} min"


def team_line(r) -> str:
    return (f"{int(r.PTS)}-{int(r.OPP_PTS)} {r.WL} · {int(r.FGM)}-{int(r.FGA)} FG, "
            f"{int(r.FG3M)}-{int(r.FG3A)} 3PT, {int(r.TOV)} TOV, {int(r.REB)} REB")


def _fmt_count(stat: str, x: float) -> str:
    return f"{int(x)} {LABELS.get(stat, stat)}"


def _fmt_pct(stat: str, makes: float, attempts: float) -> str:
    return f"{int(makes)}-{int(attempts)} {PCT_LABELS[stat]} ({makes / attempts:.0%})"


def odds(z: float) -> str:
    """A z-score as plain-language odds, e.g. "1-in-290"."""
    p = float(stats.norm.sf(abs(z)))
    if p < 1e-6:
        return "rarer than 1-in-a-million"
    n = 1 / p
    digits = len(str(int(n))) - 2
    n = round(n, -digits) if digits > 0 else round(n)
    return f"1-in-{int(n):,}"


def rarity(z: float, who: str) -> str:
    """e.g. "a 1-in-290 game for this player" or "a 1-in-160 low for this player"."""
    kind = "low" if z < 0 else "game"
    text = odds(z).replace("rarer than ", "rarer-than-")
    return f"a {text} {kind} for this {who}"


def _fmt_num(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def _fmt_share(p: float) -> str:
    pct = p * 100
    return f"{pct:.2g}%" if pct < 1 else f"{pct:.0f}%"


def milestones(r: dict, ms: dict) -> list[tuple[str, float]]:
    out = []
    for stat in ("PTS", "REB", "AST", "STL", "BLK", "FG3M"):
        for threshold, score in ms.get(stat, []):
            if r[stat] >= threshold:
                out.append((f"{int(r[stat])} {LABELS[stat]}", score))
                break
    cats = ["PTS", "REB", "AST", "STL", "BLK"]
    tens = sum(r[c] >= 10 for c in cats)
    triple = f"{int(r['PTS'])}/{int(r['REB'])}/{int(r['AST'])}"
    if tens >= 4:
        out.append((f"Quadruple-double ({triple})", ms["quadruple_double"]))
    elif tens >= 3:
        if r["PTS"] >= 40:
            out.append((f"40-point triple-double ({triple})", ms["triple_double_40"]))
        elif r["PTS"] >= 30:
            out.append((f"30-point triple-double ({triple})", ms["triple_double_30"]))
        else:
            out.append((f"Triple-double ({triple})", ms["triple_double"]))
    if all(r[c] >= 5 for c in cats):
        out.append(("5x5 (5+ in PTS/REB/AST/STL/BLK)", ms["five_by_five"]))
    if r["PTS"] >= 20 and r["REB"] >= 20:
        out.append((f"20-20 game ({int(r['PTS'])} PTS, {int(r['REB'])} REB)", ms["twenty_twenty"]))
    return out


def detect_night(d: date, player_logs: pd.DataFrame, team_logs: pd.DataFrame,
                 players: pd.DataFrame, cfg: dict) -> NightResult:
    seasons = season_window(d, cfg["baseline"]["seasons_back"])
    night_p, hist_p = _prepare(player_logs, d, seasons)
    if night_p.empty:
        return NightResult(d, [], [], 0)
    hist_p = hist_p[hist_p["MIN"] > 0]
    night_p = night_p[night_p["MIN"] > 0]

    cands = _player_candidates(d, night_p, hist_p, players, cfg)
    night_t, hist_t = _prepare(add_team_columns(team_logs), d, seasons)
    if not night_t.empty and not hist_t.empty:
        cands += _team_candidates(night_t, hist_t, cfg)
    return NightResult(d, cands, sorted(night_p["GAME_ID"].unique()), len(night_p))


def stat_note(games: pd.DataFrame, stat: str, x: float, low: bool, career_window: bool,
              n_seasons: int) -> str:
    """Plain-language context for a count stat, e.g. "season high, averaging 8.3 this season".

    `games` is the player's history before this game. A "career high" is only claimed when
    the whole career is inside the data window (`career_window`); otherwise the claim is
    limited to the seasons we have.
    """
    if games.empty:
        return ""
    cur = games[games["SEASON_IDX"] == 0][stat]
    parts = []
    if low:
        if len(cur) >= 5 and x < cur.min():
            parts.append("season low")
        elif len(cur) >= 5 and x == cur.min():
            parts.append("ties season low")
    else:
        if n_seasons > 1 and len(games) >= 10 and x > games[stat].max():
            parts.append("career high" if career_window else f"highest in {n_seasons} seasons")
        elif len(cur) >= 5 and x > cur.max():
            parts.append("season high")
        elif len(cur) >= 5 and x == cur.max():
            parts.append("ties season high")
    if len(cur) >= 3:
        parts.append(f"averaging {cur.mean():.1f} this season")
    else:
        prior = games[games["SEASON_IDX"] >= 1]
        last = prior[prior["SEASON_IDX"] == prior["SEASON_IDX"].min()][stat]
        if len(last) >= 10 and prior["SEASON_IDX"].min() == 1:
            parts.append(f"averaged {last.mean():.1f} last season")
    return ", ".join(parts)


def pct_note(games: pd.DataFrame, stat: str) -> str:
    """e.g. "28% 3PT this season (19-69)"."""
    m, a = PCT_PARTS[stat]
    cur = games[games["SEASON_IDX"] == 0] if not games.empty else games
    if cur.empty or cur[a].sum() < 10:
        return ""
    makes, attempts = int(cur[m].sum()), int(cur[a].sum())
    return f"{makes / attempts:.0%} {PCT_LABELS[stat]} this season ({makes}-{attempts})"


def _with_note(why: str, note: str) -> str:
    return f"{why}; {note}" if note else why


def _player_candidates(d, night, hist, players, cfg) -> list[Candidate]:
    b, s, u = cfg["baseline"], cfg["stats"], cfg["underperformance"]
    cap, dist = b["max_abs_z"], b["distribution"]
    ctx = build_league_context(hist, players, cfg)
    groups = {pid: g for pid, g in hist.groupby("PLAYER_ID")}
    empty = hist.iloc[0:0]
    from_year = {int(r.PLAYER_ID): r.FROM_YEAR for r in players.itertuples()}
    start_year = season_start_year(d)
    n_seasons = cfg["baseline"]["seasons_back"] + 1
    window_start = start_year - cfg["baseline"]["seasons_back"]

    out: list[Candidate] = []
    for r in night.itertuples(index=False):
        rd = r._asdict()
        pid = int(r.PLAYER_ID)
        games = groups.get(pid, empty)
        pb = PlayerBaseline(games, pid, ctx, cfg)
        fy = from_year.get(pid)
        career_window = fy is not None and not pd.isna(fy) and fy >= window_start
        em = pb.expected_minutes
        tags = []
        if from_year.get(pid) == start_year or (pb.n_prior == 0 and pb.n_current < 10):
            tags.append("Rookie" if from_year.get(pid) == start_year else "Thin history")
        early_exit = (em is not None and em >= u["early_exit_min_expected"]
                      and r.MIN < u["early_exit_fraction"] * em)
        if early_exit:
            tags.append(f"Early exit ({int(r.MIN)} of ~{em:.0f} min)")
        established = em is not None and em >= u["min_expected_minutes"]

        def add(kind, stat, value, expected, z, reason, brief=""):
            out.append(Candidate(kind, f"P:{pid}", r.PLAYER_NAME, r.TEAM_ABBREVIATION, r.MATCHUP,
                                 stat, float(value), expected, z, reason, player_line(r), list(tags),
                                 brief=brief))

        for stat in s["player_counts"]:
            exp = pb.count(stat, r.MIN)
            if exp is None:
                continue
            x = rd[stat]
            z = count_z(x, exp.mu, exp.alpha, dist, b["std_floor"][stat], cap)
            negative = stat in s["negative_stats"]
            if z > 0:
                if x < s["personal_min_value"].get(stat, 0) or (negative and not established):
                    continue
                kind = "under" if negative else "personal"
            else:
                if negative or not established or early_exit:
                    continue
                if exp.mu < u["min_expected"].get(stat, float("inf")):
                    continue
                kind = "under"
            add(kind, stat, x, exp.mu, z,
                f"{_fmt_count(stat, x)} vs {_fmt_num(exp.mu)} expected "
                f"({_with_note(rarity(z, 'player'), stat_note(games, stat, x, z < 0, career_window, n_seasons))})",
                f"{_fmt_count(stat, x)} vs {_fmt_num(exp.mu)} expected ({odds(z)})")

        for stat, min_att in s["player_pcts"].items():
            m, a = PCT_PARTS[stat]
            if rd[a] < min_att:
                continue
            p = pb.pct(stat, r.MIN)
            if p is None:
                continue
            z = pct_z(rd[m], rd[a], p, b["pct_concentration"], dist, cap)
            if z > 0:
                kind = "personal"
            elif stat in u["pct_stats"] and established and not early_exit:
                kind = "under"
            else:
                continue
            add(kind, stat, rd[m] / rd[a], p, z,
                f"{_fmt_pct(stat, rd[m], rd[a])} vs {p:.0%} expected "
                f"({_with_note(rarity(z, 'player'), pct_note(games, stat))})",
                f"{_fmt_pct(stat, rd[m], rd[a])} vs {p:.0%} expected ({odds(z)})")

        for stat in s["league_counts"]:
            z, p = empirical_z(rd[stat], ctx.sorted_values[stat], "high", cap)
            if z > 0:
                add("league", f"LEAGUE:{stat}", rd[stat], None, z,
                    f"{_fmt_count(stat, rd[stat])} (top {_fmt_share(p)} of player games"
                    f"{'; ' + n if (n := stat_note(games, stat, rd[stat], False, career_window, n_seasons)) else ''})",
                    f"{_fmt_count(stat, rd[stat])} (top {_fmt_share(p)} of player games)")

        for name, score in milestones(rd, cfg["milestones"]):
            add("league", f"MILESTONE:{name}", 0, None, score, name)
    return out


def _team_candidates(night, hist, cfg) -> list[Candidate]:
    b, s = cfg["baseline"], cfg["stats"]
    cap, dist = b["max_abs_z"], b["distribution"]
    tctx = build_team_context(hist, cfg)
    groups = {tid: g for tid, g in hist.groupby("TEAM_ID")}
    empty = hist.iloc[0:0]
    since = hist["SEASON"].min()

    out: list[Candidate] = []
    for r in night.itertuples(index=False):
        rd = r._asdict()
        games = groups.get(r.TEAM_ID, empty)

        def add(stat, value, expected, z_own, z_lg, p_lg, value_str):
            if abs(z_own) >= abs(z_lg):
                z = z_own
                why = f"vs {_fmt_expected(stat, expected)} usual, {rarity(z_own, 'team')}"
                brief = f"vs {_fmt_expected(stat, expected)} usual, {odds(z_own)}"
            else:
                side = "top" if z_lg > 0 else "bottom"
                z = z_lg
                why = brief = f"{side} {_fmt_share(p_lg)} of team games since {since}"
            out.append(Candidate("team", f"T:{r.TEAM_ID}", r.TEAM_ABBREVIATION, r.TEAM_ABBREVIATION,
                                 r.MATCHUP, stat, float(value), expected, z,
                                 f"{value_str} ({why})", team_line(r), brief=f"{value_str} ({brief})"))

        for stat in s["team_counts"]:
            rate = team_rate(games, stat, tctx, cfg)
            mu = rate * r.POSS if rate is not None else None
            z_own = count_z(rd[stat], mu, tctx.alpha[stat], dist, 1.0, cap) if mu is not None else 0.0
            z_lg, p_lg = empirical_z(team_value(rd, stat), tctx.sorted_values[stat], "both", cap)
            add(stat, rd[stat], mu, z_own, z_lg, p_lg, _fmt_count(stat, rd[stat]))

        for stat, min_att in s["team_pcts"].items():
            m, a = PCT_PARTS[stat]
            if rd[a] < min_att:
                continue
            p = team_rate(games, stat, tctx, cfg)
            z_own = pct_z(rd[m], rd[a], p, b["pct_concentration"], dist, cap) if p is not None else 0.0
            z_lg, p_lg = empirical_z(rd[m] / rd[a], tctx.sorted_values[stat], "both", cap)
            add(stat, rd[m] / rd[a], p, z_own, z_lg, p_lg, _fmt_pct(stat, rd[m], rd[a]))

        for stat in s["team_margins"]:
            z_lg, p_lg = empirical_z(rd[stat], tctx.sorted_values[stat], "both", cap)
            add(stat, rd[stat], None, 0.0, z_lg, p_lg, f"{int(rd[stat]):+d} {LABELS.get(stat, stat)}")
    return out


def _fmt_expected(stat: str, expected) -> str:
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        return "n/a"
    return f"{expected:.0%}" if stat in PCT_PARTS else _fmt_num(expected)

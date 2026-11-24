"""Multi-season baseline model (PLAN.md section 3a).

Every baseline is a *rate*: a stat per unit of exposure (minutes, games,
possessions or shot attempts). Rates are blended across this season, prior
seasons and a fill prior (cohort or league average) according to
`baseline.method`, then scaled to the exposure expected for the game.

Game frames passed in here must already be limited to games before the scored
night and carry a SEASON_IDX column: 0 = the scored night's season, 1 = last
season, and so on.
"""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

PCT_PARTS = {"FG_PCT": ("FGM", "FGA"), "FG3_PCT": ("FG3M", "FG3A"), "FT_PCT": ("FTM", "FTA")}
ALPHA_MIN, ALPHA_MAX = 1e-3, 3.0


def recency_weights(n: int, half_life: float) -> np.ndarray:
    """Weights for n games ordered oldest -> newest; the newest gets 1.0."""
    if n == 0:
        return np.zeros(0)
    if not half_life:
        return np.ones(n)
    ages = np.arange(n - 1, -1, -1)
    return 0.5 ** (ages / half_life)


def _season_weights(idx: pd.Series, decay: list[float]) -> np.ndarray:
    return np.array([decay[i - 1] if 0 < i <= len(decay) else 0.0 for i in idx], dtype=float)


def blend_rate(
    games: pd.DataFrame,
    x: str,
    e: str,
    k: float,
    fill: float | None,
    fill_k: float,
    b: dict,
) -> float | None:
    """Blended rate of column `x` per unit of column `e`.

    k      prior strength for method=bayesian, in units of `e`.
    fill   cohort/league rate used when a player's own history is thin
           (None means no fill: thin_history_prior = "none").
    fill_k strength of the fill prior for the non-bayesian methods.
    """
    method = b["method"]
    window = games[games["SEASON_IDX"] <= b["seasons_back"]]
    cur = window[window["SEASON_IDX"] == 0]
    if fill is None:
        own_games = len(cur) if method == "season_only" else len(window)
        if own_games < b["min_games_own"]:
            return None

    if method == "bayesian":
        prior = window[window["SEASON_IDX"] >= 1]
        w = _season_weights(prior["SEASON_IDX"], b["season_decay"])
        m_prior = float((w * prior[e]).sum())
        s_prior = float((w * prior[x]).sum())
        if fill is not None:
            # Prior seasons, themselves shrunk toward the fill, with a fixed strength k.
            prior_rate, strength = (s_prior + k * fill) / (m_prior + k), k
        elif m_prior > 0:
            prior_rate, strength = s_prior / m_prior, min(k, m_prior)
        else:
            prior_rate, strength = 0.0, 0.0
        wr = recency_weights(len(cur), b["recency_half_life_games"])
        m_cur = float((wr * cur[e]).sum())
        s_cur = float((wr * cur[x]).sum())
        if strength + m_cur <= 0:
            return None
        return (strength * prior_rate + s_cur) / (strength + m_cur)

    if method == "exp_decay":
        w = recency_weights(len(window), b["exp_decay_half_life_games"])
        exposure, total = float((w * window[e]).sum()), float((w * window[x]).sum())
    elif method == "fixed_weights":
        fw = b["fixed_weights"]
        num = den = exposure = 0.0
        for i, grp in window.groupby("SEASON_IDX"):
            ge = float(grp[e].sum())
            if i < len(fw) and ge > 0:
                num += fw[i] * float(grp[x].sum()) / ge
                den += fw[i]
                exposure += ge
        total = (num / den) * exposure if den else 0.0
    elif method == "season_only":
        exposure, total = float(cur[e].sum()), float(cur[x].sum())
    else:
        raise ValueError(f"Unknown baseline.method {method!r}")

    fk = fill_k if fill is not None else 0.0
    if exposure + fk <= 0:
        return None
    return (total + fk * (fill or 0.0)) / (exposure + fk)


def pos_group(position: str | None) -> str:
    """G / F / C from listings like "G-F"; X if unknown."""
    if position and position[0] in "GFC":
        return position[0]
    return "X"


def minutes_tier(mpg: float, tiers: list[float]) -> int:
    return bisect_right(tiers, mpg)


def _exposure_col(b: dict) -> str:
    return "MIN" if b["normalization"] == "per_minute" else "ONE"


def _pooled_alpha(df: pd.DataFrame, keys: list[str], stat: str, e: str, scale_col: str) -> float:
    """Method-of-moments negative binomial overdispersion, pooled over groups.

    Each group (e.g. a player-season) gets its own rate; expected value per game
    is rate * scale_col. alpha = sum((x - e)^2 - e) / sum(e^2).
    """
    g = df.groupby(keys)
    rate = g[stat].transform("sum") / g[e].transform("sum").replace(0, np.nan)
    expected = (rate * df[scale_col]).fillna(0)
    mask = g[stat].transform("size") >= 10
    resid = (df[stat] - expected) ** 2 - expected
    den = float((expected[mask] ** 2).sum())
    if den <= 0:
        return 0.1
    return float(np.clip(resid[mask].sum() / den, ALPHA_MIN, ALPHA_MAX))


@dataclass
class LeagueContext:
    """League-level and cohort-level priors built from historical player games."""

    cohort: dict
    league: dict
    alpha: dict
    pos: dict
    sorted_values: dict


def build_league_context(hist: pd.DataFrame, players: pd.DataFrame, cfg: dict) -> LeagueContext:
    b = cfg["baseline"]
    counts = cfg["stats"]["player_counts"]
    e = _exposure_col(b)
    pos = {int(r.PLAYER_ID): pos_group(r.POSITION) for r in players.itertuples()}

    h = hist.copy()
    h["GROUP"] = h["PLAYER_ID"].map(pos).fillna("X")
    mpg = h.groupby(["PLAYER_ID", "SEASON"])["MIN"].transform("mean")
    h["TIER"] = [minutes_tier(v, b["minutes_tiers"]) for v in mpg]
    h["SEASON_MPG"] = mpg

    pct_cols = [c for parts in PCT_PARTS.values() for c in parts]
    sum_cols = sorted(set(counts + pct_cols + [e]))

    def rates(sums: pd.Series) -> dict:
        out = {s: sums[s] / sums[e] if sums[e] > 0 else 0.0 for s in counts}
        for name, (m, a) in PCT_PARTS.items():
            out[name] = sums[m] / sums[a] if sums[a] > 0 else 0.45
        return out

    cohort = {key: rates(row) for key, row in h.groupby(["GROUP", "TIER"])[sum_cols].sum().iterrows()}
    league = rates(h[sum_cols].sum())

    if b["normalization"] == "per_game":
        scale = "ONE"
    elif b["minutes_basis"] == "actual":
        scale = "MIN"
    else:
        scale = "SEASON_MPG"
    # Overdispersion differs a lot by role (bench lines are far noisier), so it's estimated per minutes tier.
    alpha = {"league": {s: _pooled_alpha(h, ["PLAYER_ID", "SEASON"], s, e, scale) for s in counts}}
    for tier, g in h.groupby("TIER"):
        alpha[tier] = {s: _pooled_alpha(g, ["PLAYER_ID", "SEASON"], s, e, scale) for s in counts}

    sorted_values = {s: np.sort(h[s].to_numpy(dtype=float)) for s in cfg["stats"]["league_counts"]}
    return LeagueContext(cohort=cohort, league=league, alpha=alpha, pos=pos, sorted_values=sorted_values)


@dataclass
class CountExpectation:
    mu: float
    alpha: float


class PlayerBaseline:
    """Expectations for one player going into one game."""

    def __init__(self, games: pd.DataFrame, player_id: int, ctx: LeagueContext, cfg: dict):
        self.games = games
        self.ctx = ctx
        self.cfg = cfg
        self.b = cfg["baseline"]
        self.group = ctx.pos.get(int(player_id), "X")
        self.n_current = int((games["SEASON_IDX"] == 0).sum()) if len(games) else 0
        self.n_prior = len(games) - self.n_current
        self.expected_minutes = self._expected_minutes()

    def _expected_minutes(self) -> float | None:
        if not len(self.games):
            return None
        cur = self.games[self.games["SEASON_IDX"] == 0]
        if len(cur):
            window = self.b["expected_minutes_window"]
            recent = float(cur["MIN"].tail(window).mean())
            season = float(cur["MIN"].mean())
            w = self.b["expected_minutes_recent_weight"] if len(cur) > window else 1.0
            return w * recent + (1 - w) * season
        last = self.games[self.games["SEASON_IDX"] == self.games["SEASON_IDX"].min()]
        return float(last["MIN"].mean())

    def tier(self, actual_minutes: float) -> int:
        mpg = self.expected_minutes if self.expected_minutes is not None else actual_minutes
        return minutes_tier(mpg, self.b["minutes_tiers"])

    def _fill(self, stat: str, actual_minutes: float) -> float | None:
        mode = self.b["thin_history_prior"]
        if mode == "none":
            return None
        if mode == "cohort":
            cohort = self.ctx.cohort.get((self.group, self.tier(actual_minutes)))
            if cohort is not None:
                return cohort[stat]
        return self.ctx.league[stat]

    def count(self, stat: str, actual_minutes: float) -> CountExpectation | None:
        b = self.b
        per_game = b["normalization"] == "per_game"
        e = "ONE" if per_game else "MIN"
        k = b["prior_strength_minutes"][stat]
        fill_k = b["fill_strength_minutes"]
        if per_game:
            k, fill_k = k / b["per_game_minutes_equiv"], fill_k / b["per_game_minutes_equiv"]
        rate = blend_rate(self.games, stat, e, k, self._fill(stat, actual_minutes), fill_k, b)
        if rate is None:
            return None
        if per_game:
            mu = rate
        elif b["minutes_basis"] == "actual" or self.expected_minutes is None:
            mu = rate * actual_minutes
        else:
            mu = rate * self.expected_minutes
        return CountExpectation(mu=mu, alpha=self._alpha(stat, e, actual_minutes))

    def _alpha(self, stat: str, e: str, actual_minutes: float) -> float:
        tier_alphas = self.ctx.alpha.get(self.tier(actual_minutes), self.ctx.alpha["league"])
        prior_alpha = tier_alphas[stat]
        g = self.games[self.games["SEASON_IDX"] <= self.b["seasons_back"]]
        if len(g) < 3:
            return prior_alpha
        grp = g.groupby("SEASON_IDX")
        rate = grp[stat].transform("sum") / grp[e].transform("sum").replace(0, np.nan)
        if e == "ONE":
            scale = g["ONE"]
        elif self.b["minutes_basis"] == "actual":
            scale = g["MIN"]
        else:
            scale = grp["MIN"].transform("mean")
        expected = (rate * scale).fillna(0)
        w = np.where(g["SEASON_IDX"] == 0, 1.0, _season_weights(g["SEASON_IDX"], self.b["season_decay"]))
        den = float((w * expected ** 2).sum())
        if den <= 0:
            return prior_alpha
        own = float((w * ((g[stat] - expected) ** 2 - expected)).sum()) / den
        n_eff, n0 = float(w.sum()), self.b["dispersion_prior_games"]
        return float(np.clip((n_eff * own + n0 * prior_alpha) / (n_eff + n0), ALPHA_MIN, ALPHA_MAX))

    def pct(self, stat: str, actual_minutes: float) -> float | None:
        makes, attempts = PCT_PARTS[stat]
        b = self.b
        return blend_rate(
            self.games, makes, attempts,
            b["prior_strength_attempts"][stat],
            self._fill(stat, actual_minutes),
            b["fill_strength_attempts"], b,
        )


# --- Teams -------------------------------------------------------------------

@dataclass
class TeamContext:
    league: dict
    alpha: dict
    per_game: dict
    sorted_values: dict


def team_value(row, stat: str) -> float:
    """Value used for league-wide comparison: per 100 possessions for counts, raw otherwise."""
    if stat in PCT_PARTS:
        m, a = PCT_PARTS[stat]
        return row[m] / row[a] if row[a] else np.nan
    if stat.endswith("_MARGIN"):
        return row[stat]
    return row[stat] / row["POSS"] * 100


def build_team_context(hist: pd.DataFrame, cfg: dict) -> TeamContext:
    s = cfg["stats"]
    counts, pcts, margins = s["team_counts"], s["team_pcts"], s["team_margins"]
    league, per_game = {}, {"POSS": float(hist["POSS"].mean())}
    for stat in counts:
        league[stat] = float(hist[stat].sum() / hist["POSS"].sum())
    for stat in pcts:
        m, a = PCT_PARTS[stat]
        league[stat] = float(hist[m].sum() / hist[a].sum())
        per_game[a] = float(hist[a].mean())
    alpha = {stat: _pooled_alpha(hist, ["TEAM_ID", "SEASON"], stat, "POSS", "POSS") for stat in counts}

    sorted_values = {}
    for stat in counts:
        sorted_values[stat] = np.sort((hist[stat] / hist["POSS"] * 100).to_numpy(dtype=float))
    for stat, min_att in pcts.items():
        m, a = PCT_PARTS[stat]
        ok = hist[hist[a] >= min_att]
        sorted_values[stat] = np.sort((ok[m] / ok[a]).to_numpy(dtype=float))
    for stat in margins:
        sorted_values[stat] = np.sort(hist[stat].to_numpy(dtype=float))
    return TeamContext(league=league, alpha=alpha, per_game=per_game, sorted_values=sorted_values)


def team_rate(games: pd.DataFrame, stat: str, tctx: TeamContext, cfg: dict) -> float | None:
    """Blended per-possession rate (counts) or make rate (percentages) for a team."""
    b = dict(cfg["baseline"], min_games_own=0)
    k_games = cfg["team"]["prior_strength_games"]
    if stat in PCT_PARTS:
        m, a = PCT_PARTS[stat]
        k = k_games * tctx.per_game[a]
        return blend_rate(games, m, a, k, tctx.league[stat], k * 0.2, b)
    k = k_games * tctx.per_game["POSS"]
    return blend_rate(games, stat, "POSS", k, tctx.league[stat], k * 0.2, b)

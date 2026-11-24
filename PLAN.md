# NBA Stat Outlier Detector — Plan

## Goal

Every morning, a Python script pulls last night's NBA games and outputs a short ranked list of the **biggest** statistical outliers. That covers four kinds:

1. **Personal outliers**: a player well above (or below) their own norm, e.g. a 9 PPG bench player scoring 31.
2. **League-wide outliers**: lines that are rare by league standards whatever the player's average, e.g. 50 points, 20 rebounds, a triple-double with 20+ assists.
3. **Team outliers**: unusual team stat lines, e.g. 8% from three, 30 turnovers, +45 rebound margin.
4. **Underperformance**: established players with unusually bad nights, e.g. an All-Star going 2-for-18.

The output is meant to be selective: a **top-N list (default 10)** across all four kinds, ranked by one comparable "surprise" score. It is not meant to dump every flagged stat.

## Decisions (confirmed)

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| Delivery | Terminal output plus a saved local report file |
| Scheduling | Local Mac using `launchd` (preferred over cron on macOS because it runs missed jobs after the Mac wakes) |
| Outlier types | All four above, filtered to only the biggest |
| Stats | Box-score stats only (no advanced stats in v1) |
| Reports | Saved locally and auto-committed to git |
| Notifications | None |

---

## 1. Data source

**Primary: [`nba_api`](https://github.com/swar/nba_api)** (unofficial wrapper around stats.nba.com). It's free and needs no API key.

Endpoints needed:
- `ScoreboardV2` / `LeagueGameFinder`: list of games played on a given date.
- `BoxScoreTraditionalV3`: per-player and per-team box scores for each game.
- `PlayerGameLogs` / `LeagueGameLog`: season game logs, used to build each player's baseline and the league distributions.

Known risks and mitigations:
- stats.nba.com sometimes throttles or blocks requests. Mitigations: send proper headers, wait about 0.6–1s between calls, retry with backoff, and set a timeout.
- **Fallback source:** ESPN's public scoreboard/summary JSON, behind a small "data provider" interface so the source can be swapped without touching the analysis code.

"Last night" is defined as the previous calendar date in **US/Eastern** (NBA schedule time), computed when the script runs. A `--date YYYY-MM-DD` flag allows backfills and testing.

## 2. Local data store

Pulling full season logs every morning is slow and invites rate limits, so data is cached locally:

- **SQLite** database (`data/nba.db`) with tables such as `player_games`, `team_games`, `players`, `runs`.
- **First run:** backfill all player and team game logs for the current season plus the previous 2 seasons. `LeagueGameLog` returns a whole season in one call, so this takes only a handful of requests.
- **Daily run:** fetch only last night's box scores and append them. The DB becomes the source of truth for baselines.
- This also makes the script fast and testable offline.

## 3. Defining "outlier"

### Stats tracked

- **Player:** PTS, REB (OREB/DREB), AST, STL, BLK, TOV, 3PM, FGM/FGA, FG%, 3P%, FT%, PF, plus-minus, minutes. Box-score stats only in v1; advanced stats (usage, TS%, game score) are deferred.
- **Team:** the same counting stats, shooting percentages, pace/possessions, turnovers, rebound margin, points in a quarter (optional, needs play-by-play, so v2).

### Scoring method (one comparable "surprise score")

Each candidate gets a score so the four outlier kinds can be ranked together.

**A. Personal outliers (player vs. self)**
- Baseline = the player's **expected line** from the multi-season baseline model (see section 3a), built only from games *before* the night being scored.
- Score = how surprising the actual value is under that expectation, expressed as a z-score (see "Surprise calculation" in 3a).
- **Robustness tweaks** (needed so the list isn't junk):
  - Std floor, so a player who always gets 0 blocks doesn't post a z of 15 for one block. Optional if the count model is used, since it handles this naturally.
  - **Rank on raw values.** Rates are only used to build expectations. Fans care that a player scored 30, not that they were efficient in 4 minutes.
  - Minimum-volume rules for percentage stats (e.g. FG% only counts with 10+ FGA).

**B. League-wide outliers**
- Compare the stat line against the distribution of all player-games this season (and/or a historical reference).
- Score = percentile turned into an equivalent z, so it lives on the same scale as A.
- Also hard-coded **milestone flags** that rank highly no matter what: 40/50+ pts, 20+ reb, 15+ ast, triple-doubles, 5x5s, 10+ threes, 7+ blk/stl.

**C. Team outliers**
- Compare each team's line against (a) the team's own season baseline and (b) the league-wide team-game distribution. Use the larger of the two scores.
- Same z-scale.

**D. Underperformance**
- Same as A, but negative z-scores.
- **Only for established players** (e.g. ≥ 25 MPG and a top-N usage or scoring rank), so a deep bench player's quiet night doesn't count.
- Exclude games where the player left early (injury, ejection). Detect this as minutes well below normal; such games get an "early exit" note instead of being scored.

## 3a. Baseline model (multi-season, switchable)

A season-to-date average doesn't work in November: after 10 games a player's mean and std are noisy, and every warm night looks like a 3σ event. The baseline therefore combines **this season + the last 2 seasons**, normalizes per minute, and lets the current season gradually take over as it builds up.

### Default approach: Bayesian blend, per-minute

For each player × stat:

1. **Prior from past seasons.** Take the player's per-minute rate (e.g. points per 36) from the last 2 seasons, weighted by minutes played and a season decay. Defaults: last season 1.0, two seasons ago 0.5. A player who missed most of last season gets a weaker prior automatically, because it's minutes-weighted.
2. **Current-season evidence.** The player's per-minute rate over this season's games before the scored night.
3. **Blend (shrinkage).**
   `rate = (k · prior_rate + m · current_rate) / (k + m)`
   - `m` = minutes played this season.
   - `k` = **prior strength in minutes**, set per stat because stats stabilize at different speeds. Rebound and assist rates settle quickly; 3P% and steals are noisy. Starting values (tunable): REB/AST ≈ 300 min, PTS ≈ 400, 3PM/STL/BLK ≈ 800. In practice the prior dominates for the first few weeks and fades by around midseason.
   - Optional within-season recency weighting (e.g. a 15-game half-life), so a real role change this season counts more than October.
4. **Scale to expected minutes.** `expected_stat = rate × expected_minutes`. Expected minutes is a blend of the last ~5 games and season minutes, so a new role shows up quickly.
   - **Minutes basis option:** `expected` (default) or `actual`.
     - `expected`: a bench player who suddenly plays 40 minutes and scores 30 is a big outlier. "A breakout is a breakout," and the jump in minutes is part of the story.
     - `actual`: asks "given the minutes they actually played, was this unusual?" This isolates efficiency and filters out blowout-garbage-time effects.
   - Underperformance always uses `expected`, after early exits are excluded.
5. **Shooting percentages** use the same idea with makes/attempts (a beta-binomial prior from past seasons plus current makes and attempts), not per-minute rates.

### Surprise calculation (switchable)

- **`count` (default):** treat counting stats as a count distribution (negative binomial, with player-level overdispersion blended across seasons the same way as the mean). Compute the tail probability of seeing a value at least this extreme, then convert it to an equivalent z. This handles low-count stats (blocks, steals, threes) far better than a normal approximation, with no hand-tuned std floors.
- **`normal`:** `(value − expected) / std`, with the std blended across seasons and floored. Simpler and easier to explain, and a useful comparison point.

### Players with thin or no NBA history (rookies, call-ups, returning players)

A breakout is a breakout, so these players are **never excluded**. They just get a different prior:

- **Cohort prior:** the average per-minute rates (and dispersion) of players in the same **position group × minutes tier** over the last 2 seasons. Example: "rotation wings playing 15–25 MPG." The player's own games take over through the same `k` shrinkage.
- The cohort prior is wider (more uncertain) than a personal prior, so a rookie's 25-point night scores a bit lower than the same jump from a veteran with a long, stable history. It still ranks; it's just honest about the uncertainty.
- A **"Rookie"** tag in the report, since rookie breakouts are a story on their own.
- The league-wide check (B) runs for everyone regardless, so an absurd rookie line surfaces either way.

### Switchable options (`config.toml` → `[baseline]`)

Every option can be set in config and overridden per run (e.g. `--baseline-method fixed_weights`), so methods can be compared in backtests.

| Key | Options | Default | Notes |
|---|---|---|---|
| `method` | `bayesian`, `exp_decay`, `fixed_weights`, `season_only` | `bayesian` | `exp_decay`: game-level half-life (e.g. 25 games) across season boundaries. `fixed_weights`: e.g. 0.6 / 0.3 / 0.1 by season. `season_only`: current season only, for comparison. |
| `seasons_back` | `0`–`3` | `2` | How many prior seasons feed the prior. |
| `season_decay` | list of weights | `[1.0, 0.5]` | Weight for last season, two seasons ago. |
| `prior_strength_minutes` | per-stat table | see above | The `k` values. Higher = trust history longer. |
| `recency_half_life_games` | int or `off` | `15` | Within-season recency weighting. |
| `normalization` | `per_minute`, `per_game` | `per_minute` | `per_game` ignores role and minute changes. |
| `minutes_basis` | `expected`, `actual` | `expected` | See step 4. |
| `expected_minutes_window` | int | `5` | Recent games used for expected minutes. |
| `distribution` | `count`, `normal` | `count` | Surprise calculation. |
| `thin_history_prior` | `cohort`, `league`, `none` | `cohort` | `league`: league-average prior. `none`: skip personal scoring until `min_games`. |
| `min_games_own` | int | `0` with a prior, `8` with `none` | Games before a player's own data is used alone. |

**Backtest comparison:** `--range START END --compare-baselines bayesian,season_only,exp_decay` runs each method over the same past nights and prints their top-N lists side by side, plus summary stats (items per night, how often the same player repeats). This is how the defaults get tuned, especially on October–November nights.

**Team baselines** reuse the same machinery: team season + last 2 seasons, Bayesian blend per 100 possessions (pace-normalized), with a lower prior weight because rosters turn over between seasons.

### Picking "only the biggest"

1. Generate all candidates (player × stat, team × stat).
2. **Deduplicate per player/team:** keep their single most extreme stat and attach the others as context (e.g. "38 pts (+4.1σ), also 7 threes").
3. Apply a minimum threshold, around |z| ≥ 3.0 (configurable).
4. Rank by score with a small weight per kind (configurable) so one kind can't flood the list.
5. Cap at top N (default 10), with optional per-kind caps (e.g. max 3 team outliers).
6. If nothing clears the threshold, show the top 3 anyway and label them "quiet night."

All thresholds and weights live in a `config.toml` so they can be tuned without code changes.

## 4. Output

**Terminal:** a clean ranked table, using `rich` for color and formatting:

```
NBA Outliers — Sun, Nov 23 2026 (9 games)

 #  Type      Player / Team         Line                       Why
 1  Personal  J. Doe (BKN)          34 PTS, 12-15 FG           +4.6σ vs 11.2 avg
 2  League    A. Star (DEN)         28/21/14                   Triple-double, 99.9th pct REB
 3  Team      CHA                   4-38 3PT (10.5%)           Lowest 3P% this season
 4  Under     B. Guard (PHX)        6 PTS, 2-17 FG             −3.4σ vs 27.8 avg
 ...
```

**Saved report:** `reports/YYYY-MM-DD.md` (human-readable) plus `reports/YYYY-MM-DD.json` (machine-readable, useful if a Slack/email/dashboard layer is added later). Both are **committed to git** (see section 5).

## 5. Scheduling (macOS)

- A `launchd` agent at `~/Library/LaunchAgents/com.<you>.nba-outliers.plist`.
- Runs daily at about **8:00 AM local**. West Coast late games finish around 1:30 AM ET, so box scores are final by then.
- `launchd` runs a missed job when the Mac wakes, which cron doesn't.
- Output goes to `logs/` (stdout/stderr).
- **Auto-commit reports:** after writing the report, the script commits only the new `reports/` files with the message `Outliers for YYYY-MM-DD`. Push is off by default (`[git] push = false` in config). A failed commit is logged but never fails the run. A `--no-commit` flag disables it for manual and test runs, and backtests never commit.
- **Idempotent:** if today's report already exists, skip unless run with `--force`.
- **No-games handling:** off days, All-Star break, and the off-season exit cleanly with a "no games" message.

## 6. Project structure

```
stat-outlier-detector/
├── PLAN.md
├── README.md
├── pyproject.toml            # deps: nba_api, pandas, numpy, scipy, rich, tomli (if <3.11)
├── config.toml               # thresholds, weights, top-N, stat lists
├── src/outliers/
│   ├── cli.py                # entry point: --date, --top, --force, --backfill
│   ├── providers/
│   │   ├── base.py           # DataProvider interface
│   │   ├── nba_api_provider.py
│   │   └── espn_provider.py  # fallback
│   ├── store.py              # SQLite read/write
│   ├── baselines.py          # multi-season Bayesian/decay/fixed-weight baselines, cohort priors
│   ├── scoring.py            # count-model and normal surprise→z, floors, milestone flags
│   ├── ranking.py            # dedupe, thresholds, top-N selection
│   └── report.py             # terminal + md/json output
├── scripts/
│   └── install_launchd.sh    # writes and loads the plist
├── data/                     # nba.db (gitignored)
├── reports/                  # daily outputs (committed to git)
├── logs/                     # gitignored
└── tests/
    ├── fixtures/             # saved box scores for offline tests
    └── test_scoring.py ...
```

## 7. Implementation phases

1. **Data plumbing.** Provider interface, `nba_api` provider, SQLite store, season backfill command. *Done when:* `--backfill` fills the DB and `--date` fetches one night's box scores.
2. **Personal outliers.** Baseline model (3a): start with `bayesian` + `per_minute` + `normal`, then add the `count` distribution, cohort priors, and the alternative methods behind config switches. Add volume rules. *Done when:* outputs for a few past nights look sensible by eye.
3. **League-wide and milestones.** Distributions, percentile→z conversion, milestone flags.
4. **Team outliers.** Team baselines and league team distribution.
5. **Underperformance.** Established-player filter and early-exit detection.
6. **Ranking and report.** Dedupe, threshold, weighting, top-N, terminal and file output.
7. **Scheduling.** launchd install script, logging, idempotency, no-games handling, auto-commit of reports.
8. **Hardening.** Retries/backoff, ESPN fallback, tests built from saved fixture nights, tuning `config.toml` against 2–3 weeks of backfilled nights.

## 8. Testing and tuning

- **Unit tests** for baseline and scoring math (shrinkage blend, per-minute scaling, negative-binomial tail→z, beta-binomial for %s, percentile→z). Each baseline `method` gets known-answer tests.
- **Fixture tests:** saved box scores from known nights (e.g. a 60-point game) must show that game at #1.
- **Backtest mode:** `--range START END` runs the detector over past dates without saving, to tune thresholds so a typical night gives about 5–10 meaningful items.

## 9. Edge cases to handle

- Season start (few games): handled by the Bayesian blend (3a). Prior seasons dominate early on.
- Traded players: baseline follows the player, not the team. Per-minute normalization and expected minutes absorb the role change.
- Rookies / two-way / call-ups with no history: cohort prior (3a), tagged "Rookie" in the report.
- Players returning from a lost season: the minutes-weighted prior falls back to the older season, or to the cohort prior if there's too little data.
- Overtime games: optionally adjust counting stats per 48 minutes for team outliers.
- Postponed or suspended games: skip any game not marked Final.
- Playoffs and Play-In: use a separate baseline flag. NBA Cup games count as regular season.
- Data corrections: stats.nba.com occasionally revises box scores. Re-fetching the prior night overwrites rows.

## 10. Open questions (non-blocking; defaults noted)

1. ~~**Top-N size.**~~ Resolved: 10.
2. ~~**Baseline window.**~~ Resolved: multi-season Bayesian blend with switchable options (section 3a).
3. ~~**Advanced stats.**~~ Resolved: box-score stats only for v1.
4. ~~**Commit reports to git?**~~ Resolved: yes, auto-committed locally after each run (push off by default).
5. ~~**Notification.**~~ Resolved: no macOS notification.

## 11. Implementation notes (v1, 2026-11-24)

Built: phases 1–7, plus retries/backoff and tests from phase 8. Differences from the plan above:

- **Data:** `LeagueGameLog` (one call per season) supplies both history *and* last night's box scores, so per-game `BoxScoreTraditionalV3` calls aren't needed. `ScoreboardV3` is used only to check that every Final game made it into the stats feed. A daily run makes about 4 requests.
- **CLI:** subcommands instead of flags: `nba-outliers [run]`, `nba-outliers backfill`, `nba-outliers backtest START END [--compare-baselines ...]`.
- **Overdispersion prior** is estimated per minutes tier rather than league-wide. Bench players' lines are about 10× noisier relative to their mean than stars', and one league value overstated star surprises and understated bench ones.
- **Module layout:** `sync.py` (fetching), `detect.py` (candidates), `gitops.py` (auto-commit) replace some of the names in section 6.
- **Not built yet:** ESPN fallback provider (the `DataProvider` interface is in place). Team stats use the same baseline switches as players.

**Backtest, 2026-10-25 → 2026-11-23 (29 nights), bayesian defaults:**

| `min_score` | Avg items/night | Notes |
|---|---|---|
| 3.0 | 2.3 | 4 quiet nights; team items almost never appear (1 in 29 nights) |
| 2.75 | 3.9 | |
| 2.5 (current) | 6.7 → 6.2 after tuning | Closest to the 5–10 target in section 8 |

On opening night (2026-10-20/21), `season_only` flagged items like "Trae Young 13 AST vs 3.4 expected" (cohort fill only), while `bayesian` used prior seasons and didn't. This confirms the multi-season prior is doing its job.

**Tuning after lowering to 2.5:** steal/block minimum raised to 5 (4-steal nights were crowding the list); each stat appears once per item; plain triple-doubles demoted to context (score 2.0). Result: 6.2 items/night, no quiet nights. League-wide weight then lowered to 0.9, so league-wide items need about the top 0.3% of player games or a milestone. Result: 5.0 items/night; league-wide items fell from 76 to 29 over the month; Dončić and Wembanyama went from 8 appearances each to 4 (all 45+ pts or 6+ blk). Dropped: 37 mostly star-does-star-thing lines (40–42 pt games, 5 stl/blk, 8 threes, 13–14 ast). Gained: surprise lines such as Buddy Hield 33 pts vs 9.2 expected. Kept, because the goal is surprise, not volume.

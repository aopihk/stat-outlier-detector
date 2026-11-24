# NBA Stat Outlier Detector

Every morning, finds the biggest statistical outliers from last night's NBA games:
players far above (or below) their own norm, lines that are rare league-wide,
milestones, and unusual team stat lines. Ranked into one top-10 list.

See [PLAN.md](PLAN.md) for the design. The baseline model is described in section 3a.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/nba-outliers backfill        # ~20s: this season + 2 prior seasons into data/nba.db
```

## Usage

```sh
.venv/bin/nba-outliers                         # last night's report (prints, saves, commits)
.venv/bin/nba-outliers --date 2026-11-23       # a specific night
.venv/bin/nba-outliers --force --no-commit     # rebuild an existing report without committing
.venv/bin/nba-outliers --baseline-method exp_decay --set general.min_score=2.5
.venv/bin/nba-outliers backtest 2026-10-20 2026-11-23 --compare-baselines bayesian,season_only
```

Reports go to `reports/YYYY-MM-DD.md` and `.json` and are git-committed automatically
(`[git]` in `config.toml`; `--no-commit` to skip). If a report already exists, the run just
prints it unless `--force` is given. Nights with no games produce no report.

## Configuration

Everything tunable lives in `config.toml`: thresholds, kind weights and caps, stat lists,
milestones and all baseline switches. Any value can be overridden per run with
`--set section.key=value`. The most common baseline switches also have flags:

| Flag | Options |
|---|---|
| `--baseline-method` | `bayesian` (default), `exp_decay`, `fixed_weights`, `season_only` |
| `--baseline-distribution` | `count` (default), `normal` |
| `--baseline-minutes-basis` | `expected` (default), `actual` |
| `--baseline-normalization` | `per_minute` (default), `per_game` |
| `--baseline-thin-history-prior` | `cohort` (default), `league`, `none` |
| `--baseline-seasons-back` | `0`–`3` (default `2`) |

## Scheduling (macOS)

```sh
scripts/install_launchd.sh          # daily at 08:00
scripts/install_launchd.sh 7 30     # daily at 07:30
scripts/install_launchd.sh uninstall
```

The job runs `nba-outliers run --wait-for-data`. If games are final but the stats feed
hasn't caught up, it waits and retries (`[data]` in config). Output goes to `logs/`.
launchd runs a missed job when the Mac wakes from sleep, but not if it was shut down.

**Because this repo lives under `~/Documents`,** macOS privacy protection may block the
background job from reading it. If `logs/launchd.err.log` shows `Operation not permitted`,
either move the repo outside `~/Documents`, `~/Desktop` and `~/Downloads`, or grant Full Disk
Access to the Python binary in System Settings → Privacy & Security
(run `readlink -f .venv/bin/python` to find it).

## Tests

```sh
.venv/bin/pytest
```

Tests run offline on synthetic data.

## Data source

[`nba_api`](https://github.com/swar/nba_api) (stats.nba.com, unofficial, no key). The provider
sits behind `outliers.providers.DataProvider`, so another source can be swapped in.

"""Command line entry point.

  nba-outliers [run] [--date YYYY-MM-DD] [--force] [--no-commit] ...   daily report (default)
  nba-outliers backfill                                                 fetch all seasons into the DB
  nba-outliers backtest START END [--compare-baselines a,b,c]           evaluate past nights, no saving
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta

from rich.console import Console
from rich.table import Table

from .config import load_config, with_overrides
from .detect import detect_night
from .gitops import commit_reports
from .paths import DB_PATH, REPORTS_DIR, ROOT
from .providers import ProviderError, get_provider
from .ranking import select
from .report import print_terminal, write_reports
from .seasons import last_night, parse_date, season_window
from .store import Store
from .sync import ensure_data

log = logging.getLogger("outliers")

# Shortcut flags for the [baseline] switches; anything else goes through --set.
BASELINE_FLAGS = ["method", "distribution", "minutes_basis", "normalization", "thin_history_prior", "seasons_back"]


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value (repeatable)")
    for flag in BASELINE_FLAGS:
        p.add_argument(f"--baseline-{flag.replace('_', '-')}", dest=f"baseline_{flag}")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-outliers", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="build the report for one night (default)")
    _common(run)
    run.add_argument("--date", type=parse_date, help="night to score (default: last night, US/Eastern)")
    run.add_argument("--top", type=int, help="number of outliers to show")
    run.add_argument("--force", action="store_true", help="rebuild even if the report exists")
    run.add_argument("--no-commit", action="store_true", help="don't git-commit the report")
    run.add_argument("--no-refresh", action="store_true", help="use cached data only")
    run.add_argument("--wait-for-data", action="store_true",
                     help="if games are final but stats are missing, wait and retry (scheduled runs)")

    bf = sub.add_parser("backfill", help="fetch this season and prior seasons into the local DB")
    _common(bf)

    bt = sub.add_parser("backtest", help="run the detector over past nights without saving")
    _common(bt)
    bt.add_argument("start", type=parse_date)
    bt.add_argument("end", type=parse_date)
    bt.add_argument("--compare-baselines", help="comma-separated baseline methods to compare")
    bt.add_argument("--top", type=int, default=5, help="items to show per night")
    return parser


def _config(args) -> dict:
    overrides = list(args.set)
    for flag in BASELINE_FLAGS:
        value = getattr(args, f"baseline_{flag}", None)
        if value is not None:
            overrides.append(f"baseline.{flag}={value}" if flag == "seasons_back"
                             else f'baseline.{flag}="{value}"')
    return load_config(args.config, overrides)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "backfill", "backtest", "-h", "--help"):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = _config(args)
    try:
        return {"run": cmd_run, "backfill": cmd_backfill, "backtest": cmd_backtest}[args.command](args, cfg)
    except ProviderError as e:
        log.error("%s", e)
        return 2


def _console() -> Console:
    return Console() if sys.stdout.isatty() else Console(width=180, no_color=True)


def _load(store: Store, d: date, cfg: dict):
    seasons = season_window(d, cfg["baseline"]["seasons_back"])
    return store.load_logs("P", seasons), store.load_logs("T", seasons), store.load_players()


def cmd_run(args, cfg: dict) -> int:
    d = args.date or last_night(cfg["general"]["timezone"])
    md_path = REPORTS_DIR / f"{d.isoformat()}.md"
    if md_path.exists() and not args.force:
        print(md_path.read_text())
        log.info("Report for %s already exists (use --force to rebuild)", d)
        return 0

    provider = get_provider(cfg)
    store = Store(DB_PATH)
    warnings: list[str] = []
    try:
        try:
            games = provider.scoreboard(d)
        except ProviderError as e:
            log.warning("Scoreboard unavailable: %s", e)
            games = None
            warnings.append("Scoreboard unavailable; couldn't verify every game is included.")

        data = cfg["data"]
        attempts = data["wait_for_data_attempts"] if args.wait_for_data else 1
        missing: list = []
        for attempt in range(attempts):
            ensure_data(store, provider, d, cfg, refresh=not args.no_refresh)
            plogs, tlogs, players = _load(store, d, cfg)
            have = set(plogs.loc[plogs["GAME_DATE"] == d.isoformat(), "GAME_ID"]) if len(plogs) else set()
            missing = [g for g in (games or []) if g.final and g.game_id not in have]
            if not missing or attempt == attempts - 1:
                break
            log.info("%d final game(s) not in stats feed yet; waiting %d min",
                     len(missing), data["wait_for_data_minutes"])
            time.sleep(data["wait_for_data_minutes"] * 60)

        if missing:
            warnings.append(f"{len(missing)} final game(s) missing from the stats feed "
                            f"({', '.join(f'{g.away}@{g.home}' for g in missing)}). Rerun later with --force.")
        not_final = [g for g in (games or []) if not g.final]
        if not_final:
            warnings.append("Not final, skipped: " + ", ".join(
                f"{g.away}@{g.home} ({g.status_text or 'unknown'})" for g in not_final))

        result = detect_night(d, plogs, tlogs, players, cfg)
        n_games = len(result.game_ids)
        if n_games == 0:
            print_terminal(d, select([], cfg), 0, warnings, _console())
            log.info("No games on %s; no report written", d)
            return 0

        sel = select(result.candidates, cfg, args.top)
        print_terminal(d, sel, n_games, warnings, _console())
        files = write_reports(REPORTS_DIR, d, sel, n_games, warnings, cfg)
        log.info("Wrote %s", ", ".join(str(f.relative_to(ROOT)) for f in files))
        if cfg["git"]["commit"] and not args.no_commit:
            commit_reports(ROOT, files, f"Outliers for {d.isoformat()}", cfg["git"]["push"])
        return 0
    finally:
        store.close()


def cmd_backfill(args, cfg: dict) -> int:
    d = last_night(cfg["general"]["timezone"])
    store = Store(DB_PATH)
    try:
        ensure_data(store, get_provider(cfg), d, cfg, refresh=True)
        for season in season_window(d, cfg["baseline"]["seasons_back"]):
            n = len(store.load_logs("P", [season]))
            print(f"{season}: {n} player games")
    finally:
        store.close()
    return 0


def cmd_backtest(args, cfg: dict) -> int:
    if args.end < args.start:
        log.error("end date is before start date")
        return 1
    methods = args.compare_baselines.split(",") if args.compare_baselines else [cfg["baseline"]["method"]]
    configs = {m: with_overrides(cfg, [f'baseline.method="{m.strip()}"']) for m in methods}

    store = Store(DB_PATH)
    provider = get_provider(cfg)
    console = _console()
    stats = {m: {"nights": 0, "items": 0, "over": 0, "quiet": 0, "entities": []} for m in configs}
    try:
        ensure_data(store, provider, args.end, cfg, refresh=False)
        ensure_data(store, provider, args.start, cfg, refresh=False)
        cache: dict = {}
        d = args.start
        while d <= args.end:
            key = tuple(season_window(d, cfg["baseline"]["seasons_back"]))
            if key not in cache:
                cache[key] = _load(store, d, cfg)
            plogs, tlogs, players = cache[key]
            table = Table(title=str(d), title_justify="left")
            table.add_column("#")
            columns = []
            for m, mcfg in configs.items():
                result = detect_night(d, plogs, tlogs, players, mcfg)
                if not result.game_ids:
                    break
                sel = select(result.candidates, mcfg, args.top)
                s = stats[m]
                s["nights"] += 1
                s["items"] += 0 if sel.quiet else len(sel.items)
                s["over"] += sel.n_over_threshold
                s["quiet"] += sel.quiet
                s["entities"] += [c.entity for c in sel.items if not sel.quiet]
                table.add_column(f"{m}{' (quiet)' if sel.quiet else ''}", overflow="fold")
                columns.append([f"{c.name} · {c.reason}" for c in sel.items])
            if columns:
                for i in range(max(len(c) for c in columns)):
                    table.add_row(str(i + 1), *[c[i] if i < len(c) else "" for c in columns])
                console.print(table)
            d += timedelta(days=1)
    finally:
        store.close()

    summary = Table(title="Backtest summary", title_justify="left")
    for col in ("method", "nights", "avg listed", "avg over threshold", "quiet nights", "repeat rate"):
        summary.add_column(col)
    for m, s in stats.items():
        n = max(s["nights"], 1)
        ents = s["entities"]
        repeat = 1 - len(set(ents)) / len(ents) if ents else 0
        summary.add_row(m, str(s["nights"]), f"{s['items'] / n:.1f}", f"{s['over'] / n:.1f}",
                        str(s["quiet"]), f"{repeat:.0%}")
    console.print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

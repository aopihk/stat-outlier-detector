"""Terminal, Markdown and JSON output."""

import json
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .detect import odds
from .ranking import Selection

KIND_LABELS = {"personal": "Personal", "league": "League", "team": "Team", "under": "Under"}


def title(d: date, n_games: int) -> str:
    return f"NBA Outliers — {d.strftime('%a, %b')} {d.day} {d.year} ({n_games} game{'s' if n_games != 1 else ''})"


def _who(c) -> str:
    return c.name if c.kind == "team" else f"{c.name} ({c.team})"


def _why(c) -> str:
    parts = [c.reason] + [f"also {x}" for x in c.context]
    return "; ".join(parts)


def print_terminal(d: date, sel: Selection, n_games: int, warnings: list[str], console: Console | None = None):
    console = console or Console()
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    if not n_games:
        console.print(f"[bold]{title(d, 0)}[/bold]\nNo games were played.")
        return
    table = Table(title=title(d, n_games), title_justify="left", show_lines=False)
    for col in ("#", "Type", "Player / Team", "Line", "Why"):
        table.add_column(col, overflow="fold")
    for i, c in enumerate(sel.items, 1):
        who = _who(c) + (f" [dim]{', '.join(c.tags)}[/dim]" if c.tags else "")
        table.add_row(str(i), KIND_LABELS[c.kind], who, c.line, _why(c))
    console.print(table)
    if sel.quiet:
        console.print("[dim]Quiet night: nothing cleared the threshold, showing the closest calls.[/dim]")


def to_markdown(d: date, sel: Selection, n_games: int, warnings: list[str], cfg: dict) -> str:
    lines = [f"# {title(d, n_games)}", ""]
    lines += [f"> ⚠ {w}" for w in warnings]
    if warnings:
        lines.append("")
    if not n_games:
        return "\n".join(lines + ["No games were played.", ""])
    if sel.quiet:
        lines += ["_Quiet night: nothing cleared the threshold, showing the closest calls._", ""]
    lines += ["| # | Type | Player / Team | Line | Why |", "|---|---|---|---|---|"]
    for i, c in enumerate(sel.items, 1):
        who = _who(c) + (f" _{', '.join(c.tags)}_" if c.tags else "")
        cells = [str(i), KIND_LABELS[c.kind], who, c.line, _why(c)]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
    b = cfg["baseline"]
    lines += ["", f"<sub>Baseline: {b['method']}, {b['normalization']}, {b['minutes_basis']} minutes, "
              f"{b['distribution']} model, {b['seasons_back']} prior seasons. "
              f"{sel.n_over_threshold} entries cleared the bar (about {odds(cfg['general']['min_score'])} or rarer).</sub>", ""]
    return "\n".join(lines)


def to_json(d: date, sel: Selection, n_games: int, warnings: list[str], cfg: dict) -> str:
    return json.dumps({
        "date": d.isoformat(),
        "games": n_games,
        "quiet_night": sel.quiet,
        "n_over_threshold": sel.n_over_threshold,
        "warnings": warnings,
        "baseline": cfg["baseline"],
        "items": [dict(c.to_dict(), rank=i) for i, c in enumerate(sel.items, 1)],
    }, indent=2, default=float)


def write_reports(reports_dir: Path, d: date, sel: Selection, n_games: int,
                  warnings: list[str], cfg: dict) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md = reports_dir / f"{d.isoformat()}.md"
    js = reports_dir / f"{d.isoformat()}.json"
    md.write_text(to_markdown(d, sel, n_games, warnings, cfg))
    js.write_text(to_json(d, sel, n_games, warnings, cfg) + "\n")
    return [md, js]

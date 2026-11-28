"""Pick "only the biggest" outliers from all candidates."""

from dataclasses import dataclass

from .detect import Candidate


@dataclass
class Selection:
    items: list[Candidate]
    quiet: bool
    n_over_threshold: int


def _base_stat(stat: str) -> str:
    return stat.removeprefix("LEAGUE:").removeprefix("MILESTONE:")


def select(cands: list[Candidate], cfg: dict, top_n: int | None = None) -> Selection:
    g, kinds = cfg["general"], cfg["kinds"]
    top_n = top_n or g["top_n"]
    for c in cands:
        c.score = abs(c.z) * kinds["weights"][c.kind]

    # One entry per player/team: keep their most extreme stat, others become context.
    by_entity: dict[str, list[Candidate]] = {}
    for c in cands:
        by_entity.setdefault(c.entity, []).append(c)
    best = []
    for group in by_entity.values():
        group.sort(key=lambda c: c.score, reverse=True)
        top = group[0]
        # A stat appears once: "BLK", "LEAGUE:BLK" and "MILESTONE:BLK" describe the same number.
        seen = {_base_stat(top.stat)}
        if top.stat.startswith("MILESTONE:"):
            # A milestone like "15 AST" says nothing on its own; borrow the explanation
            # from the same stat's personal/league entry, keeping the milestone's score.
            same = next((c for c in group[1:] if _base_stat(c.stat) == _base_stat(top.stat)), None)
            if same:
                top.value, top.expected, top.reason, top.brief = same.value, same.expected, same.reason, same.brief
        top.context = []
        for c in group[1:]:
            if c.score >= g["context_score"] and _base_stat(c.stat) not in seen:
                top.context.append(c.brief)
                seen.add(_base_stat(c.stat))
        best.append(top)
    best.sort(key=lambda c: c.score, reverse=True)

    over = [c for c in best if c.score >= g["min_score"]]
    items, per_kind = [], {}
    for c in over:
        if per_kind.get(c.kind, 0) >= kinds["caps"][c.kind]:
            continue
        items.append(c)
        per_kind[c.kind] = per_kind.get(c.kind, 0) + 1
        if len(items) == top_n:
            break
    if items:
        return Selection(items, False, len(over))
    return Selection(best[: g["quiet_night_count"]], True, 0)

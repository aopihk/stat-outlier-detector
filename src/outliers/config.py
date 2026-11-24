"""Load config.toml and apply `section.key=value` overrides."""

import copy
import tomllib
from pathlib import Path

from .paths import CONFIG


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, raw = item.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _parse_value(raw.strip())


def with_overrides(cfg: dict, items: list[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in items:
        apply_override(out, item)
    return out


def _parse_value(raw: str):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw

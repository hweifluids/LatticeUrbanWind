from __future__ import annotations

"""Shared LUW deck schema and tolerant token normalizers.

Python deck IO and the Qt GUI both read the same schema file so new keys,
sections, aliases, and mode visibility rules can be added in one place.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import math
import re
from typing import Dict, List, Optional, Tuple


_KEY_SEPARATOR_RE = re.compile(r"[\s\-]+")
_KEY_REPEAT_UNDERSCORE_RE = re.compile(r"_+")

TRUE_BOOL_TOKENS = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
    "enable",
    "enabled",
}
FALSE_BOOL_TOKENS = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "off",
    "disable",
    "disabled",
}
MODE_TO_MASK = {
    "luw": 1 << 0,
    "luwdg": 1 << 1,
    "luwpf": 1 << 2,
}
MODE_MASK_ALL = sum(MODE_TO_MASK.values())


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    description: str
    aliases: Tuple[str, ...]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    section_id: str
    help: str
    kind: str
    enum_values: Tuple[str, ...] = ()
    mode_mask: int = MODE_MASK_ALL
    quoted: bool = False
    read_only: bool = False
    aliases: Tuple[str, ...] = ()


def _schema_path() -> Path:
    return Path(__file__).resolve().with_name("deck_schema.json")


def strip_quotes(raw: str) -> str:
    text = str(raw).strip()
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        return text[1:-1].strip()
    return text


def _sanitize_key(raw: str) -> str:
    text = _KEY_SEPARATOR_RE.sub("_", str(raw).strip().lower())
    text = _KEY_REPEAT_UNDERSCORE_RE.sub("_", text)
    return text.strip("_")


def parse_bool_token(raw: object) -> Optional[bool]:
    if raw is None:
        return None
    text = strip_quotes(str(raw)).strip().lower()
    if not text:
        return None
    if text in TRUE_BOOL_TOKENS:
        return True
    if text in FALSE_BOOL_TOKENS:
        return False
    try:
        numeric = float(text)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric != 0.0


@lru_cache(maxsize=1)
def _schema_payload() -> dict:
    return json.loads(_schema_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def section_specs() -> Tuple[SectionSpec, ...]:
    specs: List[SectionSpec] = []
    for item in _schema_payload()["sections"]:
        specs.append(
            SectionSpec(
                id=item["id"],
                title=item["title"],
                description=item.get("description", ""),
                aliases=tuple(item.get("aliases", [])),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def field_specs() -> Tuple[FieldSpec, ...]:
    specs: List[FieldSpec] = []
    for item in _schema_payload()["fields"]:
        mode_names = tuple(item.get("modes", MODE_TO_MASK.keys()))
        mode_mask = 0
        for name in mode_names:
            mode_mask |= MODE_TO_MASK.get(str(name).lower(), 0)
        if mode_mask == 0:
            mode_mask = MODE_MASK_ALL
        specs.append(
            FieldSpec(
                key=item["key"],
                label=item.get("label", item["key"]),
                section_id=item["section"],
                help=item.get("help", ""),
                kind=item.get("kind", "string"),
                enum_values=tuple(item.get("enum_values", [])),
                mode_mask=mode_mask,
                quoted=bool(item.get("quoted", False)),
                read_only=bool(item.get("read_only", False)),
                aliases=tuple(item.get("aliases", [])),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def field_spec_map() -> Dict[str, FieldSpec]:
    return {spec.key: spec for spec in field_specs()}


@lru_cache(maxsize=1)
def field_alias_map() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for spec in field_specs():
        aliases[_sanitize_key(spec.key)] = spec.key
        for alias in spec.aliases:
            aliases[_sanitize_key(alias)] = spec.key
    return aliases


def normalize_key(raw: str) -> str:
    sanitized = _sanitize_key(raw)
    return field_alias_map().get(sanitized, sanitized)


SECTION_ORDER: List[str] = [spec.id for spec in section_specs()]
SECTION_TITLES: Dict[str, str] = {spec.id: spec.title for spec in section_specs()}
SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {spec.id: spec.aliases for spec in section_specs()}
FIELD_TO_SECTION: Dict[str, str] = {spec.key: spec.section_id for spec in field_specs()}
FIELD_ORDER: Dict[str, List[str]] = {
    section_id: [spec.key for spec in field_specs() if spec.section_id == section_id]
    for section_id in SECTION_ORDER
}

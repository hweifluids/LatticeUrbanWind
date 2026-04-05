from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from deck_schema import (
    FIELD_ORDER,
    FIELD_TO_SECTION,
    SECTION_ALIASES,
    SECTION_ORDER,
    SECTION_TITLES,
    field_spec_map,
    normalize_key,
    parse_bool_token,
)


class DeckParseError(ValueError):
    pass


def _normalize_key(key: str) -> str:
    return normalize_key(key)


def _normalize_section_label(text: str) -> str:
    s = text.strip().lower()
    if s.startswith("[") and "]" in s:
        s = s[1 : s.index("]")]
    return " ".join(s.split())


def _comment_index(line: str) -> int:
    in_single = False
    in_double = False
    for idx in range(len(line) - 1):
        ch = line[idx]
        nxt = line[idx + 1]
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if not in_single and not in_double and ch == "/" and nxt == "/":
            return idx
    return -1


def _split_inline_comment(line: str) -> tuple[str, str]:
    idx = _comment_index(line)
    if idx < 0:
        return line, ""
    return line[:idx], line[idx:].strip()


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).strip()


def _parse_list_items(raw: str) -> List[str]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass
class DeckEntry:
    key: str
    value: str
    section_id: str
    comment: str = ""
    known: bool = True


@dataclass
class DeckDocument:
    path: Optional[Path] = None
    preamble_lines: List[str] = field(default_factory=list)
    section_loose_lines: Dict[str, List[str]] = field(default_factory=dict)
    entries: Dict[str, DeckEntry] = field(default_factory=dict)
    unknown_order: Dict[str, List[str]] = field(default_factory=dict)
    duplicates: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        path: Optional[Path] = None,
        strict_duplicates: bool = False,
    ) -> "DeckDocument":
        doc = cls(path=path)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        current_section: Optional[str] = None
        saw_structured_content = False

        for raw_line in normalized.split("\n"):
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            header_section = _match_section_header(stripped)
            if header_section is not None:
                current_section = header_section
                saw_structured_content = True
                continue

            content, comment = _split_inline_comment(line)
            match = _match_key_value(content)
            if match is not None:
                key, value = match
                section_id = FIELD_TO_SECTION.get(key, current_section or "custom")
                known = key in FIELD_TO_SECTION
                entry = DeckEntry(
                    key=key,
                    value=value,
                    section_id=section_id,
                    comment=comment,
                    known=known,
                )
                if key in doc.entries:
                    doc.duplicates.setdefault(key, [doc.entries[key].value]).append(value)
                doc.entries[key] = entry
                if not known:
                    order = doc.unknown_order.setdefault(section_id, [])
                    if key not in order:
                        order.append(key)
                saw_structured_content = True
                continue

            if not stripped:
                if not saw_structured_content and current_section is None:
                    doc.preamble_lines.append("")
                continue

            if not saw_structured_content and current_section is None:
                doc.preamble_lines.append(line)
            else:
                target_section = current_section or "custom"
                doc.section_loose_lines.setdefault(target_section, []).append(line)

        if strict_duplicates and doc.duplicates:
            dupes = ", ".join(sorted(doc.duplicates))
            raise DeckParseError(f"Duplicate deck keys are not allowed: {dupes}")
        return doc

    @classmethod
    def load(cls, path: Path | str, *, strict_duplicates: bool = False) -> "DeckDocument":
        deck_path = Path(path).expanduser().resolve()
        text = deck_path.read_text(encoding="utf-8", errors="ignore")
        return cls.from_text(text, path=deck_path, strict_duplicates=strict_duplicates)

    def has(self, key: str) -> bool:
        return _normalize_key(key) in self.entries

    def get_raw(self, key: str, default: Optional[str] = None) -> Optional[str]:
        entry = self.entries.get(_normalize_key(key))
        return entry.value if entry is not None else default

    def get_text(self, key: str, default: Optional[str] = None) -> Optional[str]:
        raw = self.get_raw(key, None)
        if raw is None:
            return default
        value = raw.strip()
        if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
            return value[1:-1].strip()
        return value

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        raw = self.get_text(key, None)
        if raw in (None, ""):
            return default
        try:
            return int(raw)
        except Exception:
            return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        raw = self.get_text(key, None)
        if raw in (None, ""):
            return default
        try:
            value = float(raw)
        except Exception:
            return default
        if value != value:
            return default
        return value

    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        raw = self.get_text(key, None)
        if raw is None:
            return default
        parsed = parse_bool_token(raw)
        return default if parsed is None else parsed

    def get_list(self, key: str) -> List[str]:
        raw = self.get_raw(key, None)
        if raw is None:
            return []
        return _parse_list_items(raw)

    def get_float_list(self, key: str) -> List[float]:
        values: List[float] = []
        for part in self.get_list(key):
            try:
                values.append(float(part))
            except Exception:
                return []
        return values

    def get_pair(self, key: str) -> Optional[tuple[float, float]]:
        values = self.get_float_list(key)
        if len(values) != 2:
            return None
        lo, hi = sorted((values[0], values[1]))
        return lo, hi

    def set_raw(
        self,
        key: str,
        value: str,
        *,
        section_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        normalized = _normalize_key(key)
        existing = self.entries.get(normalized)
        target_section = section_id or FIELD_TO_SECTION.get(normalized)
        if target_section is None and existing is not None:
            target_section = existing.section_id
        if target_section is None:
            target_section = "custom"

        if existing is not None and comment is None:
            comment = existing.comment
        entry = DeckEntry(
            key=normalized,
            value=value.strip(),
            section_id=target_section,
            comment=(comment or "").strip(),
            known=normalized in FIELD_TO_SECTION,
        )
        self.entries[normalized] = entry
        self.duplicates.pop(normalized, None)
        if not entry.known:
            order = self.unknown_order.setdefault(target_section, [])
            if normalized not in order:
                order.append(normalized)

    def set_text(
        self,
        key: str,
        value: str,
        *,
        quoted: bool = False,
        section_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        text = value.strip()
        rendered = f'"{text}"' if quoted else text
        self.set_raw(key, rendered, section_id=section_id, comment=comment)

    def set_int(self, key: str, value: int, *, section_id: Optional[str] = None) -> None:
        self.set_raw(key, str(int(value)), section_id=section_id)

    def set_float(self, key: str, value: float, *, precision: int = 6, section_id: Optional[str] = None) -> None:
        self.set_raw(key, f"{float(value):.{precision}f}", section_id=section_id)

    def set_bool(self, key: str, value: bool, *, section_id: Optional[str] = None) -> None:
        self.set_raw(key, "true" if value else "false", section_id=section_id)

    def set_list(self, key: str, values: Iterable[object], *, section_id: Optional[str] = None) -> None:
        rendered = "[" + ", ".join(_format_scalar(value) for value in values) + "]"
        self.set_raw(key, rendered, section_id=section_id)

    def set_pair(self, key: str, pair: Iterable[float], *, precision: int = 6, section_id: Optional[str] = None) -> None:
        values = list(pair)
        if len(values) != 2:
            raise ValueError(f"{key} expects exactly 2 values, got {len(values)}")
        rendered = "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"
        self.set_raw(key, rendered, section_id=section_id)

    def remove(self, key: str) -> None:
        normalized = _normalize_key(key)
        self.entries.pop(normalized, None)
        self.duplicates.pop(normalized, None)

    def duplicate_keys(self) -> List[str]:
        return sorted(self.duplicates)

    def to_dict(self) -> Dict[str, str]:
        return {key: entry.value for key, entry in self.entries.items()}

    def render(self) -> str:
        lines: List[str] = []

        if self.preamble_lines:
            lines.extend(self.preamble_lines)
            while lines and lines[-1] == "":
                lines.pop()
            if lines:
                lines.append("")
        else:
            lines.append("// LUW deck")
            lines.append("")

        for section_id in SECTION_ORDER:
            section_lines = self._render_section(section_id)
            if not section_lines:
                continue
            lines.extend(section_lines)
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path).expanduser().resolve() if path is not None else self.path
        if target is None:
            raise ValueError("No target path provided for deck save.")
        target.write_text(self.render(), encoding="utf-8")
        self.path = target
        return target

    def _render_section(self, section_id: str) -> List[str]:
        known_keys = [
            key
            for key in FIELD_ORDER.get(section_id, [])
            if key in self.entries and self.entries[key].section_id == section_id
        ]
        unknown_keys = [
            key
            for key in self.unknown_order.get(section_id, [])
            if key in self.entries and self.entries[key].section_id == section_id
        ]
        loose_lines = [line for line in self.section_loose_lines.get(section_id, []) if line.strip()]

        if not known_keys and not unknown_keys and not loose_lines:
            return []

        lines = [f"// {SECTION_TITLES.get(section_id, section_id.title())}"]
        lines.extend(loose_lines)
        for key in known_keys:
            lines.append(self._render_entry(self.entries[key]))
        for key in unknown_keys:
            lines.append(self._render_entry(self.entries[key]))
        return lines

    @staticmethod
    def _render_entry(entry: DeckEntry) -> str:
        rendered_value = entry.value
        spec = field_spec_map().get(entry.key) if entry.known else None
        if spec is not None:
            if spec.kind == "boolean":
                parsed = parse_bool_token(rendered_value)
                if parsed is not None:
                    rendered_value = "true" if parsed else "false"
            elif spec.kind in {"float_pair", "float_triplet", "uint_triplet", "float_list", "token_list"} and rendered_value.strip():
                rendered_value = "[" + ", ".join(_parse_list_items(rendered_value)) + "]"
            elif spec.quoted and rendered_value.strip():
                rendered_value = f'"{rendered_value.strip()[1:-1].strip()}"' if (
                    len(rendered_value.strip()) >= 2
                    and (
                        (rendered_value.strip()[0] == '"' and rendered_value.strip()[-1] == '"')
                        or (rendered_value.strip()[0] == "'" and rendered_value.strip()[-1] == "'")
                    )
                ) else f'"{rendered_value.strip()}"'

        line = f"{entry.key} =".rstrip()
        if rendered_value.strip():
            line += f" {rendered_value.strip()}"
        if entry.comment:
            line += f" {entry.comment}"
        return line.rstrip()


def _match_section_header(stripped_line: str) -> Optional[str]:
    if not stripped_line:
        return None
    if not (stripped_line.startswith("//") or stripped_line.startswith("#")):
        return None
    label = stripped_line[2:].strip() if stripped_line.startswith("//") else stripped_line[1:].strip()
    normalized = _normalize_section_label(label)
    for section_id, aliases in SECTION_ALIASES.items():
        if normalized == section_id:
            return section_id
        if normalized == _normalize_section_label(SECTION_TITLES.get(section_id, section_id)):
            return section_id
        for alias in aliases:
            if normalized == _normalize_section_label(alias):
                return section_id
    return None


def _match_key_value(content: str) -> Optional[tuple[str, str]]:
    stripped = content.strip()
    if not stripped or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = _normalize_key(key)
    if not key:
        return None
    return key, value.strip()


def load_deck(path: Path | str, *, strict_duplicates: bool = False) -> DeckDocument:
    return DeckDocument.load(path, strict_duplicates=strict_duplicates)


def parse_deck_text(text: str, *, strict_duplicates: bool = False) -> DeckDocument:
    return DeckDocument.from_text(text, strict_duplicates=strict_duplicates)

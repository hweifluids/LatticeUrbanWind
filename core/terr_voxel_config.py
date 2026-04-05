from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, Mapping


DEFAULT_TERR_VOXEL_APPROACH = "idw"
ALLOWED_TERR_VOXEL_APPROACHES = ("idw", "kriging_gpu", "kriging")

DECK_KEY_TERR_VOXEL_APPROACH = "terr_voxel_approach"
DECK_KEY_TERR_VOXEL_HEIGHT_FIELD = "terr_voxel_height_field"
DECK_KEY_TERR_VOXEL_IGNORE_UNDER = "terr_voxel_ignore_under"
DECK_KEY_TERR_VOXEL_GRID_RESOLUTION = "terr_voxel_grid_resolution"
DECK_KEY_TERR_VOXEL_IDW_SIGMA = "terr_voxel_idw_sigma"
DECK_KEY_TERR_VOXEL_IDW_POWER = "terr_voxel_idw_power"
DECK_KEY_TERR_VOXEL_IDW_NEIGHBORS = "terr_voxel_idw_neighbors"

DEFAULT_TERR_VOXEL_HEIGHT_FIELD = "auto"
DEFAULT_TERR_VOXEL_IGNORE_UNDER = 0.0
DEFAULT_TERR_VOXEL_GRID_RESOLUTION = 50.0
DEFAULT_TERR_VOXEL_IDW_SIGMA = 1.0
DEFAULT_TERR_VOXEL_IDW_POWER = 2.0
DEFAULT_TERR_VOXEL_IDW_NEIGHBORS = 12


@dataclass(frozen=True)
class TerrainVoxelConfig:
    approach: str = DEFAULT_TERR_VOXEL_APPROACH
    height_field: str = DEFAULT_TERR_VOXEL_HEIGHT_FIELD
    ignore_under: float = DEFAULT_TERR_VOXEL_IGNORE_UNDER
    grid_resolution: float = DEFAULT_TERR_VOXEL_GRID_RESOLUTION
    idw_sigma: float = DEFAULT_TERR_VOXEL_IDW_SIGMA
    idw_power: float = DEFAULT_TERR_VOXEL_IDW_POWER
    idw_neighbors: int = DEFAULT_TERR_VOXEL_IDW_NEIGHBORS


def _emit(warn: Callable[[str], None] | None, message: str) -> None:
    if warn:
        warn(message)


def _normalize_height_field(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "inferred":
        return DEFAULT_TERR_VOXEL_HEIGHT_FIELD
    return value


def _resolve_string(
    *,
    cli_value: object,
    deck_value: object,
    default: str,
    field_label: str,
    source_key: str,
    warn: Callable[[str], None] | None,
) -> tuple[str, str]:
    for source_name, value in (("CLI", cli_value), ("deck", deck_value)):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, source_name.lower()
        _emit(warn, f"{field_label} from {source_name} is empty. Falling back to the next source.")
    return default, source_key


def _resolve_choice(
    *,
    cli_value: object,
    deck_value: object,
    default: str,
    allowed: Iterable[str],
    field_label: str,
    warn: Callable[[str], None] | None,
) -> tuple[str, str]:
    allowed_set = {str(item).strip().lower() for item in allowed}
    for source_name, value in (("CLI", cli_value), ("deck", deck_value)):
        if value is None:
            continue
        text = str(value).strip().lower()
        if text in allowed_set:
            return text, source_name.lower()
        _emit(
            warn,
            f"{field_label} from {source_name} has unsupported value '{value}'. "
            f"Allowed values: {', '.join(sorted(allowed_set))}. Falling back to the next source.",
        )
    return default, "default"


def _resolve_float(
    *,
    cli_value: object,
    deck_value: object,
    default: float,
    field_label: str,
    min_value: float | None,
    inclusive: bool,
    warn: Callable[[str], None] | None,
) -> tuple[float, str]:
    for source_name, value in (("CLI", cli_value), ("deck", deck_value)):
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            _emit(warn, f"{field_label} from {source_name} is not numeric ('{value}'). Falling back to the next source.")
            continue
        if not math.isfinite(numeric):
            _emit(warn, f"{field_label} from {source_name} is not finite ({value}). Falling back to the next source.")
            continue
        if min_value is not None:
            valid = numeric >= min_value if inclusive else numeric > min_value
            if not valid:
                comparator = ">=" if inclusive else ">"
                _emit(
                    warn,
                    f"{field_label} from {source_name} must be {comparator} {min_value}. "
                    f"Got {numeric}. Falling back to the next source.",
                )
                continue
        return numeric, source_name.lower()
    return float(default), "default"


def _resolve_int(
    *,
    cli_value: object,
    deck_value: object,
    default: int,
    field_label: str,
    min_value: int | None,
    inclusive: bool,
    warn: Callable[[str], None] | None,
) -> tuple[int, str]:
    for source_name, value in (("CLI", cli_value), ("deck", deck_value)):
        if value is None:
            continue
        try:
            numeric = int(value)
        except Exception:
            _emit(warn, f"{field_label} from {source_name} is not an integer ('{value}'). Falling back to the next source.")
            continue
        if min_value is not None:
            valid = numeric >= min_value if inclusive else numeric > min_value
            if not valid:
                comparator = ">=" if inclusive else ">"
                _emit(
                    warn,
                    f"{field_label} from {source_name} must be {comparator} {min_value}. "
                    f"Got {numeric}. Falling back to the next source.",
                )
                continue
        return numeric, source_name.lower()
    return int(default), "default"


def resolve_terrain_voxel_config(
    deck,
    *,
    cli_overrides: Mapping[str, object] | None = None,
    warn: Callable[[str], None] | None = None,
) -> tuple[TerrainVoxelConfig, Dict[str, str]]:
    overrides = dict(cli_overrides or {})

    approach, approach_source = _resolve_choice(
        cli_value=overrides.get("approach"),
        deck_value=deck.get_text(DECK_KEY_TERR_VOXEL_APPROACH) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_APPROACH,
        allowed=ALLOWED_TERR_VOXEL_APPROACHES,
        field_label=DECK_KEY_TERR_VOXEL_APPROACH,
        warn=warn,
    )
    height_field, height_field_source = _resolve_string(
        cli_value=overrides.get("height_field"),
        deck_value=deck.get_text(DECK_KEY_TERR_VOXEL_HEIGHT_FIELD) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_HEIGHT_FIELD,
        field_label=DECK_KEY_TERR_VOXEL_HEIGHT_FIELD,
        source_key="default",
        warn=warn,
    )
    height_field = _normalize_height_field(height_field)
    ignore_under, ignore_under_source = _resolve_float(
        cli_value=overrides.get("ignore_under"),
        deck_value=deck.get_float(DECK_KEY_TERR_VOXEL_IGNORE_UNDER) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_IGNORE_UNDER,
        field_label=DECK_KEY_TERR_VOXEL_IGNORE_UNDER,
        min_value=0.0,
        inclusive=True,
        warn=warn,
    )
    grid_resolution, grid_resolution_source = _resolve_float(
        cli_value=overrides.get("grid_resolution"),
        deck_value=deck.get_float(DECK_KEY_TERR_VOXEL_GRID_RESOLUTION) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_GRID_RESOLUTION,
        field_label=DECK_KEY_TERR_VOXEL_GRID_RESOLUTION,
        min_value=0.0,
        inclusive=False,
        warn=warn,
    )
    idw_sigma, idw_sigma_source = _resolve_float(
        cli_value=overrides.get("idw_sigma"),
        deck_value=deck.get_float(DECK_KEY_TERR_VOXEL_IDW_SIGMA) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_IDW_SIGMA,
        field_label=DECK_KEY_TERR_VOXEL_IDW_SIGMA,
        min_value=0.0,
        inclusive=True,
        warn=warn,
    )
    idw_power, idw_power_source = _resolve_float(
        cli_value=overrides.get("idw_power"),
        deck_value=deck.get_float(DECK_KEY_TERR_VOXEL_IDW_POWER) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_IDW_POWER,
        field_label=DECK_KEY_TERR_VOXEL_IDW_POWER,
        min_value=0.0,
        inclusive=False,
        warn=warn,
    )
    idw_neighbors, idw_neighbors_source = _resolve_int(
        cli_value=overrides.get("idw_neighbors"),
        deck_value=deck.get_int(DECK_KEY_TERR_VOXEL_IDW_NEIGHBORS) if deck is not None else None,
        default=DEFAULT_TERR_VOXEL_IDW_NEIGHBORS,
        field_label=DECK_KEY_TERR_VOXEL_IDW_NEIGHBORS,
        min_value=1,
        inclusive=True,
        warn=warn,
    )

    return (
        TerrainVoxelConfig(
            approach=approach,
            height_field=height_field,
            ignore_under=ignore_under,
            grid_resolution=grid_resolution,
            idw_sigma=idw_sigma,
            idw_power=idw_power,
            idw_neighbors=idw_neighbors,
        ),
        {
            "approach": approach_source,
            "height_field": height_field_source,
            "ignore_under": ignore_under_source,
            "grid_resolution": grid_resolution_source,
            "idw_sigma": idw_sigma_source,
            "idw_power": idw_power_source,
            "idw_neighbors": idw_neighbors_source,
        },
    )

#!/usr/bin/env python3
"""
Synthesize a seasonal average VTK field from directional CFD average VTK files
and a windrose probability table.

Usage:
    python season_average_vtk.py <path_to_case.luw|luwdg|luwpf>
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

TARGET_HEIGHTS_M = [50, 100, 150, 200, 300, 400, 500, 600, 800]
BASE_HEIGHT_M = -50.0
LAYER_STEP_M = 10.0
ANGLE_SEQUENCE = [22.5 * i for i in range(16)]
ANGLE_TO_DIRECTION = {
    0.0: "N",
    22.5: "NNE",
    45.0: "NE",
    67.5: "ENE",
    90.0: "E",
    112.5: "ESE",
    135.0: "SE",
    157.5: "SSE",
    180.0: "S",
    202.5: "SSW",
    225.0: "SW",
    247.5: "WSW",
    270.0: "W",
    292.5: "WNW",
    315.0: "NW",
    337.5: "NNW",
}
_COMPASS_TO_ANGLE = {
    "N": 0.0,
    "NORTH": 0.0,
    "NNE": 22.5,
    "NORTHNORTHEAST": 22.5,
    "NE": 45.0,
    "NORTHEAST": 45.0,
    "ENE": 67.5,
    "EASTNORTHEAST": 67.5,
    "E": 90.0,
    "EAST": 90.0,
    "ESE": 112.5,
    "EASTSOUTHEAST": 112.5,
    "SE": 135.0,
    "SOUTHEAST": 135.0,
    "SSE": 157.5,
    "SOUTHSOUTHEAST": 157.5,
    "S": 180.0,
    "SOUTH": 180.0,
    "SSW": 202.5,
    "SOUTHSOUTHWEST": 202.5,
    "SW": 225.0,
    "SOUTHWEST": 225.0,
    "WSW": 247.5,
    "WESTSOUTHWEST": 247.5,
    "W": 270.0,
    "WEST": 270.0,
    "WNW": 292.5,
    "WESTNORTHWEST": 292.5,
    "NW": 315.0,
    "NORTHWEST": 315.0,
    "NNW": 337.5,
    "NORTHNORTHWEST": 337.5,
}
IGNORE_DIRECTION_LABELS = {
    "",
    "ALL",
    "ALLDIRECTIONS",
    "TOTAL",
    "SUM",
    "CALM",
    "CALMS",
    "VARIABLE",
    "VAR",
}
VTK_STEP_RE = re.compile(r"_avg[-_](\d+)(?:_cropped)?$", re.IGNORECASE)
SPEED_BIN_RE = re.compile(
    r"^C(?P<class_id>\d+)"
    r"_(?P<lower>[-+]?[0-9]+(?:[p\.][0-9]+)?)"
    r"_(?P<upper>[-+]?[0-9]+(?:[p\.][0-9]+)?|inf)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpeedBin:
    column_index: int
    header: str
    class_id: int
    lower: float
    upper: float | None
    target_speed: float


@dataclass(frozen=True)
class DirectionWeight:
    angle: float
    label: str
    probability: float
    velocity_weight: float
    tke_weight: float


@dataclass(frozen=True)
class VtkDirectoryChoice:
    label: str
    directory: Path
    angle_to_file: dict[float, Path]


@dataclass(frozen=True)
class ProjectLayout:
    config_path: Path
    project_dir: Path
    wind_bc_dir: Path
    profile_path: Path
    windrose_path: Path
    output_dir: Path
    figure_dir: Path
    summary_path: Path
    output_vtk_path: Path
    figure_dpi: int
    z_limit_si: float | None


@dataclass
class AccumulatorVolumes:
    use_memmap: bool
    temp_dir: Path | None
    u: np.ndarray
    vm: np.ndarray
    tke: np.ndarray

    def flush(self) -> None:
        for arr in (self.u, self.vm, self.tke):
            if hasattr(arr, "flush"):
                arr.flush()

    def cleanup(self) -> None:
        self.flush()
        if self.temp_dir is not None and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def log_rule(title: str) -> None:
    line = "=" * 24
    log(f"{line} {title} {line}")


def human_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


def format_number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _strip_inline_comment(line: str) -> str:
    return line.split("//", 1)[0].split("#", 1)[0].strip()


def _parse_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", text)
    if not match:
        return None
    value = _strip_inline_comment(match.group(1))
    if not value:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value or None


def _parse_scalar_any(text: str, keys: list[str]) -> str | None:
    for key in keys:
        value = _parse_scalar(text, key)
        if value is not None:
            return value
    return None


def _parse_pair(text: str, key: str) -> tuple[float, float] | None:
    match = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", text)
    if not match:
        return None
    value = _strip_inline_comment(match.group(1))
    if not (value.startswith("[") and value.endswith("]")):
        return None
    parts = [part.strip() for part in value[1:-1].split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def _parse_pair_any(text: str, keys: list[str]) -> tuple[float, float] | None:
    for key in keys:
        value = _parse_pair(text, key)
        if value is not None:
            return value
    return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def load_project_layout(config_path: Path) -> ProjectLayout:
    if config_path.suffix.lower() not in {".luw", ".luwdg", ".luwpf"}:
        raise ValueError("Config file must have extension .luw, .luwdg, or .luwpf")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_dir = config_path.resolve().parent
    wind_bc_dir = project_dir / "wind_bc"
    if not wind_bc_dir.is_dir():
        raise FileNotFoundError(f"wind_bc directory not found: {wind_bc_dir}")

    profile_path = wind_bc_dir / "profile.dat"
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile.dat not found: {profile_path}")

    windrose_files = sorted(wind_bc_dir.glob("windrose_*m.csv"))
    if not windrose_files:
        raise FileNotFoundError(f"No windrose_*m.csv found under: {wind_bc_dir}")
    if len(windrose_files) != 1:
        raise RuntimeError(
            "Expected exactly one windrose_*m.csv under wind_bc, found: "
            + ", ".join(path.name for path in windrose_files)
        )

    raw_config = config_path.read_text(encoding="utf-8", errors="ignore")
    vis_dpi_raw = _safe_float(_parse_scalar_any(raw_config, ["crop_vis_dpi"]))
    figure_dpi = int(round(vis_dpi_raw)) if vis_dpi_raw is not None else 1200
    if figure_dpi <= 0:
        figure_dpi = 1200

    z_limit_si = _safe_float(_parse_scalar_any(raw_config, ["z_limit"]))
    if z_limit_si is None:
        z_pair = _parse_pair_any(raw_config, ["si_z_cfd"])
        base_height = _safe_float(_parse_scalar_any(raw_config, ["base_height"]))
        if z_pair is not None and base_height is not None:
            z_limit_est = max(float(z_pair[0]), float(z_pair[1])) - float(base_height)
            if math.isfinite(z_limit_est) and z_limit_est > 0.0:
                z_limit_si = z_limit_est

    output_dir = project_dir / "RESULTS" / "season_average"
    figure_dir = output_dir / "figures"
    summary_path = output_dir / "season_average_summary.txt"
    output_vtk_path = output_dir / "season_average.vtk"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    return ProjectLayout(
        config_path=config_path.resolve(),
        project_dir=project_dir,
        wind_bc_dir=wind_bc_dir,
        profile_path=profile_path,
        windrose_path=windrose_files[0].resolve(),
        output_dir=output_dir.resolve(),
        figure_dir=figure_dir.resolve(),
        summary_path=summary_path.resolve(),
        output_vtk_path=output_vtk_path.resolve(),
        figure_dpi=figure_dpi,
        z_limit_si=z_limit_si,
    )


def parse_profile_dat(profile_path: Path, z_limit_si: float | None) -> tuple[np.ndarray, np.ndarray]:
    raw_samples: list[tuple[float, float]] = []
    with profile_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = _strip_inline_comment(raw_line)
            if not line:
                continue
            for sep in (",", ";", "\t"):
                line = line.replace(sep, " ")
            parts = [part for part in line.split() if part]
            if len(parts) < 2:
                continue
            try:
                z_val = float(parts[0])
                u_val = float(parts[1])
            except Exception:
                continue
            if not (math.isfinite(z_val) and math.isfinite(u_val)):
                log(f"WARNING: invalid profile sample at line {line_no}, skipped")
                continue
            raw_samples.append((z_val, u_val))

    if len(raw_samples) < 2:
        raise RuntimeError("profile.dat must contain at least two valid z/U samples")

    raw_samples.sort(key=lambda item: item[0])
    z_vals: list[float] = []
    u_vals: list[float] = []
    for z_val, u_val in raw_samples:
        if z_vals and abs(z_val - z_vals[-1]) < 1.0e-9:
            u_vals[-1] = u_val
            continue
        z_vals.append(z_val)
        u_vals.append(u_val)

    if len(z_vals) < 2:
        raise RuntimeError("profile.dat collapsed to fewer than two unique z samples")

    z_array = np.asarray(z_vals, dtype=np.float64)
    u_array = np.asarray(u_vals, dtype=np.float64)

    if z_limit_si is not None and z_limit_si > 1.0 and z_array[-1] <= 1.5:
        log(
            "profile.dat vertical coordinate looks normalized. "
            f"Scale z by z_limit={format_number(z_limit_si)} m."
        )
        z_array = z_array * float(z_limit_si)

    if float(np.nanmax(u_array)) <= 0.0:
        raise RuntimeError("profile.dat has non-positive wind speeds only")

    return z_array, u_array


def interpolate_profile_speed(z_vals: np.ndarray, u_vals: np.ndarray, target_height: float) -> tuple[float, str]:
    if target_height <= float(z_vals[0]):
        if math.isclose(target_height, float(z_vals[0]), rel_tol=0.0, abs_tol=1.0e-9):
            return float(u_vals[0]), "exact at lower bound"
        z0, z1 = float(z_vals[0]), float(z_vals[1])
        u0, u1 = float(u_vals[0]), float(u_vals[1])
        factor = (target_height - z0) / (z1 - z0)
        return float(u0 + factor * (u1 - u0)), "linear extrapolation below profile range"

    if target_height >= float(z_vals[-1]):
        if math.isclose(target_height, float(z_vals[-1]), rel_tol=0.0, abs_tol=1.0e-9):
            return float(u_vals[-1]), "exact at upper bound"
        z0, z1 = float(z_vals[-2]), float(z_vals[-1])
        u0, u1 = float(u_vals[-2]), float(u_vals[-1])
        factor = (target_height - z0) / (z1 - z0)
        return float(u0 + factor * (u1 - u0)), "linear extrapolation above profile range"

    idx = int(np.searchsorted(z_vals, target_height, side="left"))
    z0, z1 = float(z_vals[idx - 1]), float(z_vals[idx])
    u0, u1 = float(u_vals[idx - 1]), float(u_vals[idx])
    if math.isclose(target_height, z0, rel_tol=0.0, abs_tol=1.0e-9):
        return u0, "exact sample"
    if math.isclose(target_height, z1, rel_tol=0.0, abs_tol=1.0e-9):
        return u1, "exact sample"
    factor = (target_height - z0) / (z1 - z0)
    return float(u0 + factor * (u1 - u0)), "linear interpolation"


def parse_windrose_height(path: Path) -> float:
    matches = re.findall(r"([0-9]+(?:[p\.][0-9]+)?)m", path.stem, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError(f"Cannot parse reference height from windrose file name: {path.name}")
    return _parse_custom_float(matches[-1], allow_inf=False)


def _parse_custom_float(text: str, allow_inf: bool = True) -> float:
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.replace("m/s", "")
    cleaned = cleaned.replace("ms-1", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("p", ".")
    if allow_inf and cleaned in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def _read_csv_rows(path: Path) -> tuple[list[list[str]], str]:
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "latin-1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="strict", newline="") as handle:
                rows = [[cell.strip() for cell in row] for row in csv.reader(handle)]
            return rows, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV {path} with supported encodings: {last_error}")


def _normalize_label(text: str) -> str:
    return re.sub(r"[^A-Z0-9\.]+", "", text.upper())


def snap_angle(angle: float) -> float:
    wrapped = angle % 360.0
    best = min(ANGLE_SEQUENCE, key=lambda candidate: abs(candidate - wrapped))
    if abs(best - wrapped) > 0.51:
        raise ValueError(f"Angle {angle} cannot be snapped to the 22.5-degree direction set")
    return float(best)


def parse_direction_label(text: str) -> float | None:
    raw = str(text).strip()
    if not raw:
        return None
    normalized = _normalize_label(raw)
    if normalized in IGNORE_DIRECTION_LABELS:
        return None
    if normalized in _COMPASS_TO_ANGLE:
        return _COMPASS_TO_ANGLE[normalized]

    match = re.search(r"[-+]?\d+(?:[\.]\d+)?", raw)
    if match:
        try:
            return snap_angle(float(match.group(0)))
        except Exception:
            return None
    return None


def parse_speed_bins(headers: list[str]) -> list[SpeedBin]:
    bins_raw: list[tuple[int, str, int, float, float | None]] = []
    finite_widths: list[float] = []

    for idx, cell in enumerate(headers):
        match = SPEED_BIN_RE.match(cell.strip())
        if not match:
            continue
        header = cell.strip()
        class_id = int(match.group("class_id"))
        lower = _parse_custom_float(match.group("lower"), allow_inf=False)
        upper_token = match.group("upper")
        upper = None if upper_token.lower() == "inf" else _parse_custom_float(upper_token, allow_inf=False)
        if upper is not None:
            width = upper - lower
            if width <= 0.0:
                raise RuntimeError(f"Invalid finite windrose bin width in header: {header}")
            finite_widths.append(width)
        bins_raw.append((idx, header, class_id, lower, upper))

    if not bins_raw:
        raise RuntimeError("No speed-bin columns like C7_7p0_9p4 found in windrose CSV header")
    if not finite_widths:
        raise RuntimeError("Windrose CSV contains no finite speed bins, cannot infer target speeds")

    default_inf_half_width = 0.5 * float(sum(finite_widths))
    bins: list[SpeedBin] = []
    for idx, header, class_id, lower, upper in bins_raw:
        if upper is None:
            target_speed = lower + default_inf_half_width
            log(
                "Infinite speed bin target inferred: "
                f"{header} -> {format_number(target_speed)} m/s "
                f"(lower {format_number(lower)} + 0.5 * finite-width-sum {format_number(sum(finite_widths))})"
            )
        else:
            target_speed = 0.5 * (lower + upper)
        bins.append(
            SpeedBin(
                column_index=idx,
                header=header,
                class_id=class_id,
                lower=lower,
                upper=upper,
                target_speed=target_speed,
            )
        )
    return bins


def parse_windrose_csv(path: Path) -> tuple[list[SpeedBin], dict[float, np.ndarray], float]:
    rows, encoding = _read_csv_rows(path)
    log(f"Windrose CSV encoding used: {encoding}")
    header_row_index = None
    speed_bins: list[SpeedBin] = []

    for row_idx, row in enumerate(rows):
        bins = parse_speed_bins(row) if any(SPEED_BIN_RE.match(cell.strip()) for cell in row) else []
        if bins:
            header_row_index = row_idx
            speed_bins = bins
            break

    if header_row_index is None:
        raise RuntimeError("Failed to locate windrose CSV header row containing speed-bin columns")

    table: dict[float, np.ndarray] = {angle: np.zeros(len(speed_bins), dtype=np.float64) for angle in ANGLE_SEQUENCE}
    min_speed_col = min(bin_item.column_index for bin_item in speed_bins)

    for row_idx in range(header_row_index + 1, len(rows)):
        row = rows[row_idx]
        if not row or not any(cell.strip() for cell in row):
            continue

        angle = None
        fallback_angle = None
        for cell in row[:min_speed_col]:
            cell_text = cell.strip()
            if not cell_text:
                continue
            normalized = _normalize_label(cell_text)
            if normalized in _COMPASS_TO_ANGLE:
                angle = _COMPASS_TO_ANGLE[normalized]
                break
            if fallback_angle is None:
                fallback_angle = parse_direction_label(cell_text)
        angle = angle if angle is not None else fallback_angle
        if angle is None:
            continue

        for bin_index, bin_item in enumerate(speed_bins):
            if bin_item.column_index >= len(row):
                continue
            cell = row[bin_item.column_index].strip()
            if not cell:
                continue
            cell_norm = cell.lower()
            if cell_norm in {"-", "--", "nan", "na", "n/a"}:
                continue
            try:
                value = _parse_custom_float(cell, allow_inf=False)
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            table[angle][bin_index] += value

    raw_total = float(sum(float(values.sum()) for values in table.values()))
    if raw_total <= 0.0:
        raise RuntimeError("Windrose CSV contains no valid probability values in direction-speed cells")

    if raw_total > 1.5:
        scale = 0.01
        probability_mode = "percentage"
    else:
        scale = 1.0
        probability_mode = "fraction"

    for angle in table:
        table[angle] = table[angle] * scale

    normalized_total = float(sum(float(values.sum()) for values in table.values()))
    log(
        "Windrose raw probability total: "
        f"{format_number(raw_total)} ({probability_mode} mode -> scaled total {format_number(normalized_total)})"
    )

    return speed_bins, table, normalized_total


def compute_direction_weights(
    speed_bins: list[SpeedBin],
    joint_probabilities: dict[float, np.ndarray],
    reference_speed: float,
    total_probability: float,
) -> list[DirectionWeight]:
    if reference_speed <= 0.0:
        raise RuntimeError("Reference speed must be positive")
    if total_probability <= 0.0:
        raise RuntimeError("Total windrose probability must be positive")

    normalized_factor = 1.0 / total_probability
    target_speeds = np.asarray([item.target_speed for item in speed_bins], dtype=np.float64)
    ratios = target_speeds / float(reference_speed)

    weights: list[DirectionWeight] = []
    log_rule("Direction Weights")
    for angle in ANGLE_SEQUENCE:
        joint = np.asarray(joint_probabilities.get(angle, np.zeros(len(speed_bins))), dtype=np.float64)
        joint_norm = joint * normalized_factor
        probability = float(joint_norm.sum())
        velocity_weight = float(np.dot(joint_norm, ratios))
        tke_weight = float(np.dot(joint_norm, ratios * ratios))
        label = ANGLE_TO_DIRECTION[angle]
        weights.append(
            DirectionWeight(
                angle=angle,
                label=label,
                probability=probability,
                velocity_weight=velocity_weight,
                tke_weight=tke_weight,
            )
        )
        log(
            f"Angle {format_number(angle):>5} deg / {label:>3}: "
            f"prob={probability:.6f}, vel_weight={velocity_weight:.6f}, tke_weight={tke_weight:.6f}"
        )
    return weights


def _extract_step(stem: str) -> int | None:
    match = VTK_STEP_RE.search(stem)
    if match:
        return int(match.group(1))
    return None


def _token_to_float(token: str) -> float | None:
    text = token.strip().lower().replace("p", ".")
    try:
        value = float(text)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_angle_from_vtk_name(path: Path) -> float | None:
    tokens = [token for token in re.split(r"_+", path.stem) if token]
    avg_index = None
    for idx, token in enumerate(tokens):
        if "avg" in token.lower():
            avg_index = idx
            break
    search_tokens = tokens if avg_index is None else tokens[:avg_index]
    candidates: list[float] = []
    for token in search_tokens:
        numeric = _token_to_float(token)
        if numeric is None:
            continue
        try:
            snapped = snap_angle(numeric)
        except Exception:
            continue
        candidates.append(snapped)
    if not candidates:
        return None
    return float(candidates[-1])


def _pick_preferred_file(files: list[Path]) -> Path:
    def sort_key(path: Path) -> tuple[int, int, float, str]:
        step = _extract_step(path.stem)
        has_cropped = 1 if "cropped" in path.stem.lower() else 0
        step_sort = step if step is not None else -1
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (has_cropped, step_sort, mtime, path.name)

    return sorted(files, key=sort_key, reverse=True)[0]


def scan_vtk_directory(directory: Path, label: str) -> VtkDirectoryChoice | None:
    if not directory.is_dir():
        return None
    vtk_files = sorted(Path(item).resolve() for item in glob.glob(str(directory / "*avg*.vtk")))
    if not vtk_files:
        return None

    grouped: dict[float, list[Path]] = {}
    ignored: list[str] = []
    for vtk_file in vtk_files:
        angle = parse_angle_from_vtk_name(vtk_file)
        if angle is None:
            ignored.append(vtk_file.name)
            continue
        grouped.setdefault(angle, []).append(vtk_file)

    if ignored:
        log(
            f"{label}: ignored {len(ignored)} avg-like VTK file(s) without a resolvable angle token: "
            + ", ".join(ignored[:6])
            + (" ..." if len(ignored) > 6 else "")
        )

    if not grouped:
        return None

    selected: dict[float, Path] = {}
    for angle, files in sorted(grouped.items()):
        chosen = _pick_preferred_file(files)
        selected[angle] = chosen
        if len(files) > 1:
            log(
                f"{label}: multiple VTK files for angle {format_number(angle)} deg, choose {chosen.name} "
                f"from {len(files)} candidates."
            )

    return VtkDirectoryChoice(label=label, directory=directory.resolve(), angle_to_file=selected)


def choose_vtk_source(project_dir: Path, weights: list[DirectionWeight]) -> VtkDirectoryChoice:
    candidates = [
        scan_vtk_directory(project_dir / "RESULTS" / "crop" / "cropped_vtk", "preferred cropped_vtk"),
        scan_vtk_directory(project_dir / "RESULTS" / "vtk", "fallback RESULTS/vtk"),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise FileNotFoundError(
            "No suitable *avg*.vtk files found under "
            f"{project_dir / 'RESULTS' / 'crop' / 'cropped_vtk'} or {project_dir / 'RESULTS' / 'vtk'}"
        )

    required_angles = [item.angle for item in weights if item.probability > 1.0e-10]
    log_rule("VTK Directory Selection")
    for candidate in candidates:
        available = ", ".join(
            f"{format_number(angle)}:{path.name}" for angle, path in sorted(candidate.angle_to_file.items())
        )
        log(f"Candidate {candidate.label}: {candidate.directory}")
        log(f"  Available angles: {available}")
        missing = [angle for angle in required_angles if angle not in candidate.angle_to_file]
        if not missing:
            log(f"Selected source directory: {candidate.directory} ({candidate.label})")
            return candidate
        log("  Missing required non-zero-probability angles: " + ", ".join(format_number(angle) for angle in missing))

    raise RuntimeError(
        "No VTK directory covers all directions with non-zero windrose probability. "
        "Check cropped_vtk / RESULTS/vtk contents."
    )


def _dtype_from_vtk_token(token: str) -> np.dtype:
    key = token.strip().lower()
    mapping = {
        "float": np.dtype(">f4"),
        "double": np.dtype(">f8"),
        "int": np.dtype(">i4"),
        "unsigned_int": np.dtype(">u4"),
        "short": np.dtype(">i2"),
        "unsigned_short": np.dtype(">u2"),
        "char": np.dtype("i1"),
        "unsigned_char": np.dtype("u1"),
    }
    if key not in mapping:
        raise ValueError(f"Unsupported VTK data type: {token}")
    return mapping[key]


def parse_legacy_vtk_header(vtk_file: Path) -> dict:
    dimensions = None
    origin = None
    spacing = None
    n_points = None
    n_cells = None
    fields: dict[str, dict] = {}
    data_section_started = False
    active_assoc = None
    active_tuples = None

    with vtk_file.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("DIMENSIONS"):
                parts = line.split()
                dimensions = (int(parts[1]), int(parts[2]), int(parts[3]))
                continue
            if line.startswith("ORIGIN"):
                parts = line.split()
                origin = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if line.startswith("SPACING"):
                parts = line.split()
                spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue

            if dimensions is None or origin is None or spacing is None:
                continue

            if line.startswith("POINT_DATA"):
                parts = line.split()
                n_points = int(parts[1])
                active_assoc = "POINT_DATA"
                active_tuples = n_points
                data_section_started = True
                continue

            if line.startswith("CELL_DATA"):
                parts = line.split()
                n_cells = int(parts[1])
                active_assoc = "CELL_DATA"
                active_tuples = n_cells
                data_section_started = True
                continue

            if not data_section_started:
                continue

            if line.startswith("SCALARS"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid SCALARS line in {vtk_file}: {line}")
                name = parts[1]
                vtk_type_token = parts[2]
                dtype = _dtype_from_vtk_token(vtk_type_token)
                n_comp = int(parts[3]) if len(parts) > 3 else 1
                tuple_count = active_tuples if active_tuples is not None else n_points
                if tuple_count is None:
                    raise ValueError(f"SCALARS without POINT_DATA/CELL_DATA in {vtk_file}: {line}")
                lookup_line = handle.readline().decode("utf-8", errors="replace").strip()
                if not lookup_line.startswith("LOOKUP_TABLE"):
                    raise ValueError(f"Expected LOOKUP_TABLE after SCALARS {name}, got: {lookup_line}")
                data_offset = handle.tell()
                fields[name] = {
                    "offset": data_offset,
                    "components": n_comp,
                    "dtype": dtype,
                    "kind": "SCALARS",
                    "vtk_type_token": vtk_type_token,
                    "tuples": int(tuple_count),
                    "association": active_assoc,
                }
                handle.seek(data_offset + int(tuple_count) * n_comp * dtype.itemsize)
                continue

            if line.startswith("VECTORS"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid VECTORS line in {vtk_file}: {line}")
                name = parts[1]
                vtk_type_token = parts[2]
                dtype = _dtype_from_vtk_token(vtk_type_token)
                tuple_count = active_tuples if active_tuples is not None else n_points
                if tuple_count is None:
                    raise ValueError(f"VECTORS without POINT_DATA/CELL_DATA in {vtk_file}: {line}")
                data_offset = handle.tell()
                fields[name] = {
                    "offset": data_offset,
                    "components": 3,
                    "dtype": dtype,
                    "kind": "VECTORS",
                    "vtk_type_token": vtk_type_token,
                    "tuples": int(tuple_count),
                    "association": active_assoc,
                }
                handle.seek(data_offset + int(tuple_count) * 3 * dtype.itemsize)
                continue

            if line.startswith("FIELD"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid FIELD line in {vtk_file}: {line}")
                n_arrays = int(parts[2])
                for _ in range(n_arrays):
                    arr_line = handle.readline().decode("utf-8", errors="replace").strip()
                    while arr_line == "":
                        arr_line = handle.readline().decode("utf-8", errors="replace").strip()
                    arr_parts = arr_line.split()
                    if len(arr_parts) < 4:
                        raise ValueError(f"Invalid FIELD array line in {vtk_file}: {arr_line}")
                    name = arr_parts[0]
                    n_comp = int(arr_parts[1])
                    n_tuples = int(arr_parts[2])
                    vtk_type_token = arr_parts[3]
                    dtype = _dtype_from_vtk_token(vtk_type_token)
                    data_offset = handle.tell()
                    fields[name] = {
                        "offset": data_offset,
                        "components": n_comp,
                        "dtype": dtype,
                        "kind": "FIELD",
                        "vtk_type_token": vtk_type_token,
                        "tuples": n_tuples,
                        "association": active_assoc,
                    }
                    handle.seek(data_offset + n_comp * n_tuples * dtype.itemsize)
                continue

    if dimensions is None or origin is None or spacing is None:
        raise ValueError(f"Invalid VTK header in {vtk_file}: missing DIMENSIONS/ORIGIN/SPACING")
    if n_points is None:
        nx, ny, nz = dimensions
        n_points = int(nx * ny * nz)

    return {
        "dimensions": dimensions,
        "origin": origin,
        "spacing": spacing,
        "n_points": n_points,
        "n_cells": n_cells,
        "fields": fields,
    }


def _normalize_field_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_vector_field_name(vtk_info: dict) -> str:
    fields = list(vtk_info["fields"].keys())
    if "u_avg" in vtk_info["fields"]:
        return "u_avg"

    candidates = ["u_avg", "velocity", "wind", "uvw", "u"]
    by_lower = {field.lower(): field for field in fields}
    by_norm = {_normalize_field_token(field): field for field in fields}
    for candidate in candidates:
        found = by_lower.get(candidate.lower()) or by_norm.get(_normalize_field_token(candidate))
        if found is not None:
            return found

    n_points = int(vtk_info.get("n_points", 0))
    for name, meta in vtk_info["fields"].items():
        tuples = int(meta.get("tuples", n_points))
        if tuples == n_points and int(meta.get("components", 1)) >= 3:
            return name
    raise ValueError(f"VTK missing wind vector field. Available fields={fields}")


def resolve_tke_field_name(vtk_info: dict) -> str:
    fields = list(vtk_info["fields"].keys())
    if "tke" in vtk_info["fields"]:
        return "tke"

    candidates = ["tke", "k", "tke_avg", "k_avg", "sgs_tke", "turbulence_k", "tkenergy"]
    by_lower = {field.lower(): field for field in fields}
    by_norm = {_normalize_field_token(field): field for field in fields}
    for candidate in candidates:
        found = by_lower.get(candidate.lower()) or by_norm.get(_normalize_field_token(candidate))
        if found is not None:
            return found
    raise ValueError(f"VTK missing TKE scalar field. Available fields={fields}")


def validate_headers(vtk_choice: VtkDirectoryChoice, weights: list[DirectionWeight]) -> tuple[dict, dict[float, dict]]:
    reference_header = None
    header_by_angle: dict[float, dict] = {}

    for item in weights:
        if item.probability <= 1.0e-10:
            continue
        vtk_file = vtk_choice.angle_to_file[item.angle]
        log(f"Read header for angle {format_number(item.angle)} deg from {vtk_file.name}")
        info = parse_legacy_vtk_header(vtk_file)
        vector_name = resolve_vector_field_name(info)
        tke_name = resolve_tke_field_name(info)
        log(
            "  Grid signature: "
            f"dims={info['dimensions']}, origin={info['origin']}, spacing={info['spacing']}, "
            f"vector={vector_name}, tke={tke_name}"
        )
        if reference_header is None:
            reference_header = info
        else:
            signature = (
                tuple(reference_header["dimensions"]),
                tuple(reference_header["origin"]),
                tuple(reference_header["spacing"]),
            )
            current_signature = (tuple(info["dimensions"]), tuple(info["origin"]), tuple(info["spacing"]))
            if current_signature != signature:
                raise RuntimeError(
                    "Input VTK grids are not identical across directions. "
                    f"Reference={signature}, current={current_signature}, file={vtk_file}"
                )
        header_by_angle[item.angle] = info

    if reference_header is None:
        raise RuntimeError("No non-zero-probability direction remained after windrose weighting")
    return reference_header, header_by_angle


def create_accumulators(output_dir: Path, dims: tuple[int, int, int]) -> AccumulatorVolumes:
    nx, ny, nz = dims
    n_points = int(nx) * int(ny) * int(nz)
    total_bytes = n_points * 5 * np.dtype(np.float32).itemsize
    use_memmap = total_bytes > 512 * 1024 * 1024

    if use_memmap:
        temp_dir = Path(tempfile.mkdtemp(prefix="season_average_tmp_", dir=output_dir))
        log(
            "Accumulator strategy: disk-backed memmap "
            f"(estimated data volume {human_bytes(total_bytes)}) at {temp_dir}"
        )
        u_arr = np.memmap(temp_dir / "u_accum.bin", mode="w+", dtype=np.float32, shape=(nz, ny, nx, 3))
        vm_arr = np.memmap(temp_dir / "vm_accum.bin", mode="w+", dtype=np.float32, shape=(nz, ny, nx))
        tke_arr = np.memmap(temp_dir / "tke_accum.bin", mode="w+", dtype=np.float32, shape=(nz, ny, nx))
        u_arr[:] = 0.0
        vm_arr[:] = 0.0
        tke_arr[:] = 0.0
        return AccumulatorVolumes(True, temp_dir, u_arr, vm_arr, tke_arr)

    log(f"Accumulator strategy: in-memory arrays (estimated data volume {human_bytes(total_bytes)})")
    return AccumulatorVolumes(
        False,
        None,
        np.zeros((nz, ny, nx, 3), dtype=np.float32),
        np.zeros((nz, ny, nx), dtype=np.float32),
        np.zeros((nz, ny, nx), dtype=np.float32),
    )


def _field_meta(vtk_info: dict, field_name: str) -> tuple[int, int, int, int, np.dtype, dict]:
    if field_name not in vtk_info["fields"]:
        raise KeyError(f"Field '{field_name}' not found")
    nx, ny, nz = vtk_info["dimensions"]
    field = vtk_info["fields"][field_name]
    n_comp = int(field["components"])
    dtype = field["dtype"]
    tuple_count = int(field.get("tuples", vtk_info["n_points"]))
    if tuple_count != int(vtk_info["n_points"]):
        assoc = field.get("association")
        raise RuntimeError(
            f"Field '{field_name}' has tuples={tuple_count}, association={assoc}; only POINT_DATA is supported"
        )
    return nx, ny, nz, n_comp, dtype, field


def iter_field_slices(vtk_path: Path, vtk_info: dict, field_name: str):
    nx, ny, nz, n_comp, dtype, field = _field_meta(vtk_info, field_name)
    n_values_per_slice = nx * ny * n_comp
    with vtk_path.open("rb") as handle:
        handle.seek(int(field["offset"]))
        for z_index in range(nz):
            raw = np.fromfile(handle, dtype=dtype, count=n_values_per_slice)
            if raw.size != n_values_per_slice:
                raise RuntimeError(f"Failed to read full z-slice for field '{field_name}' at z={z_index}")
            native = raw.astype(np.float32, copy=False)
            if n_comp == 1:
                yield z_index, native.reshape((ny, nx))
            else:
                yield z_index, native.reshape((ny, nx, n_comp))


def progress_step_for_total(total: int) -> int:
    if total <= 50:
        return 1
    if total <= 150:
        return 2
    return 5


def log_progress(prefix: str, current: int, total: int, last_percent: int) -> int:
    percent = int(round(100.0 * current / max(total, 1)))
    step = progress_step_for_total(total)
    if current == total or last_percent < 0 or percent - last_percent >= step:
        log(f"{prefix}: {current}/{total} ({percent}%)")
        return percent
    return last_percent


def accumulate_direction(
    vtk_path: Path,
    vtk_info: dict,
    vector_field: str,
    tke_field: str,
    velocity_weight: float,
    tke_weight: float,
    accum: AccumulatorVolumes,
    angle_label: str,
) -> None:
    _, _, nz = vtk_info["dimensions"]

    if abs(velocity_weight) > 0.0:
        log(
            f"Start vector accumulation for {angle_label}: field={vector_field}, "
            f"velocity_weight={velocity_weight:.6f}"
        )
        last_percent = -1
        vel_weight32 = np.float32(velocity_weight)
        for z_index, vel in iter_field_slices(vtk_path, vtk_info, vector_field):
            if vel.ndim != 3 or vel.shape[2] < 3:
                raise RuntimeError(f"Vector field '{vector_field}' in {vtk_path.name} must have 3 components")
            accum.u[z_index] += vel_weight32 * vel[:, :, :3]
            mag = np.sqrt(vel[:, :, 0] * vel[:, :, 0] + vel[:, :, 1] * vel[:, :, 1] + vel[:, :, 2] * vel[:, :, 2])
            accum.vm[z_index] += vel_weight32 * mag
            last_percent = log_progress(f"  {angle_label} vector progress", z_index + 1, nz, last_percent)
    else:
        log(f"Skip vector accumulation for {angle_label}: velocity_weight is zero")

    if abs(tke_weight) > 0.0:
        log(f"Start TKE accumulation for {angle_label}: field={tke_field}, tke_weight={tke_weight:.6f}")
        last_percent = -1
        tke_weight32 = np.float32(tke_weight)
        for z_index, tke in iter_field_slices(vtk_path, vtk_info, tke_field):
            if tke.ndim != 2:
                raise RuntimeError(f"TKE field '{tke_field}' in {vtk_path.name} must be scalar")
            accum.tke[z_index] += tke_weight32 * tke
            last_percent = log_progress(f"  {angle_label} TKE progress", z_index + 1, nz, last_percent)
    else:
        log(f"Skip TKE accumulation for {angle_label}: tke_weight is zero")


def synthesize_average_fields(
    vtk_choice: VtkDirectoryChoice,
    weights: list[DirectionWeight],
    header_by_angle: dict[float, dict],
    reference_header: dict,
    output_dir: Path,
) -> AccumulatorVolumes:
    dims = tuple(int(v) for v in reference_header["dimensions"])
    accum = create_accumulators(output_dir, dims)

    try:
        for item in weights:
            if item.probability <= 1.0e-10:
                log(
                    f"Skip angle {format_number(item.angle)} deg / {item.label}: "
                    "windrose probability is zero"
                )
                continue

            vtk_path = vtk_choice.angle_to_file[item.angle]
            vtk_info = header_by_angle[item.angle]
            vector_field = resolve_vector_field_name(vtk_info)
            tke_field = resolve_tke_field_name(vtk_info)

            log_rule(f"Accumulate {format_number(item.angle)} deg / {item.label}")
            log(f"Source VTK: {vtk_path}")
            log(f"Probability contribution: {item.probability:.6f}")
            accumulate_direction(
                vtk_path=vtk_path,
                vtk_info=vtk_info,
                vector_field=vector_field,
                tke_field=tke_field,
                velocity_weight=item.velocity_weight,
                tke_weight=item.tke_weight,
                accum=accum,
                angle_label=f"{format_number(item.angle)} deg / {item.label}",
            )
        accum.flush()
        return accum
    except Exception:
        accum.cleanup()
        raise


def write_legacy_vtk(output_path: Path, reference_header: dict, accum: AccumulatorVolumes, windrose_name: str) -> None:
    nx, ny, nz = reference_header["dimensions"]
    ox, oy, oz = reference_header["origin"]
    dx, dy, dz = reference_header["spacing"]
    n_points = int(nx) * int(ny) * int(nz)

    log_rule("Write Output VTK")
    log(f"Writing synthesized VTK to: {output_path}")
    with output_path.open("wb") as handle:
        header_lines = [
            "# vtk DataFile Version 3.0",
            f"Season average synthesized from {windrose_name}",
            "BINARY",
            "DATASET STRUCTURED_POINTS",
            f"DIMENSIONS {nx} {ny} {nz}",
            f"ORIGIN {ox} {oy} {oz}",
            f"SPACING {dx} {dy} {dz}",
            "",
            f"POINT_DATA {n_points}",
        ]
        handle.write(("\n".join(header_lines) + "\n").encode("utf-8"))

        log("  Write VECTORS u_avg")
        handle.write(b"VECTORS u_avg float\n")
        last_percent = -1
        for z_index in range(nz):
            handle.write(np.asarray(accum.u[z_index], dtype=">f4").tobytes(order="C"))
            last_percent = log_progress("  u_avg write progress", z_index + 1, nz, last_percent)

        scalar_specs = [
            ("u", accum.u[..., 0]),
            ("v", accum.u[..., 1]),
            ("w", accum.u[..., 2]),
            ("vm", accum.vm),
            ("tke", accum.tke),
        ]
        for field_name, field_array in scalar_specs:
            log(f"  Write SCALARS {field_name}")
            handle.write(f"SCALARS {field_name} float\n".encode("utf-8"))
            handle.write(b"LOOKUP_TABLE default\n")
            last_percent = -1
            for z_index in range(nz):
                handle.write(np.asarray(field_array[z_index], dtype=">f4").tobytes(order="C"))
                last_percent = log_progress(f"  {field_name} write progress", z_index + 1, nz, last_percent)

    log(f"Finished writing synthesized VTK: {output_path}")


def height_to_z_index(height_m: float) -> int:
    return int(round((height_m - BASE_HEIGHT_M) / LAYER_STEP_M))


def build_height_plan(vtk_info: dict, target_heights: list[int]) -> list[dict]:
    nz = int(vtk_info["dimensions"][2])
    plan = []
    for height in target_heights:
        z_index = height_to_z_index(height)
        plan.append(
            {
                "target_height": int(height),
                "z_index": int(z_index),
                "valid": 0 <= z_index < nz,
                "mapped_height": BASE_HEIGHT_M + LAYER_STEP_M * z_index,
            }
        )
    return plan


def read_z_slice(vtk_file: Path, vtk_info: dict, field_name: str, z_index: int) -> np.ndarray:
    nx, ny, nz = vtk_info["dimensions"]
    if not (0 <= z_index < nz):
        raise ValueError(f"z index out of range: {z_index}")
    _, _, _, n_comp, dtype, field = _field_meta(vtk_info, field_name)
    n_values = nx * ny * n_comp
    byte_offset = int(field["offset"]) + z_index * n_values * dtype.itemsize
    with vtk_file.open("rb") as handle:
        handle.seek(byte_offset)
        raw = np.fromfile(handle, dtype=dtype, count=n_values)
    if raw.size != n_values:
        raise RuntimeError(f"Read failure for {field_name} at z={z_index} in {vtk_file}")
    native = raw.astype(np.float32, copy=False)
    if n_comp == 1:
        return native.reshape((ny, nx))
    return native.reshape((ny, nx, n_comp))


def _grid_3x3():
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    return fig, axes.ravel()


def plot_scalar_figure(
    vtk_file: Path,
    vtk_info: dict,
    field_name: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    output_path: Path,
    dpi: int,
    symmetric: bool = False,
) -> None:
    log_rule(f"Plot {field_name}")
    log(f"Generate figure for field {field_name} -> {output_path.name}")

    height_plan = build_height_plan(vtk_info, TARGET_HEIGHTS_M)
    ox, oy, _ = vtk_info["origin"]
    dx, dy, _ = vtk_info["spacing"]
    nx, ny, _ = vtk_info["dimensions"]
    extent = [float(ox), float(ox + dx * (nx - 1)), float(oy), float(oy + dy * (ny - 1))]

    slices: dict[int, np.ndarray] = {}
    lows: list[float] = []
    highs: list[float] = []

    for item in height_plan:
        if not item["valid"]:
            continue
        z_index = item["z_index"]
        arr = read_z_slice(vtk_file, vtk_info, field_name, z_index)
        if arr.ndim != 2:
            raise RuntimeError(f"Scalar field '{field_name}' expected at z={z_index}")
        slices[z_index] = arr
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            lows.append(float(np.nanpercentile(valid, 2)))
            highs.append(float(np.nanpercentile(valid, 98)))
        log(f"  Loaded z={z_index} ({item['target_height']} m) for figure {field_name}")

    if lows and highs:
        if symmetric:
            abs_max = max(abs(min(lows)), abs(max(highs)))
            if not math.isfinite(abs_max) or abs_max <= 0.0:
                abs_max = 1.0
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        else:
            vmin = min(lows)
            vmax = max(highs)
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0) if symmetric else Normalize(vmin=0.0, vmax=1.0)

    fig, axes = _grid_3x3()
    mappable = None
    for ax, item in zip(axes, height_plan):
        height = item["target_height"]
        z_index = item["z_index"]
        if not item["valid"]:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{height}m\nz={z_index}\nOUT_OF_RANGE", ha="center", va="center", fontsize=12)
            continue
        im = ax.imshow(slices[z_index], origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{height}m (z={z_index})", fontsize=11)
        mappable = im

    for ax in axes[len(height_plan):]:
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.86, pad=0.02)
        cbar.set_label(colorbar_label)

    fig.suptitle(
        f"{title} (9 layers) | {vtk_file.name}\n"
        f"height=-50+10*z, local grid={nx}x{ny}",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    log(f"Saved figure: {output_path}")


def write_summary(
    summary_path: Path,
    layout: ProjectLayout,
    reference_speed: float,
    reference_speed_mode: str,
    windrose_height: float,
    speed_bins: list[SpeedBin],
    weights: list[DirectionWeight],
    vtk_choice: VtkDirectoryChoice,
    reference_header: dict,
    total_probability: float,
) -> None:
    lines: list[str] = []
    lines.append("Season Average Summary")
    lines.append("======================")
    lines.append(f"Config path: {layout.config_path}")
    lines.append(f"Project dir: {layout.project_dir}")
    lines.append(f"Windrose file: {layout.windrose_path}")
    lines.append(f"Profile file: {layout.profile_path}")
    lines.append(f"Selected VTK dir: {vtk_choice.directory}")
    lines.append(f"Output VTK: {layout.output_vtk_path}")
    lines.append(f"Figure dir: {layout.figure_dir}")
    lines.append("")
    lines.append(f"Windrose reference height: {format_number(windrose_height)} m")
    lines.append(f"Reference speed from profile: {format_number(reference_speed)} m/s ({reference_speed_mode})")
    lines.append(f"Normalized total probability used: {total_probability:.9f}")
    lines.append(
        "Grid signature: "
        f"dims={reference_header['dimensions']}, origin={reference_header['origin']}, "
        f"spacing={reference_header['spacing']}"
    )
    lines.append("")
    lines.append("Speed bins:")
    for item in speed_bins:
        upper_text = "inf" if item.upper is None else format_number(item.upper)
        lines.append(
            f"  {item.header}: lower={format_number(item.lower)} m/s, "
            f"upper={upper_text} m/s, target={format_number(item.target_speed)} m/s"
        )
    lines.append("")
    lines.append("Direction weights:")
    for item in weights:
        vtk_name = vtk_choice.angle_to_file.get(item.angle)
        vtk_text = vtk_name.name if vtk_name is not None else "-"
        lines.append(
            f"  {format_number(item.angle):>5} deg / {item.label:>3}: "
            f"prob={item.probability:.9f}, "
            f"vel_weight={item.velocity_weight:.9f}, "
            f"tke_weight={item.tke_weight:.9f}, "
            f"vtk={vtk_text}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        usage="%(prog)s <config_path>",
        description="Synthesize seasonal average wind/tke VTK from directional average VTK files and windrose probabilities.",
    )
    parser.add_argument("config_path", help="Path to LUW-family config (*.luw, *.luwdg, *.luwpf)")
    args = parser.parse_args()

    start_time = time.time()
    layout = load_project_layout(Path(args.config_path))

    log_rule("Season Average Synthesis")
    log(f"Config path: {layout.config_path}")
    log(f"Project dir: {layout.project_dir}")
    log(f"wind_bc dir: {layout.wind_bc_dir}")
    log(f"profile.dat: {layout.profile_path}")
    log(f"windrose CSV: {layout.windrose_path}")
    log(f"Output dir: {layout.output_dir}")
    log(f"Figures dir: {layout.figure_dir}")
    log(f"Figure DPI: {layout.figure_dpi}")
    if layout.z_limit_si is not None:
        log(f"Profile z_limit candidate from config: {format_number(layout.z_limit_si)} m")
    else:
        log("Profile z_limit candidate from config: not available")

    z_vals, u_vals = parse_profile_dat(layout.profile_path, layout.z_limit_si)
    log(
        f"profile.dat parsed: {len(z_vals)} unique samples, "
        f"z range {format_number(float(z_vals[0]))} -> {format_number(float(z_vals[-1]))} m, "
        f"U range {format_number(float(np.min(u_vals)))} -> {format_number(float(np.max(u_vals)))} m/s"
    )

    windrose_height = parse_windrose_height(layout.windrose_path)
    reference_speed, reference_speed_mode = interpolate_profile_speed(z_vals, u_vals, windrose_height)
    log(
        f"Windrose reference height parsed from file name: {format_number(windrose_height)} m; "
        f"profile reference speed = {format_number(reference_speed)} m/s ({reference_speed_mode})"
    )

    speed_bins, joint_probabilities, total_probability = parse_windrose_csv(layout.windrose_path)
    log_rule("Speed Bins")
    for item in speed_bins:
        upper_text = "inf" if item.upper is None else format_number(item.upper)
        log(
            f"{item.header}: lower={format_number(item.lower)} m/s, "
            f"upper={upper_text} m/s, target={format_number(item.target_speed)} m/s"
        )

    weights = compute_direction_weights(
        speed_bins=speed_bins,
        joint_probabilities=joint_probabilities,
        reference_speed=reference_speed,
        total_probability=total_probability,
    )

    vtk_choice = choose_vtk_source(layout.project_dir, weights)
    reference_header, header_by_angle = validate_headers(vtk_choice, weights)

    accum = synthesize_average_fields(
        vtk_choice=vtk_choice,
        weights=weights,
        header_by_angle=header_by_angle,
        reference_header=reference_header,
        output_dir=layout.output_dir,
    )

    try:
        write_legacy_vtk(
            output_path=layout.output_vtk_path,
            reference_header=reference_header,
            accum=accum,
            windrose_name=layout.windrose_path.name,
        )

        write_summary(
            summary_path=layout.summary_path,
            layout=layout,
            reference_speed=reference_speed,
            reference_speed_mode=reference_speed_mode,
            windrose_height=windrose_height,
            speed_bins=speed_bins,
            weights=weights,
            vtk_choice=vtk_choice,
            reference_header=reference_header,
            total_probability=total_probability,
        )

        output_info = parse_legacy_vtk_header(layout.output_vtk_path)
        plot_scalar_figure(
            vtk_file=layout.output_vtk_path,
            vtk_info=output_info,
            field_name="u",
            title="Season Average U Component",
            colorbar_label="u (m/s)",
            cmap="coolwarm",
            output_path=layout.figure_dir / "season_average_u_9layers.png",
            dpi=max(80, int(layout.figure_dpi)),
            symmetric=True,
        )
        plot_scalar_figure(
            vtk_file=layout.output_vtk_path,
            vtk_info=output_info,
            field_name="v",
            title="Season Average V Component",
            colorbar_label="v (m/s)",
            cmap="coolwarm",
            output_path=layout.figure_dir / "season_average_v_9layers.png",
            dpi=max(80, int(layout.figure_dpi)),
            symmetric=True,
        )
        plot_scalar_figure(
            vtk_file=layout.output_vtk_path,
            vtk_info=output_info,
            field_name="w",
            title="Season Average W Component",
            colorbar_label="w (m/s)",
            cmap="coolwarm",
            output_path=layout.figure_dir / "season_average_w_9layers.png",
            dpi=max(80, int(layout.figure_dpi)),
            symmetric=True,
        )
        plot_scalar_figure(
            vtk_file=layout.output_vtk_path,
            vtk_info=output_info,
            field_name="vm",
            title="Season Average Velocity Magnitude",
            colorbar_label="vm (m/s)",
            cmap="turbo",
            output_path=layout.figure_dir / "season_average_vm_9layers.png",
            dpi=max(80, int(layout.figure_dpi)),
            symmetric=False,
        )
        plot_scalar_figure(
            vtk_file=layout.output_vtk_path,
            vtk_info=output_info,
            field_name="tke",
            title="Season Average TKE",
            colorbar_label="tke (m^2/s^2)",
            cmap="magma",
            output_path=layout.figure_dir / "season_average_tke_9layers.png",
            dpi=max(80, int(layout.figure_dpi)),
            symmetric=False,
        )
    finally:
        accum.cleanup()

    elapsed = time.time() - start_time
    log_rule("Finished")
    log(f"Synthesized VTK: {layout.output_vtk_path}")
    log(f"Summary file: {layout.summary_path}")
    log(f"Figures directory: {layout.figure_dir}")
    log(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()

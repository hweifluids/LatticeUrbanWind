#!/usr/bin/env python3
"""
Geo-only batch VTK crop / visualize workflow.

Logic:
1) Read one config path from the command line as the only accepted argument.
2) Accept `.luw`, `.luwdg`, or `.luwpf` config files with the same parsing logic.
3) Only geo mode is supported here. Local-XY mode is intentionally removed.
4) Auto-discover the input VTK directory relative to the config file:
   `<config_dir>/RESULTS/vtk`, then `<config_dir>/RESULTS`,
   then `<config_dir>/proj_temp/vtk`, then `<config_dir>`.
   If `crop_debug_input_dir` is set, it overrides this fallback logic.
5) Read visualization DPI from `crop_vis_dpi`; if missing, fall back to `1200`.
6) Use the geo bounds in the config file to crop every discovered VTK.
7) Export one cropped VTK and one set of figures for each source file.

Expected keys in the config file and typical values:
    crop_debug_file_glob = "*_avg-*.vtk,*_avg_*.vtk"

    target_crs = "EPSG:32651"
    crop_min_lon = 121.4
    crop_max_lon = 121.6
    crop_min_lat = 31.1800
    crop_max_lat = 31.3050

    // Full geographic extent of the raw VTK domain used in local<->lon/lat mapping.
    cut_lon_manual = [121.3, 121.7]
    cut_lat_manual = [31.1, 31.4]
    utm_crs = "EPSG:32651"
    rotate_deg = 0.774936

    crop_cell_size = 10
    crop_vis_dpi = 1200
    crop_debug_input_dir = "RESULTS/vtk"

Notes:
- `crop_min_lon/crop_max_lon/crop_min_lat/crop_max_lat` define the crop area.
- `cut_lon_manual/cut_lat_manual` define the full raw VTK geographic domain.
- If `crop_debug_file_glob` is missing, the script falls back to `*_avg-*.vtk,*_avg_*.vtk`.
- If `crop_cell_size` is missing, the script falls back to `cell_size`.
- If `crop_vis_dpi` is missing, the script falls back to `1200`.
- `crop_debug_input_dir` is resolved relative to `<config_dir>` even if it
  starts with `/` or `\\`.
- If `target_crs` is missing, the script falls back to `utm_crs`.
- Output is written to `<config_dir>/RESULTS/crop`.
- Existing output files with the same name are overwritten.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pyproj import Transformer
from scipy.ndimage import map_coordinates

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

RAW_AVG_BASENAME_RE = re.compile(r"^.+_avg[-_](\d+)\.vtk$")
TARGET_HEIGHTS_M = [50, 100, 150, 200, 300, 400, 500, 600, 800]
BASE_HEIGHT_M = -50.0
LAYER_STEP_M = 10.0
MASK_CHUNK_ROWS = 256
BOUNDARY_SAMPLES = 800


@dataclass
class RuntimeConfig:
    CONFIG_PATH: str
    DATA_DIR: str
    DATA_DIR_SOURCE: str
    OUTPUT_DIR: str
    TARGET_CRS: str
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    cell_size: float = 10.0
    vis_dpi: int = 1200
    crop_debug_file_glob: str = "*_avg-*.vtk,*_avg_*.vtk"
    WIND_VTK_CONFIG: dict = field(default_factory=dict)


def _strip_inline_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _parse_scalar(text: str, key: str) -> str | None:
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", text)
    if not m:
        return None
    val = _strip_inline_comment(m.group(1))
    if not val:
        return None
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val or None


def _parse_scalar_any(text: str, keys: list[str]) -> str | None:
    for key in keys:
        val = _parse_scalar(text, key)
        if val is not None:
            return val
    return None


def _parse_pair(text: str, key: str) -> tuple[float, float] | None:
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", text)
    if not m:
        return None
    val = _strip_inline_comment(m.group(1))
    if not (val.startswith("[") and val.endswith("]")):
        return None
    parts = [p.strip() for p in val[1:-1].split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def _parse_pair_any(text: str, keys: list[str]) -> tuple[float, float] | None:
    for key in keys:
        val = _parse_pair(text, key)
        if val is not None:
            return val
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


def _infer_utm_from_lonlat(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    if lat >= 0.0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def _resolve_debug_input_dir(base_dir: Path, raw_path: str) -> Path:
    normalized = re.sub(r"^[\\/]+", "", raw_path.strip())
    target = (base_dir / normalized).resolve()
    if not target.is_dir():
        raise FileNotFoundError(
            "crop_debug_input_dir resolved to a non-directory path: "
            f"{target}"
        )
    return target


def _discover_default_input_dir(base_dir: Path) -> tuple[Path, str]:
    candidates = [
        base_dir / "RESULTS" / "vtk",
        base_dir / "RESULTS",
        base_dir / "proj_temp" / "vtk",
        base_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            rel = candidate.relative_to(base_dir)
            rel_text = str(rel).replace("\\", "/")
            return resolved, rel_text if rel_text else "."
    raise FileNotFoundError(
        "Could not auto-discover an input VTK directory relative to the config file. "
        "Expected one of: <config_dir>/RESULTS/vtk, <config_dir>/RESULTS, "
        "<config_dir>/proj_temp/vtk, or <config_dir>."
    )


def _load_runtime_config(luw_path: Path) -> RuntimeConfig:
    if not luw_path.is_file():
        raise FileNotFoundError(f"Config file not found: {luw_path}")

    raw = luw_path.read_text(encoding="utf-8", errors="ignore")
    base_dir = luw_path.resolve().parent

    crop_debug_file_glob = _parse_scalar_any(raw, ["crop_debug_file_glob"]) or "*_avg-*.vtk,*_avg_*.vtk"

    domain_lon = _parse_pair_any(raw, ["cut_lon_manual", "domain_lon"])
    domain_lat = _parse_pair_any(raw, ["cut_lat_manual", "domain_lat"])

    crop_min_lon = _safe_float(_parse_scalar_any(raw, ["crop_min_lon"]))
    crop_max_lon = _safe_float(_parse_scalar_any(raw, ["crop_max_lon"]))
    crop_min_lat = _safe_float(_parse_scalar_any(raw, ["crop_min_lat"]))
    crop_max_lat = _safe_float(_parse_scalar_any(raw, ["crop_max_lat"]))

    if (crop_min_lon is None) != (crop_max_lon is None):
        raise ValueError("Config must define both crop_min_lon and crop_max_lon together.")
    if (crop_min_lat is None) != (crop_max_lat is None):
        raise ValueError("Config must define both crop_min_lat and crop_max_lat together.")

    if crop_min_lon is None and crop_max_lon is None:
        if domain_lon is None:
            raise ValueError("Config missing crop_min_lon/crop_max_lon and cut_lon_manual/domain_lon")
        min_lon, max_lon = sorted(domain_lon)
    else:
        min_lon, max_lon = sorted((float(crop_min_lon), float(crop_max_lon)))

    if crop_min_lat is None and crop_max_lat is None:
        if domain_lat is None:
            raise ValueError("Config missing crop_min_lat/crop_max_lat and cut_lat_manual/domain_lat")
        min_lat, max_lat = sorted(domain_lat)
    else:
        min_lat, max_lat = sorted((float(crop_min_lat), float(crop_max_lat)))

    if domain_lon is None:
        domain_lon = (min_lon, max_lon)
    if domain_lat is None:
        domain_lat = (min_lat, max_lat)

    lon_lo, lon_hi = sorted(domain_lon)
    lat_lo, lat_hi = sorted(domain_lat)

    target_crs = _parse_scalar_any(raw, ["target_crs", "TARGET_CRS", "utm_crs"])
    if not target_crs:
        lon_c = 0.5 * (min_lon + max_lon)
        lat_c = 0.5 * (min_lat + max_lat)
        target_crs = _infer_utm_from_lonlat(lon_c, lat_c)

    utm_crs = _parse_scalar_any(raw, ["utm_crs", "UTM_CRS"]) or target_crs
    rotate_deg = _safe_float(_parse_scalar_any(raw, ["rotate_deg"]))
    cell_size = _safe_float(_parse_scalar_any(raw, ["crop_cell_size", "cell_size"])) or 10.0
    vis_dpi_raw = _safe_float(_parse_scalar_any(raw, ["crop_vis_dpi"]))
    vis_dpi = int(round(vis_dpi_raw)) if vis_dpi_raw is not None else 1200
    if vis_dpi <= 0:
        vis_dpi = 1200
    crop_debug_input_dir = _parse_scalar_any(raw, ["crop_debug_input_dir"])

    if crop_debug_input_dir is not None:
        data_dir_path = _resolve_debug_input_dir(base_dir, crop_debug_input_dir)
        data_dir_source = f"crop_debug_input_dir={crop_debug_input_dir}"
    else:
        data_dir_path, matched_rel = _discover_default_input_dir(base_dir)
        data_dir_source = f"auto:{matched_rel}"

    data_dir = str(data_dir_path)
    output_dir = (base_dir / "RESULTS" / "crop").resolve()

    wind_cfg = {
        "cut_lon_manual": [float(lon_lo), float(lon_hi)],
        "cut_lat_manual": [float(lat_lo), float(lat_hi)],
        "utm_crs": str(utm_crs),
    }
    if rotate_deg is not None:
        wind_cfg["rotate_deg"] = float(rotate_deg)

    return RuntimeConfig(
        CONFIG_PATH=str(luw_path.resolve()),
        DATA_DIR=str(data_dir),
        DATA_DIR_SOURCE=str(data_dir_source),
        OUTPUT_DIR=str(output_dir),
        TARGET_CRS=str(target_crs),
        min_lon=float(min_lon),
        max_lon=float(max_lon),
        min_lat=float(min_lat),
        max_lat=float(max_lat),
        cell_size=float(cell_size),
        vis_dpi=int(vis_dpi),
        crop_debug_file_glob=str(crop_debug_file_glob),
        WIND_VTK_CONFIG=wind_cfg,
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


def _vtk_token_from_dtype(dtype: np.dtype) -> str:
    dt = np.dtype(dtype)
    key = dt.newbyteorder(">")
    mapping = {
        np.dtype(">f4"): "float",
        np.dtype(">f8"): "double",
        np.dtype(">i4"): "int",
        np.dtype(">u4"): "unsigned_int",
        np.dtype(">i2"): "short",
        np.dtype(">u2"): "unsigned_short",
        np.dtype("i1"): "char",
        np.dtype("u1"): "unsigned_char",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype for VTK export: {dtype}")
    return mapping[key]


def parse_legacy_vtk_header(vtk_file: str) -> dict:
    dimensions = None
    origin = None
    spacing = None
    n_points = None
    n_cells = None
    fields = {}
    data_section_started = False
    active_assoc = None
    active_tuples = None

    with open(vtk_file, "rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("DIMENSIONS"):
                p = line.split()
                dimensions = (int(p[1]), int(p[2]), int(p[3]))
                continue
            if line.startswith("ORIGIN"):
                p = line.split()
                origin = (float(p[1]), float(p[2]), float(p[3]))
                continue
            if line.startswith("SPACING"):
                p = line.split()
                spacing = (float(p[1]), float(p[2]), float(p[3]))
                continue

            if dimensions is None or origin is None or spacing is None:
                continue

            if line.startswith("POINT_DATA"):
                p = line.split()
                if len(p) < 2:
                    raise ValueError(f"Invalid POINT_DATA line: {line}")
                n_points = int(p[1])
                active_assoc = "POINT_DATA"
                active_tuples = n_points
                data_section_started = True
                continue

            if line.startswith("CELL_DATA"):
                p = line.split()
                if len(p) < 2:
                    raise ValueError(f"Invalid CELL_DATA line: {line}")
                n_cells = int(p[1])
                active_assoc = "CELL_DATA"
                active_tuples = n_cells
                data_section_started = True
                continue

            if not data_section_started:
                continue

            if line.startswith("SCALARS"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid SCALARS line: {line}")
                name = parts[1]
                vtk_type_token = parts[2]
                dtype = _dtype_from_vtk_token(parts[2])
                n_comp = int(parts[3]) if len(parts) > 3 else 1
                tuple_count = active_tuples if active_tuples is not None else n_points
                if tuple_count is None:
                    raise ValueError(f"SCALARS without active POINT_DATA/CELL_DATA section: {line}")

                lookup_line = f.readline().decode("utf-8", errors="replace").strip()
                if not lookup_line.startswith("LOOKUP_TABLE"):
                    raise ValueError(f"Expected LOOKUP_TABLE after SCALARS {name}, got: {lookup_line}")

                data_offset = f.tell()
                fields[name] = {
                    "offset": data_offset,
                    "components": n_comp,
                    "dtype": dtype,
                    "kind": "SCALARS",
                    "vtk_type_token": vtk_type_token,
                    "tuples": int(tuple_count),
                    "association": active_assoc,
                }

                byte_count = int(tuple_count) * n_comp * dtype.itemsize
                f.seek(data_offset + byte_count)
                continue

            if line.startswith("VECTORS"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid VECTORS line: {line}")
                name = parts[1]
                vtk_type_token = parts[2]
                dtype = _dtype_from_vtk_token(parts[2])
                tuple_count = active_tuples if active_tuples is not None else n_points
                if tuple_count is None:
                    raise ValueError(f"VECTORS without active POINT_DATA/CELL_DATA section: {line}")

                data_offset = f.tell()
                fields[name] = {
                    "offset": data_offset,
                    "components": 3,
                    "dtype": dtype,
                    "kind": "VECTORS",
                    "vtk_type_token": vtk_type_token,
                    "tuples": int(tuple_count),
                    "association": active_assoc,
                }

                byte_count = int(tuple_count) * 3 * dtype.itemsize
                f.seek(data_offset + byte_count)
                continue

            if line.startswith("FIELD"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid FIELD line: {line}")
                n_arrays = int(parts[2])
                for _ in range(n_arrays):
                    arr_raw = f.readline()
                    if not arr_raw:
                        raise ValueError("Unexpected EOF while reading FIELD arrays")
                    arr_line = arr_raw.decode("utf-8", errors="replace").strip()
                    while arr_line == "":
                        arr_raw = f.readline()
                        if not arr_raw:
                            raise ValueError("Unexpected EOF while reading FIELD arrays")
                        arr_line = arr_raw.decode("utf-8", errors="replace").strip()

                    arr_parts = arr_line.split()
                    if len(arr_parts) < 4:
                        raise ValueError(f"Invalid FIELD array line: {arr_line}")
                    name = arr_parts[0]
                    n_comp = int(arr_parts[1])
                    n_tuples = int(arr_parts[2])
                    vtk_type_token = arr_parts[3]
                    dtype = _dtype_from_vtk_token(vtk_type_token)

                    data_offset = f.tell()
                    fields[name] = {
                        "offset": data_offset,
                        "components": n_comp,
                        "dtype": dtype,
                        "kind": "FIELD",
                        "vtk_type_token": vtk_type_token,
                        "tuples": n_tuples,
                        "association": active_assoc,
                    }

                    byte_count = n_comp * n_tuples * dtype.itemsize
                    f.seek(data_offset + byte_count)
                continue

    if dimensions is None or origin is None or spacing is None:
        raise ValueError("Invalid VTK header: missing DIMENSIONS/ORIGIN/SPACING")
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


def _read_z_slice_native(vtk_file: str, vtk_info: dict, field_name: str, z_index: int) -> np.ndarray:
    if field_name not in vtk_info["fields"]:
        raise KeyError(f"Field '{field_name}' not found")

    nx, ny, nz = vtk_info["dimensions"]
    if not (0 <= z_index < nz):
        raise ValueError(f"z index out of range: {z_index}, valid=[0,{nz - 1}]")

    field = vtk_info["fields"][field_name]
    n_comp = field["components"]
    dtype = field["dtype"]
    tuple_count = int(field.get("tuples", vtk_info["n_points"]))
    if tuple_count != int(vtk_info["n_points"]):
        assoc = field.get("association")
        raise ValueError(
            f"Field '{field_name}' has tuples={tuple_count}, association={assoc}. "
            "Z-slice reader only supports POINT_DATA fields with tuples == n_points."
        )

    n_values = nx * ny * n_comp
    byte_offset = field["offset"] + z_index * n_values * dtype.itemsize

    with open(vtk_file, "rb") as f:
        f.seek(byte_offset)
        raw = np.fromfile(f, dtype=dtype, count=n_values)

    if raw.size != n_values:
        raise ValueError(f"Read failure for {field_name} at z={z_index}")

    if n_comp == 1:
        return raw.reshape((ny, nx))
    return raw.reshape((ny, nx, n_comp))


def read_z_slice(vtk_file: str, vtk_info: dict, field_name: str, z_index: int) -> np.ndarray:
    arr = _read_z_slice_native(vtk_file, vtk_info, field_name, z_index)
    return arr.astype(np.float32, copy=False)


def height_to_z_index(height_m: float) -> int:
    return int(round((height_m - BASE_HEIGHT_M) / LAYER_STEP_M))


def build_height_plan(vtk_info: dict, target_heights: list[int]) -> list[dict]:
    nz = vtk_info["dimensions"][2]
    plan = []
    for h in target_heights:
        z = height_to_z_index(h)
        valid = 0 <= z < nz
        plan.append({
            "target_height": int(h),
            "z_index": int(z),
            "valid": valid,
            "mapped_height": BASE_HEIGHT_M + LAYER_STEP_M * z,
        })
    return plan


def _grid_3x3():
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    return fig, axes.ravel()


def _resolve_wind_domain_bounds(cfg) -> tuple[float, float, float, float]:
    wind_cfg = getattr(cfg, "WIND_VTK_CONFIG", {}) or {}

    if "cut_lon_manual" in wind_cfg and len(wind_cfg["cut_lon_manual"]) >= 2:
        lon_a, lon_b = wind_cfg["cut_lon_manual"][0], wind_cfg["cut_lon_manual"][1]
    elif "domain_lon" in wind_cfg and len(wind_cfg["domain_lon"]) >= 2:
        lon_a, lon_b = wind_cfg["domain_lon"][0], wind_cfg["domain_lon"][1]
    elif "domain_lon_min" in wind_cfg and "domain_lon_max" in wind_cfg:
        lon_a, lon_b = wind_cfg["domain_lon_min"], wind_cfg["domain_lon_max"]
    else:
        lon_a = wind_cfg.get("min_lon", cfg.min_lon)
        lon_b = wind_cfg.get("max_lon", cfg.max_lon)

    if "cut_lat_manual" in wind_cfg and len(wind_cfg["cut_lat_manual"]) >= 2:
        lat_a, lat_b = wind_cfg["cut_lat_manual"][0], wind_cfg["cut_lat_manual"][1]
    elif "domain_lat" in wind_cfg and len(wind_cfg["domain_lat"]) >= 2:
        lat_a, lat_b = wind_cfg["domain_lat"][0], wind_cfg["domain_lat"][1]
    elif "domain_lat_min" in wind_cfg and "domain_lat_max" in wind_cfg:
        lat_a, lat_b = wind_cfg["domain_lat_min"], wind_cfg["domain_lat_max"]
    else:
        lat_a = wind_cfg.get("min_lat", cfg.min_lat)
        lat_b = wind_cfg.get("max_lat", cfg.max_lat)

    lon_lo, lon_hi = sorted((float(lon_a), float(lon_b)))
    lat_lo, lat_hi = sorted((float(lat_a), float(lat_b)))
    return lon_lo, lon_hi, lat_lo, lat_hi


@dataclass
class GeoTransformModel:
    utm_crs: str
    rotate_deg: float
    pivot_x: float
    pivot_y: float
    x_origin_rot: float
    y_origin_rot: float
    ll_to_utm: Transformer
    utm_to_ll: Transformer

    @staticmethod
    def _rotate(
        x: np.ndarray,
        y: np.ndarray,
        deg: float,
        pivot_x: float,
        pivot_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        th = math.radians(deg)
        c = math.cos(th)
        s = math.sin(th)
        xr = c * (x - pivot_x) - s * (y - pivot_y) + pivot_x
        yr = s * (x - pivot_x) + c * (y - pivot_y) + pivot_y
        return xr, yr

    def local_xy_to_lonlat(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rot = x_local + self.x_origin_rot
        y_rot = y_local + self.y_origin_rot
        x_utm, y_utm = self._rotate(x_rot, y_rot, -self.rotate_deg, self.pivot_x, self.pivot_y)
        lon, lat = self.utm_to_ll.transform(x_utm, y_utm)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def lonlat_to_local(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_utm, y_utm = self.ll_to_utm.transform(lon, lat)
        x_rot, y_rot = self._rotate(np.asarray(x_utm), np.asarray(y_utm), self.rotate_deg, self.pivot_x, self.pivot_y)
        x_local = x_rot - self.x_origin_rot
        y_local = y_rot - self.y_origin_rot
        return np.asarray(x_local, dtype=np.float64), np.asarray(y_local, dtype=np.float64)


def _build_geo_transform_model(cfg) -> GeoTransformModel:
    lon_lo, lon_hi, lat_lo, lat_hi = _resolve_wind_domain_bounds(cfg)
    wind_cfg = getattr(cfg, "WIND_VTK_CONFIG", {}) or {}
    utm_crs = str(wind_cfg.get("utm_crs", cfg.TARGET_CRS))

    ll_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    x00, y00 = ll_to_utm.transform(lon_lo, lat_lo)
    x10, y10 = ll_to_utm.transform(lon_hi, lat_lo)
    x11, y11 = ll_to_utm.transform(lon_hi, lat_hi)
    x01, y01 = ll_to_utm.transform(lon_lo, lat_hi)

    rotate_cfg = wind_cfg.get("rotate_deg")
    if rotate_cfg is None:
        rotate_deg = -math.degrees(math.atan2(y10 - y00, x10 - x00))
    else:
        rotate_deg = float(rotate_cfg)

    pts_x = np.asarray([x00, x10, x11, x01], dtype=np.float64)
    pts_y = np.asarray([y00, y10, y11, y01], dtype=np.float64)
    pivot_x = float(np.mean(pts_x))
    pivot_y = float(np.mean(pts_y))

    xr, yr = GeoTransformModel._rotate(pts_x, pts_y, rotate_deg, pivot_x, pivot_y)
    x_origin_rot = float(np.min(xr))
    y_origin_rot = float(np.min(yr))

    return GeoTransformModel(
        utm_crs=utm_crs,
        rotate_deg=rotate_deg,
        pivot_x=pivot_x,
        pivot_y=pivot_y,
        x_origin_rot=x_origin_rot,
        y_origin_rot=y_origin_rot,
        ll_to_utm=ll_to_utm,
        utm_to_ll=utm_to_ll,
    )


def _build_domain_boundary_target_xy(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    model: GeoTransformModel,
    ll_to_target: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    nx = x_axis.size
    ny = y_axis.size
    sx = np.linspace(0, nx - 1, min(nx, BOUNDARY_SAMPLES)).round().astype(np.int32)
    sy = np.linspace(0, ny - 1, min(ny, BOUNDARY_SAMPLES)).round().astype(np.int32)

    xb = x_axis[sx]
    yb = np.full_like(xb, y_axis[0], dtype=np.float64)
    xr = np.full_like(sy, x_axis[-1], dtype=np.float64)
    yr = y_axis[sy]
    xt = x_axis[sx[::-1]]
    yt = np.full_like(xt, y_axis[-1], dtype=np.float64)
    xl = np.full_like(sy[::-1], x_axis[0], dtype=np.float64)
    yl = y_axis[sy[::-1]]

    x_path = np.concatenate([xb, xr, xt, xl, xb[:1]])
    y_path = np.concatenate([yb, yr, yt, yl, yb[:1]])

    lon, lat = model.local_xy_to_lonlat(x_path, y_path)
    x_target, y_target = ll_to_target.transform(lon, lat)
    return np.asarray(x_target, dtype=np.float64), np.asarray(y_target, dtype=np.float64)


def resolve_xy_bounds_from_config_and_args(args: argparse.Namespace, cfg) -> dict:
    transformer = Transformer.from_crs("EPSG:4326", cfg.TARGET_CRS, always_xy=True)
    default_min_x, default_min_y = transformer.transform(cfg.min_lon, cfg.min_lat)
    default_max_x, default_max_y = transformer.transform(cfg.max_lon, cfg.max_lat)

    min_x = default_min_x if args.min_x is None else float(args.min_x)
    max_x = default_max_x if args.max_x is None else float(args.max_x)
    min_y = default_min_y if args.min_y is None else float(args.min_y)
    max_y = default_max_y if args.max_y is None else float(args.max_y)

    min_x, max_x = sorted((min_x, max_x))
    min_y, max_y = sorted((min_y, max_y))

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "default_min_x": default_min_x,
        "default_max_x": default_max_x,
        "default_min_y": default_min_y,
        "default_max_y": default_max_y,
    }


def compute_visualization_crop(vtk_info: dict, cfg, xy_bounds: dict) -> dict:
    nx, ny, _ = vtk_info["dimensions"]
    ox, oy, _ = vtk_info["origin"]
    dx, dy, _ = vtk_info["spacing"]

    x_axis = ox + np.arange(nx, dtype=np.float64) * dx
    y_axis = oy + np.arange(ny, dtype=np.float64) * dy

    transform_model = _build_geo_transform_model(cfg)
    ll_to_target = Transformer.from_crs("EPSG:4326", cfg.TARGET_CRS, always_xy=True)

    min_x = float(xy_bounds["min_x"])
    max_x = float(xy_bounds["max_x"])
    min_y = float(xy_bounds["min_y"])
    max_y = float(xy_bounds["max_y"])
    sel_rect_x = np.asarray([min_x, max_x, max_x, min_x, min_x], dtype=np.float64)
    sel_rect_y = np.asarray([min_y, min_y, max_y, max_y, min_y], dtype=np.float64)

    domain_poly_x, domain_poly_y = _build_domain_boundary_target_xy(
        x_axis=x_axis,
        y_axis=y_axis,
        model=transform_model,
        ll_to_target=ll_to_target,
    )

    in_range = np.zeros((ny, nx), dtype=bool)
    for y0 in range(0, ny, MASK_CHUNK_ROWS):
        y1 = min(ny, y0 + MASK_CHUNK_ROWS)
        y_block = y_axis[y0:y1]
        xx, yy = np.meshgrid(x_axis, y_block, indexing="xy")
        lon, lat = transform_model.local_xy_to_lonlat(xx, yy)
        x_proj, y_proj = ll_to_target.transform(lon, lat)
        in_range[y0:y1, :] = (
            (x_proj >= min_x)
            & (x_proj <= max_x)
            & (y_proj >= min_y)
            & (y_proj <= max_y)
        )

    if not np.any(in_range):
        raise ValueError(
            "Visualization range does not overlap transformed VTK domain after "
            "local->lon/lat->TARGET_CRS mapping."
        )

    x_idx = np.where(np.any(in_range, axis=0))[0]
    y_idx = np.where(np.any(in_range, axis=1))[0]
    x_start = int(x_idx[0])
    x_end = int(x_idx[-1] + 1)
    y_start = int(y_idx[0])
    y_end = int(y_idx[-1] + 1)

    crop_mask = in_range[y_start:y_end, x_start:x_end]
    inside_count = int(np.count_nonzero(crop_mask))
    total_count = int(crop_mask.size)
    inside_ratio = float(inside_count / total_count) if total_count > 0 else 0.0

    return {
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
        "mask": crop_mask,
        "inside_count": inside_count,
        "inside_ratio": inside_ratio,
        "utm_crs": transform_model.utm_crs,
        "rotate_deg": transform_model.rotate_deg,
        "domain_lonlat": _resolve_wind_domain_bounds(cfg),
        "domain_poly_x": domain_poly_x,
        "domain_poly_y": domain_poly_y,
        "selected_rect_x": sel_rect_x,
        "selected_rect_y": sel_rect_y,
        "transform_model": transform_model,
        "nx_crop": x_end - x_start,
        "ny_crop": y_end - y_start,
    }


def build_regular_target_grid(vtk_info: dict, cfg, xy_bounds: dict, crop: dict) -> dict:
    nx, ny, _ = vtk_info["dimensions"]
    ox, oy, _ = vtk_info["origin"]
    dx, dy, _ = vtk_info["spacing"]
    model = crop["transform_model"]

    min_x = float(xy_bounds["min_x"])
    max_x = float(xy_bounds["max_x"])
    min_y = float(xy_bounds["min_y"])
    max_y = float(xy_bounds["max_y"])

    grid_step = float(getattr(cfg, "cell_size", 0.0))
    if not np.isfinite(grid_step) or grid_step <= 0:
        grid_step = min(abs(float(dx)), abs(float(dy)))
    if grid_step <= 0:
        grid_step = 10.0

    nx_out = max(2, int(round((max_x - min_x) / grid_step)) + 1)
    ny_out = max(2, int(round((max_y - min_y) / grid_step)) + 1)
    x_vec = np.linspace(min_x, max_x, nx_out, dtype=np.float64)
    y_vec = np.linspace(min_y, max_y, ny_out, dtype=np.float64)
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="xy")

    target_to_ll = Transformer.from_crs(cfg.TARGET_CRS, "EPSG:4326", always_xy=True)
    lon, lat = target_to_ll.transform(xx, yy)
    x_local, y_local = model.lonlat_to_local(lon, lat)

    x_idx = (x_local - float(ox)) / float(dx)
    y_idx = (y_local - float(oy)) / float(dy)
    valid_mask = (
        (x_idx >= 0.0)
        & (x_idx <= float(nx - 1))
        & (y_idx >= 0.0)
        & (y_idx <= float(ny - 1))
    )

    x_idx_clip = np.clip(x_idx, 0.0, float(nx - 1))
    y_idx_clip = np.clip(y_idx, 0.0, float(ny - 1))
    coords_for_map = np.vstack([y_idx_clip.ravel(), x_idx_clip.ravel()]).astype(np.float64, copy=False)

    th = math.radians(-float(model.rotate_deg))
    c = math.cos(th)
    s = math.sin(th)

    return {
        "x_vec": x_vec,
        "y_vec": y_vec,
        "xx": xx,
        "yy": yy,
        "shape": (ny_out, nx_out),
        "extent": [float(x_vec[0]), float(x_vec[-1]), float(y_vec[0]), float(y_vec[-1])],
        "valid_mask": valid_mask,
        "coords_for_map": coords_for_map,
        "vector_rot_c": c,
        "vector_rot_s": s,
        "grid_step": grid_step,
    }


def _resample_scalar_to_target(local_field: np.ndarray, target_grid: dict) -> np.ndarray:
    ny_out, nx_out = target_grid["shape"]
    interp_flat = map_coordinates(
        local_field,
        target_grid["coords_for_map"],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    out = np.asarray(interp_flat, dtype=np.float32).reshape(ny_out, nx_out)
    out[~target_grid["valid_mask"]] = np.nan
    return out


def plot_range_diagnostic(
    crop: dict,
    output_path: str,
    target_crs: str,
    show: bool,
    dpi: int,
) -> None:
    poly_x = crop.get("domain_poly_x")
    poly_y = crop.get("domain_poly_y")
    rect_x = crop.get("selected_rect_x")
    rect_y = crop.get("selected_rect_y")
    if poly_x is None or poly_y is None or rect_x is None or rect_y is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 8), constrained_layout=True)
    ax.plot(poly_x, poly_y, color="#1f77b4", linewidth=2.0, label="VTK total domain")
    ax.plot(rect_x, rect_y, color="#d62728", linewidth=2.0, label="Selected range")
    ax.fill(rect_x, rect_y, color="#d62728", alpha=0.10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({target_crs})")
    ax.set_ylabel(f"Y ({target_crs})")
    ax.set_title("Range Diagnostic: VTK Total Domain vs Selected Range")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved range diagnostic figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_wind_figure(
    vtk_file: str,
    vtk_info: dict,
    height_plan: list[dict],
    crop: dict,
    target_grid: dict,
    quiver_step: int,
    output_path: str,
    show: bool,
    dpi: int,
) -> None:
    if "u_avg" not in vtk_info["fields"]:
        raise ValueError("VTK missing 'u_avg' field")

    ny_out, nx_out = target_grid["shape"]
    x_vec = target_grid["x_vec"]
    y_vec = target_grid["y_vec"]
    extent = target_grid["extent"]
    valid_mask = target_grid["valid_mask"]
    c = target_grid["vector_rot_c"]
    s = target_grid["vector_rot_s"]

    step = max(1, int(quiver_step))
    x_idx = np.arange(0, nx_out, step, dtype=int)
    y_idx = np.arange(0, ny_out, step, dtype=int)
    xx, yy = np.meshgrid(x_vec[x_idx], y_vec[y_idx], indexing="xy")

    sampled = {}
    speed_maps = {}
    speeds_for_norm = []

    for item in height_plan:
        if not item["valid"]:
            continue
        z = item["z_index"]
        vel = read_z_slice(vtk_file, vtk_info, "u_avg", z)
        if vel.ndim != 3 or vel.shape[2] < 2:
            raise ValueError("Field 'u_avg' must have at least 2 components")

        u_local = _resample_scalar_to_target(vel[:, :, 0], target_grid)
        v_local = _resample_scalar_to_target(vel[:, :, 1], target_grid)
        w = _resample_scalar_to_target(
            vel[:, :, 2] if vel.shape[2] >= 3 else np.zeros((vel.shape[0], vel.shape[1]), dtype=np.float32),
            target_grid,
        )

        u = c * u_local - s * v_local
        v = s * u_local + c * v_local
        speed = np.sqrt(u * u + v * v + w * w)

        u_s = u[np.ix_(y_idx, x_idx)]
        v_s = v[np.ix_(y_idx, x_idx)]
        sampled_mask = valid_mask[np.ix_(y_idx, x_idx)]
        u_s = np.where(sampled_mask, u_s, np.nan)
        v_s = np.where(sampled_mask, v_s, np.nan)

        sampled[z] = (u_s, v_s)
        speed_maps[z] = speed
        speed_sample = speed[::4, ::4]
        speed_sample = speed_sample[np.isfinite(speed_sample)]
        if speed_sample.size > 0:
            speeds_for_norm.append(speed_sample)

    if speeds_for_norm:
        all_speed = np.concatenate(speeds_for_norm)
        vmin = float(np.nanpercentile(all_speed, 2))
        vmax = float(np.nanpercentile(all_speed, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(all_speed)) if np.isfinite(np.nanmin(all_speed)) else 0.0
            vmax = float(np.nanmax(all_speed)) if np.isfinite(np.nanmax(all_speed)) else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = _grid_3x3()
    mappable = None

    for ax, item in zip(axes, height_plan):
        h = item["target_height"]
        z = item["z_index"]

        if not item["valid"]:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{h}m\nz={z}\nOUT_OF_RANGE", ha="center", va="center", fontsize=12)
            continue

        speed = speed_maps[z]
        u_s, v_s = sampled[z]

        im = ax.imshow(speed, origin="lower", extent=extent, cmap="turbo", norm=norm, aspect="equal")
        ax.quiver(
            xx,
            yy,
            u_s,
            v_s,
            color="white",
            alpha=0.88,
            pivot="mid",
            headwidth=3.8,
            headlength=5.2,
            headaxislength=4.6,
            width=0.0022,
        )
        mappable = im

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{h}m (z={z})", fontsize=11)

    for ax in axes[len(height_plan):]:
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("3D wind speed (m/s)")

    fig.suptitle(
        f"3D Wind Arrows (9 layers) | {Path(vtk_file).name}\n"
        f"height=-50+10*z, quiver_step={step}, target_grid={nx_out}x{ny_out}",
        fontsize=14,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved wind figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tke_figure(
    vtk_file: str,
    vtk_info: dict,
    height_plan: list[dict],
    crop: dict,
    target_grid: dict,
    output_path: str,
    show: bool,
    dpi: int,
) -> None:
    if "tke" not in vtk_info["fields"]:
        raise ValueError("VTK missing 'tke' field")

    ny_out, nx_out = target_grid["shape"]
    extent = target_grid["extent"]

    slices = {}
    lows = []
    highs = []

    for item in height_plan:
        if not item["valid"]:
            continue
        z = item["z_index"]
        tke = read_z_slice(vtk_file, vtk_info, "tke", z)
        if tke.ndim != 2:
            raise ValueError("Field 'tke' must be scalar")
        tke = _resample_scalar_to_target(tke, target_grid)
        slices[z] = tke
        tke_valid = tke[np.isfinite(tke)]
        if tke_valid.size > 0:
            lows.append(float(np.nanpercentile(tke_valid, 2)))
            highs.append(float(np.nanpercentile(tke_valid, 98)))

    if lows and highs:
        vmin = min(lows)
        vmax = max(highs)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            values = [arr[np.isfinite(arr)] for arr in slices.values() if np.any(np.isfinite(arr))]
            if values:
                merged = np.concatenate(values)
                vmin = float(np.nanmin(merged)) if np.isfinite(np.nanmin(merged)) else 0.0
                vmax = float(np.nanmax(merged)) if np.isfinite(np.nanmax(merged)) else 1.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = _grid_3x3()
    mappable = None

    for ax, item in zip(axes, height_plan):
        h = item["target_height"]
        z = item["z_index"]

        if not item["valid"]:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{h}m\nz={z}\nOUT_OF_RANGE", ha="center", va="center", fontsize=12)
            continue

        tke = slices[z]
        im = ax.imshow(tke, origin="lower", extent=extent, cmap="magma", norm=norm, aspect="equal")
        mappable = im
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{h}m (z={z})", fontsize=11)

    for ax in axes[len(height_plan):]:
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("TKE (m^2/s^2)")

    fig.suptitle(
        f"TKE (9 layers) | {Path(vtk_file).name}\n"
        f"height=-50+10*z, target_grid={nx_out}x{ny_out}",
        fontsize=14,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved TKE figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_cropped_vtk(
    vtk_file: str,
    vtk_info: dict,
    crop: dict,
    output_path: str,
) -> None:
    nx, ny, nz = vtk_info["dimensions"]
    ox, oy, oz = vtk_info["origin"]
    dx, dy, dz = vtk_info["spacing"]
    x_start = int(crop["x_start"])
    x_end = int(crop["x_end"])
    y_start = int(crop["y_start"])
    y_end = int(crop["y_end"])

    if not (0 <= x_start < x_end <= nx and 0 <= y_start < y_end <= ny):
        raise ValueError("Invalid crop window for VTK export")

    nx_out = x_end - x_start
    ny_out = y_end - y_start
    nz_out = nz
    ox_out = float(ox + x_start * dx)
    oy_out = float(oy + y_start * dy)
    oz_out = float(oz)
    n_points = int(vtk_info["n_points"])
    n_points_out = nx_out * ny_out * nz_out

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        header = [
            "# vtk DataFile Version 3.0",
            f"Cropped from {Path(vtk_file).name}",
            "BINARY",
            "DATASET STRUCTURED_POINTS",
            f"DIMENSIONS {nx_out} {ny_out} {nz_out}",
            f"ORIGIN {ox_out:.10f} {oy_out:.10f} {oz_out:.10f}",
            f"SPACING {float(dx):.10f} {float(dy):.10f} {float(dz):.10f}",
            f"POINT_DATA {n_points_out}",
        ]
        f.write(("\n".join(header) + "\n").encode("utf-8"))

        for field_name, field in vtk_info["fields"].items():
            field_tuples = int(field.get("tuples", n_points))
            if field_tuples != n_points:
                print(
                    f"Skip field '{field_name}' during export: "
                    f"tuples={field_tuples} != n_points={n_points}"
                )
                continue

            n_comp = int(field["components"])
            vtk_type_token = str(field.get("vtk_type_token") or _vtk_token_from_dtype(field["dtype"]))
            kind = str(field.get("kind") or ("VECTORS" if n_comp == 3 else "SCALARS"))

            if kind == "VECTORS" and n_comp == 3:
                f.write(f"VECTORS {field_name} {vtk_type_token}\n".encode("utf-8"))
            else:
                if n_comp == 1:
                    f.write(f"SCALARS {field_name} {vtk_type_token}\n".encode("utf-8"))
                else:
                    f.write(f"SCALARS {field_name} {vtk_type_token} {n_comp}\n".encode("utf-8"))
                f.write(b"LOOKUP_TABLE default\n")

            for z in range(nz_out):
                arr = _read_z_slice_native(vtk_file, vtk_info, field_name, z)
                arr_crop = arr[y_start:y_end, x_start:x_end, ...]
                arr_out = np.ascontiguousarray(arr_crop)
                f.write(arr_out.tobytes(order="C"))

    print(f"Saved cropped VTK: {output_path}")


def _extract_step(vtk_path: str) -> int | None:
    name = Path(vtk_path).name
    m = RAW_AVG_BASENAME_RE.match(name)
    if m:
        return int(m.group(1))
    return None


def _list_vtk_files(input_dir: str, patterns_arg: str) -> list[str]:
    patterns = [p.strip() for p in str(patterns_arg).split(",") if p.strip()]
    if not patterns:
        patterns = ["*_avg-*.vtk", "*_avg_*.vtk"]

    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(input_dir, pattern)))

    uniq = sorted(set(os.path.abspath(p) for p in candidates if os.path.isfile(p)))
    if not uniq:
        raise FileNotFoundError(f"No VTK files found in {input_dir} with patterns={patterns}")

    def _sort_key(path: str) -> tuple[int, int | float, str]:
        step = _extract_step(path)
        if step is None:
            return (1, os.path.getmtime(path), Path(path).name)
        return (0, step, Path(path).name)

    uniq.sort(key=_sort_key)
    return uniq


def _normalize_field_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_vector_field_name(vtk_info: dict) -> str:
    fields = list(vtk_info["fields"].keys())
    if "u_avg" in vtk_info["fields"]:
        return "u_avg"

    candidates = ["u_avg", "velocity", "wind", "uvw", "u"]
    by_lower = {f.lower(): f for f in fields}
    by_norm = {_normalize_field_token(f): f for f in fields}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
        found = by_norm.get(_normalize_field_token(candidate))
        if found is not None:
            return found

    n_points = int(vtk_info.get("n_points", 0))
    for name, meta in vtk_info["fields"].items():
        tuples = int(meta.get("tuples", n_points))
        if tuples == n_points and int(meta.get("components", 1)) >= 3:
            return name

    raise ValueError(f"VTK missing wind vector field. Available fields={fields}")


def _resolve_tke_field_name(vtk_info: dict) -> str:
    fields = list(vtk_info["fields"].keys())
    if "tke" in vtk_info["fields"]:
        return "tke"

    candidates = ["tke", "k", "tke_avg", "k_avg", "sgs_tke", "turbulence_k", "tkenergy"]
    by_lower = {f.lower(): f for f in fields}
    by_norm = {_normalize_field_token(f): f for f in fields}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
        found = by_norm.get(_normalize_field_token(candidate))
        if found is not None:
            return found

    raise ValueError(f"VTK missing TKE scalar field. Available fields={fields}")


def _alias_visualization_fields(vtk_info: dict) -> dict:
    aliased = dict(vtk_info)
    aliased_fields = dict(vtk_info["fields"])

    vector_name = _resolve_vector_field_name(vtk_info)
    if vector_name != "u_avg":
        aliased_fields["u_avg"] = aliased_fields[vector_name]

    tke_name = _resolve_tke_field_name(vtk_info)
    if tke_name != "tke":
        aliased_fields["tke"] = aliased_fields[tke_name]

    aliased["fields"] = aliased_fields
    return aliased


def _config_kind_label(config_path: str) -> str:
    suffix = Path(config_path).suffix.lower()
    if suffix in (".luw", ".luwdg", ".luwpf"):
        return suffix[1:]
    if suffix:
        return suffix[1:]
    return "config"


def _log_output_intent(label: str, path: str) -> None:
    action = "Overwriting" if os.path.exists(path) else "Writing"
    print(f"  - {action} {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        usage="%(prog)s <config_path>",
        description="Geo-only batch crop and visualize legacy wind VTK files from a LUW-family config."
    )
    parser.add_argument(
        "config_path",
        help="Path to config (*.luw, *.luwdg, *.luwpf)",
    )
    args = parser.parse_args()

    config_luw = Path(args.config_path).resolve()
    cfg = _load_runtime_config(config_luw)

    output_root = os.path.abspath(cfg.OUTPUT_DIR)
    cropped_dir = os.path.join(output_root, "cropped_vtk")
    figure_dir = os.path.join(output_root, "figures")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("=== Batch Geo Crop / Visualization ===")
    print(f"Config path: {cfg.CONFIG_PATH}")
    print(f"Config type: {_config_kind_label(cfg.CONFIG_PATH)}")
    print(f"Config directory: {Path(cfg.CONFIG_PATH).parent}")
    print(
        "Input directory resolution order: "
        "<config_dir>/RESULTS/vtk -> <config_dir>/RESULTS -> "
        "<config_dir>/proj_temp/vtk -> <config_dir>"
    )
    print("Input directory override key: crop_debug_input_dir (resolved relative to <config_dir>)")
    print(f"Resolved input directory: {cfg.DATA_DIR}")
    print(f"Input directory source: {cfg.DATA_DIR_SOURCE}")
    print(f"Resolved output root: {output_root}")
    print("Output policy: existing files with the same name will be overwritten.")
    print(f"crop_debug_file_glob: {cfg.crop_debug_file_glob}")
    print(
        "Crop lon/lat bounds: "
        f"lon=[{cfg.min_lon:.6f}, {cfg.max_lon:.6f}], "
        f"lat=[{cfg.min_lat:.6f}, {cfg.max_lat:.6f}]"
    )
    print(
        "VTK domain lon/lat: "
        f"lon=[{cfg.WIND_VTK_CONFIG['cut_lon_manual'][0]:.6f}, {cfg.WIND_VTK_CONFIG['cut_lon_manual'][1]:.6f}], "
        f"lat=[{cfg.WIND_VTK_CONFIG['cut_lat_manual'][0]:.6f}, {cfg.WIND_VTK_CONFIG['cut_lat_manual'][1]:.6f}]"
    )
    print(f"Target CRS: {cfg.TARGET_CRS}")
    print(f"Crop grid step: {cfg.cell_size:.3f} m")
    print(f"Figure DPI: {cfg.vis_dpi}")

    print("Discovering input VTK files...")
    vtk_files = _list_vtk_files(cfg.DATA_DIR, cfg.crop_debug_file_glob)
    if not vtk_files:
        raise ValueError("No VTK files selected after filtering.")

    xy_bounds = resolve_xy_bounds_from_config_and_args(
        argparse.Namespace(min_x=None, max_x=None, min_y=None, max_y=None),
        cfg,
    )

    print(f"Found {len(vtk_files)} VTK file(s) to process.")
    print(
        f"Using XY bounds ({cfg.TARGET_CRS}): "
        f"X=[{xy_bounds['min_x']:.2f}, {xy_bounds['max_x']:.2f}], "
        f"Y=[{xy_bounds['min_y']:.2f}, {xy_bounds['max_y']:.2f}]"
    )

    crop_cache: dict[tuple[tuple, tuple, tuple], dict] = {}
    grid_cache: dict[tuple[tuple, tuple, tuple], dict] = {}

    for idx, vtk_file in enumerate(vtk_files, start=1):
        print("")
        print(f"[{idx}/{len(vtk_files)}] Starting {Path(vtk_file).name}")
        info = parse_legacy_vtk_header(vtk_file)
        print(
            "  - Grid signature: "
            f"dimensions={info['dimensions']}, origin={info['origin']}, spacing={info['spacing']}"
        )
        print(f"  - Available fields: {list(info['fields'].keys())}")

        vector_field_name = _resolve_vector_field_name(info)
        tke_field_name = _resolve_tke_field_name(info)
        print(f"  - Wind field selected for visualization: {vector_field_name}")
        print(f"  - TKE field selected for visualization: {tke_field_name}")

        vis_info = _alias_visualization_fields(info)
        key = (tuple(info["dimensions"]), tuple(info["origin"]), tuple(info["spacing"]))

        if key not in crop_cache:
            print("  - Computing crop window and target grid for this grid signature...")
            crop_cache[key] = compute_visualization_crop(info, cfg, xy_bounds)
            grid_cache[key] = build_regular_target_grid(info, cfg, xy_bounds, crop_cache[key])
            print(
                "  - Crop computed: "
                f"X_idx=[{crop_cache[key]['x_start']}, {crop_cache[key]['x_end']}), "
                f"Y_idx=[{crop_cache[key]['y_start']}, {crop_cache[key]['y_end']}), "
                f"size={crop_cache[key]['nx_crop']}x{crop_cache[key]['ny_crop']}"
            )
            print(
                "  - Target grid computed: "
                f"{grid_cache[key]['shape'][1]}x{grid_cache[key]['shape'][0]}, "
                f"step={grid_cache[key]['grid_step']:.2f}m, "
                f"valid={np.mean(grid_cache[key]['valid_mask']) * 100:.2f}%"
            )
        else:
            print("  - Reusing cached crop window and target grid for matching grid signature.")

        crop = crop_cache[key]
        target_grid = grid_cache[key]
        height_plan = build_height_plan(info, TARGET_HEIGHTS_M)

        stem = Path(vtk_file).stem
        cropped_out = os.path.join(cropped_dir, f"{stem}_cropped.vtk")
        range_out = os.path.join(figure_dir, f"{stem}_range_diagnostic.png")
        wind_out = os.path.join(figure_dir, f"{stem}_wind_arrows_9layers.png")
        tke_out = os.path.join(figure_dir, f"{stem}_tke_9layers.png")

        _log_output_intent("cropped VTK", cropped_out)
        export_cropped_vtk(vtk_file=vtk_file, vtk_info=info, crop=crop, output_path=cropped_out)
        _log_output_intent("range diagnostic figure", range_out)
        plot_range_diagnostic(
            crop=crop,
            output_path=range_out,
            target_crs=cfg.TARGET_CRS,
            show=False,
            dpi=max(80, int(cfg.vis_dpi)),
        )
        _log_output_intent("wind figure", wind_out)
        plot_wind_figure(
            vtk_file=vtk_file,
            vtk_info=vis_info,
            height_plan=height_plan,
            crop=crop,
            target_grid=target_grid,
            quiver_step=50,
            output_path=wind_out,
            show=False,
            dpi=max(80, int(cfg.vis_dpi)),
        )
        _log_output_intent("TKE figure", tke_out)
        plot_tke_figure(
            vtk_file=vtk_file,
            vtk_info=vis_info,
            height_plan=height_plan,
            crop=crop,
            target_grid=target_grid,
            output_path=tke_out,
            show=False,
            dpi=max(80, int(cfg.vis_dpi)),
        )

        print(f"[{idx}/{len(vtk_files)}] Finished {Path(vtk_file).name}")

    print("")
    print("Processing complete.")
    print(f"Processed files: {len(vtk_files)}")
    print(f"Cropped VTK directory: {cropped_dir}")
    print(f"Figure directory: {figure_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert FluidX3D legacy VTK files back to geographic NetCDF grids.

Input: one .luw path
Output: NetCDF files in <project>/RESULTS for all matched VTK files
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.ndimage import map_coordinates

# Limit thread over-subscription from native libs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str) -> None:
    print(f"[INFO] {_ts()} | {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"[WARN] {_ts()} | {msg}", flush=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {_ts()} | {msg}", flush=True)
    raise SystemExit(code)


def _strip_inline_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _parse_scalar(text: str, key: str) -> Optional[str]:
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", text)
    if not m:
        return None
    val = _strip_inline_comment(m.group(1))
    if not val:
        return None
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        val = val[1:-1].strip()
    return val


def _parse_pair(text: str, key: str) -> Optional[Tuple[float, float]]:
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*\[([^\]]+)\]\s*$", text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) < 2:
        return None
    try:
        a = float(parts[0])
        b = float(parts[1])
    except Exception:
        return None
    return (a, b)


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _infer_utm_from_lonlat(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    if lat >= 0.0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


@dataclass
class CaseConfig:
    luw_path: Path
    project_home: Path
    casename: str
    datetime: Optional[str]
    cut_lon: Tuple[float, float]
    cut_lat: Tuple[float, float]
    utm_crs: str
    rotate_deg: float


def parse_luw_config(luw_path: Path) -> CaseConfig:
    if not luw_path.is_file():
        fail(f"LUW file not found: {luw_path}")
    raw = luw_path.read_text(encoding="utf-8", errors="ignore")

    casename = _parse_scalar(raw, "casename")
    if not casename:
        fail("Missing key 'casename' in LUW file.")

    datetime_str = _parse_scalar(raw, "datetime")

    lon_pair = _parse_pair(raw, "cut_lon_manual")
    lat_pair = _parse_pair(raw, "cut_lat_manual")
    if lon_pair is None or lat_pair is None:
        fail("Missing 'cut_lon_manual' or 'cut_lat_manual' in LUW file.")

    lon_lo, lon_hi = sorted((lon_pair[0], lon_pair[1]))
    lat_lo, lat_hi = sorted((lat_pair[0], lat_pair[1]))

    utm_crs = _parse_scalar(raw, "utm_crs")
    if not utm_crs:
        lon_c = 0.5 * (lon_lo + lon_hi)
        lat_c = 0.5 * (lat_lo + lat_hi)
        utm_crs = _infer_utm_from_lonlat(lon_c, lat_c)
        log_warn(f"'utm_crs' not found in LUW. Auto-inferred CRS: {utm_crs}")

    rotate_conf = _safe_float(_parse_scalar(raw, "rotate_deg"))

    # Recompute from geographic lower edge when value is missing.
    tf_ll2utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x00, y00 = tf_ll2utm.transform(lon_lo, lat_lo)
    x10, y10 = tf_ll2utm.transform(lon_hi, lat_lo)
    dx0 = x10 - x00
    dy0 = y10 - y00
    rotate_est = -math.degrees(math.atan2(dy0, dx0))
    rotate_deg = rotate_conf if rotate_conf is not None else rotate_est

    if rotate_conf is not None:
        diff = abs(rotate_conf - rotate_est)
        log_info(
            f"Rotation from LUW={rotate_conf:.9f} deg, recomputed={rotate_est:.9f} deg, abs diff={diff:.9e}."
        )
    else:
        log_info(f"Rotation not found in LUW. Using recomputed rotation {rotate_deg:.9f} deg.")

    return CaseConfig(
        luw_path=luw_path.resolve(),
        project_home=luw_path.resolve().parent,
        casename=casename,
        datetime=datetime_str,
        cut_lon=(lon_lo, lon_hi),
        cut_lat=(lat_lo, lat_hi),
        utm_crs=utm_crs,
        rotate_deg=rotate_deg,
    )


def discover_case_vtk_files(cfg: CaseConfig) -> List[Path]:
    candidates: List[Path] = []

    vtk_dir = cfg.project_home / "proj_temp" / "vtk"
    if vtk_dir.is_dir():
        all_vtk = sorted(vtk_dir.glob("*.vtk"))
        if cfg.datetime:
            dt_vtk = [p for p in all_vtk if cfg.datetime in p.name]
            if dt_vtk:
                candidates.extend(dt_vtk)
            else:
                log_warn(
                    f"No VTK in {vtk_dir} matched datetime={cfg.datetime}. "
                    "Falling back to all VTK files in this folder."
                )
                candidates.extend(all_vtk)
        else:
            candidates.extend(all_vtk)

    if not candidates:
        results_dir = cfg.project_home / "RESULTS"
        if results_dir.is_dir():
            all_vtk = sorted(results_dir.glob("*.vtk"))
            if cfg.datetime:
                dt_vtk = [p for p in all_vtk if cfg.datetime in p.name]
                candidates.extend(dt_vtk if dt_vtk else all_vtk)
            else:
                candidates.extend(all_vtk)

    # Keep unique and deterministic order.
    uniq = sorted({p.resolve() for p in candidates})
    if not uniq:
        fail(
            f"No VTK files found for case '{cfg.casename}'. "
            f"Checked: {cfg.project_home / 'proj_temp' / 'vtk'} and {cfg.project_home / 'RESULTS'}"
        )

    log_info(f"Discovered {len(uniq)} VTK file(s) for processing.")
    for p in uniq:
        log_info(f"  - {p}")
    return uniq


_VTK_DTYPE_MAP: Dict[str, np.dtype] = {
    "char": np.dtype(">i1"),
    "unsigned_char": np.dtype(">u1"),
    "short": np.dtype(">i2"),
    "unsigned_short": np.dtype(">u2"),
    "int": np.dtype(">i4"),
    "unsigned_int": np.dtype(">u4"),
    # FluidX3D outputs float; long types are kept for compatibility.
    "long": np.dtype(">i4"),
    "unsigned_long": np.dtype(">u4"),
    "float": np.dtype(">f4"),
    "double": np.dtype(">f8"),
}


@dataclass
class VTKField:
    name: str
    vtk_type: str
    n_components: int
    dtype: np.dtype
    offset: int
    n_points: int
    file_path: Path
    _map: Optional[np.memmap] = field(default=None, init=False, repr=False)

    def view(self) -> np.memmap:
        if self._map is None:
            self._map = np.memmap(
                str(self.file_path),
                mode="r",
                dtype=self.dtype,
                offset=self.offset,
                shape=(self.n_points, self.n_components),
                order="C",
            )
        return self._map


@dataclass
class VTKStructuredPoints:
    path: Path
    dims: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    n_points: int
    fields: List[VTKField]


def _read_ascii_line(fh) -> Optional[str]:
    line = fh.readline()
    if not line:
        return None
    return line.decode("ascii", errors="ignore").strip()


def parse_legacy_structured_points(vtk_path: Path) -> VTKStructuredPoints:
    if not vtk_path.is_file():
        fail(f"VTK file not found: {vtk_path}")

    dims: Optional[Tuple[int, int, int]] = None
    origin: Optional[Tuple[float, float, float]] = None
    spacing: Optional[Tuple[float, float, float]] = None
    n_points: Optional[int] = None
    fields: List[VTKField] = []

    with open(vtk_path, "rb") as fh:
        line1 = _read_ascii_line(fh)
        _ = _read_ascii_line(fh)
        line3 = _read_ascii_line(fh)
        line4 = _read_ascii_line(fh)
        if line1 is None or "vtk datafile version" not in line1.lower():
            fail(f"Invalid VTK header in file: {vtk_path}")
        if line3 is None or line3.upper() != "BINARY":
            fail(f"Only BINARY VTK is supported. File: {vtk_path}")
        if line4 is None or "DATASET STRUCTURED_POINTS" not in line4.upper():
            fail(f"Only DATASET STRUCTURED_POINTS is supported. File: {vtk_path}")

        while True:
            line = _read_ascii_line(fh)
            if line is None:
                break
            if not line:
                continue
            parts = line.split()
            key = parts[0].upper()
            if key == "DIMENSIONS" and len(parts) >= 4:
                dims = (int(parts[1]), int(parts[2]), int(parts[3]))
            elif key == "ORIGIN" and len(parts) >= 4:
                origin = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif key == "SPACING" and len(parts) >= 4:
                spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif key == "POINT_DATA" and len(parts) >= 2:
                n_points = int(parts[1])
                break

        if dims is None or origin is None or spacing is None or n_points is None:
            fail(f"Missing DIMENSIONS/ORIGIN/SPACING/POINT_DATA metadata in {vtk_path}")

        nx, ny, nz = dims
        if nx <= 0 or ny <= 0 or nz <= 0:
            fail(f"Invalid dimensions in {vtk_path}: {dims}")
        if nx * ny * nz != n_points:
            fail(
                f"POINT_DATA mismatch in {vtk_path}. "
                f"DIMENSIONS product={nx*ny*nz}, POINT_DATA={n_points}"
            )

        file_size = vtk_path.stat().st_size

        while True:
            line = _read_ascii_line(fh)
            if line is None:
                break
            if not line:
                continue
            parts = line.split()
            if parts[0].upper() != "SCALARS":
                fail(f"Unsupported VTK data section header '{line}' in {vtk_path}")

            if len(parts) < 3:
                fail(f"Malformed SCALARS header '{line}' in {vtk_path}")
            arr_name = parts[1]
            vtk_type = parts[2].lower()
            n_comp = int(parts[3]) if len(parts) >= 4 else 1

            if vtk_type not in _VTK_DTYPE_MAP:
                fail(f"Unsupported VTK scalar type '{vtk_type}' in {vtk_path}")
            if n_comp <= 0:
                fail(f"Invalid component count '{n_comp}' for field '{arr_name}' in {vtk_path}")

            lookup = _read_ascii_line(fh)
            if lookup is None or not lookup.upper().startswith("LOOKUP_TABLE"):
                fail(f"Expected LOOKUP_TABLE after SCALARS '{arr_name}' in {vtk_path}")

            dtype = _VTK_DTYPE_MAP[vtk_type]
            offset = fh.tell()
            n_values = n_points * n_comp
            n_bytes = n_values * dtype.itemsize
            next_offset = offset + n_bytes
            if next_offset > file_size:
                fail(
                    f"Field '{arr_name}' exceeds file size in {vtk_path}. "
                    f"Needed end offset={next_offset}, file size={file_size}"
                )

            fields.append(
                VTKField(
                    name=arr_name,
                    vtk_type=vtk_type,
                    n_components=n_comp,
                    dtype=dtype,
                    offset=offset,
                    n_points=n_points,
                    file_path=vtk_path,
                )
            )
            fh.seek(next_offset)

    if not fields:
        fail(f"No SCALARS fields found in VTK file: {vtk_path}")

    return VTKStructuredPoints(
        path=vtk_path,
        dims=dims,
        origin=origin,
        spacing=spacing,
        n_points=n_points,
        fields=fields,
    )


@dataclass
class TransformModel:
    utm_crs: str
    rotate_deg: float
    pivot_x: float
    pivot_y: float
    x_origin_rot: float
    y_origin_rot: float
    ll_to_utm: Transformer
    utm_to_ll: Transformer

    def _rotate(self, x: np.ndarray, y: np.ndarray, deg: float) -> Tuple[np.ndarray, np.ndarray]:
        th = math.radians(deg)
        c = math.cos(th)
        s = math.sin(th)
        xr = c * (x - self.pivot_x) - s * (y - self.pivot_y) + self.pivot_x
        yr = s * (x - self.pivot_x) + c * (y - self.pivot_y) + self.pivot_y
        return xr, yr

    def local_to_lonlat(self, x_local: np.ndarray, y_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_rot = x_local + self.x_origin_rot
        y_rot = y_local + self.y_origin_rot
        x_utm, y_utm = self._rotate(x_rot, y_rot, -self.rotate_deg)
        lon, lat = self.utm_to_ll.transform(x_utm, y_utm)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def lonlat_to_local(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_utm, y_utm = self.ll_to_utm.transform(lon, lat)
        x_rot, y_rot = self._rotate(np.asarray(x_utm), np.asarray(y_utm), self.rotate_deg)
        x_local = x_rot - self.x_origin_rot
        y_local = y_rot - self.y_origin_rot
        return np.asarray(x_local, dtype=np.float64), np.asarray(y_local, dtype=np.float64)


def build_transform_model(cfg: CaseConfig) -> TransformModel:
    lon_lo, lon_hi = cfg.cut_lon
    lat_lo, lat_hi = cfg.cut_lat

    ll_to_utm = Transformer.from_crs("EPSG:4326", cfg.utm_crs, always_xy=True)
    utm_to_ll = Transformer.from_crs(cfg.utm_crs, "EPSG:4326", always_xy=True)

    x00, y00 = ll_to_utm.transform(lon_lo, lat_lo)
    x10, y10 = ll_to_utm.transform(lon_hi, lat_lo)
    x11, y11 = ll_to_utm.transform(lon_hi, lat_hi)
    x01, y01 = ll_to_utm.transform(lon_lo, lat_hi)

    pts_x = np.asarray([x00, x10, x11, x01], dtype=np.float64)
    pts_y = np.asarray([y00, y10, y11, y01], dtype=np.float64)
    cx = float(np.mean(pts_x))
    cy = float(np.mean(pts_y))

    th = math.radians(cfg.rotate_deg)
    c = math.cos(th)
    s = math.sin(th)
    xr = c * (pts_x - cx) - s * (pts_y - cy) + cx
    yr = s * (pts_x - cx) + c * (pts_y - cy) + cy
    x_min_rot = float(np.min(xr))
    y_min_rot = float(np.min(yr))

    log_info(
        "Transform model built: "
        f"utm_crs={cfg.utm_crs}, rotate_deg={cfg.rotate_deg:.9f}, "
        f"pivot=({cx:.3f}, {cy:.3f}), local_origin_rot=({x_min_rot:.3f}, {y_min_rot:.3f})."
    )

    return TransformModel(
        utm_crs=cfg.utm_crs,
        rotate_deg=cfg.rotate_deg,
        pivot_x=cx,
        pivot_y=cy,
        x_origin_rot=x_min_rot,
        y_origin_rot=y_min_rot,
        ll_to_utm=ll_to_utm,
        utm_to_ll=utm_to_ll,
    )


def _sanitize_var_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    if not s:
        s = "var"
    if s[0].isdigit():
        s = "v_" + s
    return s.lower()


@dataclass
class OutputVarSpec:
    out_name: str
    src_field: VTKField
    component: int
    long_name: str
    units: str


def build_output_specs(vtk_meta: VTKStructuredPoints) -> List[OutputVarSpec]:
    reserved = {"height", "lat", "lon"}
    used: Dict[str, int] = {}

    def unique(name: str) -> str:
        base = name
        if base in reserved:
            base = f"{base}_var"
        idx = used.get(base, 0)
        if idx == 0:
            used[base] = 1
            return base
        out = f"{base}_{idx+1}"
        used[base] = idx + 1
        return out

    specs: List[OutputVarSpec] = []

    vector_fields = [f for f in vtk_meta.fields if f.n_components == 3]
    main_vec: Optional[VTKField] = None
    vec_priority = {"data", "u", "u_avg", "uvw", "velocity"}
    for f in vector_fields:
        if f.name.lower() in vec_priority:
            main_vec = f
            break
    if main_vec is None and vector_fields:
        main_vec = vector_fields[0]

    if main_vec is not None:
        specs.extend(
            [
                OutputVarSpec("u", main_vec, 0, f"{main_vec.name} component x", "m s-1"),
                OutputVarSpec("v", main_vec, 1, f"{main_vec.name} component y", "m s-1"),
                OutputVarSpec("w", main_vec, 2, f"{main_vec.name} component z", "m s-1"),
            ]
        )
        used["u"] = 1
        used["v"] = 1
        used["w"] = 1

    for field in vtk_meta.fields:
        if field is main_vec:
            continue
        base = _sanitize_var_name(field.name)

        if field.n_components == 1:
            if base == "data":
                stem = vtk_meta.path.stem.lower()
                if "rho" in stem:
                    base = "rho"
                else:
                    base = "scalar"
            out = unique(base)
            specs.append(
                OutputVarSpec(
                    out_name=out,
                    src_field=field,
                    component=0,
                    long_name=f"{field.name} scalar",
                    units="",
                )
            )
            continue

        if field.n_components == 3:
            for c, suf in enumerate(("x", "y", "z")):
                out = unique(f"{base}_{suf}")
                specs.append(
                    OutputVarSpec(
                        out_name=out,
                        src_field=field,
                        component=c,
                        long_name=f"{field.name} component {suf}",
                        units="",
                    )
                )
            continue

        for c in range(field.n_components):
            out = unique(f"{base}_c{c}")
            specs.append(
                OutputVarSpec(
                    out_name=out,
                    src_field=field,
                    component=c,
                    long_name=f"{field.name} component {c}",
                    units="",
                )
            )

    if not specs:
        fail(f"No supported fields found in VTK file: {vtk_meta.path}")

    log_info(f"Output variables for {vtk_meta.path.name}: {', '.join(s.out_name for s in specs)}")
    return specs


def _build_axis(origin: float, spacing: float, n: int) -> np.ndarray:
    return origin + np.arange(n, dtype=np.float64) * spacing


def compute_complete_lonlat_bounds(
    model: TransformModel, x_axis: np.ndarray, y_axis: np.ndarray
) -> Tuple[float, float, float, float]:
    # Left and right edges for complete lon range.
    x_l = np.full_like(y_axis, x_axis[0], dtype=np.float64)
    x_r = np.full_like(y_axis, x_axis[-1], dtype=np.float64)
    lon_l, _ = model.local_to_lonlat(x_l, y_axis)
    lon_r, _ = model.local_to_lonlat(x_r, y_axis)

    lon_low = np.minimum(lon_l, lon_r)
    lon_high = np.maximum(lon_l, lon_r)
    lon_min_complete = float(np.max(lon_low))
    lon_max_complete = float(np.min(lon_high))

    # Bottom and top edges for complete lat range.
    y_b = np.full_like(x_axis, y_axis[0], dtype=np.float64)
    y_t = np.full_like(x_axis, y_axis[-1], dtype=np.float64)
    _, lat_b = model.local_to_lonlat(x_axis, y_b)
    _, lat_t = model.local_to_lonlat(x_axis, y_t)

    lat_low = np.minimum(lat_b, lat_t)
    lat_high = np.maximum(lat_b, lat_t)
    lat_min_complete = float(np.max(lat_low))
    lat_max_complete = float(np.min(lat_high))

    lon_lo, lon_hi = sorted((lon_min_complete, lon_max_complete))
    lat_lo, lat_hi = sorted((lat_min_complete, lat_max_complete))

    if not (math.isfinite(lon_lo) and math.isfinite(lon_hi) and math.isfinite(lat_lo) and math.isfinite(lat_hi)):
        fail("Non-finite complete lon/lat bounds detected.")
    if lon_hi <= lon_lo or lat_hi <= lat_lo:
        fail(
            "Failed to derive a valid complete lon/lat range. "
            f"lon=[{lon_lo}, {lon_hi}], lat=[{lat_lo}, {lat_hi}]"
        )

    return lon_lo, lon_hi, lat_lo, lat_hi


def estimate_lonlat_resolution(
    model: TransformModel, x_axis: np.ndarray, y_axis: np.ndarray, lon_span: float, lat_span: float
) -> Tuple[float, float]:
    x_mid = x_axis[len(x_axis) // 2]
    y_mid = y_axis[len(y_axis) // 2]

    lon_mid, _ = model.local_to_lonlat(x_axis, np.full_like(x_axis, y_mid))
    _, lat_mid = model.local_to_lonlat(np.full_like(y_axis, x_mid), y_axis)

    dlon = float(np.nanmedian(np.abs(np.diff(lon_mid)))) if lon_mid.size > 1 else math.nan
    dlat = float(np.nanmedian(np.abs(np.diff(lat_mid)))) if lat_mid.size > 1 else math.nan

    if not (math.isfinite(dlon) and dlon > 0.0):
        dlon = lon_span / max(1, len(x_axis) - 1)
    if not (math.isfinite(dlat) and dlat > 0.0):
        dlat = lat_span / max(1, len(y_axis) - 1)

    if dlon <= 0.0 or dlat <= 0.0:
        fail("Failed to estimate valid lon/lat resolution.")
    return dlon, dlat


@dataclass
class TargetGridMapping:
    lon: np.ndarray
    lat: np.ndarray
    coords_for_map: np.ndarray
    outside_ratio: float


def build_target_grid_mapping(
    model: TransformModel,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    lon_bounds: Tuple[float, float],
    lat_bounds: Tuple[float, float],
) -> TargetGridMapping:
    lon_lo, lon_hi = lon_bounds
    lat_lo, lat_hi = lat_bounds

    nx = x_axis.size
    ny = y_axis.size
    x0, y0 = float(x_axis[0]), float(y_axis[0])
    sx = float(x_axis[1] - x_axis[0]) if nx > 1 else 1.0
    sy = float(y_axis[1] - y_axis[0]) if ny > 1 else 1.0
    if abs(sx) < 1e-12 or abs(sy) < 1e-12:
        fail(f"Invalid source spacing for mapping: sx={sx}, sy={sy}")

    full_lon_span = lon_hi - lon_lo
    full_lat_span = lat_hi - lat_lo
    dlon_ref, dlat_ref = estimate_lonlat_resolution(model, x_axis, y_axis, full_lon_span, full_lat_span)

    current = [lon_lo, lon_hi, lat_lo, lat_hi]
    last_outside = 1.0

    for attempt in range(12):
        lon_lo_c, lon_hi_c, lat_lo_c, lat_hi_c = current
        if lon_hi_c <= lon_lo_c or lat_hi_c <= lat_lo_c:
            fail("Cropping collapsed while searching complete lon/lat mapping range.")

        nx_out = max(2, int(round((lon_hi_c - lon_lo_c) / dlon_ref)) + 1)
        ny_out = max(2, int(round((lat_hi_c - lat_lo_c) / dlat_ref)) + 1)
        nx_out = min(nx_out, max(2, nx * 4))
        ny_out = min(ny_out, max(2, ny * 4))

        lon_vec = np.linspace(lon_lo_c, lon_hi_c, nx_out, dtype=np.float64)
        lat_vec = np.linspace(lat_lo_c, lat_hi_c, ny_out, dtype=np.float64)

        lon2d, lat2d = np.meshgrid(lon_vec, lat_vec, indexing="xy")
        x_local, y_local = model.lonlat_to_local(lon2d, lat2d)

        x_idx = (x_local - x0) / sx
        y_idx = (y_local - y0) / sy
        inside = (x_idx >= -1e-6) & (x_idx <= (nx - 1) + 1e-6) & (y_idx >= -1e-6) & (y_idx <= (ny - 1) + 1e-6)
        outside_ratio = 1.0 - float(np.mean(inside))

        log_info(
            f"Target mapping attempt {attempt+1}: grid={nx_out}x{ny_out}, "
            f"outside ratio={outside_ratio:.6e}, lon=[{lon_lo_c:.8f}, {lon_hi_c:.8f}], "
            f"lat=[{lat_lo_c:.8f}, {lat_hi_c:.8f}]"
        )

        if outside_ratio <= 0.0:
            x_idx = np.clip(x_idx, 0.0, float(nx - 1))
            y_idx = np.clip(y_idx, 0.0, float(ny - 1))
            coords = np.vstack([y_idx.ravel(), x_idx.ravel()]).astype(np.float64, copy=False)
            return TargetGridMapping(
                lon=lon_vec,
                lat=lat_vec,
                coords_for_map=coords,
                outside_ratio=outside_ratio,
            )

        if outside_ratio >= last_outside - 1e-9 and attempt >= 2:
            fail(
                f"Could not build a complete Cartesian lon/lat target grid. "
                f"Last outside ratio={outside_ratio:.6e}"
            )
        last_outside = outside_ratio

        lon_margin = max(dlon_ref * 2.0, (lon_hi_c - lon_lo_c) * 0.02)
        lat_margin = max(dlat_ref * 2.0, (lat_hi_c - lat_lo_c) * 0.02)
        current = [lon_lo_c + lon_margin, lon_hi_c - lon_margin, lat_lo_c + lat_margin, lat_hi_c - lat_margin]

    fail("Exceeded max attempts while searching a complete lon/lat mapping range.")
    raise RuntimeError("unreachable")


def _extract_plane(field: VTKField, level_k: int, nx: int, ny: int, component: int) -> np.ndarray:
    nxy = nx * ny
    i0 = level_k * nxy
    i1 = i0 + nxy
    src = field.view()[i0:i1, component]
    # Convert to native float32 for scipy interpolation.
    plane = np.asarray(src, dtype=np.float32).reshape(ny, nx)
    return plane


def interpolate_level(
    k: int,
    nx: int,
    ny: int,
    out_nx: int,
    out_ny: int,
    coords_for_map: np.ndarray,
    specs: Sequence[OutputVarSpec],
) -> Tuple[int, Dict[str, np.ndarray]]:
    out: Dict[str, np.ndarray] = {}
    for spec in specs:
        plane = _extract_plane(spec.src_field, k, nx, ny, spec.component)
        interp_flat = map_coordinates(
            plane,
            coords_for_map,
            order=3,
            mode="nearest",
            prefilter=True,
        )
        out[spec.out_name] = np.asarray(interp_flat, dtype=np.float32).reshape(out_ny, out_nx)
    return k, out


def process_single_vtk(
    vtk_meta: VTKStructuredPoints,
    model: TransformModel,
    out_dir: Path,
    workers: int,
) -> Path:
    nx, ny, nz = vtk_meta.dims
    ox, oy, oz = vtk_meta.origin
    sx, sy, sz = vtk_meta.spacing

    log_info(
        f"Processing {vtk_meta.path.name}: dims=({nx}, {ny}, {nz}), "
        f"origin=({ox:.6f}, {oy:.6f}, {oz:.6f}), spacing=({sx:.6f}, {sy:.6f}, {sz:.6f})"
    )

    x_axis = _build_axis(ox, sx, nx)
    y_axis = _build_axis(oy, sy, ny)
    z_axis = _build_axis(oz, sz, nz)

    lon_lo, lon_hi, lat_lo, lat_hi = compute_complete_lonlat_bounds(model, x_axis, y_axis)
    log_info(
        f"Largest complete lon/lat axis-aligned range: "
        f"lon=[{lon_lo:.8f}, {lon_hi:.8f}], lat=[{lat_lo:.8f}, {lat_hi:.8f}]"
    )

    mapping = build_target_grid_mapping(
        model=model,
        x_axis=x_axis,
        y_axis=y_axis,
        lon_bounds=(lon_lo, lon_hi),
        lat_bounds=(lat_lo, lat_hi),
    )
    out_lon = mapping.lon
    out_lat = mapping.lat
    out_nx = out_lon.size
    out_ny = out_lat.size
    log_info(
        f"Target Cartesian lon/lat grid built: nx={out_nx}, ny={out_ny}, "
        f"outside ratio={mapping.outside_ratio:.6e}"
    )

    specs = build_output_specs(vtk_meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / f"{vtk_meta.path.stem}.nc"
    chunk_y = min(out_ny, 256)
    chunk_x = min(out_nx, 256)

    with Dataset(out_nc, "w", format="NETCDF4") as nc:
        nc.createDimension("height", nz)
        nc.createDimension("lat", out_ny)
        nc.createDimension("lon", out_nx)

        v_height = nc.createVariable("height", "f4", ("height",), zlib=True, complevel=3)
        v_lat = nc.createVariable("lat", "f8", ("lat",), zlib=True, complevel=3)
        v_lon = nc.createVariable("lon", "f8", ("lon",), zlib=True, complevel=3)

        v_height.long_name = "height above local model origin"
        v_height.units = "m"
        v_lat.long_name = "latitude"
        v_lat.units = "degree_north"
        v_lon.long_name = "longitude"
        v_lon.units = "degree_east"

        v_height[:] = z_axis.astype(np.float32)
        v_lat[:] = out_lat
        v_lon[:] = out_lon

        var_handles: Dict[str, object] = {}
        for spec in specs:
            v = nc.createVariable(
                spec.out_name,
                "f4",
                ("height", "lat", "lon"),
                zlib=True,
                complevel=3,
                shuffle=True,
                chunksizes=(1, chunk_y, chunk_x),
                fill_value=np.float32(np.nan),
            )
            if spec.long_name:
                v.long_name = spec.long_name
            if spec.units:
                v.units = spec.units
            var_handles[spec.out_name] = v

        nc.source_vtk = str(vtk_meta.path)
        nc.utm_crs = model.utm_crs
        nc.rotate_deg = f"{model.rotate_deg:.12f}"
        nc.transform_pivot_xy_utm = f"{model.pivot_x:.6f},{model.pivot_y:.6f}"
        nc.transform_local_origin_rot_xy = f"{model.x_origin_rot:.6f},{model.y_origin_rot:.6f}"
        nc.note = (
            "Converted from FluidX3D legacy VTK to strict Cartesian lon/lat grid "
            "using inverse UTM-rotation-origin-shift chain."
        )

        log_info(
            f"Start cubic interpolation for {vtk_meta.path.name}: levels={nz}, "
            f"variables={len(specs)}, workers={workers}"
        )

        t0 = time.time()
        next_write = 0
        pending: Dict[int, Dict[str, np.ndarray]] = {}

        if workers <= 1 or nz <= 1:
            for k in range(nz):
                _, block = interpolate_level(
                    k=k,
                    nx=nx,
                    ny=ny,
                    out_nx=out_nx,
                    out_ny=out_ny,
                    coords_for_map=mapping.coords_for_map,
                    specs=specs,
                )
                for name, arr in block.items():
                    var_handles[name][k, :, :] = arr
                if (k + 1) % max(1, nz // 20) == 0 or (k + 1) == nz:
                    elapsed = time.time() - t0
                    log_info(f"Interpolation progress: {k+1}/{nz} levels ({elapsed:.1f}s elapsed).")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        interpolate_level,
                        k,
                        nx,
                        ny,
                        out_nx,
                        out_ny,
                        mapping.coords_for_map,
                        specs,
                    ): k
                    for k in range(nz)
                }
                done_count = 0
                for fut in as_completed(futures):
                    k_done, block = fut.result()
                    pending[k_done] = block
                    done_count += 1

                    while next_write in pending:
                        block_write = pending.pop(next_write)
                        for name, arr in block_write.items():
                            var_handles[name][next_write, :, :] = arr
                        next_write += 1

                    if done_count % max(1, nz // 20) == 0 or done_count == nz:
                        elapsed = time.time() - t0
                        log_info(
                            f"Interpolation progress: {done_count}/{nz} levels completed "
                            f"({elapsed:.1f}s elapsed)."
                        )

        elapsed_total = time.time() - t0
        log_info(f"Finished interpolation and NetCDF writing in {elapsed_total:.1f}s: {out_nc}")

    return out_nc


def main() -> None:
    if len(sys.argv) != 2:
        fail("Usage: python vtk2nc_new.py <path_to_case.luw>")

    luw_path = Path(sys.argv[1]).resolve()
    cfg = parse_luw_config(luw_path)

    log_info(f"LUW path: {cfg.luw_path}")
    log_info(f"Case name: {cfg.casename}")
    if cfg.datetime:
        log_info(f"Case datetime: {cfg.datetime}")
    else:
        log_warn("Case datetime is missing in LUW. VTK discovery falls back to all available VTK files.")

    model = build_transform_model(cfg)
    vtk_files = discover_case_vtk_files(cfg)
    out_dir = cfg.project_home / "RESULTS"

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(cpu_n, 12))
    log_info(f"Runtime worker threads set to {workers} (CPU count={cpu_n}).")

    ok: List[Path] = []
    failed: List[Tuple[Path, str]] = []

    for i, vtk_path in enumerate(vtk_files, start=1):
        log_info("=" * 88)
        log_info(f"[{i}/{len(vtk_files)}] Start file: {vtk_path}")
        try:
            vtk_meta = parse_legacy_structured_points(vtk_path)
            out_nc = process_single_vtk(vtk_meta, model, out_dir, workers=workers)
            ok.append(out_nc)
        except SystemExit:
            raise
        except Exception as exc:
            msg = str(exc)
            failed.append((vtk_path, msg))
            log_warn(f"Failed file {vtk_path}: {msg}")

    log_info("=" * 88)
    log_info(f"Conversion summary: success={len(ok)}, failed={len(failed)}")
    for p in ok:
        log_info(f"  [OK] {p}")
    for p, m in failed:
        log_warn(f"  [FAIL] {p} | {m}")

    if failed:
        fail(f"{len(failed)} file(s) failed during conversion.", code=2)


if __name__ == "__main__":
    main()

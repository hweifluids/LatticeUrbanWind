#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert cropped per-angle average VTK files to NetCDF on UTM coordinates.

This converter is intended for RESULTS/crop/cropped_vtk_raw_assembled outputs:
- remove the 0-50 m pedestal/base layers by keeping VTK z >= pedestal height;
- shift the pedestal top to zero and add the case terrain-minimum ASL;
- convert horizontal local rotated CFD coordinates to true UTM easting/northing;
- rotate horizontal wind components from model x/y axes to UTM east/north axes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset
from pyproj import CRS

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from vtk2nc_new import (  # noqa: E402
    CaseConfig,
    VTKField,
    VTKStructuredPoints,
    _build_axis,
    build_transform_model,
    parse_legacy_structured_points,
    parse_luw_config,
)


CASE_ORDER = ("beijing", "chongqing", "shanghai", "singapore")
DEFAULT_INPUT_SUBDIR = Path("RESULTS") / "crop" / "cropped_vtk_raw_assembled"
DEFAULT_OUTPUT_SUBDIR = Path("RESULTS") / "nc_utm_asl"
DEFAULT_INPUT_GLOB = "ANG_*_avg-*_cropped.vtk"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[INFO] {_ts()} | {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {_ts()} | {msg}", flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def parse_range_asl(range_path: Path, required_cases: Optional[Sequence[str]] = None) -> Dict[str, float]:
    if not range_path.is_file():
        fail(f"Range file not found: {range_path}")

    values: Dict[str, float] = {}
    current_case: Optional[str] = None
    city_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
    explicit_re = re.compile(r"terrain_min_asl_m\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
    plain_m_re = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*m\s*$", re.IGNORECASE)

    wanted = [c.lower() for c in required_cases] if required_cases else list(CASE_ORDER)
    number_m_re = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)

    for raw in range_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        lower = line.lower()
        for case_name in wanted:
            if case_name in lower:
                matches = number_m_re.findall(line)
                if matches:
                    values[case_name] = float(matches[-1])
                    current_case = None
                break

        m_city = city_re.match(line)
        if m_city:
            current_case = m_city.group(1).lower()
            continue

        if current_case is None:
            continue

        m_explicit = explicit_re.search(line)
        if m_explicit:
            values[current_case] = float(m_explicit.group(1))
            continue

        m_plain = plain_m_re.match(line)
        if m_plain:
            values[current_case] = float(m_plain.group(1))

    if required_cases:
        missing = [c.lower() for c in required_cases if c.lower() not in values]
    else:
        missing = []
    if missing:
        fail(f"Missing ASL values in {range_path}: {', '.join(missing)}")
    return values


def read_prep_summary_asl(case_dir: Path) -> Optional[float]:
    summary_path = case_dir / "prep_summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"Failed to read {summary_path}: {exc}")
        return None
    value = data.get("dem_absolute_min_m")
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def find_case_config(case_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        conf = Path(explicit).resolve()
        if not conf.is_file():
            fail(f"Config file not found: {conf}")
        return conf

    for name in ("conf.luwpf", "conf.luwdg", "conf.luw"):
        conf = case_dir / name
        if conf.is_file():
            return conf
    fail(f"No conf.luwpf/conf.luwdg/conf.luw found in {case_dir}")
    raise RuntimeError("unreachable")


def resolve_terrain_min_asl(
    case_dir: Path,
    case_name: str,
    explicit_value: Optional[float],
    range_path: Optional[Path],
) -> float:
    if explicit_value is not None:
        if not math.isfinite(explicit_value):
            fail(f"Invalid --terrain-min-asl: {explicit_value}")
        return float(explicit_value)

    prep_value = read_prep_summary_asl(case_dir)
    if prep_value is not None:
        log(f"Terrain min ASL from prep_summary.json: {prep_value:.9f} m")
        return prep_value

    if range_path is not None and range_path.is_file():
        values = parse_range_asl(range_path, required_cases=None)
        if case_name.lower() in values:
            return values[case_name.lower()]

    fail(
        "Terrain minimum ASL was not provided and could not be inferred. "
        "Use --terrain-min-asl or provide prep_summary.json with dem_absolute_min_m."
    )
    raise RuntimeError("unreachable")


def discover_cases(root: Path, requested: Optional[Sequence[str]]) -> List[Path]:
    wanted = {c.lower() for c in requested} if requested else set(CASE_ORDER)
    cases: List[Path] = []
    for name in CASE_ORDER:
        if name not in wanted:
            continue
        case_dir = root / name
        if not (case_dir / "conf.luwpf").is_file():
            warn(f"Skip case without conf.luwpf: {case_dir}")
            continue
        cases.append(case_dir)

    extra = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.lower() in wanted and p.name.lower() not in CASE_ORDER
    )
    for case_dir in extra:
        if (case_dir / "conf.luwpf").is_file():
            cases.append(case_dir)

    if not cases:
        fail(f"No case directories found under {root}")
    return cases


def discover_vtk_files(case_dir: Path, input_subdir: Path, input_glob: str) -> List[Path]:
    vtk_dir = case_dir / input_subdir
    if not vtk_dir.is_dir():
        fail(f"VTK input directory not found: {vtk_dir}")
    files = sorted(vtk_dir.glob(input_glob))
    if not files:
        fail(f"No avg VTK files matching '{input_glob}' found in {vtk_dir}")
    return files


def angle_token(path: Path) -> str:
    m = re.search(r"ANG_([^_]+)_", path.name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"DG_[^_]+_([^_]+)_", path.name, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _field_by_name(meta: VTKStructuredPoints, name: str) -> Optional[VTKField]:
    needle = name.lower()
    for field in meta.fields:
        if field.name.lower() == needle:
            return field
    return None


@dataclass
class FieldComponent:
    field: VTKField
    component: int


@dataclass
class WindFields:
    u: FieldComponent
    v: FieldComponent
    w: FieldComponent
    scalar_fields: List[VTKField]


def select_wind_fields(meta: VTKStructuredPoints) -> WindFields:
    u = _field_by_name(meta, "u")
    v = _field_by_name(meta, "v")
    w = _field_by_name(meta, "w")
    main_vector: Optional[VTKField] = None

    if u is not None and v is not None and w is not None:
        if u.n_components != 1 or v.n_components != 1 or w.n_components != 1:
            fail(f"{meta.path.name}: scalar u/v/w fields must have one component each")
        wind = WindFields(
            u=FieldComponent(u, 0),
            v=FieldComponent(v, 0),
            w=FieldComponent(w, 0),
            scalar_fields=[],
        )
        excluded = {"u", "v", "w"}
    else:
        vector_fields = [f for f in meta.fields if f.n_components >= 3]
        priority = ("u_avg", "u", "data", "uvw", "velocity")
        for name in priority:
            main_vector = next((f for f in vector_fields if f.name.lower() == name), None)
            if main_vector is not None:
                break
        if main_vector is None and vector_fields:
            main_vector = vector_fields[0]
        if main_vector is None:
            names = ", ".join(f"{f.name}(ncomp={f.n_components})" for f in meta.fields)
            fail(f"{meta.path.name}: expected u/v/w scalars or a 3-component wind field. Available: {names}")
        wind = WindFields(
            u=FieldComponent(main_vector, 0),
            v=FieldComponent(main_vector, 1),
            w=FieldComponent(main_vector, 2),
            scalar_fields=[],
        )
        excluded = {main_vector.name.lower()}

    wind.scalar_fields = [
        f for f in meta.fields
        if f.n_components == 1 and f.name.lower() not in excluded
    ]
    return wind


def read_scalar_plane(field: VTKField, k: int, nx: int, ny: int, component: int = 0) -> np.ndarray:
    nxy = nx * ny
    start = k * nxy
    stop = start + nxy
    arr = field.view()[start:stop, component]
    return np.asarray(arr, dtype=np.float32).reshape(ny, nx)


def build_utm_coordinate_grids(cfg: CaseConfig, meta: VTKStructuredPoints) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, _ = meta.dims
    ox, oy, _ = meta.origin
    sx, sy, _ = meta.spacing

    x_local = _build_axis(ox, sx, nx)
    y_local = _build_axis(oy, sy, ny)

    model = build_transform_model(cfg)
    x2d_local, y2d_local = np.meshgrid(x_local, y_local, indexing="xy")
    x_rot = x2d_local + model.x_origin_rot
    y_rot = y2d_local + model.y_origin_rot
    easting, northing = model._rotate(x_rot, y_rot, -model.rotate_deg)
    return (
        x_local.astype(np.float64, copy=False),
        y_local.astype(np.float64, copy=False),
        np.asarray(easting, dtype=np.float64),
        np.asarray(northing, dtype=np.float64),
    )


def create_output_dataset(
    out_nc: Path,
    cfg: CaseConfig,
    meta: VTKStructuredPoints,
    terrain_min_asl_m: float,
    pedestal_height_m: float,
    compression_level: int,
    kept_z_indices: np.ndarray,
    z_asl: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    easting: np.ndarray,
    northing: np.ndarray,
    wind: WindFields,
):
    nx, ny, _ = meta.dims
    nz_out = len(kept_z_indices)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    chunk_y = min(ny, 256)
    chunk_x = min(nx, 256)
    chunk_z = 1

    ds = Dataset(out_nc, "w", format="NETCDF4")
    ds.createDimension("z", nz_out)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    x_var = ds.createVariable("x", "f8", ("x",), zlib=True, complevel=compression_level)
    y_var = ds.createVariable("y", "f8", ("y",), zlib=True, complevel=compression_level)
    z_var = ds.createVariable("z", "f4", ("z",), zlib=True, complevel=compression_level)
    e_var = ds.createVariable("easting", "f8", ("y", "x"), zlib=True, complevel=compression_level)
    n_var = ds.createVariable("northing", "f8", ("y", "x"), zlib=True, complevel=compression_level)

    x_var.long_name = "local CFD x coordinate before UTM inverse rotation"
    x_var.units = "m"
    y_var.long_name = "local CFD y coordinate before UTM inverse rotation"
    y_var.units = "m"
    z_var.standard_name = "altitude"
    z_var.long_name = "altitude above mean sea level"
    z_var.units = "m"
    z_var.positive = "up"
    e_var.standard_name = "projection_x_coordinate"
    e_var.long_name = f"UTM easting ({cfg.utm_crs})"
    e_var.units = "m"
    n_var.standard_name = "projection_y_coordinate"
    n_var.long_name = f"UTM northing ({cfg.utm_crs})"
    n_var.units = "m"

    x_var[:] = x_local
    y_var[:] = y_local
    z_var[:] = z_asl.astype(np.float32)
    e_var[:, :] = easting
    n_var[:, :] = northing

    crs_var = ds.createVariable("crs", "i4")
    crs = CRS.from_user_input(cfg.utm_crs)
    crs_var.long_name = "coordinate reference system"
    crs_var.grid_mapping_name = "transverse_mercator"
    crs_var.spatial_ref = crs.to_wkt()
    epsg = crs.to_epsg()
    if epsg is not None:
        crs_var.epsg_code = int(epsg)
        crs_var.crs_wkt = crs.to_wkt()

    def make_data_var(name: str, long_name: str, units: str) -> object:
        var = ds.createVariable(
            name,
            "f4",
            ("z", "y", "x"),
            zlib=True,
            complevel=compression_level,
            shuffle=True,
            chunksizes=(chunk_z, chunk_y, chunk_x),
            fill_value=np.float32(np.nan),
        )
        var.long_name = long_name
        if units:
            var.units = units
        var.coordinates = "z easting northing"
        var.grid_mapping = "crs"
        return var

    data_vars: Dict[str, object] = {
        "u": make_data_var("u", "eastward wind component in UTM CRS", "m s-1"),
        "v": make_data_var("v", "northward wind component in UTM CRS", "m s-1"),
        "w": make_data_var("w", "vertical wind component", "m s-1"),
    }
    data_vars["u"].standard_name = "eastward_wind"
    data_vars["v"].standard_name = "northward_wind"
    data_vars["w"].standard_name = "upward_air_velocity"
    data_vars["u"].source_field = f"{wind.u.field.name}[{wind.u.component}]"
    data_vars["v"].source_field = f"{wind.v.field.name}[{wind.v.component}]"
    data_vars["w"].source_field = f"{wind.w.field.name}[{wind.w.component}]"

    used = {"u", "v", "w"}
    scalar_var_names: List[Tuple[VTKField, str]] = []
    for field in wind.scalar_fields:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", field.name.strip()).lower() or "scalar"
        if base[0].isdigit():
            base = "v_" + base
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        long_name = field.name
        units = "m2 s-2" if name == "tke" else ""
        data_vars[name] = make_data_var(name, long_name, units)
        data_vars[name].source_field = f"{field.name}[0]"
        scalar_var_names.append((field, name))

    ds.source_vtk = str(meta.path)
    ds.casename = cfg.casename
    if cfg.datetime:
        ds.case_datetime = cfg.datetime
    ds.angle = angle_token(meta.path)
    ds.utm_crs = cfg.utm_crs
    ds.rotate_deg = float(cfg.rotate_deg)
    ds.vtk_origin = ",".join(f"{v:.12g}" for v in meta.origin)
    ds.vtk_spacing = ",".join(f"{v:.12g}" for v in meta.spacing)
    ds.terrain_min_asl_m = float(terrain_min_asl_m)
    ds.pedestal_height_removed_m = float(pedestal_height_m)
    ds.vertical_processing = (
        "Kept VTK layers where z_vtk >= pedestal_height_removed_m; "
        "stored z = z_vtk - pedestal_height_removed_m + terrain_min_asl_m."
    )
    ds.vector_processing = (
        "Horizontal VTK u/v components were interpreted as model rotated x/y components "
        "and rotated by -rotate_deg into UTM east/north components."
    )
    ds.history = f"{_ts()} converted by vtk_avg_to_utm_asl_nc.py"

    return ds, data_vars, scalar_var_names


def convert_one(
    vtk_path: Path,
    cfg: CaseConfig,
    terrain_min_asl_m: float,
    out_dir: Path,
    pedestal_height_m: float,
    compression_level: int,
    overwrite: bool,
) -> Path:
    meta = parse_legacy_structured_points(vtk_path)
    wind = select_wind_fields(meta)
    nx, ny, nz = meta.dims
    _, _, oz = meta.origin
    _, _, sz = meta.spacing
    z_vtk = _build_axis(oz, sz, nz)
    kept = np.where(z_vtk >= pedestal_height_m - 1e-6)[0]
    if kept.size == 0:
        fail(f"{vtk_path.name}: no z layers remain after removing pedestal height {pedestal_height_m} m")
    z_asl = z_vtk[kept] - pedestal_height_m + terrain_min_asl_m

    out_nc = out_dir / f"{vtk_path.stem}_utm_asl.nc"
    if out_nc.exists() and not overwrite:
        log(f"Skip existing: {out_nc}")
        return out_nc

    x_local, y_local, easting, northing = build_utm_coordinate_grids(cfg, meta)
    ds, data_vars, scalar_var_names = create_output_dataset(
        out_nc=out_nc,
        cfg=cfg,
        meta=meta,
        terrain_min_asl_m=terrain_min_asl_m,
        pedestal_height_m=pedestal_height_m,
        compression_level=compression_level,
        kept_z_indices=kept,
        z_asl=z_asl,
        x_local=x_local,
        y_local=y_local,
        easting=easting,
        northing=northing,
        wind=wind,
    )

    theta = math.radians(float(cfg.rotate_deg))
    c = math.cos(theta)
    s = math.sin(theta)

    try:
        t0 = time.time()
        for out_k, src_k in enumerate(kept):
            u_model = read_scalar_plane(wind.u.field, int(src_k), nx, ny, wind.u.component)
            v_model = read_scalar_plane(wind.v.field, int(src_k), nx, ny, wind.v.component)
            data_vars["u"][out_k, :, :] = c * u_model + s * v_model
            data_vars["v"][out_k, :, :] = -s * u_model + c * v_model
            data_vars["w"][out_k, :, :] = read_scalar_plane(wind.w.field, int(src_k), nx, ny, wind.w.component)

            for field, out_name in scalar_var_names:
                data_vars[out_name][out_k, :, :] = read_scalar_plane(field, int(src_k), nx, ny, 0)

            if (out_k + 1) % max(1, len(kept) // 10) == 0 or (out_k + 1) == len(kept):
                log(
                    f"{vtk_path.name}: wrote {out_k + 1}/{len(kept)} z layers "
                    f"({time.time() - t0:.1f}s elapsed)"
                )
    finally:
        ds.close()

    log(f"Wrote NetCDF: {out_nc}")
    return out_nc


def iter_selected_files(files: Sequence[Path], limit: Optional[int]) -> Iterable[Path]:
    if limit is None or limit <= 0:
        yield from files
    else:
        yield from files[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cropped avg VTK files to UTM/ASL NetCDF.")
    parser.add_argument(
        "--root",
        default=r"D:\averagedField\New0518_average",
        help="Root folder containing city case directories.",
    )
    parser.add_argument("--case-dir", default=None, help="Process one case directory instead of the default multi-case root.")
    parser.add_argument("--config", default=None, help="Explicit conf.luw/conf.luwdg/conf.luwpf path for --case-dir mode.")
    parser.add_argument("--range-file", default=None, help="Path to Range.txt with terrain ASL values.")
    parser.add_argument("--terrain-min-asl", type=float, default=None, help="Terrain minimum true ASL in meters.")
    parser.add_argument("--cases", nargs="*", default=None, help="Case names to process. Default: four known cases.")
    parser.add_argument("--input-subdir", default=str(DEFAULT_INPUT_SUBDIR), help="Input VTK subdirectory under each case.")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB, help="Input VTK glob under --input-subdir.")
    parser.add_argument("--output-subdir", default=str(DEFAULT_OUTPUT_SUBDIR), help="Output NetCDF subdirectory under each case.")
    parser.add_argument("--pedestal-height", type=float, default=50.0, help="Pedestal/base height to remove in meters.")
    parser.add_argument("--compression-level", type=int, default=3, help="NetCDF zlib compression level 0-9.")
    parser.add_argument("--limit", type=int, default=None, help="Limit files per case for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NetCDF outputs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    range_path = Path(args.range_file).resolve() if args.range_file else root / "Range.txt"
    input_subdir = Path(args.input_subdir)
    output_subdir = Path(args.output_subdir)
    compression_level = max(0, min(9, int(args.compression_level)))

    if args.case_dir:
        case_dir = Path(args.case_dir).resolve()
        conf = find_case_config(case_dir, args.config)
        cfg = parse_luw_config(conf)
        terrain_min = resolve_terrain_min_asl(
            case_dir=case_dir,
            case_name=cfg.casename,
            explicit_value=args.terrain_min_asl,
            range_path=range_path if range_path.is_file() else None,
        )
        files = discover_vtk_files(case_dir, input_subdir, args.input_glob)
        selected = list(iter_selected_files(files, args.limit))
        out_dir = case_dir / output_subdir

        log(f"Case dir: {case_dir}")
        log(f"Config: {conf}")
        log(f"Input glob: {case_dir / input_subdir / args.input_glob}")
        log(f"Pedestal height removed: {args.pedestal_height:.3f} m")
        log(f"Files: {len(selected)}/{len(files)}, terrain_min_asl_m={terrain_min:.9f}, out={out_dir}")

        ok: List[Path] = []
        failed: List[Tuple[Path, str]] = []
        for idx, vtk_path in enumerate(selected, start=1):
            log(f"[{idx}/{len(selected)}] {vtk_path.name}")
            try:
                ok.append(
                    convert_one(
                        vtk_path=vtk_path,
                        cfg=cfg,
                        terrain_min_asl_m=terrain_min,
                        out_dir=out_dir,
                        pedestal_height_m=float(args.pedestal_height),
                        compression_level=compression_level,
                        overwrite=bool(args.overwrite),
                    )
                )
            except Exception as exc:
                failed.append((vtk_path, str(exc)))
                warn(f"Failed {vtk_path}: {exc}")

        log("=" * 88)
        log(f"Summary: ok={len(ok)}, failed={len(failed)}")
        for path, msg in failed:
            warn(f"FAIL {path} | {msg}")
        return 2 if failed else 0

    asl_values = parse_range_asl(range_path, required_cases=CASE_ORDER)
    cases = discover_cases(root, args.cases)

    total_ok: List[Path] = []
    total_failed: List[Tuple[Path, str]] = []

    log(f"Root: {root}")
    log(f"Range file: {range_path}")
    log(f"Pedestal height removed: {args.pedestal_height:.3f} m")

    for case_dir in cases:
        case_name = case_dir.name.lower()
        terrain_min = asl_values[case_name]
        conf = case_dir / "conf.luwpf"
        cfg = parse_luw_config(conf)
        files = discover_vtk_files(case_dir, input_subdir, args.input_glob)
        selected = list(iter_selected_files(files, args.limit))
        out_dir = case_dir / output_subdir
        log("=" * 88)
        log(f"Case {case_name}: files={len(selected)}/{len(files)}, terrain_min_asl_m={terrain_min:.6f}, out={out_dir}")

        for idx, vtk_path in enumerate(selected, start=1):
            log(f"[{case_name} {idx}/{len(selected)}] {vtk_path.name}")
            try:
                out_nc = convert_one(
                    vtk_path=vtk_path,
                    cfg=cfg,
                    terrain_min_asl_m=terrain_min,
                    out_dir=out_dir,
                    pedestal_height_m=float(args.pedestal_height),
                    compression_level=compression_level,
                    overwrite=bool(args.overwrite),
                )
                total_ok.append(out_nc)
            except Exception as exc:
                total_failed.append((vtk_path, str(exc)))
                warn(f"Failed {vtk_path}: {exc}")

    log("=" * 88)
    log(f"Summary: ok={len(total_ok)}, failed={len(total_failed)}")
    for path, msg in total_failed:
        warn(f"FAIL {path} | {msg}")
    return 2 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make post translate for VTK files so that the minimum bounds move to the origin.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pyvista as pv

import pickle

import numpy as np

# ------------------------------ Core geometry ops ------------------------------

def compute_translate_offset(mesh: pv.DataSet) -> Tuple[float, float, float]:
    """Compute the translation offset that moves the minimum bounds to the origin."""
    xmin, ymin, zmin = mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]
    return (-xmin, -ymin, -zmin - 50.0)


def translate_mesh(mesh: pv.DataSet, offset: Tuple[float, float, float]) -> pv.DataSet:
    """Return a translated copy of mesh using the provided offset."""
    if offset == (0.0, 0.0, 0.0):
        return mesh
    return mesh.translate(offset, inplace=False)


# ------------------------------ File discovery ------------------------------

def find_latest_by_pattern(folder: Path, datetime_str: str, kind: str) -> Optional[Path]:
    """
    Find the file with the largest 9 digit index matching:
    {datetime}_raw_{kind}-#########.vtk
    kind must be 'u' or 'rho'.
    Return None when not found.
    """
    if kind not in {"u", "rho"}:
        raise ValueError("kind must be 'u' or 'rho'")
    if not folder.is_dir():
        return None

    pat = re.compile(rf"^{re.escape(datetime_str)}_raw_{kind}-(\d{{9}})\.vtk$", re.IGNORECASE)
    candidates: Dict[int, Path] = {}
    for f in folder.iterdir():
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if m:
            idx = int(m.group(1))
            candidates[idx] = f
    if not candidates:
        return None
    return candidates[max(candidates.keys())]


def derive_case_datetime_from_vtk(vtk_path: Path) -> Tuple[str, str, str]:
    """
    Derive casename, datetime and prefix from a single vtk filename.
    casename uses the parent folder name.
    datetime tries to parse from either:
      <datetime>_raw_u-#########.vtk  or  <datetime>_raw_rho-#########.vtk
      fall back to stem if no pattern matches.
    prefix is 'uvw' for u pattern, 'rho' for rho pattern, 'mesh' as a fallback.
    """
    casename = vtk_path.parent.name
    stem = vtk_path.stem
    m_u = re.match(r"^(.+?)_raw_u-\d{9}$", stem, flags=re.IGNORECASE)
    m_rho = re.match(r"^(.+?)_raw_rho-\d{9}$", stem, flags=re.IGNORECASE)
    if m_u:
        return casename, m_u.group(1), "uvw"
    if m_rho:
        return casename, m_rho.group(1), "rho"
    return casename, stem, "mesh"


# ------------------------------ Text config parsing ------------------------------

def parse_case_from_text(txt_path: Path) -> Tuple[str, str]:
    """
    Parse casename and datetime in a case insensitive manner from a text file.
    Expected patterns:
      casename = <value>
      datetime = <value>
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"Config file not found: {txt_path}")
    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    m_case = re.search(r"casename\s*=\s*([^\s]+)", content, flags=re.IGNORECASE)
    m_dt = re.search(r"datetime\s*=\s*([^\s]+)", content, flags=re.IGNORECASE)
    if not m_case or not m_dt:
        raise ValueError("Failed to parse 'casename' or 'datetime' from the config file")
    return m_case.group(1), m_dt.group(1)


# ------------------------------ Output path builder ------------------------------

def build_output_path(parent_dir: Path, prefix: str, casename: str, datetime_str: str, tailname: str) -> Path:
    """
    Build output path under parent_dir/RESULTS.
    Filename formats:
      <prefix>-<casename>_<datetime>.vtk
      <prefix>-<casename>_<datetime>_<tailname>.vtk when tailname is not empty.
    """
    results_dir = parent_dir / "RESULTS"
    results_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}-{casename}_{datetime_str}"
    if tailname:
        base = f"{base}_{tailname}"
    return results_dir / f"{base}.vtk"


# ------------------------------ Merge helpers ------------------------------
def load_dem_grid_pkl(pkl_path: Path) -> Optional[dict]:
    """
    Load DEM grid data from pkl file.
    Expected keys: dem_grid, x_grid, y_grid, x_min, y_min
    """
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as f:
        dem_data = pickle.load(f)

    needed = {"dem_grid", "x_grid", "y_grid", "x_min", "y_min"}
    missing = [k for k in needed if k not in dem_data]
    if missing:
        raise ValueError(f"DEM pkl missing keys: {missing}")

    return dem_data


def convert_pkl_dem_to_points(dem_data: dict) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    """
    Convert pkl DEM grid to point samples in the same XY frame as the translated VTK.
    Returns:
      points_xy: (N, 2)
      elevations: (N,)
      x_min, y_min, x_max, y_max, z_max
    """
    dem_grid = np.asarray(dem_data["dem_grid"], dtype=float)
    x_grid = np.asarray(dem_data["x_grid"], dtype=float)
    y_grid = np.asarray(dem_data["y_grid"], dtype=float)
    x_min = float(dem_data["x_min"])
    y_min = float(dem_data["y_min"])

    X, Y = np.meshgrid(x_grid + x_min, y_grid + y_min, indexing="xy")
    points_xy = np.column_stack([X.ravel(), Y.ravel()])
    elevations = dem_grid.ravel()

    x_max = float(x_min + x_grid.max()) if x_grid.size else x_min
    y_max = float(y_min + y_grid.max()) if y_grid.size else y_min
    z_max = float(np.nanmax(dem_grid)) if dem_grid.size else 0.0

    return points_xy, elevations, x_min, y_min, x_max, y_max, z_max


def idw_interpolate(query_xy: np.ndarray, src_xy: np.ndarray, src_val: np.ndarray, k: int = 12, power: float = 2.0) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation using cKDTree.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise RuntimeError("scipy is required for IDW interpolation but cannot be imported") from e

    if src_xy.shape[0] == 0:
        return np.zeros((query_xy.shape[0],), dtype=float)

    kk = int(min(max(k, 1), src_xy.shape[0]))
    tree = cKDTree(src_xy)
    distances, indices = tree.query(query_xy, k=kk)

    if kk == 1:
        return src_val[indices].astype(float)

    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / (distances ** power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / weights_sum
    vals = np.sum(weights * src_val[indices], axis=1)
    return vals.astype(float)


def build_wodem_datetime(datetime_str: str) -> str:
    return f"woDEM_{datetime_str}"


def make_wodem_pointcloud(mesh: pv.DataSet, dem_points_xy: np.ndarray, dem_elev: np.ndarray,
                          dem_x_min: float, dem_y_min: float, dem_x_max: float, dem_y_max: float,
                          k: int = 12, power: float = 2.0) -> pv.PolyData:
    """
    Apply woDEM transform on top of the existing translation (including the extra -50).
    This outputs an unstructured point cloud (PolyData points only).
    Steps:
      1) IDW interpolate DEM height for each mesh point by its XY
      2) new_z = z - dem_height
      3) drop points with new_z < 0
      4) attach all point_data arrays and also attach dem height field
    """
    mesh_for_points = mesh.copy(deep=True)
    try:
        mesh_for_points = mesh_for_points.cell_data_to_point_data(pass_cell_data=True)
    except Exception:
        pass

    pts = np.asarray(mesh_for_points.points, dtype=float)
    qxy = pts[:, 0:2]

    inside = (
        (qxy[:, 0] >= dem_x_min) & (qxy[:, 0] <= dem_x_max) &
        (qxy[:, 1] >= dem_y_min) & (qxy[:, 1] <= dem_y_max)
    )

    dem_h = np.zeros((pts.shape[0],), dtype=float)
    if np.any(inside):
        dem_h[inside] = idw_interpolate(qxy[inside], dem_points_xy, dem_elev, k=k, power=power)

    new_z = pts[:, 2] - dem_h
    keep = new_z >= 0.0

    new_pts = pts[keep].copy()
    new_pts[:, 2] = new_z[keep]

    out = pv.PolyData(new_pts)

    for name, arr in mesh_for_points.point_data.items():
        a = np.asarray(arr)
        if a.shape[0] == pts.shape[0]:
            out.point_data[name] = a[keep].copy()

    out.point_data["dem_height"] = dem_h[keep].copy()

    return out

def copy_point_and_cell_data(src: pv.DataSet, dst: pv.DataSet, suffix: str = "") -> None:
    """
    Copy all point_data and cell_data arrays from src to dst.
    """
    for name, arr in src.point_data.items():
        out_name = name if name not in dst.point_data else f"{name}{suffix}"
        dst.point_data[out_name] = arr.copy()

    for name, arr in src.cell_data.items():
        out_name = name if name not in dst.cell_data else f"{name}{suffix}"
        dst.cell_data[out_name] = arr.copy()


def ensure_same_topology(a: pv.DataSet, b: pv.DataSet) -> None:
    """
    Ensure that two datasets have the same number of points and cells.
    Raise an informative error when not compatible for a safe merge.
    """
    if a.n_points != b.n_points:
        raise ValueError(f"Point count mismatch when merging: {a.n_points} vs {b.n_points}")
    if a.n_cells != b.n_cells:
        raise ValueError(f"Cell count mismatch when merging: {a.n_cells} vs {b.n_cells}")


# ------------------------------ Processing paths ------------------------------

def process_single_vtk(vtk_file: Path, tailname: str) -> Path:
    """
    Translate only this vtk and save the result.
    Output under the parent/RESULTS directory of the input vtk.
    Prefix derived from filename pattern.
    """
    if not vtk_file.is_file():
        raise FileNotFoundError(f"VTK file not found: {vtk_file}")
    casename, datetime_str, prefix = derive_case_datetime_from_vtk(vtk_file)
    mesh_in = pv.read(vtk_file)
    offset = compute_translate_offset(mesh_in)
    mesh_out = translate_mesh(mesh_in, offset)
    out_path = build_output_path(vtk_file.parent, prefix, casename, datetime_str, tailname)
    mesh_out.save(out_path, binary=True)
    print(f"[OK] Translated single VTK using offset {offset} and wrote: {out_path}")
    return out_path


def process_with_conf(conf_file: Path, tailname: str, absonly: bool) -> Path:

    """
    Read casename and datetime from conf file, look under parent/proj_temp/vtk
    for latest u and rho files, apply the same translation, and save a single output file.
    Prefix rules
      uvwrho when both exist, uvw when only u exists, rho when only rho exists.
    """
    casename, datetime_str = parse_case_from_text(conf_file)
    parent_dir = conf_file.parent
    search_dir = parent_dir / "proj_temp" / "vtk"

    vtk_u = find_latest_by_pattern(search_dir, datetime_str, "u")
    vtk_rho = find_latest_by_pattern(search_dir, datetime_str, "rho")

    if vtk_u is None and vtk_rho is None:
        raise FileNotFoundError(f"No matching VTK files found under: {search_dir}")

    # Prefer using 'u' as base when both exist for a stable merge target
    base_path = vtk_u if vtk_u is not None else vtk_rho
    base_mesh_in = pv.read(base_path)
    offset = compute_translate_offset(base_mesh_in)
    base_mesh = translate_mesh(base_mesh_in, offset)

    prefix = "uvw" if vtk_rho is None else "uvwrho" if vtk_u is not None else "rho"

    if vtk_u is not None and base_path != vtk_u:
        mesh_u = translate_mesh(pv.read(vtk_u), offset)
        ensure_same_topology(base_mesh, mesh_u)
        copy_point_and_cell_data(mesh_u, base_mesh, suffix="_u")

    if vtk_rho is not None and base_path != vtk_rho:
        mesh_rho = translate_mesh(pv.read(vtk_rho), offset)
        ensure_same_topology(base_mesh, mesh_rho)
        copy_point_and_cell_data(mesh_rho, base_mesh, suffix="_rho")

    # When base is 'u', ensure arrays from base 'u' do not need suffixing
    # When base is 'rho', ensure arrays from base 'rho' do not need suffixing
    # Arrays from the other mesh are suffixed to avoid name collisions

    out_path = build_output_path(parent_dir, prefix, casename, datetime_str, tailname)
    base_mesh.save(out_path, binary=True)
    dem_pkl_path = parent_dir / "proj_temp" / "dem_grid.pkl"
    if (not absonly) and dem_pkl_path.is_file():

        try:
            dem_data = load_dem_grid_pkl(dem_pkl_path)
            dem_points_xy, dem_elev, dx_min, dy_min, dx_max, dy_max, dz_max = convert_pkl_dem_to_points(dem_data)

            dem_points_xy[:, 0] = np.subtract(dem_points_xy[:, 0], dx_min)
            dem_points_xy[:, 1] = np.subtract(dem_points_xy[:, 1], dy_min)

            dem_x_min = 0.0
            dem_y_min = 0.0
            dem_x_max = float(np.subtract(dx_max, dx_min))
            dem_y_max = float(np.subtract(dy_max, dy_min))

            vx_max, vy_max, vz_max = base_mesh.bounds[1], base_mesh.bounds[3], base_mesh.bounds[5]
            print(f"[INFO] VTK max corner after translation: ({vx_max:.3f}, {vy_max:.3f}, {vz_max:.3f})")
            print(f"[INFO] PKL DEM XY range after subtracting x_min y_min: x=[0.000, {dem_x_max:.3f}], y=[0.000, {dem_y_max:.3f}]")
            print(f"[INFO] PKL DEM Z range: min {float(np.nanmin(dem_elev)):.3f}, max {dz_max:.3f}")


            wodem_datetime = build_wodem_datetime(datetime_str)
            wodem_out_path = build_output_path(parent_dir, prefix, casename, wodem_datetime, tailname)
            wodem_cloud = make_wodem_pointcloud(
                base_mesh,
                dem_points_xy,
                dem_elev,
                dem_x_min,
                dem_y_min,
                dem_x_max,
                dem_y_max,
                k=12,
                power=2.0
            )

            wodem_cloud = wodem_cloud.extract_points(wodem_cloud.points[:, 2] >= 0.0, include_cells=False)

            wodem_cloud.save(wodem_out_path, binary=True)

            print(f"[OK] Wrote woDEM point-cloud VTK with DEM field to: {wodem_out_path}")
        except Exception as e:
            print(f"[WARN] DEM pkl exists but woDEM generation failed: {e}")

    details = []
    if vtk_u is not None:
        details.append(f"u={vtk_u.name}")
    if vtk_rho is not None:
        details.append(f"rho={vtk_rho.name}")
    print(f"[OK] Translated with shared offset {offset}; merged {' and '.join(details)} into: {out_path}")
    return out_path


# ------------------------------ CLI ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate VTK datasets so that the minimum bounds move to the origin. "
                    "Accepts a single input path plus an optional --tailname."
    )
    parser.add_argument("input_path", help="Path to a .vtk file or a text config file containing casename and datetime")
    parser.add_argument("--tailname", default="", help="Optional tail string appended to the output filename")
    parser.add_argument("--absonly", action="store_true", help="Skip woDEM output when dem_grid.pkl exists")

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    tailname = args.tailname.strip()
    print(f"[INFO] Processing input: {input_path} with tailname: '{tailname}'")

    if not input_path.exists():
        sys.exit(f"[ERROR] Input path does not exist: {input_path}")

    try:
        if input_path.suffix.lower() == ".vtk":
            process_single_vtk(input_path, tailname)
        else:
            process_with_conf(input_path, tailname, args.absonly)
    except Exception as e:
        sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()

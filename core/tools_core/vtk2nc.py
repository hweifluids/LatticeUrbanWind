#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import sys
import re
from pathlib import Path
import numpy as np
import xarray as xr

try:
    import vtk
    from vtk.util import numpy_support
except Exception as e:
    print(f"[Error] VTK import failed: {e}")
    sys.exit(1)

try:
    from pyproj import Transformer
except Exception as e:
    print(f"[Error] pyproj import failed: {e}")
    sys.exit(1)


def soft_exit(msg: str, code: int = 1) -> None:
    """Print a message and exit with a non zero code."""
    print(f"[Exit] {msg}")
    sys.exit(code)


def parse_luw(cfg_path: Path) -> dict:
    """
    Parse the .luw text file. Expected keys:
      casename = <string without spaces>
      datetime = <string without spaces>
      si_x_cfd = [min_x, max_x]  optional
      si_y_cfd = [min_y, max_y]  optional
    Returns a dict with found values.
    """
    txt = cfg_path.read_text(encoding="utf-8", errors="ignore")

    def find_scalar(key: str) -> str | None:
        m = re.search(rf"{key}\s*=\s*([^\s]+)", txt)
        return m.group(1) if m else None

    def find_pair(key: str) -> tuple[float, float] | None:
        m = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", txt)
        if not m:
            return None
        try:
            a, b = [float(v) for v in m.group(1).split(",")]
            return a, b
        except Exception:
            return None

    cfg = {
        "casename": find_scalar("casename"),
        "datetime": find_scalar("datetime"),
        "si_x_cfd": find_pair("si_x_cfd"),
        "si_y_cfd": find_pair("si_y_cfd"),
        "cut_lon_manual": find_pair("cut_lon_manual"),
        "cut_lat_manual": find_pair("cut_lat_manual"),
    }

    return cfg


def locate_results_dir(cfg_path: Path) -> Path:
    """
    The VTK file is located inside the RESULTS folder under the parent directory of the config file.
    Example:
      /path/to/config.luw
      /path/to/RESULTS/uvw-<casename>_<datetime>.vtk
    """
    parent_dir = cfg_path.parent
    results_dir = parent_dir / "RESULTS"
    if not results_dir.is_dir():
        soft_exit(f"RESULTS folder not found at: {results_dir}")
    return results_dir


def find_vtk_file(results_dir: Path, casename: str, datetime_str: str) -> Path:
    """
    Try both naming patterns:
      uvw-{casename}_{datetime}.vtk
      uvwrho-{casename}_{datetime}.vtk
    Prefer the file that actually exists. If both exist, prefer uvwrho so that we can ignore data_rho and still read data.
    """
    candidates = [
        results_dir / f"uvwrho-{casename}_{datetime_str}.vtk",
        results_dir / f"uvw-{casename}_{datetime_str}.vtk",
    ]
    for p in candidates:
        if p.is_file():
            return p
    soft_exit(
        f"No VTK file found. Tried: {candidates[0].name} and {candidates[1].name} under {results_dir}"
    )
    return Path()  # unreachable


def read_vtk_dataset(vtk_path: Path):
    """Read the VTK legacy file as a vtkDataSet and return it."""
    try:
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(str(vtk_path))
        reader.Update()
        data = reader.GetOutput()
        if data is None:
            soft_exit(f"VTK reader returned no output for file: {vtk_path}")
        return data
    except Exception as e:
        soft_exit(f"Failed to read VTK file: {vtk_path}, reason: {e}")
    return None  # unreachable


def select_point_array_as_wind(pointdata) -> "vtk.vtkDataArray":
    """
    Select the 3 component array for wind.
    Priority:
      1) Array named "data" with at least 3 components.
      2) First array with at least 3 components, excluding an array named "data_rho".
    """
    n_arrays = pointdata.GetNumberOfArrays()
    if n_arrays == 0:
        soft_exit("No point arrays found in VTK dataset.")
    # Pass 1: look for array named "data"
    for i in range(n_arrays):
        arr = pointdata.GetArray(i)
        name = arr.GetName() if arr else None
        comps = arr.GetNumberOfComponents() if arr else 0
        if name == "data" and comps >= 3:
            return arr
    # Pass 2: first 3 component array excluding "data_rho"
    for i in range(n_arrays):
        arr = pointdata.GetArray(i)
        if not arr:
            continue
        name = arr.GetName()
        comps = arr.GetNumberOfComponents()
        if name == "data_rho":
            continue
        if comps >= 3:
            return arr
    soft_exit("No suitable 3 component wind array found. Expected an array named 'data' or any 3 component array.")
    return None  # unreachable


def build_coordinates_from_origin_spacing(dims: tuple[int, int, int], origin: tuple[float, float, float],
                                          spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 1D coordinate arrays from VTK origin and spacing.
    """
    nx, ny, nz = dims
    ox, oy, oz = origin
    sx, sy, sz = spacing
    xs = ox + np.arange(nx, dtype=np.float64) * sx
    ys = oy + np.arange(ny, dtype=np.float64) * sy
    zs = oz + np.arange(nz, dtype=np.float64) * sz
    return xs, ys, zs


def main():
    # Argument parsing
    if len(sys.argv) != 2:
        soft_exit("Usage: python script.py /path/to/config.luw")
    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.is_file():
        soft_exit(f"Config file not found: {cfg_path}")
    if cfg_path.suffix.lower() != ".luw":
        print(f"[Warn] Config file does not have .luw suffix: {cfg_path.name}")

    # Parse config
    cfg = parse_luw(cfg_path)
    casename = cfg.get("casename")
    datetime_str = cfg.get("datetime")
    if not casename or not datetime_str:
        soft_exit("Config must provide both 'casename' and 'datetime'.")

    print(f"[Info] Loaded config, casename={casename}, datetime={datetime_str}")

    # Locate VTK
    results_dir = locate_results_dir(cfg_path)
    vtk_path = find_vtk_file(results_dir, casename, datetime_str)
    print(f"[Info] Using VTK file: {vtk_path}")

    # Read dataset
    data = read_vtk_dataset(vtk_path)

    # Basic grid info
    try:
        dims = data.GetDimensions()
        origin = data.GetOrigin()
        spacing = data.GetSpacing()
    except Exception as e:
        soft_exit(f"Dataset is missing dimensions, origin, or spacing: {e}")

    if not all(isinstance(v, (int, np.integer)) for v in dims) or any(d <= 0 for d in dims):
        soft_exit(f"Invalid grid dimensions: {dims}")
    if any(abs(s) < 1e-12 for s in spacing):
        soft_exit(f"Invalid grid spacing: {spacing}")

    print(f"[Info] Grid dims: {dims}")
    print(f"[Info] Grid origin: {origin}")
    print(f"[Info] Grid spacing: {spacing}")

    # Coordinates
    xs, ys, zs = build_coordinates_from_origin_spacing(dims, origin, spacing)

    # Optionally override horizontal min max from config only when no manual lon/lat anchor is provided
    if cfg.get("si_x_cfd") and cfg.get("si_y_cfd") and not (cfg.get("cut_lon_manual") and cfg.get("cut_lat_manual")):
        try:
            min_x, max_x = cfg["si_x_cfd"]
            min_y, max_y = cfg["si_y_cfd"]
            xs = np.linspace(min_x, max_x, dims[0], dtype=np.float64)
            ys = np.linspace(min_y, max_y, dims[1], dtype=np.float64)
            print(f"[Info] Using horizontal range from config, x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        except Exception as e:
            soft_exit(f"Failed to apply si_x_cfd and si_y_cfd from config: {e}")
    else:
        print("[Info] Using meter coordinates relative to VTK origin for x and y.")


    # Height handling
    # Adjust heights by subtracting 50 m for the ground pedestal, then remove layers below zero after this shift
    z_after_ground = zs - 50.0
    mask = z_after_ground >= 0.0
    if not np.any(mask):
        soft_exit("After removing the lowest 50 m, no vertical layers remain. Check your grid and spacing.")
    print(f"[Info] Vertical layers before cut: {len(zs)}, after cut: {int(mask.sum())}")

    # Read point data and select wind array
    pointdata = data.GetPointData()
    if pointdata is None:
        soft_exit("No point data found in VTK dataset.")
    wind_arr = select_point_array_as_wind(pointdata)

    # Convert to numpy and reshape
    np_wind = numpy_support.vtk_to_numpy(wind_arr)
    n_points = data.GetNumberOfPoints()
    if np_wind.shape[0] != n_points:
        soft_exit(f"Wind array row count {np_wind.shape[0]} does not match number of points {n_points}.")

    nx, ny, nz = dims
    expected_points = nx * ny * nz
    if n_points != expected_points:
        soft_exit(f"Number of points {n_points} does not match dims product {expected_points}.")

    # VTK uses (X, Y, Z) ordering for point indexing and NumPy reshape requires (Z, Y, X) for natural slicing
    try:
        # First three components are u, v, w
        u_flat = np_wind[:, 0]
        v_flat = np_wind[:, 1]
        w_flat = np_wind[:, 2]

        u = u_flat.reshape(nz, ny, nx)
        v = v_flat.reshape(nz, ny, nx)
        w = w_flat.reshape(nz, ny, nx)
    except Exception as e:
        soft_exit(f"Failed to reshape wind arrays with dims {dims}: {e}")

    # Apply vertical cut, keep layers where z_after_ground >= 0
    kept_idx = np.where(mask)[0]
    z_kept = z_after_ground[mask]
    try:
        u = u[kept_idx, :, :]
        v = v[kept_idx, :, :]
        w = w[kept_idx, :, :]
    except Exception as e:
        soft_exit(f"Failed to slice vertical layers after 50 m cut: {e}")

    # Build latitude and longitude. If manual lon/lat anchor is provided, treat (x=0,y=0) as (lon_min,lat_min).
    try:
        if cfg.get("cut_lon_manual") and cfg.get("cut_lat_manual"):
            lon_min, _ = cfg["cut_lon_manual"]
            lat_min, _ = cfg["cut_lat_manual"]

            # Use zero-based meter grid so that the first node maps exactly to the anchor
            xs_rel = np.arange(dims[0], dtype=np.float64) * spacing[0]
            ys_rel = np.arange(dims[1], dtype=np.float64) * spacing[1]
            xx_m, yy_m = np.meshgrid(xs_rel, ys_rel, indexing="xy")

            # Forward: WGS84 -> UTM 50N at the anchor point
            tf_fwd = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
            e0, n0 = tf_fwd.transform(lon_min, lat_min)

            # Shift relative meters by the anchor's absolute UTM coordinates
            xx_abs = e0 + xx_m
            yy_abs = n0 + yy_m

            # Inverse: UTM 50N -> WGS84
            tf_inv = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
            lons, lats = tf_inv.transform(xx_abs.ravel(), yy_abs.ravel())
            print(f"[Info] Anchored lon/lat using cut_lon_manual[0]={lon_min}, cut_lat_manual[0]={lat_min}")
        else:
            # Fallback: interpret xs and ys as absolute UTM 50N meters
            xx_m, yy_m = np.meshgrid(xs, ys, indexing="xy")
            tf_inv = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
            lons, lats = tf_inv.transform(xx_m.ravel(), yy_m.ravel())
            print("[Info] No manual lon/lat anchor provided. Interpreting x,y as absolute UTM 50N.")

        lons = np.asarray(lons, dtype=np.float64).reshape((dims[1], dims[0]))
        lats = np.asarray(lats, dtype=np.float64).reshape((dims[1], dims[0]))
    except Exception as e:
        soft_exit(f"Geodetic transform failed: {e}")



    # Prepare Dataset
    try:
        ds = xr.Dataset(
            data_vars={
                "u": (("h", "y", "x"), u, {"long_name": "u wind", "units": "m s-1"}),
                "v": (("h", "y", "x"), v, {"long_name": "v wind", "units": "m s-1"}),
                "w": (("h", "y", "x"), w, {"long_name": "w wind", "units": "m s-1"}),
                "lat": (("y", "x"), lats, {"long_name": "latitude", "units": "degree_north"}),
                "lon": (("y", "x"), lons, {"long_name": "longitude", "units": "degree_east"}),
            },
            coords={
                "x": ("x", xs.astype(np.float32), {"long_name": "x", "units": "meters"}),
                "y": ("y", ys.astype(np.float32), {"long_name": "y", "units": "meters"}),
                "h": ("h", z_kept.astype(np.float32), {"long_name": "height above ground", "units": "meters"}),
            },
            attrs={
                "institution": "National Meteorological Information Center, CMA, Beijing, China",
                "source": "CFD-LBM V1.0",
                "contact": "Dr. Han Shuai, hans@cma.gov.cn, 010-58993096",
                "history": "Created by Dr. Han Shuai, NMIC/CMA, 2025-07-27",
                "note": "VTK vertical coordinate adjusted by subtracting 50 m then removing below ground layers.",
            },
        )
    except Exception as e:
        soft_exit(f"Failed to build xarray Dataset: {e}")

    # Encoding
    encoding = {
        "x": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "y": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "h": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "lat": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "lon": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "u": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "v": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
        "w": {"dtype": "float32", "zlib": True, "complevel": 3, "_FillValue": np.nan},
    }

    # Output path in RESULTS
    out_nc = results_dir / f"uvw-{casename}_{datetime_str}.nc"
    try:
        ds.to_netcdf(out_nc, mode="w", format="NETCDF4", encoding=encoding)
    except Exception as e:
        soft_exit(f"Failed to write NetCDF file: {e}")

    print(f"[OK] Saved NetCDF: {out_nc}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        soft_exit(f"Unhandled error: {exc}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wind field processor driven by a single LUW config file.

This version implements:
1) Ground definition uses zmin + 50 m. The slab [zmin, zmin+50) is removed.
   Real height is h = z_vtk - (zmin + 50). Therefore z_vtk = zmin + 50 maps to h = 0 m.
2) Ask user for the number of sections to export.
3) Confirmation accepts ENTER, 'y', or 'yes' as affirmative.
4) Save one figure per selected height layer as wind_{height_m}m.png under RESULTS/sections.
5) Axes are labeled in longitude and latitude. For each figure, the axis limits are set to
   that layer's lon/lat min and max computed from the mapped grid.
6) Vector field selection priority: U, velocity, Velocity, UVW, data, Data, then the first 3-component array.
7) Ignore data_rho if present. Soft landing on foreseeable errors. All prompts and comments are in English.
8) Optional 3D NetCDF export in lon/lat coordinates to RESULTS as <vtk_basename>_visluw.nc.
"""

import os
import sys
import re
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pyvista as pv
from pyproj import Transformer, CRS


# ------------------------- Utilities -------------------------

def soft_exit(msg: str, code: int = 1) -> None:
    """Print an error message and exit gracefully."""
    print(f"[ERROR] {msg}")
    sys.exit(code)


def read_luw(path: str) -> Dict[str, Any]:
    """
    Parse a minimal LUW-like text file with simple `key = value` entries.
    Supported keys include, but are not limited to:
        casename = foo
        datetime = 20250101090000
        cut_lon_manual=[121.0,121.5]
        cut_lat_manual=[31.0,31.3]
        utm_epsg = 32651
        utm = 32651
        utm_zone = 51
        utm_hemisphere = N
        center_lon = 121.5
        center_lat = 31.25
    Lines starting with // or # are ignored. Standalone ... lines are ignored.
    """
    if not os.path.isfile(path):
        soft_exit(f"LUW file not found: {path}")

    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("#") or line == "...":
            continue
        m = re.match(r'^([A-Za-z0-9_]+)\s*=\s*(.+)$', line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()

        # Try simple JSON-like list [a,b]
        if v.startswith("[") and v.endswith("]"):
            try:
                parts = [p.strip() for p in v[1:-1].split(",") if p.strip()]
                data[k] = [float(p) for p in parts]
                continue
            except Exception:
                pass

        # Try int
        if re.fullmatch(r'[+-]?\d+', v):
            try:
                data[k] = int(v)
                continue
            except Exception:
                pass

        # Try float
        if re.fullmatch(r'[+-]?\d+\.\d*', v):
            try:
                data[k] = float(v)
                continue
            except Exception:
                pass

        # Fallback string
        data[k] = v.strip().strip('"').strip("'")

    return data


def infer_epsg_from_luw(cfg: Dict[str, Any],
                        fallback_lon: Optional[float],
                        fallback_lat: Optional[float]) -> int:
    """
    Infer UTM EPSG code. Preference:
    1) utm_epsg, epsg
    2) utm
    3) utm_zone + utm_hemisphere
    4) Derive from a fallback lon/lat if provided
    """
    for key in ("utm_epsg", "epsg", "UTM_EPSG", "UTM_epsg", "utm_code"):
        if key in cfg:
            return int(cfg[key])
    if "utm" in cfg:
        return int(cfg["utm"])
    if "utm_zone" in cfg:
        zone = int(cfg["utm_zone"])
        hemi = str(cfg.get("utm_hemisphere", "N")).upper()
        return (32600 if hemi == "N" else 32700) + zone
    if fallback_lon is not None and fallback_lat is not None:
        zone = int((fallback_lon + 180.0) // 6) + 1
        hemi_north = 1 if fallback_lat >= 0.0 else 0
        return (32600 if hemi_north else 32700) + zone
    soft_exit("Cannot infer UTM EPSG from LUW. Please provide utm_epsg, utm, or utm_zone/utm_hemisphere.")
    return 0


def input_float_with_default(prompt: str, default: Optional[float]) -> float:
    """Prompt user for a float with optional default when empty input is given."""
    suffix = f" [default {default}]" if default is not None else ""
    while True:
        s = input(f"{prompt}{suffix}: ").strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except Exception:
            print("Invalid number. Please re-enter.")


def input_int_with_default(prompt: str, default: Optional[int]) -> int:
    """Prompt user for a positive integer with optional default when empty input is given."""
    suffix = f" [default {default}]" if default is not None else ""
    while True:
        s = input(f"{prompt}{suffix}: ").strip()
        if s == "" and default is not None:
            return int(default)
        try:
            val = int(s)
            if val <= 0:
                print("Please enter a positive integer.")
                continue
            return val
        except Exception:
            print("Invalid integer. Please re-enter.")


def confirm_proceed(prompt: str) -> bool:
    """
    Ask for confirmation.
    Accept ENTER, 'y', 'yes' as affirmative.
    Any other input is treated as negative.
    """
    ans = input(f"{prompt} ").strip().lower()
    return ans in ("", "y", "yes")


def find_results_dir(luw_path: str) -> str:
    """
    Prefer RESULTS under the LUW's parent.
    If not present, try RESULTS under the LUW's grandparent.
    Create the first path under parent if none exists.
    """
    p = os.path.abspath(luw_path)
    parent = os.path.dirname(p)
    candidate1 = os.path.join(parent, "RESULTS")
    candidate2 = os.path.join(os.path.dirname(parent), "RESULTS")

    if os.path.isdir(candidate1):
        return candidate1
    if os.path.isdir(candidate2):
        return candidate2
    os.makedirs(candidate1, exist_ok=True)
    return candidate1


def locate_vtk(results_dir: str, casename: str, dtstr: str) -> Tuple[str, str]:
    """
    Locate uvw-{casename}_{datetime}.vtk or uvwrho-{casename}_{datetime}.vtk in RESULTS.
    Return (vtk_path, basename_without_ext).
    """
    cand1 = os.path.join(results_dir, f"uvw-{casename}_{dtstr}.vtk")
    cand2 = os.path.join(results_dir, f"uvwrho-{casename}_{dtstr}.vtk")
    if os.path.isfile(cand1):
        return cand1, os.path.splitext(os.path.basename(cand1))[0]
    if os.path.isfile(cand2):
        return cand2, os.path.splitext(os.path.basename(cand2))[0]
    vtks = [f for f in os.listdir(results_dir) if f.endswith(".vtk")]
    for name in vtks:
        if name.startswith(f"uvw-{casename}_{dtstr}") or name.startswith(f"uvwrho-{casename}_{dtstr}"):
            path = os.path.join(results_dir, name)
            return path, os.path.splitext(name)[0]
    soft_exit(f"VTK not found in RESULTS. Expected uvw-{casename}_{dtstr}.vtk or uvwrho-{casename}_{dtstr}.vtk")
    return "", ""


def choose_vector_name(mesh: pv.DataSet) -> str:
    """
    Choose wind vector array with priority:
    U, velocity, Velocity, UVW, data, Data, then the first 3-component array.
    Ignore data_rho if present.
    """
    names = list(mesh.array_names)
    names_no_rho = [n for n in names if n.lower() != "data_rho"]
    priority = ["U", "velocity", "Velocity", "UVW", "data", "Data"]
    for cand in priority:
        if cand in names_no_rho:
            arr = mesh[cand]
            if arr.ndim == 2 and arr.shape[1] == 3:
                return cand
    for n in names_no_rho:
        arr = mesh[n]
        if arr.ndim == 2 and arr.shape[1] == 3:
            return n
    soft_exit(f"No 3-component vector array found. Available arrays: {names}")
    return ""


# ------------------------- Core Processor -------------------------

class WindFieldProcessor:
    """Wind field processor using PyVista structured grid and index-based cropping."""

    def __init__(self,
                vtk_file_path: str,
                transformer_fwd: Transformer,
                transformer_inv: Transformer,
                center_lon: float,
                center_lat: float,
                si_x_cfd: Optional[List[float]] = None,
                si_y_cfd: Optional[List[float]] = None) -> None:
        self.vtk_file_path = vtk_file_path
        self.mesh: Optional[pv.StructuredGrid] = None
        self.wind_data: Optional[np.ndarray] = None
        self.grid_info: Optional[Dict[str, Any]] = None

        # Geodesy
        self.transformer_fwd = transformer_fwd   # lon/lat -> UTM
        self.transformer_inv = transformer_inv   # UTM -> lon/lat
        self.center_lon = float(center_lon)      # SW corner lon
        self.center_lat = float(center_lat)      # SW corner lat
        self.center_utm = self.transformer_fwd.transform(self.center_lon, self.center_lat)

        # Optional CFD meter ranges, e.g., [0.0, 38156.18], [0.0, 33276.73]
        self.si_x_cfd = si_x_cfd
        self.si_y_cfd = si_y_cfd

        # Vertical info
        self.zmin_vtk: Optional[float] = None
        self.zmax_vtk: Optional[float] = None

        # Crop bounds in VTK-relative coordinates
        self.crop_bounds: Optional[Dict[str, float]] = None

        # For plotting extent equal to the per-layer mapped min/max
        self.crop_lonlat_extent: Optional[Tuple[float, float, float, float]] = None


    def load_vtk(self) -> None:
        """Load VTK using PyVista and cache bounds."""
        print(f"[INFO] Reading VTK: {self.vtk_file_path}")
        self.mesh = pv.read(self.vtk_file_path)
        if not hasattr(self.mesh, "dimensions"):
            soft_exit("Input VTK is not a structured grid. Conversion is not implemented.")

        print(f"[INFO] Mesh dimensions: {self.mesh.dimensions}")
        print(f"[INFO] Mesh origin: {self.mesh.origin}")
        print(f"[INFO] Mesh spacing: {self.mesh.spacing}")
        bounds = self.mesh.bounds
        self.zmin_vtk = float(bounds[4])
        self.zmax_vtk = float(bounds[5])
        print(f"[INFO] Mesh bounds: X [{bounds[0]:.3f}, {bounds[1]:.3f}], "
              f"Y [{bounds[2]:.3f}, {bounds[3]:.3f}], "
              f"Z [{bounds[4]:.3f}, {bounds[5]:.3f}]")

    def plan_crop_from_lonlat(self,
                              lon_min: float, lon_max: float,
                              lat_min: float, lat_max: float) -> None:
        """
        Compute VTK-relative crop bounds from a lon/lat bounding box.
        Ground handling uses zmin + 50 m. Keep slices with h >= 0 where h = z_vtk - (zmin + 50).
        """
        if self.mesh is None or self.zmin_vtk is None or self.zmax_vtk is None:
            soft_exit("Mesh is not loaded.")

        # Store the requested lon/lat crop as a reference
        self.crop_lonlat_extent = (float(lon_min), float(lon_max), float(lat_min), float(lat_max))

        # Compute full-grid XY extents in VTK coordinates
        nx_full, ny_full, nz_full = self.mesh.dimensions
        origin = self.mesh.origin
        spacing = self.mesh.spacing
        x_full_min = float(origin[0])
        x_full_max = float(origin[0] + (nx_full - 1) * spacing[0])
        y_full_min = float(origin[1])
        y_full_max = float(origin[1] + (ny_full - 1) * spacing[1])

        # Optional scale factors from VTK units to CFD meters
        sx = 1.0
        sy = 1.0
        if self.si_x_cfd and len(self.si_x_cfd) == 2:
            sx = (self.si_x_cfd[1] - self.si_x_cfd[0]) / max(x_full_max - x_full_min, 1e-12)
        if self.si_y_cfd and len(self.si_y_cfd) == 2:
            sy = (self.si_y_cfd[1] - self.si_y_cfd[0]) / max(y_full_max - y_full_min, 1e-12)

        # Convert lon/lat box to UTM meters
        xmin_utm, ymin_utm = self.transformer_fwd.transform(lon_min, lat_min)
        xmax_utm, ymax_utm = self.transformer_fwd.transform(lon_max, lat_max)

        # Map UTM meters to VTK coordinates referenced to SW origin
        x_vtk_min = x_full_min + (xmin_utm - self.center_utm[0]) / sx
        x_vtk_max = x_full_min + (xmax_utm - self.center_utm[0]) / sx
        y_vtk_min = y_full_min + (ymin_utm - self.center_utm[1]) / sy
        y_vtk_max = y_full_min + (ymax_utm - self.center_utm[1]) / sy


        # Build full coordinate axes in VTK coordinates
        nx_full, ny_full, nz_full = self.mesh.dimensions
        origin = self.mesh.origin
        spacing = self.mesh.spacing
        z_coords_vtk_full = origin[2] + np.arange(nz_full) * spacing[2]

        # Ground at zmin + 50 m. Real height h = z_vtk - (zmin + 50).
        zmin = float(self.zmin_vtk)
        height = z_coords_vtk_full - (zmin + 50.0)
        z_mask = height >= 0.0  # keep layers at or above 0 m real height

        z_vtk_kept = z_coords_vtk_full[z_mask]
        if z_vtk_kept.size == 0:
            soft_exit("After removing the ground slab [zmin, zmin+50), no vertical slices remain.")

        self.crop_bounds = {
            "x_vtk_min": float(min(x_vtk_min, x_vtk_max)),
            "x_vtk_max": float(max(x_vtk_min, x_vtk_max)),
            "y_vtk_min": float(min(y_vtk_min, y_vtk_max)),
            "y_vtk_max": float(max(y_vtk_min, y_vtk_max)),
            "z_vtk_min": float(np.min(z_vtk_kept)),
            "z_vtk_max": float(np.max(z_vtk_kept))
        }

        print("[INFO] Planned crop in VTK-relative coordinates:")
        print(f"       X [{self.crop_bounds['x_vtk_min']:.3f}, {self.crop_bounds['x_vtk_max']:.3f}]")
        print(f"       Y [{self.crop_bounds['y_vtk_min']:.3f}, {self.crop_bounds['y_vtk_max']:.3f}]")
        print(f"       Ground reference: zmin + 50 m; kept vertical layers >= 0 m real height")

    def extract_structured_grid(self) -> None:
        """Extract cropped 3D vector field into an ndarray of shape (nx, ny, nz, 3) and cache metadata."""
        if self.mesh is None or self.crop_bounds is None or self.zmin_vtk is None:
            soft_exit("Mesh or crop bounds are not ready.")

        nx_full, ny_full, nz_full = self.mesh.dimensions
        origin = self.mesh.origin
        spacing = self.mesh.spacing

        # Full coordinate axes in VTK coordinates
        x_coords_full = origin[0] + np.arange(nx_full) * spacing[0]
        y_coords_full = origin[1] + np.arange(ny_full) * spacing[1]
        z_coords_vtk_full = origin[2] + np.arange(nz_full) * spacing[2]

        # Index masks by closed intervals
        x_mask = (x_coords_full >= self.crop_bounds["x_vtk_min"]) & (x_coords_full <= self.crop_bounds["x_vtk_max"])
        y_mask = (y_coords_full >= self.crop_bounds["y_vtk_min"]) & (y_coords_full <= self.crop_bounds["y_vtk_max"])
        z_mask_range = (z_coords_vtk_full >= self.crop_bounds["z_vtk_min"]) & (z_coords_vtk_full <= self.crop_bounds["z_vtk_max"])

        x_idx = np.where(x_mask)[0]
        y_idx = np.where(y_mask)[0]
        z_idx = np.where(z_mask_range)[0]

        if len(x_idx) == 0 or len(y_idx) == 0 or len(z_idx) == 0:
            soft_exit("Crop box does not intersect the dataset after index mapping.")

        # Choose wind vector array
        vec_name = choose_vector_name(self.mesh)
        vec_flat = self.mesh[vec_name]

        # Reshape to (nx, ny, nz, 3) from PyVista's flat order (z, y, x, 3)
        wind_full = vec_flat.reshape((nz_full, ny_full, nx_full, 3))
        wind_full = np.transpose(wind_full, (2, 1, 0, 3))

        # Apply index cropping
        wind = wind_full[np.ix_(x_idx, y_idx, z_idx)]

        # Compute height values for the kept z indices using h = z_vtk - (zmin + 50)
        z_vtk_kept = z_coords_vtk_full[z_idx]
        zmin = float(self.zmin_vtk)
        z_heights = z_vtk_kept - (zmin + 50.0)

        # Cache arrays and metadata
        self.wind_data = wind
        self.grid_info = {
            "x_coords_vtk": x_coords_full[x_idx],
            "y_coords_vtk": y_coords_full[y_idx],
            "z_coords_vtk": z_vtk_kept,
            "z_heights": z_heights,
            "zmin_vtk": zmin,
            "zmax_vtk": float(self.zmax_vtk) if self.zmax_vtk is not None else None,
            "center_lon": float(self.center_lon),
            "center_lat": float(self.center_lat),
            "center_utm": tuple(self.center_utm),
            "vector_name": vec_name
        }

        print(f"[INFO] Cropped wind array shape: {self.wind_data.shape}")
        print(f"[INFO] Lowest plotted height: {float(z_heights.min()):.3f} m")

    def save_npz(self, out_path: str) -> None:
        """Save compressed NPZ with keys wind_data and grid_info."""
        if self.wind_data is None or self.grid_info is None:
            soft_exit("No data to save. Extraction not completed.")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, wind_data=self.wind_data, grid_info=self.grid_info)
        size_mb = os.path.getsize(out_path) / 1024.0 / 1024.0
        print(f"[INFO] Saved NPZ: {out_path} ({size_mb:.1f} MB)")

    def export_netcdf(self, out_path: str, epsg: int, casename: str, dtstr: str, vtk_basename: str) -> None:
        """
        Export a 3D NetCDF file with lon/lat coordinates and height dimension.
        Variables: u_grid, v_grid, w with dims (height, y, x). Coordinate variables: lon(y,x), lat(y,x), height(height).
        The wind components follow the model grid axes, not rotated to true east and north.
        """
        if self.wind_data is None or self.grid_info is None:
            print("[WARN] No data to export. Skipping NetCDF.")
            return

        try:
            from netCDF4 import Dataset
        except Exception:
            print("[WARN] netCDF4 is not available. Please install 'netCDF4' to enable NetCDF export. Skipping.")
            return

        # Shapes and coordinates
        nx = self.wind_data.shape[0]
        ny = self.wind_data.shape[1]
        nz = self.wind_data.shape[2]

        x_vtk = np.array(self.grid_info["x_coords_vtk"])  # length nx
        y_vtk = np.array(self.grid_info["y_coords_vtk"])  # length ny

        # Map VTK x/y to absolute UTM using SW origin and optional CFD meter scaling
        nx_full, ny_full, _ = self.mesh.dimensions
        origin = self.mesh.origin
        spacing = self.mesh.spacing
        x_full_min = float(origin[0])
        x_full_max = float(origin[0] + (nx_full - 1) * spacing[0])
        y_full_min = float(origin[1])
        y_full_max = float(origin[1] + (ny_full - 1) * spacing[1])

        sx = 1.0
        sy = 1.0
        if self.si_x_cfd and len(self.si_x_cfd) == 2:
            sx = (self.si_x_cfd[1] - self.si_x_cfd[0]) / max(x_full_max - x_full_min, 1e-12)
        if self.si_y_cfd and len(self.si_y_cfd) == 2:
            sy = (self.si_y_cfd[1] - self.si_y_cfd[0]) / max(y_full_max - y_full_min, 1e-12)

        swx, swy = self.center_utm
        x_abs = swx + (x_vtk - x_full_min) * sx
        y_abs = swy + (y_vtk - y_full_min) * sy

        X_abs, Y_abs = np.meshgrid(x_abs, y_abs, indexing="xy")
        lon_flat, lat_flat = self.transformer_inv.transform(X_abs.ravel(), Y_abs.ravel())

        lon2d = np.asarray(lon_flat, dtype=np.float64).reshape((ny, nx))
        lat2d = np.asarray(lat_flat, dtype=np.float64).reshape((ny, nx))
        
        print(f"[INFO] NetCDF lon/lat range after mapping: "
              f"lon[{lon2d.min():.6f}, {lon2d.max():.6f}], lat[{lat2d.min():.6f}, {lat2d.max():.6f}]")

        # Prepare wind components as (height, y, x)
        wind_z_y_x_3 = np.transpose(self.wind_data, (2, 1, 0, 3))
        u3 = wind_z_y_x_3[:, :, :, 0].astype(np.float32)
        v3 = wind_z_y_x_3[:, :, :, 1].astype(np.float32)
        w3 = wind_z_y_x_3[:, :, :, 2].astype(np.float32)

        height = np.array(self.grid_info["z_heights"], dtype=np.float32)  # length nz

        # Write NetCDF
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ds = Dataset(out_path, "w", format="NETCDF4")
        try:
            # Dimensions
            ds.createDimension("height", nz)
            ds.createDimension("y", ny)
            ds.createDimension("x", nx)

            # Dimension coordinate variables
            var_y = ds.createVariable("y", "i4", ("y",))
            var_y.long_name = "grid y index"
            var_y.units = "1"
            var_y[:] = np.arange(ny, dtype=np.int32)

            var_x = ds.createVariable("x", "i4", ("x",))
            var_x.long_name = "grid x index"
            var_x.units = "1"
            var_x[:] = np.arange(nx, dtype=np.int32)

            # Coordinate variables
            var_height = ds.createVariable("height", "f4", ("height",))
            var_height.units = "m"
            var_height.long_name = "height above ground, reference zmin + 50 m"
            var_height.axis = "Z"
            var_height.positive = "up"
            var_height[:] = height

            var_lat = ds.createVariable("lat", "f8", ("y", "x"), zlib=True, complevel=4)
            var_lat.units = "degrees_north"
            var_lat.standard_name = "latitude"
            var_lat.long_name = "latitude on curvilinear grid"
            var_lat[:, :] = lat2d

            var_lon = ds.createVariable("lon", "f8", ("y", "x"), zlib=True, complevel=4)
            var_lon.units = "degrees_east"
            var_lon.standard_name = "longitude"
            var_lon.long_name = "longitude on curvilinear grid"
            var_lon[:, :] = lon2d

            # Data variables
            var_u = ds.createVariable("u_grid", "f4", ("height", "y", "x"), zlib=True, complevel=4)
            var_u.units = "m s-1"
            var_u.long_name = "wind component along model x axis"
            var_u.coordinates = "lon lat height"
            var_u[:, :, :] = u3

            var_v = ds.createVariable("v_grid", "f4", ("height", "y", "x"), zlib=True, complevel=4)
            var_v.units = "m s-1"
            var_v.long_name = "wind component along model y axis"
            var_v.coordinates = "lon lat height"
            var_v[:, :, :] = v3

            var_w = ds.createVariable("w", "f4", ("height", "y", "x"), zlib=True, complevel=4)
            var_w.units = "m s-1"
            var_w.long_name = "vertical wind component"
            var_w.coordinates = "lon lat height"
            var_w[:, :, :] = w3

            # Optional CRS variable
            crs = ds.createVariable("crs", "i4")
            crs.long_name = "coordinate reference system"
            crs.spatial_ref = f"EPSG:{int(epsg)}"
            crs.grid_mapping_name = "latitude_longitude"

            # Global attributes
            ds.Conventions = "CF-1.8"
            ds.title = "Wind field on lon/lat grid with height from zmin + 50 m"
            ds.history = "Created by wind_from_luw.py"
            ds.source_vtk = vtk_basename
            ds.crs_epsg = int(epsg)
            ds.center_lon = float(self.center_lon)
            ds.center_lat = float(self.center_lat)
            ds.note = "Components are aligned with model x and y axes, not rotated to true east and north."
            ds.case = str(casename)
            ds.datetime_tag = str(dtstr)

        finally:
            ds.close()

        size_mb = os.path.getsize(out_path) / 1024.0 / 1024.0
        print(f"[INFO] Saved NetCDF: {out_path} ({size_mb:.1f} MB)")

    def visualize(self, out_dir: str, num_layers: int) -> None:
        """
        Make per-layer quicklook figures and a histogram into out_dir.
        Each selected layer is saved as wind_{height_m}m.png with lon/lat axes.
        For each figure, axes are limited to that layer's lon/lat min and max computed from the mapped grid.
        """
        if self.wind_data is None or self.grid_info is None:
            soft_exit("No data to visualize. Extraction not completed.")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)

        speeds = np.sqrt(np.sum(self.wind_data ** 2, axis=-1))
        nz = speeds.shape[2]

        # Determine layer indices to export
        n_layers = max(1, min(num_layers, nz))
        layers = np.linspace(0, nz - 1, num=n_layers, dtype=int)
        layers = np.unique(layers)

        # Build lon/lat grid by the same linear mapping used for NetCDF so axes match exactly
        x_vtk = np.array(self.grid_info["x_coords_vtk"])
        y_vtk = np.array(self.grid_info["y_coords_vtk"])
        
        # Build lon/lat grid via SW-origin mapping with optional CFD scaling
        nx_full, ny_full, _ = self.mesh.dimensions
        origin = self.mesh.origin
        spacing = self.mesh.spacing
        x_full_min = float(origin[0])
        x_full_max = float(origin[0] + (nx_full - 1) * spacing[0])
        y_full_min = float(origin[1])
        y_full_max = float(origin[1] + (ny_full - 1) * spacing[1])

        sx = 1.0
        sy = 1.0
        if self.si_x_cfd and len(self.si_x_cfd) == 2:
            sx = (self.si_x_cfd[1] - self.si_x_cfd[0]) / max(x_full_max - x_full_min, 1e-12)
        if self.si_y_cfd and len(self.si_y_cfd) == 2:
            sy = (self.si_y_cfd[1] - self.si_y_cfd[0]) / max(y_full_max - y_full_min, 1e-12)

        swx, swy = self.center_utm
        x_abs = swx + (x_vtk - x_full_min) * sx
        y_abs = swy + (y_vtk - y_full_min) * sy
        X_abs, Y_abs = np.meshgrid(x_abs, y_abs, indexing="xy")
        lon_flat, lat_flat = self.transformer_inv.transform(X_abs.ravel(), Y_abs.ravel())
        lon2d = np.asarray(lon_flat, dtype=np.float64).reshape(Y_abs.shape)
        lat2d = np.asarray(lat_flat, dtype=np.float64).reshape(Y_abs.shape)


        # Common color scale
        vmin = 0.0
        vmax = float(np.max(speeds))

        used_names = set()
        for k in layers:
            img = speeds[:, :, k].T
            height_val = float(self.grid_info["z_heights"][k])

            # Per-layer axis limits taken from the mapped lon/lat grid
            lon_min_layer = float(np.nanmin(lon2d))
            lon_max_layer = float(np.nanmax(lon2d))
            lat_min_layer = float(np.nanmin(lat2d))
            lat_max_layer = float(np.nanmax(lat2d))

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                img,
                origin="lower",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="bilinear",
                extent=[lon_min_layer, lon_max_layer, lat_min_layer, lat_max_layer],
            )
            ax.set_title(f"Wind speed at {height_val:.1f} m")
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")

            # Enforce exact layer-specific limits
            ax.set_xlim(lon_min_layer, lon_max_layer)
            ax.set_ylim(lat_min_layer, lat_max_layer)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Wind Speed (m/s)")

            h_rounded = int(round(height_val))
            fname = f"wind_{h_rounded}m.png"
            if fname in used_names:
                fname = f"wind_{h_rounded}m_{k}.png"
            used_names.add(fname)
            out_png = os.path.join(out_dir, fname)
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved section: {out_png}")

        # Histogram of speeds for reference
        fig, ax = plt.subplots(figsize=(10, 6))
        flat = speeds.flatten()
        flat = flat[flat > 0.1]
        ax.hist(flat, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")
        ax.set_title("Wind Speed Distribution")
        ax.grid(True, alpha=0.3)
        out_hist = os.path.join(out_dir, "wind_speed_distribution.png")
        plt.savefig(out_hist, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved distribution: {out_hist}")


# ------------------------- Main -------------------------

def main() -> None:
    # Validate CLI
    if len(sys.argv) != 2:
        print("Usage: python wind_from_luw.py <path/to/conf.luw>")
        sys.exit(2)

    luw_path = sys.argv[1]
    print(f"[INFO] LUW: {luw_path}")

    # Read LUW
    try:
        cfg = read_luw(luw_path)
        si_x_cfd = cfg.get("si_x_cfd", None)
        si_y_cfd = cfg.get("si_y_cfd", None)
    except SystemExit:
        raise
    except Exception as e:
        soft_exit(f"Failed to read LUW. Detail: {e}")

    # Extract casename and datetime
    casename = str(cfg.get("casename", "")).strip()
    dtstr = str(cfg.get("datetime", "")).strip()
    if not casename:
        soft_exit("Missing casename in LUW. Please add: casename = your_case")
    if not dtstr or not re.fullmatch(r'\d{14}', dtstr):
        soft_exit("Missing or invalid datetime in LUW. Expected format yyyymmddHHMMSS. Run tranluw first.")

    # Defaults for interactive crop from LUW if available
    lon_defaults = cfg.get("cut_lon_manual", None)
    lat_defaults = cfg.get("cut_lat_manual", None)
    lon_min_def = float(lon_defaults[0]) if isinstance(lon_defaults, list) and len(lon_defaults) == 2 else None
    lon_max_def = float(lon_defaults[1]) if isinstance(lon_defaults, list) and len(lon_defaults) == 2 else None
    lat_min_def = float(lat_defaults[0]) if isinstance(lat_defaults, list) and len(lat_defaults) == 2 else None
    lat_max_def = float(lat_defaults[1]) if isinstance(lat_defaults, list) and len(lat_defaults) == 2 else None

    # Determine center for VTK-relative mapping
    center_lon_luw = cfg.get("center_lon", None)
    center_lat_luw = cfg.get("center_lat", None)

    # Determine UTM CRS
    lon_avg_for_epsg = None
    lat_avg_for_epsg = None
    if lon_min_def is not None and lon_max_def is not None and lat_min_def is not None and lat_max_def is not None:
        lon_avg_for_epsg = 0.5 * (lon_min_def + lon_max_def)
        lat_avg_for_epsg = 0.5 * (lat_min_def + lat_max_def)
    epsg = infer_epsg_from_luw(cfg, lon_avg_for_epsg, lat_avg_for_epsg)
    try:
        crs_dst = CRS.from_epsg(epsg)
    except Exception:
        soft_exit(f"Invalid EPSG in LUW: {epsg}")
    transformer_fwd = Transformer.from_crs("EPSG:4326", crs_dst, always_xy=True)
    transformer_inv = Transformer.from_crs(crs_dst, "EPSG:4326", always_xy=True)
    print(f"[INFO] Using UTM CRS EPSG:{epsg}")

    # Interactive crop inputs
    try:
        print("[INPUT] Please provide target crop in lon/lat.")
        lon_min = input_float_with_default("Enter target longitude min", lon_min_def)
        lon_max = input_float_with_default("Enter target longitude max", lon_max_def)
        lat_min = input_float_with_default("Enter target latitude min", lat_min_def)
        lat_max = input_float_with_default("Enter target latitude max", lat_max_def)
    except KeyboardInterrupt:
        soft_exit("Aborted by user during input.")
    except Exception as e:
        soft_exit(f"Invalid inputs. Detail: {e}")

    # Echo and confirm
    print(f"[CONFIRM] Crop box lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")
    if not confirm_proceed("Press ENTER to proceed, or type 'y'/'yes'. Any other key to abort:"):
        soft_exit("User did not confirm the crop box.")

    # Set mapping origin to the SW corner of the CFD grid. This origin corresponds to VTK (0,0).
    if center_lon_luw is not None and center_lat_luw is not None:
        center_lon = float(center_lon_luw)
        center_lat = float(center_lat_luw)
    elif (lon_min_def is not None and lat_min_def is not None):
        center_lon = float(lon_min_def)
        center_lat = float(lat_min_def)
    else:
        center_lon = float(lon_min)
        center_lat = float(lat_min)
    print(f"[INFO] Mapping origin set to SW corner at lon {center_lon}, lat {center_lat}")


    # Ask for number of layers to export
    try:
        n_layers = input_int_with_default("Enter number of layers to export", 6)
    except KeyboardInterrupt:
        soft_exit("Aborted by user during input.")
    except Exception as e:
        soft_exit(f"Invalid inputs. Detail: {e}")

    # Locate RESULTS and VTK
    results_dir = find_results_dir(luw_path)
    print(f"[INFO] RESULTS directory: {results_dir}")
    vtk_path, base_name = locate_vtk(results_dir, casename, dtstr)
    print(f"[INFO] VTK located: {vtk_path}")

    # Prepare outputs
    npz_out = os.path.join(results_dir, f"{base_name}.npz")
    sections_dir = os.path.join(results_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)

    # Process
    try:
        processor = WindFieldProcessor(
            vtk_path, transformer_fwd, transformer_inv, center_lon, center_lat,
            si_x_cfd=si_x_cfd, si_y_cfd=si_y_cfd
        )
        processor.load_vtk()
        processor.plan_crop_from_lonlat(lon_min, lon_max, lat_min, lat_max)
        processor.extract_structured_grid()
        processor.save_npz(npz_out)
        processor.visualize(sections_dir, num_layers=n_layers)

        # Optional NetCDF export
        print("[INPUT] Do you want to export a 3D NetCDF file in lon/lat coordinates to RESULTS?")
        if confirm_proceed("Press ENTER, 'y', or 'yes' to export. Any other key to skip:"):
            nc_out = os.path.join(results_dir, f"{base_name}_visluw.nc")
            processor.export_netcdf(nc_out, epsg=epsg, casename=casename, dtstr=dtstr, vtk_basename=base_name)
        else:
            print("[INFO] NetCDF export skipped by user choice.")

    except SystemExit:
        raise
    except Exception as e:
        soft_exit(f"Processing failed. Detail: {e}")

    print("[INFO] Completed successfully.")
    print(f"[INFO] NPZ: {npz_out}")
    print(f"[INFO] Figures directory: {sections_dir}")


if __name__ == "__main__":
    main()

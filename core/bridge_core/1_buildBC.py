# 1_buildBC_with_dem.py
# Modified version of 1_buildBC.py with DEM terrain support
from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, Optional
import math
try:
    import vtk
    from vtk.util import numpy_support as nps
except Exception:
    vtk = None
    nps = None
import geopandas as gpd
from auto_UTM import get_utm_crs_from_conf_raw
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
from terrainreader import ensure_dem_shp_from_tif
import shutil
import os
import sys
import re
import time

# limit threads for numpy/scipy to avoid over-subscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import xarray as xr
import pickle

from concurrent.futures import ThreadPoolExecutor, as_completed
from dask import config as dask_config
from pyproj import Transformer
from scipy.interpolate import griddata, LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay, cKDTree

def _pick_first_existing_var(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _isel_first_time_if_present(da: xr.DataArray) -> xr.DataArray:
    for tdim in ("Time", "time"):
        if tdim in da.dims:
            return da.isel({tdim: 0})
    return da


def _detect_and_rename_wrf_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize common WRF dimension names to internal names.
    Geographic lon/lat should be handled via lon_geo/lat_geo, not via dim coords.
    """
    stag_dims = []
    for d in ("west_east_stag", "south_north_stag", "bottom_top_stag"):
        if d in ds.dims:
            stag_dims.append(d)
    if stag_dims:
        _log_info(f"Detected WRF staggered dimensions in input: {', '.join(stag_dims)}")

    rename_map = {}
    if "west_east" in ds.dims and "lon" not in ds.dims:
        rename_map["west_east"] = "lon"
    if "south_north" in ds.dims and "lat" not in ds.dims:
        rename_map["south_north"] = "lat"
    if "west_east_stag" in ds.dims and "lon_stag" not in ds.dims:
        rename_map["west_east_stag"] = "lon_stag"
    if "south_north_stag" in ds.dims and "lat_stag" not in ds.dims:
        rename_map["south_north_stag"] = "lat_stag"

    if rename_map:
        ds = ds.rename(rename_map)
        _log_info(f"Renamed WRF dims to internal names: {rename_map}")
    return ds


def _ensure_lonlat_geo_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure ds has lon_geo and lat_geo coordinates.
    lon_geo and lat_geo can be 1D or 2D.
    Priority:
      1) existing lon/lat
      2) WRF XLONG/XLAT
    """
    if "lon_geo" in ds.coords and "lat_geo" in ds.coords:
        return ds

    if "lon" in ds.variables and "lat" in ds.variables:
        lon_da = ds["lon"]
        lat_da = ds["lat"]
        lon_da = _isel_first_time_if_present(lon_da)
        lat_da = _isel_first_time_if_present(lat_da)
        ds = ds.assign_coords(lon_geo=lon_da, lat_geo=lat_da)
        return ds

    lon_name = _pick_first_existing_var(ds, ("XLONG", "XLONG_M", "XLONG_U", "XLONG_V"))
    lat_name = _pick_first_existing_var(ds, ("XLAT", "XLAT_M", "XLAT_U", "XLAT_V"))
    if lon_name and lat_name:
        lon_da = _isel_first_time_if_present(ds[lon_name])
        lat_da = _isel_first_time_if_present(ds[lat_name])
        ds = ds.assign_coords(lon_geo=lon_da, lat_geo=lat_da)
        _log_info(f"Using WRF lon/lat coords: {lon_name}/{lat_name}")
        return ds

    raise KeyError("Cannot find lon/lat or XLONG/XLAT in dataset")


def _resolve_wind_var_name(ds: xr.Dataset, preferred: str, fallbacks: Tuple[str, ...], kind: str) -> str:
    if preferred in ds.variables:
        return preferred
    for name in fallbacks:
        if name in ds.variables:
            _log_warn(f"Wind variable {preferred} not found, use {name} for {kind}")
            return name
    raise KeyError(f"Wind variable {preferred} not found, and fallbacks also missing for {kind}")


def _resolve_temperature_var_name(ds: xr.Dataset) -> Tuple[Optional[str], bool]:
    """
    Resolve temperature variable from raw NC variable names.
    Returns:
      (var_name, is_celsius)
      - is_celsius=False: variable is already Kelvin
      - is_celsius=True:  variable is Celsius and should be converted to Kelvin
    """
    kelvin_candidates = ("TK", "tk", "T", "t", "Temp", "temp", "Temperature", "temperature")
    celsius_candidates = ("TC", "tc")

    # Prefer data variables to avoid accidentally selecting coordinates.
    t_kelvin = next((name for name in kelvin_candidates if name in ds.data_vars), None)
    t_celsius = next((name for name in celsius_candidates if name in ds.data_vars), None)
    if t_kelvin is None:
        t_kelvin = _pick_first_existing_var(ds, kelvin_candidates)
    if t_celsius is None:
        t_celsius = _pick_first_existing_var(ds, celsius_candidates)

    if t_celsius is not None:
        if t_kelvin is not None:
            _log_warn(
                f"Both Kelvin and Celsius temperature fields found ({t_kelvin}, {t_celsius}); "
                f"use {t_celsius} (Celsius) with conversion to Kelvin"
            )
        return t_celsius, True

    if t_kelvin is not None:
        return t_kelvin, False

    return None, False


def _destagger_wrf(da: xr.DataArray) -> xr.DataArray:
    """
    Destagger common WRF staggered dimensions by averaging adjacent grid points.
    IMPORTANT: Do not rely on xarray coordinate alignment when averaging; use positional averaging.
    """
    def _avg_along_dim_positional(x: xr.DataArray, dim_in: str, dim_out: str) -> xr.DataArray:
        if dim_in not in x.dims:
            return x

        x0 = x.isel({dim_in: slice(0, -1)}).data
        x1 = x.isel({dim_in: slice(1, None)}).data
        data = 0.5 * (x0 + x1)

        dims_out = list(x.dims)
        dim_idx = dims_out.index(dim_in)
        dims_out[dim_idx] = dim_out

        coords_out = {}
        for d in dims_out:
            if d == dim_out:
                if dim_in in x.coords:
                    c = np.asarray(x.coords[dim_in].values)
                    if c.ndim == 1 and c.size == x.sizes[dim_in]:
                        c_mid = 0.5 * (c[:-1] + c[1:])
                        coords_out[dim_out] = (dim_out, c_mid.astype(np.float32))
                    else:
                        coords_out[dim_out] = (dim_out, np.arange(data.shape[dim_idx], dtype=np.float32))
                else:
                    coords_out[dim_out] = (dim_out, np.arange(data.shape[dim_idx], dtype=np.float32))
            else:
                if d in x.coords:
                    coords_out[d] = x.coords[d]

        out = xr.DataArray(
            data,
            dims=tuple(dims_out),
            coords=coords_out,
            attrs=x.attrs,
            name=x.name,
        )
        return out

    if "lon_stag" in da.dims:
        _log_info("Destagger along lon_stag to lon for WRF staggered grid")
        da = _avg_along_dim_positional(da, "lon_stag", "lon")
    if "lat_stag" in da.dims:
        _log_info("Destagger along lat_stag to lat for WRF staggered grid")
        da = _avg_along_dim_positional(da, "lat_stag", "lat")
    if "bottom_top_stag" in da.dims:
        _log_info("Destagger along bottom_top_stag to bottom_top for WRF staggered grid")
        da = _avg_along_dim_positional(da, "bottom_top_stag", "bottom_top")

    return da


def _ensure_dim_index_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure every internal dimension used by interp/isel has a numeric 1D coordinate.
    This is required for WRF-style datasets where renamed dims often have no coord variable.
    """
    dims_need = []
    for d in ("lon", "lat", "lon_stag", "lat_stag", "bottom_top", "bottom_top_stag"):
        if d in ds.dims and d not in ds.coords:
            dims_need.append(d)

    if not dims_need:
        return ds

    coords = {d: (d, np.arange(ds.sizes[d], dtype=np.float32)) for d in dims_need}
    return ds.assign_coords(coords)

def _ensure_wrf_height_agl_1d(ds: xr.Dataset) -> xr.Dataset:
    """
    For WRF: build a 1D vertical height (AGL) coordinate for bottom_top using PH/PHB (and HGT if present).
    Saved as coord name: height_agl_1d with dim bottom_top.
    IMPORTANT: Use positional averaging on bottom_top_stag to avoid coordinate-alignment NaNs.
    """
    if "bottom_top" not in ds.dims:
        return ds
    if "height_agl_1d" in ds.coords:
        return ds
    if "bottom_top_stag" not in ds.dims:
        return ds
    if ("PH" not in ds.variables) or ("PHB" not in ds.variables):
        return ds

    try:
        ph = _isel_first_time_if_present(ds["PH"])
        phb = _isel_first_time_if_present(ds["PHB"])

        g = np.float32(9.81)
        z_w = (ph + phb) / g  # (bottom_top_stag, lat, lon) in meters

        z0 = z_w.isel(bottom_top_stag=slice(0, -1)).data
        z1 = z_w.isel(bottom_top_stag=slice(1, None)).data
        z_m_data = 0.5 * (z0 + z1)

        coords_out = {}
        if "bottom_top_stag" in z_w.coords:
            c = np.asarray(z_w.coords["bottom_top_stag"].values)
            if c.ndim == 1 and c.size == ds.sizes["bottom_top_stag"]:
                c_mid = 0.5 * (c[:-1] + c[1:])
                coords_out["bottom_top"] = ("bottom_top", c_mid.astype(np.float32))
            else:
                coords_out["bottom_top"] = ("bottom_top", np.arange(ds.sizes["bottom_top"], dtype=np.float32))
        else:
            coords_out["bottom_top"] = ("bottom_top", np.arange(ds.sizes["bottom_top"], dtype=np.float32))

        if "lat" in z_w.dims and "lat" in z_w.coords:
            coords_out["lat"] = z_w.coords["lat"]
        if "lon" in z_w.dims and "lon" in z_w.coords:
            coords_out["lon"] = z_w.coords["lon"]

        z_m = xr.DataArray(
            z_m_data,
            dims=("bottom_top",) + tuple(d for d in z_w.dims if d != "bottom_top_stag"),
            coords=coords_out,
            name="z_mass",
        )

        if "HGT" in ds.variables:
            hgt = _isel_first_time_if_present(ds["HGT"])
            z_m = z_m - hgt

        if ("lat" in z_m.dims) and ("lon" in z_m.dims):
            z1d = z_m.mean(dim=("lat", "lon"), skipna=True).astype(np.float32).values
        else:
            z1d = z_m.astype(np.float32).values
            if z1d.ndim > 1:
                z1d = np.nanmean(z1d.reshape(z1d.shape[0], -1), axis=1).astype(np.float32)

        if z1d.ndim != 1 or z1d.size != ds.sizes["bottom_top"]:
            return ds

        if not np.isfinite(z1d).all():
            return ds

        # Enforce monotonic increasing (required by later 1D interpolation)
        z1d = z1d.astype(np.float32)
        for k in range(1, z1d.size):
            if z1d[k] <= z1d[k - 1]:
                z1d[k] = z1d[k - 1] + np.float32(1e-3)

        da_h = xr.DataArray(z1d, dims=("bottom_top",), name="height_agl_1d")
        da_h.attrs["units"] = "m"
        da_h.attrs["description"] = "WRF AGL height (domain-mean) on mass levels"

        return ds.assign_coords(height_agl_1d=da_h)

    except Exception:
        return ds

def _ensure_wrf_pressure_hpa_1d(ds: xr.Dataset) -> xr.Dataset:
    """
    For WRF: provide a 1D pressure (hPa) coordinate on bottom_top for reference (domain-mean).
    This does NOT replace height-based z used by VTK/CFD.
    """
    if "bottom_top" not in ds.dims:
        return ds
    if "pressure_hpa_1d" in ds.coords:
        return ds
    if ("P" not in ds.variables) or ("PB" not in ds.variables):
        return ds

    try:
        p = _isel_first_time_if_present(ds["P"] + ds["PB"])  # Pa
        if ("lat" in p.dims) and ("lon" in p.dims):
            p1d = (p.mean(dim=("lat", "lon"), skipna=True).astype(np.float32).values) / np.float32(100.0)
        else:
            v = p.astype(np.float32).values
            if v.ndim > 1:
                v = np.nanmean(v.reshape(v.shape[0], -1), axis=1).astype(np.float32)
            p1d = v / np.float32(100.0)

        if p1d.ndim != 1 or p1d.size != ds.sizes["bottom_top"]:
            return ds

        if not np.isfinite(p1d).all():
            return ds

        da_p = xr.DataArray(p1d.astype(np.float32), dims=("bottom_top",), name="pressure_hpa_1d")
        da_p.attrs["units"] = "hPa"
        da_p.attrs["description"] = "WRF pressure (domain-mean) on mass levels"

        return ds.assign_coords(pressure_hpa_1d=da_p)

    except Exception:
        return ds

def _detect_vertical_dim_and_levels(ds: xr.Dataset) -> Tuple[xr.Dataset, str, np.ndarray]:
    """
    Return (ds, vert_dim_name, lev_1d_values).
    lev_1d_values must be 1D and length == ds.sizes[vert_dim_name].
    """
    # prefer true vertical dims first
    for cand in ("lev", "height_agl", "height", "elevation"):
        if cand in ds.dims:
            lev = np.asarray(ds[cand].values, dtype=np.float32)
            if lev.ndim == 1 and lev.size == ds.sizes[cand]:
                return ds, cand, lev

    # WRF mass vertical dimension
    if "bottom_top" in ds.dims:
        ds = _ensure_wrf_pressure_hpa_1d(ds)
        ds = _ensure_wrf_height_agl_1d(ds)

        if "height_agl_1d" in ds.coords:
            lev = np.asarray(ds["height_agl_1d"].values, dtype=np.float32)
            if lev.ndim == 1 and lev.size == ds.sizes["bottom_top"]:
                return ds, "bottom_top", lev

        if "bottom_top" in ds.coords:
            lev = np.asarray(ds["bottom_top"].values, dtype=np.float32)
            if lev.ndim == 1 and lev.size == ds.sizes["bottom_top"]:
                return ds, "bottom_top", lev

        lev = np.arange(ds.sizes["bottom_top"], dtype=np.float32)
        return ds, "bottom_top", lev

    raise ValueError("Vertical coord not found. Expect lev or height_agl or height or elevation or bottom_top")


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _parse_target_lonlat_from_conf(conf_raw: str | None) -> Optional[Tuple[float, float, float, float]]:
    if not conf_raw:
        return None
    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", conf_raw)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", conf_raw)
    if not (m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip()):
        return None

    lon_min_c, lon_max_c = [float(v) for v in m_lon.group(1).split(",")]
    lat_min_c, lat_max_c = [float(v) for v in m_lat.group(1).split(",")]
    lon_min, lon_max = sorted((lon_min_c, lon_max_c))
    lat_min, lat_max = sorted((lat_min_c, lat_max_c))
    return lon_min, lon_max, lat_min, lat_max


def _format_lonlat_bounds(bounds: Tuple[float, float, float, float]) -> str:
    lon_min, lon_max, lat_min, lat_max = bounds
    return f"lon[{lon_min:.6f}, {lon_max:.6f}] lat[{lat_min:.6f}, {lat_max:.6f}]"


def _bbox_contains(target_bounds: Tuple[float, float, float, float],
                   input_bounds: Tuple[float, float, float, float]) -> bool:
    t_lon_min, t_lon_max, t_lat_min, t_lat_max = target_bounds
    i_lon_min, i_lon_max, i_lat_min, i_lat_max = input_bounds
    return (i_lon_min <= t_lon_min) and (i_lon_max >= t_lon_max) and (i_lat_min <= t_lat_min) and (i_lat_max >= t_lat_max)


def _bbox_max_miss_percent(target_bounds: Tuple[float, float, float, float],
                           input_bounds: Tuple[float, float, float, float]) -> float:
    """
    Return the max under-coverage percentage of input_bounds relative to target_bounds.
    This is intended to tolerate tiny floating/rounding mismatches.
    """
    t_lon_min, t_lon_max, t_lat_min, t_lat_max = target_bounds
    i_lon_min, i_lon_max, i_lat_min, i_lat_max = input_bounds

    lon_span = abs(t_lon_max - t_lon_min)
    lat_span = abs(t_lat_max - t_lat_min)
    if lon_span <= 0.0 or lat_span <= 0.0:
        return 100.0

    miss_lon_min = max(0.0, i_lon_min - t_lon_min)
    miss_lon_max = max(0.0, t_lon_max - i_lon_max)
    miss_lat_min = max(0.0, i_lat_min - t_lat_min)
    miss_lat_max = max(0.0, t_lat_max - i_lat_max)

    lon_pct = (max(miss_lon_min, miss_lon_max) / lon_span) * 100.0
    lat_pct = (max(miss_lat_min, miss_lat_max) / lat_span) * 100.0
    return max(lon_pct, lat_pct)


def _timed_input_line(prompt: str, timeout_s: float) -> Optional[str]:
    """
    Read a line from stdin with timeout.
    Returns the line (without trailing newline) or None if timeout/EOF happens.

    On Windows, uses msvcrt to avoid blocking forever on console input.
    On non-Windows, falls back to select() where available.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()

    try:
        import msvcrt  # type: ignore
    except Exception:
        msvcrt = None  # type: ignore

    if msvcrt is not None:
        buf: list[str] = []
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()

                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(buf)

                if ch == "\x03":
                    raise KeyboardInterrupt

                if ch == "\b":
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue

                if ch in ("\x00", "\xe0"):
                    try:
                        msvcrt.getwch()
                    except Exception:
                        pass
                    continue

                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()

            time.sleep(0.05)

        sys.stdout.write("\n")
        sys.stdout.flush()
        return None

    try:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], float(timeout_s))
        if not ready:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return None
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\r\n")
    except Exception:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return None


def _confirm_bbox_coverage(kind: str,
                           target_bounds: Tuple[float, float, float, float],
                           input_bounds: Tuple[float, float, float, float]) -> None:
    if _bbox_contains(target_bounds, input_bounds):
        return

    max_miss_pct = _bbox_max_miss_percent(target_bounds, input_bounds)
    if max_miss_pct < 0.1:
        _log_warn(f"{kind} bounds are slightly smaller than target (max miss {max_miss_pct:.4f}% < 0.1%). Continue without interruption.")
        return

    _log_warn(f"{kind} bounds do not fully cover the target area.")
    _log_warn(f"Target lon/lat bounds: {_format_lonlat_bounds(target_bounds)}")
    _log_warn(f"Input  lon/lat bounds: {_format_lonlat_bounds(input_bounds)}")

    timeout_s = 5.0
    ans_raw = _timed_input_line(
        f"Continue anyway? (Y/N) [auto-continue in {int(timeout_s)}s]: ",
        timeout_s=timeout_s,
    )
    if ans_raw is None:
        _log_warn(f"No input received (timeout {int(timeout_s)}s). Continuing by default.")
        return

    ans = ans_raw.strip().lower()
    if ans in ("n", "no"):
        _log_info("User canceled. Exiting.")
        sys.exit(1)

    if ans in ("y", "yes", ""):
        _log_warn("Continuing despite bounds mismatch.")
        return

    _log_warn(f"Invalid input '{ans_raw}'. Continuing by default.")
    return


def _load_dem_data(
    project_home: Path,
    work_crs: str,
    elevation_field: str = "elevation",
    conf_raw: str | None = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load DEM data from shapefile in terrain_db.
    If no shapefile exists, try to generate one from GeoTIFF.
    Returns (points_xy, elevations) in work_crs or (None, None) if not found.
    Elevations are adjusted so minimum = 0.

    Args:
        project_home: Project directory containing DEM folder
        work_crs: Target CRS (e.g., "EPSG:32651")
        elevation_field: Name of elevation field in shapefile
    """
    dem_folder = project_home / "terrain_db"
    shp_files = list(dem_folder.glob("*.shp")) if dem_folder.exists() else []

    if not shp_files:
        _log_warn("No DEM shapefile found in terrain_db. Trying GeoTIFF fallback...")
        if conf_raw:
            created = ensure_dem_shp_from_tif(conf_raw, project_home)
            if created and created.exists():
                shp_files = list(dem_folder.glob("*.shp"))
        if not shp_files:
            _log_warn("No DEM shapefile available, proceeding without terrain data")
            return None, None

    dem_shp = shp_files[0]
    _log_info(f"Loading DEM from: {dem_shp}")

    try:
        dem_gdf = gpd.read_file(dem_shp)

        # Force geometries to 2D to avoid Z-coordinate issues during reprojection
        def force_2d(geom):
            if geom is None:
                return None
            return shapely_transform(lambda x, y, z=None: (x, y), geom)

        dem_gdf['geometry'] = dem_gdf['geometry'].apply(force_2d)

        # Validate DEM coverage against target bounds (lon/lat) if available
        target_bounds = _parse_target_lonlat_from_conf(conf_raw)
        if target_bounds:
            try:
                if dem_gdf.crs is None:
                    _log_warn("DEM shapefile has no CRS; assume EPSG:4326 for coverage check.")
                    dem_wgs84 = dem_gdf
                else:
                    dem_wgs84 = dem_gdf
                    if dem_gdf.crs.to_epsg() != 4326:
                        dem_wgs84 = dem_gdf.to_crs(epsg=4326)
                ib_minx, ib_miny, ib_maxx, ib_maxy = dem_wgs84.total_bounds
                input_bounds = (float(ib_minx), float(ib_maxx), float(ib_miny), float(ib_maxy))
                _confirm_bbox_coverage("DEM SHP", target_bounds, input_bounds)
            except Exception as e:
                _log_warn(f"Failed to validate DEM SHP bounds: {e}")

        # Reproject to working CRS if needed
        if dem_gdf.crs != work_crs:
            _log_info(f"Reprojecting DEM from {dem_gdf.crs} to {work_crs}")
            dem_gdf = dem_gdf.to_crs(work_crs)

        # Find elevation field
        elev_col = None
        for col in ['elevation', 'Elevation', 'ELEVATION', 'height', 'Height', 'z', 'Z']:
            if col in dem_gdf.columns:
                elev_col = col
                break

        if elev_col is None:
            _log_warn(f"Elevation field not found in DEM shapefile")
            return None, None

        _log_info(f"Using elevation column: {elev_col}")

        # Extract points and elevations
        points = []
        elevations = []

        for idx, row in dem_gdf.iterrows():
            geom = row['geometry']
            elev = row[elev_col]

            if geom is None or geom.is_empty:
                continue

            try:
                elev_val = float(elev)
                if math.isnan(elev_val):
                    continue
            except (ValueError, TypeError):
                continue

            # Use centroid for all geometry types (Point, Polygon, etc.)
            centroid = geom.centroid
            if centroid is not None and not centroid.is_empty:
                points.append([centroid.x, centroid.y])
                elevations.append(elev_val)

        if not points:
            _log_warn("No valid DEM points found")
            return None, None

        points_xy = np.array(points)
        elevations = np.array(elevations)

        _log_info(f"Loaded {len(elevations)} DEM points")
        _log_info(f"Original elevation range (sea level): {elevations.min():.2f} to {elevations.max():.2f} meters")

        # Adjust elevations so minimum = 0 (datum adjustment)
        min_elev = elevations.min()
        elevations = elevations - min_elev

        _log_info(f"Adjusted to datum (lowest point = 0): {elevations.min():.2f} to {elevations.max():.2f} meters")
        _log_info(f"Datum offset applied: {min_elev:.2f}m")

        return points_xy, elevations

    except Exception as e:
        _log_warn(f"Failed to load DEM data: {e}")
        return None, None


def _interpolate_dem_to_grid(dem_points: np.ndarray, dem_elevations: np.ndarray,
                             x_grid: np.ndarray, y_grid: np.ndarray,
                             rotate_deg: float, pivot_xy: Tuple[float, float],
                             x_min: float, y_min: float,
                             idw_power: float = 2.0, idw_neighbors: int = 12) -> np.ndarray:
    """
    Interpolate DEM elevations to wind field grid using IDW.
    DEM points are already shifted to relative coordinates (origin at 0,0).

    Args:
        dem_points: Nx2 array of DEM points in shifted coordinates (origin at 0,0)
        dem_elevations: N array of elevation values (already adjusted to datum)
        x_grid: 1D array of x coordinates in wind grid (after rotation and shift)
        y_grid: 1D array of y coordinates in wind grid (after rotation and shift)
        rotate_deg: rotation angle applied to wind grid
        pivot_xy: rotation pivot point in absolute UTM coordinates
        x_min: minimum x in absolute UTM coordinates (for shifting pivot)
        y_min: minimum y in absolute UTM coordinates (for shifting pivot)
        idw_power: IDW power parameter
        idw_neighbors: number of nearest neighbors

    Returns:
        2D array (ny, nx) of interpolated elevations
    """
    # Shift pivot to relative coordinates
    pivot_shifted = (pivot_xy[0] - x_min, pivot_xy[1] - y_min)

    # Rotate DEM points to match wind grid
    th = math.radians(rotate_deg)
    c = math.cos(th)
    s = math.sin(th)

    dem_x = dem_points[:, 0]
    dem_y = dem_points[:, 1]

    dem_x_rot = c * (dem_x - pivot_shifted[0]) - s * (dem_y - pivot_shifted[1]) + pivot_shifted[0]
    dem_y_rot = s * (dem_x - pivot_shifted[0]) + c * (dem_y - pivot_shifted[1]) + pivot_shifted[1]

    dem_points_rot = np.column_stack([dem_x_rot, dem_y_rot])

    # Crop DEM points to wind grid bounding box (with buffer)
    x_min_grid = x_grid.min()
    x_max_grid = x_grid.max()
    y_min_grid = y_grid.min()
    y_max_grid = y_grid.max()

    # Add 10% buffer to ensure coverage
    x_range = x_max_grid - x_min_grid
    y_range = y_max_grid - y_min_grid
    buffer = max(x_range, y_range) * 0.1

    mask = (
        (dem_x_rot >= x_min_grid - buffer) &
        (dem_x_rot <= x_max_grid + buffer) &
        (dem_y_rot >= y_min_grid - buffer) &
        (dem_y_rot <= y_max_grid + buffer)
    )

    dem_points_rot_cropped = dem_points_rot[mask]
    dem_elevations_cropped = dem_elevations[mask]

    _log_info(f"Cropped DEM from {len(dem_elevations)} to {len(dem_elevations_cropped)} points within grid bounds")

    if len(dem_elevations_cropped) == 0:
        _log_warn("No DEM points within grid bounds, using zeros")
        return np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)

    # Create grid points
    nx = len(x_grid)
    ny = len(y_grid)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # IDW interpolation using cropped DEM points
    tree = cKDTree(dem_points_rot_cropped)
    distances, indices = tree.query(grid_points, k=min(idw_neighbors, len(dem_elevations_cropped)))
    distances = np.maximum(distances, 1e-10)

    weights = 1.0 / (distances ** idw_power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum

    Z = np.sum(weights_normalized * dem_elevations_cropped[indices], axis=1)
    Z = Z.reshape(ny, nx)

    # Ensure minimum is at least 0
    if Z.min() < 0:
        _log_warn(f"Interpolation created negative values (min={Z.min():.2f}m), clipping to 0")
        Z = np.maximum(Z, 0.0)

    _log_info(f"DEM interpolated to grid: absolute elevation range {Z.min():.2f} to {Z.max():.2f} meters")

    # Convert to elevation difference (relative to minimum in this grid)
    z_min_grid = Z.min()
    Z_diff = Z - z_min_grid

    _log_info(f"DEM elevation difference range: 0.00 to {Z_diff.max():.2f} meters (relative to grid minimum)")
    _log_info(f"Grid minimum elevation (datum): {z_min_grid:.2f} meters")

    return Z_diff


def _forward_fill_whole_layer(primary: np.ndarray, *companions: np.ndarray) -> None:
    """
    Forward-fill entire vertical layers to eliminate NaNs.
    NaN-layer detection is based on primary array. The same layer copy
    operation is applied to all companions.
    """
    nz = primary.shape[0]
    arrays = (primary,) + companions

    first_valid = None
    for k in range(nz):
        if not np.isnan(primary[k]).all():
            first_valid = k
            break

    if first_valid is None:
        raise ValueError("All vertical layers are NaN, invalid data")

    for k in range(first_valid):
        for arr in arrays:
            arr[k] = arr[first_valid]

    for k in range(first_valid + 1, nz):
        if np.isnan(primary[k]).all():
            for arr in arrays:
                arr[k] = arr[k - 1]

def _fill_nan_3d_nearest(arr: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """
    Fill NaNs/Inf in a (nz, ny, nx) array using nearest neighbor in (x,y) space per level.
    """
    ny, nx = x2d.shape
    pts = np.column_stack([x2d.ravel(), y2d.ravel()]).astype(np.float64)

    for k in range(arr.shape[0]):
        flat = arr[k].astype(np.float64).ravel()
        good = np.isfinite(flat)
        if good.all():
            continue
        if not good.any():
            continue

        tree = cKDTree(pts[good])
        _, idx = tree.query(pts[~good], k=1)

        flat_bad = flat[~good]
        flat_good = flat[good]
        flat_bad[:] = flat_good[idx]
        flat[~good] = flat_bad

        arr[k] = flat.reshape(ny, nx).astype(np.float32)

    return arr

def _read_conf_and_nc_path(conf_path: Union[str, Path]) -> Tuple[Path | None, dict, Path]:
    conf_file = Path(conf_path).expanduser().resolve()
    if not conf_file.exists():
        _log_error(f"conf file not found: {conf_file}")
        raise FileNotFoundError(str(conf_file))

    txt = conf_file.read_text(encoding="utf-8", errors="ignore")
    conf: dict = {"__raw__": txt}

    m_case = re.search(r"casename\s*=\s*([^\s]+)", txt)
    if not m_case:
        _log_error("casename not found in conf")
        raise RuntimeError("casename missing in conf")
    casename = m_case.group(1)
    conf["casename"] = casename

    m_dt = re.search(r"datetime\s*=\s*([0-9]{14})", txt)
    if m_dt:
        dt_str = m_dt.group(1)
    else:
        dt_str = "20990101120000"
        _log_warn("datetime not provided in conf, use 20990101120000 and write it back")
        # 追加写回到现有 conf
        with open(conf_file, "a", encoding="utf-8") as f:
            if not txt.endswith("\n"):
                f.write("\n")
            f.write(f"datetime = {dt_str}\n")
        # 刷新原文
        txt = conf_file.read_text(encoding="utf-8", errors="ignore")
        conf["__raw__"] = txt
    conf["datetime"] = dt_str

    project_home = conf_file.parent
    wind_dir = project_home / "wind_bc"
    expect_nc = wind_dir / f"{casename}_{dt_str}.nc"

    if expect_nc.exists():
        return expect_nc, conf, conf_file

    if not wind_dir.exists():
        _log_error(f"wind_bc folder not found under {project_home}")
        raise FileNotFoundError(str(expect_nc))

    nc_list = sorted(wind_dir.glob("*.nc"))
    if len(nc_list) == 1:
        _log_warn(f"Expected {expect_nc.name} not found. Use the only nc file {nc_list[0].name} under wind_bc.")
        return nc_list[0], conf, conf_file

    if len(nc_list) == 0:
        _log_error(f"No nc file found under {wind_dir}. Expected {expect_nc.name}.")
    else:
        _log_error(f"Expected {expect_nc.name} not found and multiple nc files present under {wind_dir}.")
    raise FileNotFoundError(str(expect_nc))



def _crop_wind_by_utm_bounds(ds: xr.Dataset, conf_raw: str | None, utm_crs: str) -> xr.Dataset:
    """
    Crop wind data in UTM coordinate space (same as DEM).
    Converts lon/lat bounds from config to UTM, then crops wind grid points.
    Supports both 1D lon/lat and 2D lon/lat mesh (WRF XLONG/XLAT).
    Also slices staggered dims lon_stag and lat_stag when present.
    """
    if not conf_raw:
        _log_info("No manual crop range from conf, skip cropping")
        return ds

    ds = _ensure_lonlat_geo_coords(ds)

    m_utm_x = re.search(r"cut_utm_x\s*=\s*\[([^\]]+)\]", conf_raw)
    m_utm_y = re.search(r"cut_utm_y\s*=\s*\[([^\]]+)\]", conf_raw)

    if m_utm_x and m_utm_y and m_utm_x.group(1).strip() and m_utm_y.group(1).strip():
        x_lo, x_hi = [float(v) for v in m_utm_x.group(1).split(",")]
        y_lo, y_hi = [float(v) for v in m_utm_y.group(1).split(",")]
        _log_info(f"Using direct UTM bounds from config: X=[{x_lo}, {x_hi}], Y=[{y_lo}, {y_hi}]")
    else:
        m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", conf_raw)
        m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", conf_raw)
        if not (m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip()):
            _log_info("cut_lon_manual or cut_lat_manual not provided, skip cropping")
            return ds

        lon_min_c, lon_max_c = [float(v) for v in m_lon.group(1).split(",")]
        lat_min_c, lat_max_c = [float(v) for v in m_lat.group(1).split(",")]
        lon_lo, lon_hi = sorted((lon_min_c, lon_max_c))
        lat_lo, lat_hi = sorted((lat_min_c, lat_max_c))

        transformer_ll2utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        x_lo, y_lo = transformer_ll2utm.transform(lon_lo, lat_lo)
        x_hi, y_hi = transformer_ll2utm.transform(lon_hi, lat_hi)

        _log_info(f"Converted lon/lat bounds [{lon_lo:.6f}, {lon_hi:.6f}] x [{lat_lo:.6f}, {lat_hi:.6f}]")
        _log_info(f"  to UTM bounds X=[{x_lo:.2f}, {x_hi:.2f}], Y=[{y_lo:.2f}, {y_hi:.2f}]")

    lon_geo = ds["lon_geo"].values
    lat_geo = ds["lat_geo"].values

    transformer_ll2utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    if lon_geo.ndim == 1 and lat_geo.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lon_geo, lat_geo, indexing="xy")
    elif lon_geo.ndim == 2 and lat_geo.ndim == 2:
        if lon_geo.shape != lat_geo.shape:
            _log_warn("lon_geo and lat_geo have mismatched 2D shapes, skip cropping")
            return ds
        Lon2d, Lat2d = lon_geo, lat_geo
    else:
        _log_warn("lon/lat geo dims are not both 1D or both 2D, skip cropping")
        return ds

    x_utm, y_utm = transformer_ll2utm.transform(Lon2d.ravel(), Lat2d.ravel())
    x_utm = x_utm.reshape(Lon2d.shape)
    y_utm = y_utm.reshape(Lat2d.shape)

    mask = (x_utm >= x_lo) & (x_utm <= x_hi) & (y_utm >= y_lo) & (y_utm <= y_hi)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        _log_warn("No wind data points within UTM bounds, returning original dataset")
        return ds

    lat_idx_min, lat_idx_max = int(rows[0]), int(rows[-1])
    lon_idx_min, lon_idx_max = int(cols[0]), int(cols[-1])

    sel = {
        "lat": slice(lat_idx_min, lat_idx_max + 1),
        "lon": slice(lon_idx_min, lon_idx_max + 1),
    }

    if "lat_stag" in ds.dims:
        sel["lat_stag"] = slice(lat_idx_min, lat_idx_max + 2)
    if "lon_stag" in ds.dims:
        sel["lon_stag"] = slice(lon_idx_min, lon_idx_max + 2)

    _log_info(
        f"Cropped wind data by UTM bounds: lat [{lat_idx_min}, {lat_idx_max}] lon [{lon_idx_min}, {lon_idx_max}]"
    )

    return ds.isel(**sel)


def _get_fixed_utm_crs(conf_raw: str | None) -> str:
    """
    Determine UTM CRS from deck file if possible.
    Fallback to EPSG:32651 for backward compatibility.
    """
    if conf_raw:
        try:
            return get_utm_crs_from_conf_raw(conf_raw, default_epsg="EPSG:32651")
        except Exception as e:
            _log_warn(f"Failed to auto detect UTM CRS from conf, fallback to EPSG:32651. Reason: {e}")
            return "EPSG:32651"
    _log_warn("conf_raw is empty when detecting UTM CRS, use EPSG:32651 as fallback")
    return "EPSG:32651"

def _project_bbox_and_rotation(lon_min: float, lon_max: float, lat_min: float, lat_max: float, utm_crs) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
    """
    Project bbox corners to UTM, estimate rotation angle against X axis, return center as pivot.
    Returns:
      pts_xy: (4,2) projected corners in order (xmin,ymin) (xmax,ymin) (xmax,ymax) (xmin,ymax)
      center_xy: (2,) center
      rotate_deg: counter-clockwise positive, to align bottom edge to X axis
      pivot_xy: rotation center
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x00, y00 = transformer.transform(lon_min, lat_min)
    x10, y10 = transformer.transform(lon_max, lat_min)
    x11, y11 = transformer.transform(lon_max, lat_max)
    x01, y01 = transformer.transform(lon_min, lat_max)

    pts_xy = np.array([[x00, y00], [x10, y10], [x11, y11], [x01, y01]], dtype=float)
    cx = float(pts_xy[:, 0].mean())
    cy = float(pts_xy[:, 1].mean())

    dx = x10 - x00
    dy = y10 - y00
    angle_rad = math.atan2(dy, dx)
    rotate_deg = - math.degrees(angle_rad)

    return pts_xy, np.array([cx, cy], dtype=float), rotate_deg, (cx, cy)


def _rotate_xy(x: np.ndarray, y: np.ndarray, deg: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate around (cx, cy) by deg.
    """
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)
    xr = c * (x - cx) - s * (y - cy) + cx
    yr = s * (x - cx) + c * (y - cy) + cy
    return xr, yr


def _project_rotate_grid(lon: np.ndarray, lat: np.ndarray, utm_crs, rotate_deg: float, pivot_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project lon/lat grid to UTM, then rotate around pivot.
    Supports lon/lat as 1D vectors or 2D mesh (WRF XLONG/XLAT).
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat, indexing="xy")
    elif lon.ndim == 2 and lat.ndim == 2:
        if lon.shape != lat.shape:
            raise ValueError("lon and lat 2D shapes mismatch")
        Lon, Lat = lon, lat
    else:
        raise ValueError("lon/lat must be both 1D or both 2D")

    x, y = transformer.transform(Lon, Lat)
    xr, yr = _rotate_xy(x, y, rotate_deg, pivot_xy[0], pivot_xy[1])
    return xr, yr


def _progress_draw(completed: int, total: int, last_percent: int, last_len: int) -> Tuple[int, int]:
    percent = int(completed * 100 / total) if total > 0 else 100
    if percent != last_percent:
        tens = percent // 10
        units = percent % 10
        bar = "#" * tens + str(units)
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            print(f"\r{bar}", end="", flush=True)
        else:
            print(f"{completed}/{total} {percent:3d}% {bar}", flush=True)
        return percent, len(bar)
    return last_percent, last_len


def _interp_to_uniform_meter_grid(u: np.ndarray, v: np.ndarray,
                                  src_x: np.ndarray, src_y: np.ndarray,
                                  nx: int, ny: int,
                                  x_min: float, x_max: float, y_min: float, y_max: float,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Interpolate rotated UTM wind fields u v onto a uniform Cartesian grid.
    Linear in-plane interpolation with nearest fallback outside convex hull.
    Returns u v on target grid and dx dy.
    Geometry and barycentric weights are precomputed once for all levels to accelerate evaluation.
    """
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xg, Yg = np.meshgrid(xi, yi, indexing="xy")

    pts = np.column_stack([src_x.ravel(), src_y.ravel()])  # source points
    qpts = np.column_stack([Xg.ravel(), Yg.ravel()])       # target points

    if u.shape[1] * u.shape[2] != pts.shape[0]:
        raise ValueError(
            f"Source grid size mismatch: u has {u.shape[1]}x{u.shape[2]}={u.shape[1]*u.shape[2]} points, "
            f"but src_x/src_y provide {pts.shape[0]} points"
        )

    valid_mask = np.isfinite(pts).all(axis=1)
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        _log_warn(f"Found {n_bad} NaN/Inf points in src grid, drop them for triangulation")

    pts_valid = pts[valid_mask]
    if pts_valid.shape[0] < 3:
        raise ValueError("Not enough valid points for triangulation after dropping NaNs")

    # build geometry once (on valid points only)
    tri = Delaunay(pts_valid)
    tree = cKDTree(pts_valid)
    _, nn_idx = tree.query(qpts, k=1)  # nearest neighbor for fallback

    # for all target points, find containing simplex and precompute barycentric weights
    simp = tri.find_simplex(qpts)  # -1 means outside
    inside = simp >= 0
    idx_tri = tri.simplices[simp[inside]]                       # (N_in, 3)
    T = tri.transform[simp[inside]]                             # (N_in, 3, 2)
    r = np.einsum("...ij,...j->...i", T[:, :2, :], qpts[inside] - T[:, 2, :])  # (N_in, 2)
    w0 = r[:, 0]
    w1 = r[:, 1]
    w2 = 1.0 - w0 - w1

    i0 = idx_tri[:, 0]
    i1 = idx_tri[:, 1]
    i2 = idx_tri[:, 2]

    nz = u.shape[0]
    u_out = np.empty((nz, ny, nx), dtype=np.float32)
    v_out = np.empty((nz, ny, nx), dtype=np.float32)

    M = qpts.shape[0]

    import sys

    if verbose:
        _log_info(f"Total vertical levels to interpolate: {nz}")
        _log_info("Start horizontal interpolation onto uniform grid: precomputed barycentric weights with nearest fallback")
        sys.stdout.write("[INFO] Horizontal interpolation progress (# is 5%): |")
        sys.stdout.flush()

    next_percent_idx = 1  # 1..20

    for k in range(nz):
        uk_all = u[k].astype(np.float64).ravel()
        vk_all = v[k].astype(np.float64).ravel()
        uk = uk_all[valid_mask]
        vk = vk_all[valid_mask]


        out_u_flat = np.empty(M, dtype=np.float32)
        out_v_flat = np.empty(M, dtype=np.float32)

        # inside convex hull: barycentric blend of triangle vertices
        vals_u_in = (w0 * uk[i0] + w1 * uk[i1] + w2 * uk[i2]).astype(np.float32)
        vals_v_in = (w0 * vk[i0] + w1 * vk[i1] + w2 * vk[i2]).astype(np.float32)
        out_u_flat[inside] = vals_u_in
        out_v_flat[inside] = vals_v_in

        # outside convex hull: nearest neighbor fallback
        nn_sel = nn_idx[~inside]
        if nn_sel.size:
            out_u_flat[~inside] = uk[nn_sel].astype(np.float32)
            out_v_flat[~inside] = vk[nn_sel].astype(np.float32)

        u_out[k] = out_u_flat.reshape(ny, nx)
        v_out[k] = out_v_flat.reshape(ny, nx)

        if verbose:
            progress_ratio = (k + 1) / nz if nz > 0 else 1.0
            while next_percent_idx <= 20 and progress_ratio >= (next_percent_idx / 20.0):
                sys.stdout.write("#")
                if next_percent_idx % 4 == 0:
                    sys.stdout.write(f"|{next_percent_idx * 5}%|")
                sys.stdout.flush()
                next_percent_idx += 1

    if verbose:
        if next_percent_idx <= 20:
            while next_percent_idx <= 20:
                sys.stdout.write("#")
                if next_percent_idx % 4 == 0:
                    sys.stdout.write(f"|{next_percent_idx * 5}%|")
                next_percent_idx += 1
            sys.stdout.flush()

        sys.stdout.write("|Finished|\n")
        sys.stdout.flush()
        _log_info("Horizontal interpolation finished")

    dx = (x_max - x_min) / (nx - 1) if nx > 1 else 0.0
    dy = (y_max - y_min) / (ny - 1) if ny > 1 else 0.0
    return u_out, v_out, dx, dy

def buildBC_dev(
    conf_path: Union[str, Path],
    var_u: str = "u",
    var_v: str = "v",
    elevation_scale: float = 1.0,
    write_vtk: bool = False,
) -> None:
    try:
        # conf and path
        nc_path_resolved, conf, conf_file = _read_conf_and_nc_path(conf_path)
        if nc_path_resolved is None:
            raise FileNotFoundError("Cannot determine input nc path. Provide casename in deck file or pass nc_path")
        _log_info(f"Open NetCDF: {nc_path_resolved}")

        # parallel config for dask: auto threads equals CPU count
        max_workers = min(int(os.environ.get("BC_WORKERS", "160")), os.cpu_count() or 1)
        dask_config.set(scheduler="threads", num_workers=max_workers)

        _log_info(f"Set parallel workers: {max_workers}")

        # Get UTM CRS first
        utm_crs = _get_fixed_utm_crs(conf.get("__raw__"))

        # Load DEM data (will be reprojected to UTM)
        project_home = conf_file.parent
        dem_points_utm, dem_elevations = _load_dem_data(project_home, utm_crs, conf_raw=conf.get("__raw__"))

        # Get UTM CRS first
        utm_crs = _get_fixed_utm_crs(conf.get("__raw__"))

        ds = xr.open_dataset(nc_path_resolved, chunks="auto")
        temp_name, temp_is_celsius = _resolve_temperature_var_name(ds)
        if temp_name is None:
            _log_info("No temperature field found in input NC; CSV will keep columns X,Y,Z,u,v,w")
        elif temp_is_celsius:
            _log_info(f"Detected temperature field {temp_name} (Celsius); will convert to Kelvin for CSV")
        else:
            _log_info(f"Detected temperature field {temp_name} (Kelvin)")

        ds = _detect_and_rename_wrf_dims(ds)
        ds = _ensure_lonlat_geo_coords(ds)
        ds = _ensure_dim_index_coords(ds)
        if "lon" in ds.dims and "lat" in ds.dims:
            ds = ds.chunk({"lon": 64, "lat": 64})

        # Validate wind data coverage before cropping
        target_bounds = _parse_target_lonlat_from_conf(conf.get("__raw__"))
        if target_bounds:
            try:
                lon_geo = ds["lon_geo"].values
                lat_geo = ds["lat_geo"].values
                input_bounds = (
                    float(np.nanmin(lon_geo)),
                    float(np.nanmax(lon_geo)),
                    float(np.nanmin(lat_geo)),
                    float(np.nanmax(lat_geo)),
                )
                _confirm_bbox_coverage("Wind NC", target_bounds, input_bounds)
            except Exception as e:
                _log_warn(f"Failed to validate wind NC bounds: {e}")

        # UTM-based crop (same coordinate system as DEM)
        ds = _crop_wind_by_utm_bounds(ds, conf.get("__raw__"), utm_crs)

        # vertical coordinate detection (support WRF: bottom_top as dim without coord)
        ds, vert, lev = _detect_vertical_dim_and_levels(ds)



        has_dem_input = (dem_points_utm is not None) and (dem_elevations is not None)

        # upsample if too small (support WRF staggered dims: lon_stag/lat_stag)
        target_n_map = {"lon": 200, "lat": 200, vert: 10}
        if "lon_stag" in ds.dims:
            target_n_map["lon_stag"] = target_n_map["lon"] + 1
        if "lat_stag" in ds.dims:
            target_n_map["lat_stag"] = target_n_map["lat"] + 1

        dims_to_check = ["lon", "lat"]
        if "lon_stag" in ds.dims:
            dims_to_check.append("lon_stag")
        if "lat_stag" in ds.dims:
            dims_to_check.append("lat_stag")
        if vert not in dims_to_check:
            dims_to_check.append(vert)

        new_coords = {}
        for dim in dims_to_check:
            if dim not in ds.dims:
                continue
            target_n = target_n_map.get(dim, ds.sizes[dim])
            if ds.sizes[dim] < target_n:
                coords = ds[dim].values
                new_coords[dim] = np.linspace(coords[0], coords[-1], target_n, dtype=coords.dtype)
                _log_info(f"Dimension {dim} has {ds.sizes[dim]} points, upsample to {target_n}")

        if new_coords:
            if has_dem_input:
                _log_info(
                    "Skip Dense Compute interpolation because terrain-first BC sampling "
                    "must use native NC spacing before horizontal interpolation"
                )
            else:
                _log_info("Start interpolation on lon, lat and vertical (Dense Compute)...")
                if len(new_coords) == 1:
                    ds = ds.interp(new_coords, method="pchip")
                else:
                    horizontal_keys = ["lon", "lat", "lon_stag", "lat_stag"]
                    horizontal_coords = {k: v for k, v in new_coords.items() if k in horizontal_keys}
                    vertical_coords = {k: v for k, v in new_coords.items() if k not in horizontal_keys}

                    if horizontal_coords:
                        _log_info("Interpolating horizontal dimensions with linear method...")
                        ds = ds.interp(horizontal_coords, method="linear")

                    if vertical_coords:
                        _log_info("Interpolating vertical dimension with pchip method...")
                        ds = ds.interp(vertical_coords, method="pchip")

                ds = ds.persist()
                _log_info("Finished lon/lat/vertical interpolation")

                # IMPORTANT: ds.interp() may change the vertical coordinate values/length
                # (e.g., height 4 -> 10). Refresh (vert, lev) to stay consistent with
                # the interpolated dataset; otherwise the later vertical remap may fall
                # back to index-based z levels and collapse z_top.
                ds, vert, lev = _detect_vertical_dim_and_levels(ds)
                _log_info(f"Refreshed vertical levels after interpolation: {vert} ({int(np.asarray(lev).size)} levels)")

        # extract variables (support WRF names and WRF staggered grids)
        u_name = _resolve_wind_var_name(ds, var_u, ("u", "U"), "u")
        v_name = _resolve_wind_var_name(ds, var_v, ("v", "V"), "v")
        if (temp_name is not None) and (temp_name not in ds.variables):
            _log_warn(f"Temperature field {temp_name} is unavailable after preprocessing; skip temperature export")
            temp_name = None

        w_name = None
        if "w" in ds.variables:
            w_name = "w"
        elif "W" in ds.variables:
            w_name = "W"
            _log_warn("Wind variable w not found, use W for w")
        else:
            w_name = None

        da_u = _destagger_wrf(_isel_first_time_if_present(ds[u_name]))
        da_v = _destagger_wrf(_isel_first_time_if_present(ds[v_name]))

        if w_name is None:
            da_w = None
        else:
            da_w = _destagger_wrf(_isel_first_time_if_present(ds[w_name]))

        if temp_name is None:
            da_t = None
        else:
            da_t = _destagger_wrf(_isel_first_time_if_present(ds[temp_name]))

        if ("lat" in da_u.dims) and ("lon" in da_u.dims):
            if ("lat" in da_v.dims) and ("lon" in da_v.dims):
                if (da_v.sizes["lat"] != da_u.sizes["lat"]) or (da_v.sizes["lon"] != da_u.sizes["lon"]):
                    da_v = da_v.interp(lat=da_u["lat"], lon=da_u["lon"], method="linear")
            if (da_w is not None) and ("lat" in da_w.dims) and ("lon" in da_w.dims):
                if (da_w.sizes["lat"] != da_u.sizes["lat"]) or (da_w.sizes["lon"] != da_u.sizes["lon"]):
                    da_w = da_w.interp(lat=da_u["lat"], lon=da_u["lon"], method="linear")
            if (da_t is not None) and ("lat" in da_t.dims) and ("lon" in da_t.dims):
                if (da_t.sizes["lat"] != da_u.sizes["lat"]) or (da_t.sizes["lon"] != da_u.sizes["lon"]):
                    da_t = da_t.interp(lat=da_u["lat"], lon=da_u["lon"], method="linear")

        u = da_u.transpose(vert, "lat", "lon").astype(np.float32).values
        v = da_v.transpose(vert, "lat", "lon").astype(np.float32).values
        if da_w is None:
            w = np.zeros_like(u)
        else:
            w = da_w.transpose(vert, "lat", "lon").astype(np.float32).values

        temp_k = None
        if da_t is not None:
            extra_t_dims = [d for d in da_t.dims if d not in (vert, "lat", "lon")]
            if extra_t_dims:
                _log_warn(
                    f"Temperature field {temp_name} has extra dims {extra_t_dims}; "
                    "use the first index on those dims"
                )
                da_t = da_t.isel({d: 0 for d in extra_t_dims})

            if (vert not in da_t.dims) or ("lat" not in da_t.dims) or ("lon" not in da_t.dims):
                _log_warn(
                    f"Temperature field {temp_name} dims {tuple(da_t.dims)} are incompatible with "
                    f"required ({vert}, lat, lon); skip temperature export"
                )
            else:
                temp_k = da_t.transpose(vert, "lat", "lon").astype(np.float32).values
                if temp_is_celsius:
                    temp_k = (temp_k + np.float32(273.15)).astype(np.float32)

        # lev is resolved earlier by _detect_vertical_dim_and_levels(ds)
        lev = np.asarray(lev, dtype=np.float32)

        lon_geo_da = ds["lon_geo"]
        lat_geo_da = ds["lat_geo"]

        if (
            (lon_geo_da.ndim == 2) and (lat_geo_da.ndim == 2) and
            ("lat" in lon_geo_da.dims) and ("lon" in lon_geo_da.dims) and
            ("lat" in da_u.dims) and ("lon" in da_u.dims)
        ):
            lon_geo_on = lon_geo_da.interp(lat=da_u["lat"], lon=da_u["lon"], method="linear")
            lat_geo_on = lat_geo_da.interp(lat=da_u["lat"], lon=da_u["lon"], method="linear")
            lon = lon_geo_on.values
            lat = lat_geo_on.values
        else:
            lon = lon_geo_da.values
            lat = lat_geo_da.values



        # original lon/lat range (use geographic lon_geo/lat_geo)
        orig_lon_min = float(np.nanmin(lon))
        orig_lon_max = float(np.nanmax(lon))
        orig_lat_min = float(np.nanmin(lat))
        orig_lat_max = float(np.nanmax(lat))

        nz, ny_src, nx_src = u.shape
        _log_info(f"Data shape nz={nz} ny={ny_src} nx={nx_src}")

        # fill NaNs
        if np.isnan(u).any() or np.isnan(v).any():
            _log_warn("NaNs found. Apply vertical forward fill")
            _forward_fill_whole_layer(u, v)
        if (temp_k is not None) and np.isnan(temp_k).any():
            _log_warn("NaNs found in temperature field. Apply vertical forward fill")
            _forward_fill_whole_layer(temp_k)

        # projection and rotation, use UTM CRS for CFD domain
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        _log_info(f"Using UTM CRS for CFD domain: {utm_crs}")

        pts_xy, center_xy, rotate_deg, pivot_xy = _project_bbox_and_rotation(lon_min, lon_max, lat_min, lat_max, utm_crs)
        _log_info(f"Estimated convergence angle = {rotate_deg:.6f} deg, use projected bbox centroid as pivot")

        # Calculate target domain and rotation from config bounds when available
        # In rotated CFD coordinates, the domain size must be computed after applying rotate_deg.
        target_domain_size = None
        domain_origin_rot = None
        if conf.get("__raw__"):
            conf_raw = conf.get("__raw__")
            m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", conf_raw)
            m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", conf_raw)
            if m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip():
                lon_min_c, lon_max_c = [float(v) for v in m_lon.group(1).split(",")]
                lat_min_c, lat_max_c = [float(v) for v in m_lat.group(1).split(",")]
                lon_lo, lon_hi = sorted((lon_min_c, lon_max_c))
                lat_lo, lat_hi = sorted((lat_min_c, lat_max_c))

                transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                x00, y00 = transformer.transform(lon_lo, lat_lo)
                x10, y10 = transformer.transform(lon_hi, lat_lo)
                x11, y11 = transformer.transform(lon_hi, lat_hi)
                x01, y01 = transformer.transform(lon_lo, lat_hi)

                pts_xy_conf = np.array([[x00, y00], [x10, y10], [x11, y11], [x01, y01]], dtype=float)
                cx = float(pts_xy_conf[:, 0].mean())
                cy = float(pts_xy_conf[:, 1].mean())

                dx0 = x10 - x00
                dy0 = y10 - y00
                angle_rad = math.atan2(dy0, dx0)
                rotate_deg = - math.degrees(angle_rad)
                pivot_xy = (cx, cy)
                _log_info(f"Override rotation/pivot from config bounds: rotate_deg={rotate_deg:.6f}, pivot=({cx:.3f}, {cy:.3f})")

                xr_c, yr_c = _rotate_xy(pts_xy_conf[:, 0], pts_xy_conf[:, 1], rotate_deg, cx, cy)
                x_min_conf, x_max_conf = float(np.min(xr_c)), float(np.max(xr_c))
                y_min_conf, y_max_conf = float(np.min(yr_c)), float(np.max(yr_c))

                si_x_target = x_max_conf - x_min_conf
                si_y_target = y_max_conf - y_min_conf

                target_domain_size = (si_x_target, si_y_target)
                domain_origin_rot = (x_min_conf, y_min_conf)
                _log_info(
                    f"Target domain (rotated) from config: {si_x_target:.3f} x {si_y_target:.3f} m; "
                    f"origin=({x_min_conf:.3f}, {y_min_conf:.3f})"
                )


        # project each grid point and rotate
        x_rot, y_rot = _project_rotate_grid(lon, lat, utm_crs, rotate_deg, pivot_xy)

        # global min/max in meters after rotation, then shift to zero origin
        x_min_data, x_max_data = float(x_rot.min()), float(x_rot.max())
        y_min_data, y_max_data = float(y_rot.min()), float(y_rot.max())
        _log_info(f"Wind data range after rotation X [{x_min_data:.3f}, {x_max_data:.3f}] Y [{y_min_data:.3f}, {y_max_data:.3f}]")

        if domain_origin_rot is not None:
            x_origin, y_origin = float(domain_origin_rot[0]), float(domain_origin_rot[1])
            _log_info(f"Use config-rotated origin shift: X0={x_origin:.3f} Y0={y_origin:.3f}")
        else:
            x_origin, y_origin = x_min_data, y_min_data

        x_src = x_rot - x_origin
        y_src = y_rot - y_origin

        if (x_src.shape == u[0].shape) and (y_src.shape == u[0].shape):
            has_nan_temp = (temp_k is not None) and np.isnan(temp_k).any()
            if np.isnan(u).any() or np.isnan(v).any() or np.isnan(w).any() or has_nan_temp:
                _log_warn("NaNs found in wind/temperature fields, fill horizontally by nearest neighbor in projected space")
                u = _fill_nan_3d_nearest(u, x_src, y_src)
                v = _fill_nan_3d_nearest(v, x_src, y_src)
                w = _fill_nan_3d_nearest(w, x_src, y_src)
                if temp_k is not None:
                    temp_k = _fill_nan_3d_nearest(temp_k, x_src, y_src)
        else:
            _log_warn("Skip NaN horizontal fill because projected grid shape mismatch")

        # Use target domain size if available, otherwise use data range
        if target_domain_size is not None:
            x_range_target, y_range_target = target_domain_size
            _log_info(f"Using target domain size for interpolation: {x_range_target:.3f} x {y_range_target:.3f} m")
        else:
            x_range_target = x_max_data - x_min_data
            y_range_target = y_max_data - y_min_data
            _log_info(f"Using data range for interpolation: {x_range_target:.3f} x {y_range_target:.3f} m")

        # Choose target grid resolution in each horizontal direction:
        # make the spacing close to midmesh_basesize (default 50 m) and exactly divide the domain length
        conf_raw = conf.get("__raw__")

        base_height = 50.0
        if conf_raw:
            m_bh = re.search(r"base_height\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", conf_raw)
            if m_bh:
                try:
                    base_height = float(m_bh.group(1))
                except Exception:
                    base_height = 50.0
        if (not math.isfinite(base_height)) or (base_height < 0.0):
            base_height = 50.0

        z_limit_agl = None
        if conf_raw:
            m_zlim = re.search(r"z_limit\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", conf_raw)
            if m_zlim:
                try:
                    z_limit_agl = float(m_zlim.group(1))
                except Exception:
                    z_limit_agl = None
        if (z_limit_agl is not None) and ((not math.isfinite(z_limit_agl)) or (z_limit_agl <= 0.0)):
            z_limit_agl = None

        mesh_base = 50.0

        if conf_raw:
            m_mb = re.search(r"midmesh_basesize\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", conf_raw)
            if m_mb:
                try:
                    mesh_base = float(m_mb.group(1))
                except Exception:
                    mesh_base = 50.0
        if (not math.isfinite(mesh_base)) or (mesh_base <= 0.0):
            mesh_base = 50.0

        _log_info(f"midmesh_basesize for output grid = {mesh_base:.3f} m (default 50.0 m if not set)")

        def _choose_n_points_for_approx_spacing(length_m: float, target_spacing: float) -> int:
            if not math.isfinite(length_m) or length_m <= 0.0:
                return 1
            n_cells = max(1, int(round(length_m / target_spacing)))
            return n_cells + 1

        nx = _choose_n_points_for_approx_spacing(x_range_target, mesh_base)
        ny = _choose_n_points_for_approx_spacing(y_range_target, mesh_base)

        _log_info(
            "Target grid points (nx, ny) = "
            f"({nx}, {ny}) -> nominal spacing ~ "
            f"{x_range_target / max(nx - 1, 1):.3f} m x {y_range_target / max(ny - 1, 1):.3f} m"
        )

        if (not np.isfinite(x_src).all()) or (not np.isfinite(y_src).all()):
            _log_warn("src_x/src_y contain NaN/Inf after projection. Fallback to structured meter grid for interpolation.")
            x1 = np.linspace(0.0, x_range_target, nx_src, dtype=np.float32)
            y1 = np.linspace(0.0, y_range_target, ny_src, dtype=np.float32)
            x_src, y_src = np.meshgrid(x1, y1, indexing="xy")

        # Keep source-grid fields for terrain-first sampling on boundary points.
        u_src_native = u
        v_src_native = v
        w_src_native = w
        temp_src_native = temp_k

        # interpolate to uniform grid using target domain size and chosen resolution
        u_m, v_m, dx, dy = _interp_to_uniform_meter_grid(
            u, v, x_src, y_src,
            nx=nx, ny=ny,
            x_min=0.0, x_max=x_range_target, y_min=0.0, y_max=y_range_target
        )


        # horizontally interpolate w onto the same uniform grid (silent)
        w_m, _, _, _ = _interp_to_uniform_meter_grid(
            w, w, x_src, y_src,
            nx=nx, ny=ny,
            x_min=0.0, x_max=x_range_target, y_min=0.0, y_max=y_range_target,
            verbose=False
        )
        w = w_m

        if temp_k is not None:
            temp_m, _, _, _ = _interp_to_uniform_meter_grid(
                temp_k, temp_k, x_src, y_src,
                nx=nx, ny=ny,
                x_min=0.0, x_max=x_range_target, y_min=0.0, y_max=y_range_target,
                verbose=False
            )
            temp_k = temp_m

        dz = 0.0




        # Interpolate DEM to wind grid (if available)
        dem_grid = None
        dem_points_shifted = None
        dem_data_to_save = None
        if dem_points_utm is not None and dem_elevations is not None:
            _log_info("Interpolating DEM to wind field grid")
            x_grid = np.linspace(0.0, x_range_target, nx)
            y_grid = np.linspace(0.0, y_range_target, ny)
            # Shift DEM points to match wind grid origin
            # dem_points_shifted = dem_points_utm - np.array([x_min_data, y_min_data])
            # dem_grid = _interpolate_dem_to_grid(
            #     dem_points_shifted, dem_elevations,
            #     x_grid, y_grid,
            #     rotate_deg, pivot_xy,
            #     x_min_data, y_min_data
            # )
            # Rotate DEM points into the same rotated UTM frame as wind grid, then shift to local origin
            dem_x = dem_points_utm[:, 0]
            dem_y = dem_points_utm[:, 1]
            dem_x_rot, dem_y_rot = _rotate_xy(dem_x, dem_y, rotate_deg, pivot_xy[0], pivot_xy[1])
            dem_points_shifted = np.column_stack([dem_x_rot - x_origin, dem_y_rot - y_origin])


            # DEM points are already in rotated local coordinates, so set rotate_deg=0 here
            dem_grid = _interpolate_dem_to_grid(
                dem_points_shifted, dem_elevations,
                x_grid, y_grid,
                0.0, (0.0, 0.0),
                0.0, 0.0
            )

            # Prepare DEM data for saving (will save later after proj_temp is created)
            # IMPORTANT: Always save UNSCALED data to pkl for reusability
            dem_data_to_save = {
                'dem_grid': dem_grid,  # (ny, nx) elevation difference array (UNSCALED)
                'x_grid': x_grid,      # 1D x coordinates
                'y_grid': y_grid,      # 1D y coordinates
                'dx': dx,
                'dy': dy,
                'base_height': base_height,
                'rotate_deg': rotate_deg,
                'pivot_xy': pivot_xy,
                'x_min': x_origin,
                'y_min': y_origin,
                'x_max': x_origin + float(x_range_target),
                'y_max': y_origin + float(y_range_target)

            }
        else:
            _log_info("No DEM data, proceeding with flat terrain")

        # Remap vertical levels to a uniform z grid in meters (required by VTK/CFD).
        # Use lev (meters) when available; keep fallback for older inputs.
        lev_m = np.asarray(lev, dtype=np.float32)

        if (lev_m.ndim != 1) or (lev_m.size != nz) or (not np.isfinite(lev_m).all()):
            _log_warn("Invalid vertical levels detected, fallback to index-based z levels (1 m spacing)")
            lev_m = np.arange(nz, dtype=np.float32)

        # Treat lev as pressure only when metadata strongly suggests pressure (units/name).
        units = ""
        try:
            if isinstance(vert, str) and ((vert in ds.variables) or (vert in ds.coords)):
                units = str(ds[vert].attrs.get("units", "")).lower()
        except Exception:
            units = ""

        vert_lc = str(vert).lower()
        is_length_units = (units in ("m", "meter", "meters", "metre", "metres")) or (units.strip() == "m")
        is_pressure_units = (units in ("pa", "hpa", "mb")) or ("mbar" in units) or ("millibar" in units)
        is_pressure_name = vert_lc in ("plev", "pressure", "isobaric", "isobaric1", "isobaric_inpa", "isobaric_inhpa")

        if is_pressure_units or (is_pressure_name and (not is_length_units)):
            lev_min = float(np.nanmin(lev_m))
            lev_max = float(np.nanmax(lev_m))
            if (lev_min > 10.0) and (lev_max < 2000.0) and (lev_max - lev_min > 10.0):
                _log_warn("Vertical levels appear to be pressure; keep index-based meters for VTK/CFD compatibility")
                lev_m = np.arange(nz, dtype=np.float32)


        # Ensure increasing upward
        if lev_m.size >= 2 and lev_m[1] < lev_m[0]:
            lev_m = lev_m[::-1].copy()
            u_m = u_m[::-1].copy()
            v_m = v_m[::-1].copy()
            w   = w[::-1].copy()
            if temp_k is not None:
                temp_k = temp_k[::-1].copy()
            u_src_native = u_src_native[::-1].copy()
            v_src_native = v_src_native[::-1].copy()
            w_src_native = w_src_native[::-1].copy()
            if temp_src_native is not None:
                temp_src_native = temp_src_native[::-1].copy()

        # Build strictly increasing z source (AGL, do NOT prepend z=0 here)
        z_src_raw = lev_m.copy().astype(np.float32)
        for k in range(1, z_src_raw.size):
            if z_src_raw[k] <= z_src_raw[k - 1]:
                z_src_raw[k] = z_src_raw[k - 1] + np.float32(1e-3)

        z_min_raw = float(z_src_raw[0])
        z_top = float(z_src_raw[-1])

        if (not math.isfinite(z_top)) or (z_top <= 0.0):
            _log_warn("Invalid z_top detected, fallback to 1 m vertical spacing")
            z_src_raw = np.arange(nz, dtype=np.float32)
            z_min_raw = float(z_src_raw[0])
            z_top = float(z_src_raw[-1])

        n_cell_z = max(1, int(round(z_top / mesh_base)))
        nz_new = n_cell_z + 1
        z_new = np.linspace(0.0, z_top, nz_new, dtype=np.float32)

        _log_info(
            f"Resample vertical grid to ~{mesh_base:.3f} m with zmin-below nearest fill and ground=0: "
            f"z_min_raw={z_min_raw:.3f} m, z_top={z_top:.3f} m, nz_src={int(z_src_raw.size)} -> "
            f"nz_new={nz_new}, dz_new={z_new[1] - z_new[0]:.3f} m"
        )

        # Nearest fill outside range: below zmin -> first layer; above top -> last layer
        f_u_z = interp1d(z_src_raw, u_m, axis=0, bounds_error=False, fill_value=(u_m[0], u_m[-1]))
        f_v_z = interp1d(z_src_raw, v_m, axis=0, bounds_error=False, fill_value=(v_m[0], v_m[-1]))
        f_w_z = interp1d(z_src_raw, w,   axis=0, bounds_error=False, fill_value=(w[0],   w[-1]))
        if temp_k is not None:
            f_t_z = interp1d(
                z_src_raw, temp_k, axis=0, bounds_error=False, fill_value=(temp_k[0], temp_k[-1])
            )

        u_m = f_u_z(z_new).astype(np.float32)
        v_m = f_v_z(z_new).astype(np.float32)
        w   = f_w_z(z_new).astype(np.float32)
        if temp_k is not None:
            temp_k = f_t_z(z_new).astype(np.float32)

        # Keep a vertical-remapped copy on the source horizontal grid.
        f_u_z_src = interp1d(
            z_src_raw, u_src_native, axis=0, bounds_error=False,
            fill_value=(u_src_native[0], u_src_native[-1])
        )
        f_v_z_src = interp1d(
            z_src_raw, v_src_native, axis=0, bounds_error=False,
            fill_value=(v_src_native[0], v_src_native[-1])
        )
        f_w_z_src = interp1d(
            z_src_raw, w_src_native, axis=0, bounds_error=False,
            fill_value=(w_src_native[0], w_src_native[-1])
        )
        if temp_src_native is not None:
            f_t_z_src = interp1d(
                z_src_raw, temp_src_native, axis=0, bounds_error=False,
                fill_value=(temp_src_native[0], temp_src_native[-1])
            )

        u_src_native = f_u_z_src(z_new).astype(np.float32)
        v_src_native = f_v_z_src(z_new).astype(np.float32)
        w_src_native = f_w_z_src(z_new).astype(np.float32)
        if temp_src_native is not None:
            temp_src_native = f_t_z_src(z_new).astype(np.float32)

        nz = int(nz_new)
        dz = float(z_new[1] - z_new[0]) if nz_new > 1 else 0.0

        # Build a single global top plane for both CSV export and si_z_cfd:
        # clip at (highest terrain + z_limit), then keep consistent everywhere.
        z_top_from_wind = float((nz - 1) * dz) + float(base_height) if nz > 1 else float(base_height)
        ground_z_max = float(base_height)
        if dem_grid is not None:
            dem_max_scaled = float(np.nanmax(dem_grid)) * float(elevation_scale)
            if (not math.isfinite(dem_max_scaled)) or (dem_max_scaled < 0.0):
                dem_max_scaled = 0.0
            ground_z_max = float(base_height) + dem_max_scaled

        z_top_output = float(z_top_from_wind)
        if z_limit_agl is not None:
            z_cap_from_terrain = float(ground_z_max) + float(z_limit_agl)
            if z_top_output > z_cap_from_terrain:
                z_top_output = z_cap_from_terrain
                _log_info(
                    f"Apply global top clipping by highest terrain + z_limit: "
                    f"ground_max={ground_z_max:.3f} m, z_limit={z_limit_agl:.3f} m -> top={z_top_output:.3f} m"
                )
            else:
                _log_info(
                    f"Skip global top clipping: input top {z_top_output:.3f} m <= "
                    f"highest-terrain cap {z_cap_from_terrain:.3f} m"
                )



        # volume-mean velocity
        um_vol = np.array([
            float(np.nanmean(u_m)),
            float(np.nanmean(v_m)),
            float(np.nanmean(w))
        ], dtype=float)

        # write cropped lon/lat and SI ranges back to conf
        if conf_file.exists():
            conf_lines = conf_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            while len(conf_lines) < 49:
                conf_lines.append("")
            conf_lines[19] = "// WRF Data Range in lon/lat"
            # conf_lines[20] = f"cut_lon_wrf = [{orig_lon_min:.6f}, {orig_lon_max:.6f}]"
            # conf_lines[21] = f"cut_lat_wrf = [{orig_lat_min:.6f}, {orig_lat_max:.6f}]"
            conf_lines[14] = "// Projected SI Range after rotation"

            # Use target_domain_size if available (from config bounds), otherwise use data bounds
            if target_domain_size is not None:
                si_x_range, si_y_range = target_domain_size
                _log_info(f"Using target domain size for si_x/y_cfd: {si_x_range:.3f} x {si_y_range:.3f} m")
            else:
                si_x_range = x_max_data - x_min_data
                si_y_range = y_max_data - y_min_data
                _log_info(f"Using data bounds for si_x/y_cfd: {si_x_range:.3f} x {si_y_range:.3f} m")

            conf_lines[15] = f"si_x_cfd = [0.000000, {si_x_range:.6f}]"
            conf_lines[16] = f"si_y_cfd = [0.000000, {si_y_range:.6f}]"
            conf_lines[17] = f"si_z_cfd = [0.000000, {z_top_output:.6f}]"


            conf_lines[20:23] = [
                f"utm_crs = \"{utm_crs}\"",
                f"rotate_deg = {rotate_deg:.6f}",
                "origin_shift_applied = true",
            ]
            conf_file.write_text("\n".join(conf_lines), encoding="utf-8")
            _log_info("Wrote lon/lat and SI ranges to deck file")

        proj_temp = project_home / "proj_temp"
        proj_temp.mkdir(parents=True, exist_ok=True)

        if write_vtk:
            if vtk is None or nps is None:
                raise RuntimeError("VTK output requested (--write-vtk) but vtk Python package is not installed")

            # build VTK uniform grid, origin at 0 0 0
            grid = vtk.vtkUniformGrid()
            grid.SetOrigin(0.0, 0.0, 0.0)
            grid.SetSpacing(dx, dy, dz)
            grid.SetDimensions(nx, ny, nz)

            def _add(arr: np.ndarray, name: str):
                vtk_arr = nps.numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT)
                vtk_arr.SetName(name)
                grid.GetPointData().AddArray(vtk_arr)

            _add(u_m, "u")
            _add(v_m, "v")
            _add(w, "w")
            grid.GetPointData().SetActiveScalars("u")

            writer = (
                vtk.vtkXMLUniformGridWriter()
                if hasattr(vtk, "vtkXMLUniformGridWriter")
                else vtk.vtkXMLImageDataWriter()
            )
            vti_path = proj_temp / (Path(nc_path_resolved).with_suffix(".vti").name)
            writer.SetFileName(str(vti_path))
            writer.SetInputData(grid)
            writer.SetDataModeToBinary()
            writer.Write()
            mb = Path(vti_path).stat().st_size / 1e6
            _log_info(f"Wrote VTI: {vti_path} size ~ {mb:.3f} MB")
        else:
            _log_info("Skip VTI output (default). Use --write-vtk to enable.")

        # Save DEM data as pkl for 3_voxelization_with_dem.py
        if dem_data_to_save is not None:
            dem_pkl_path = proj_temp / "dem_grid.pkl"
            with open(dem_pkl_path, 'wb') as f:
                pickle.dump(dem_data_to_save, f)
            _log_info(f"Saved DEM grid data to: {dem_pkl_path}")

        # Prepare interpolation for terrain-adjusted wind field.
        # Use the same top as si_z_cfd upper bound to keep CFD domain and CSV consistent.
        z_max_output = float(z_top_output)

        z_original = np.arange(nz, dtype=np.float32) * dz  # AGL in wind grid (0, dz, 2*dz, ...)


        if dem_grid is not None:
            _log_info(f"Applying terrain-following coordinates with fixed top at {z_max_output:.2f}m")
            _log_info(f"DEM elevation difference range: {dem_grid.min():.2f}m to {dem_grid.max():.2f}m")
            _log_info(f"Bottom elevation range: {base_height + dem_grid.min():.2f}m to {base_height + dem_grid.max():.2f}m")

            dem_max_scaled = float(np.nanmax(dem_grid)) * float(elevation_scale)
            if math.isfinite(dem_max_scaled):
                ground_z_max = float(base_height) + dem_max_scaled
                if z_max_output <= ground_z_max + 1e-6:
                    _log_warn(
                        "Top boundary is not above terrain everywhere: "
                        f"z_top={z_max_output:.2f}m <= ground_max={ground_z_max:.2f}m. "
                        "Boundary CSV may contain very few points; check vertical levels in the input NC."
                    )
        else:
            _log_info(f"No DEM data, applying flat base offset: Z = Z_grid + {base_height}m")

        def _idw_interp_1d(col: np.ndarray, z_query: float, z_src: np.ndarray) -> float:
            if z_query <= z_src[0]:
                return float(col[0])
            if z_query >= z_src[-1]:
                return float(col[-1])

            k_upper = int(np.searchsorted(z_src, z_query))
            k_lower = k_upper - 1

            z_lower = float(z_src[k_lower])
            z_upper = float(z_src[k_upper])
            d_lower = abs(z_query - z_lower)
            d_upper = abs(z_query - z_upper)

            if d_lower < 1e-6:
                return float(col[k_lower])
            if d_upper < 1e-6:
                return float(col[k_upper])

            w_lower = 1.0 / d_lower
            w_upper = 1.0 / d_upper
            w_sum = w_lower + w_upper
            return float((w_lower * float(col[k_lower]) + w_upper * float(col[k_upper])) / w_sum)

        def _interpolate_wind_at_z(
            u_col: np.ndarray,
            v_col: np.ndarray,
            w_col: np.ndarray,
            z_target: float,
            z_src: np.ndarray,
            dem_offset: float,
            t_col: Optional[np.ndarray] = None,
        ) -> Tuple[float, float, float, Optional[float]]:
            """
            Interpolate wind (and optional temperature) at target z using IDW between bracketing levels.
            z_target: absolute target z in CSV coordinates
            z_src: original z levels in AGL coordinates
            dem_offset: local ground z (base_height + local terrain)
            """
            # Apply local terrain uplift to source z first, then interpolate at target absolute z.
            z_src_terrain = z_src + np.float32(dem_offset)
            z_target_clamped = float(
                np.clip(z_target, float(z_src_terrain[0]), float(z_src_terrain[-1]))
            )

            u_interp = _idw_interp_1d(u_col, z_target_clamped, z_src_terrain)
            v_interp = _idw_interp_1d(v_col, z_target_clamped, z_src_terrain)
            w_interp = _idw_interp_1d(w_col, z_target_clamped, z_src_terrain)

            if t_col is None:
                return u_interp, v_interp, w_interp, None

            t_interp = _idw_interp_1d(t_col, z_target_clamped, z_src_terrain)
            return u_interp, v_interp, w_interp, t_interp

        use_terrain_first_sampling = False
        src_interp_idx = None
        src_interp_w = None
        src_interp_n = 0
        src_dem_flat = None
        u_src_flat = None
        v_src_flat = None
        w_src_flat = None
        t_src_flat = None

        if (dem_grid is not None) and (dem_points_shifted is not None):
            try:
                src_pts = np.column_stack([x_src.ravel(), y_src.ravel()]).astype(np.float64)
                src_valid = np.isfinite(src_pts).all(axis=1)
                if src_valid.sum() < 1:
                    raise RuntimeError("no valid source points")

                src_pts_valid = src_pts[src_valid]
                src_valid_idx = np.flatnonzero(src_valid).astype(np.int32)

                # Terrain uplift is evaluated on source NC points (coarse), then interpolated to target points.
                dem_k = int(min(12, dem_points_shifted.shape[0]))
                if dem_k < 1:
                    raise RuntimeError("DEM points are empty after coordinate transform")

                dem_tree = cKDTree(dem_points_shifted)
                dem_d, dem_i = dem_tree.query(src_pts_valid, k=dem_k)
                if dem_k == 1:
                    dem_src_valid = dem_elevations[dem_i].astype(np.float64)
                else:
                    dem_d = np.maximum(dem_d, 1e-10)
                    dem_w = 1.0 / (dem_d ** 2.0)
                    dem_w /= dem_w.sum(axis=1, keepdims=True)
                    dem_src_valid = np.sum(dem_w * dem_elevations[dem_i], axis=1)
                dem_src_valid = np.asarray(dem_src_valid, dtype=np.float32)
                dem_src_valid = dem_src_valid - np.float32(np.nanmin(dem_src_valid))
                dem_src_valid = np.maximum(dem_src_valid, np.float32(0.0))

                src_dem_flat = np.zeros(src_pts.shape[0], dtype=np.float32)
                src_dem_flat[src_valid_idx] = dem_src_valid

                x_q = np.linspace(0.0, x_range_target, nx, dtype=np.float64)
                y_q = np.linspace(0.0, y_range_target, ny, dtype=np.float64)
                Xq, Yq = np.meshgrid(x_q, y_q, indexing="xy")
                qpts = np.column_stack([Xq.ravel(), Yq.ravel()])
                m_q = qpts.shape[0]
                # High-order smooth interpolation from coarse->fine:
                # local quadratic MLS (Moving Least Squares) with Wendland-C2 kernel.
                # This is continuous and avoids IDW-style artifacts/staircasing.
                src_h_mls_k = int(min(24, src_pts_valid.shape[0]))
                if src_h_mls_k < 1:
                    raise RuntimeError("no source points for horizontal MLS")

                basis_dim = 6 if src_h_mls_k >= 6 else (3 if src_h_mls_k >= 3 else 1)
                p0 = np.zeros((basis_dim,), dtype=np.float64)
                p0[0] = 1.0

                tree_src = cKDTree(src_pts_valid)
                src_dist, src_idx_local = tree_src.query(qpts, k=src_h_mls_k)

                if src_h_mls_k == 1:
                    src_dist = src_dist[:, None]
                    src_idx_local = src_idx_local[:, None]

                src_interp_idx = src_valid_idx[src_idx_local].astype(np.int32)
                src_interp_w = np.empty((m_q, src_h_mls_k), dtype=np.float32)
                src_interp_n = int(src_h_mls_k)

                for qi in range(m_q):
                    qx = float(qpts[qi, 0])
                    qy = float(qpts[qi, 1])
                    ids_local = src_idx_local[qi]
                    d_local = np.asarray(src_dist[qi], dtype=np.float64)

                    px = src_pts_valid[ids_local, 0].astype(np.float64)
                    py = src_pts_valid[ids_local, 1].astype(np.float64)

                    h = float(np.max(d_local))
                    if (not math.isfinite(h)) or (h <= 1e-12):
                        h = 1.0
                    h *= 1.000001

                    xn = (px - qx) / h
                    yn = (py - qy) / h
                    r = np.sqrt(xn * xn + yn * yn)

                    # Wendland C2 compact kernel: w = (1-r)^4 * (4r+1), r in [0,1]
                    t = np.clip(1.0 - r, 0.0, None)
                    wk = (t ** 4) * (4.0 * r + 1.0)

                    wk_sum = float(np.sum(wk))
                    if (not math.isfinite(wk_sum)) or (wk_sum <= 1e-14):
                        wk = np.ones_like(r, dtype=np.float64)
                        wk_sum = float(np.sum(wk))

                    if basis_dim == 1:
                        c = wk / wk_sum
                    else:
                        if basis_dim == 3:
                            b = np.column_stack([
                                np.ones_like(xn),
                                xn,
                                yn,
                            ])
                        else:
                            b = np.column_stack([
                                np.ones_like(xn),
                                xn,
                                yn,
                                xn * xn,
                                xn * yn,
                                yn * yn,
                            ])

                        bw = b * wk[:, None]
                        m_mat = b.T @ bw
                        reg = 1e-10 * (float(np.trace(m_mat)) / float(basis_dim)) + 1e-12
                        m_mat = m_mat.copy()
                        m_mat[np.diag_indices(basis_dim)] += reg

                        try:
                            t_coef = np.linalg.solve(m_mat, p0)
                            c = wk * (b @ t_coef)
                            c_sum = float(np.sum(c))
                            if (not math.isfinite(c_sum)) or (abs(c_sum) <= 1e-14):
                                c = wk / wk_sum
                            else:
                                c = c / c_sum

                            # Shape-preserving limiter to suppress oscillatory ringing from negative lobes.
                            neg_mass = float(np.sum(np.abs(c[c < 0.0])))
                            if neg_mass > 0.08:
                                c_pos = wk / wk_sum
                                alpha = min(1.0, (neg_mass - 0.08) / 0.30)
                                c = (1.0 - alpha) * c + alpha * c_pos
                                c = c / float(np.sum(c))
                        except Exception:
                            c = wk / wk_sum

                    src_interp_w[qi, :] = c.astype(np.float32)

                u_src_flat = u_src_native.reshape(nz, -1)
                v_src_flat = v_src_native.reshape(nz, -1)
                w_src_flat = w_src_native.reshape(nz, -1)
                if temp_src_native is not None:
                    t_src_flat = temp_src_native.reshape(nz, -1)

                use_terrain_first_sampling = True
                _log_info(
                    "Boundary sampling order switched to terrain-first: "
                    f"uplift on source columns first, then horizontal MLS interpolation "
                    f"(quadratic, Wendland-C2, k={src_interp_n}) to target points"
                )
            except Exception as e:
                _log_warn(
                    f"Terrain-first boundary sampling initialization failed ({e}); "
                    "fallback to target-column sampling"
                )
                use_terrain_first_sampling = False

        def _sample_wind_at_point(
            j: int,
            i: int,
            z_target: float,
            ground_z_local: float,
            write_temp_flag: bool,
        ) -> Tuple[float, float, float, Optional[float]]:
            if use_terrain_first_sampling:
                q_idx = int(j * nx + i)
                ids = src_interp_idx[q_idx]
                wxy = src_interp_w[q_idx]

                u_acc = 0.0
                v_acc = 0.0
                w_acc = 0.0
                t_acc = 0.0

                for n in range(src_interp_n):
                    w_n = float(wxy[n])
                    if w_n == 0.0:
                        continue
                    sid = int(ids[n])
                    src_ground = float(base_height) + float(src_dem_flat[sid]) * float(elevation_scale)
                    z_query = float(np.clip(z_target - src_ground, float(z_original[0]), float(z_original[-1])))

                    u_acc += w_n * _idw_interp_1d(u_src_flat[:, sid], z_query, z_original)
                    v_acc += w_n * _idw_interp_1d(v_src_flat[:, sid], z_query, z_original)
                    w_acc += w_n * _idw_interp_1d(w_src_flat[:, sid], z_query, z_original)
                    if write_temp_flag and (t_src_flat is not None):
                        t_acc += w_n * _idw_interp_1d(t_src_flat[:, sid], z_query, z_original)

                if write_temp_flag:
                    return u_acc, v_acc, w_acc, t_acc
                return u_acc, v_acc, w_acc, None

            u_col = u_m[:, j, i]
            v_col = v_m[:, j, i]
            w_col = w[:, j, i]
            t_col = temp_k[:, j, i] if write_temp_flag else None
            return _interpolate_wind_at_z(
                u_col, v_col, w_col, z_target, z_original, ground_z_local, t_col
            )

        bc_sum = np.zeros(3, dtype=float)
        bc_cnt = 0
        csv_path = proj_temp / "SurfData_Latest.csv"
        write_temp = temp_k is not None

        # Log elevation scale if not 1.0
        if elevation_scale != 1.0:
            _log_info(f"Applying elevation scale {elevation_scale}x to CSV output")
        if write_temp:
            _log_info("Append temperature column T (Kelvin) to boundary CSV")
        _log_info("Append patch column to boundary CSV: top=1, south=2, north=3, west=4, east=5, bottom=0")

        patch_bottom = 0
        patch_top = 1
        patch_south = 2
        patch_north = 3
        patch_west = 4
        patch_east = 5

        if dz > 0.0 and math.isfinite(dz):
            ground_eps = max(1e-3, min(0.1, 0.05 * dz))
        else:
            ground_eps = 0.05
        _log_info(f"Bottom patch points are sampled at ground + {ground_eps:.4f} m")

        def _local_ground_and_cap_z(j: int, i: int) -> Tuple[float, float]:
            dem_diff = float(dem_grid[j, i]) if dem_grid is not None else 0.0
            dem_diff_scaled = dem_diff * float(elevation_scale)
            ground_z = float(base_height) + dem_diff_scaled
            z_cap_here = float(z_max_output)
            return ground_z, z_cap_here

        def _write_csv_row(
            fp,
            x: float,
            y: float,
            z: float,
            uval: float,
            vval: float,
            wval: float,
            patch: int,
            tval: Optional[float] = None,
        ) -> None:
            if write_temp:
                fp.write(f"{x:.3f},{y:.3f},{z:.3f},{uval},{vval},{wval},{float(tval)},{int(patch)}\n")
            else:
                fp.write(f"{x:.3f},{y:.3f},{z:.3f},{uval},{vval},{wval},{int(patch)}\n")

        with open(csv_path, "w", encoding="utf-8") as f:
            if write_temp:
                f.write("X,Y,Z,u,v,w,T,patch\n")
            else:
                f.write("X,Y,Z,u,v,w,patch\n")

            # bottom face: sample slightly above local terrain
            for j in range(ny):
                y = j * dy
                for i in range(nx):
                    x = i * dx

                    ground_z, z_top_here = _local_ground_and_cap_z(j, i)
                    if (not math.isfinite(ground_z)) or (not math.isfinite(z_top_here)) or (z_top_here <= ground_z):
                        continue

                    z_span = z_top_here - ground_z
                    z_bottom_here = ground_z + min(ground_eps, 0.5 * z_span)
                    if z_bottom_here >= z_top_here - 1e-6:
                        continue

                    uval, vval, wval, tval = _sample_wind_at_point(
                        j, i, z_bottom_here, ground_z, write_temp
                    )

                    _write_csv_row(
                        f, x, y, z_bottom_here, uval, vval, wval, patch_bottom, tval
                    )
                    bc_sum += np.array([uval, vval, wval])
                    bc_cnt += 1

            # top face: keep as a flat plane (do not follow terrain)
            for j in range(ny):
                y = j * dy
                for i in range(nx):
                    x = i * dx
                    ground_z, z_top_here = _local_ground_and_cap_z(j, i)

                    if (not math.isfinite(ground_z)) or (not math.isfinite(z_top_here)) or (ground_z >= z_top_here):
                        continue

                    uval, vval, _w_interp, tval = _sample_wind_at_point(
                        j, i, z_top_here, ground_z, write_temp
                    )

                    _write_csv_row(
                        f, x, y, z_top_here, uval, vval, 0.0, patch_top, tval
                    )
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1

            # south and north faces
            for j in (0, ny - 1):
                y = j * dy
                patch_side = patch_south if j == 0 else patch_north
                for i in range(nx):
                    x = i * dx
                    ground_z, z_top_here = _local_ground_and_cap_z(j, i)

                    if (not math.isfinite(ground_z)) or (not math.isfinite(z_top_here)) or (z_top_here <= ground_z):
                        continue

                    # ground point (always)
                    uval, vval, _w_interp, tval = _sample_wind_at_point(
                        j, i, ground_z, ground_z, write_temp
                    )
                    _write_csv_row(
                        f, x, y, ground_z, uval, vval, 0.0, patch_side, tval
                    )
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1

                    if dz <= 0.0 or (not math.isfinite(dz)):
                        continue

                    # step upward from ground by dz, exclude the very top plane (handled by top face)
                    max_agl = z_top_here - ground_z
                    k_max_here = int(math.floor(max_agl / dz + 1e-6))
                    if k_max_here < 1:
                        continue

                    if k_max_here > (nz - 1):
                        k_max_here = nz - 1

                    for k_agl in range(1, k_max_here + 1):
                        z_out = ground_z + float(k_agl) * dz
                        if z_out >= z_top_here - 1e-6:
                            continue

                        uval, vval, _w_interp, tval = _sample_wind_at_point(
                            j, i, z_out, ground_z, write_temp
                        )
                        _write_csv_row(
                            f, x, y, z_out, uval, vval, 0.0, patch_side, tval
                        )
                        bc_sum += np.array([uval, vval, 0.0])
                        bc_cnt += 1


            # west and east faces
            for i in (0, nx - 1):
                x = i * dx
                patch_side = patch_west if i == 0 else patch_east
                for j in range(ny):
                    y = j * dy
                    ground_z, z_top_here = _local_ground_and_cap_z(j, i)

                    if (not math.isfinite(ground_z)) or (not math.isfinite(z_top_here)) or (z_top_here <= ground_z):
                        continue

                    # ground point (always)
                    uval, vval, _w_interp, tval = _sample_wind_at_point(
                        j, i, ground_z, ground_z, write_temp
                    )
                    _write_csv_row(
                        f, x, y, ground_z, uval, vval, 0.0, patch_side, tval
                    )
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1

                    if dz <= 0.0 or (not math.isfinite(dz)):
                        continue

                    # step upward from ground by dz, exclude the very top plane (handled by top face)
                    max_agl = z_top_here - ground_z
                    k_max_here = int(math.floor(max_agl / dz + 1e-6))
                    if k_max_here < 1:
                        continue

                    if k_max_here > (nz - 1):
                        k_max_here = nz - 1

                    for k_agl in range(1, k_max_here + 1):
                        z_out = ground_z + float(k_agl) * dz
                        if z_out >= z_top_here - 1e-6:
                            continue

                        uval, vval, _w_interp, tval = _sample_wind_at_point(
                            j, i, z_out, ground_z, write_temp
                        )
                        _write_csv_row(
                            f, x, y, z_out, uval, vval, 0.0, patch_side, tval
                        )
                        bc_sum += np.array([uval, vval, 0.0])
                        bc_cnt += 1


        _log_info(f"Wrote boundary CSV: {csv_path}")

        # copy timestamped CSV
        if conf_file.exists():
            txt = conf.get("__raw__", conf_file.read_text(encoding="utf-8", errors="ignore"))
            m_dt = re.search(r"datetime\s*=\s*([0-9]{14})", txt)
            if m_dt:
                dt_str = m_dt.group(1)
            else:
                dt_str = "20990101120000"
                _log_warn("datetime not provided in conf, use 20990101120000")
            dst_path = proj_temp / f"SurfData_{dt_str}.csv"
            shutil.copyfile(csv_path, dst_path)
            _log_info(f"Copied timestamped CSV: {dst_path}")

        # boundary mean and downstream face
        um_bc = bc_sum / bc_cnt if bc_cnt > 0 else np.zeros(3, dtype=float)
        mean_u, mean_v = um_vol[0], um_vol[1]

        if abs(mean_u) >= abs(mean_v):
            downstream_face = "+x" if mean_u >= 0 else "-x"
            parallel = mean_u if mean_u >= 0 else -mean_u
            perp = mean_v
            sign = 1.0 if perp >= 0 else -1.0
        else:
            downstream_face = "+y" if mean_v >= 0 else "-y"
            parallel = mean_v if mean_v >= 0 else -mean_v
            perp = mean_u
            sign = 1.0 if perp >= 0 else -1.0

        theta = math.degrees(math.atan2(abs(perp), abs(parallel))) if parallel != 0 else 90.0
        yaw_angle = sign * theta

        if conf_file.exists():
            conf_lines = conf_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            while len(conf_lines) < 49:
                conf_lines.append("")
            conf_lines[24] = "// Volume-mean uvw and downstream boundary with yaw angle"
            conf_lines[25] = f"um_vol = [{um_vol[0]:.6f}, {um_vol[1]:.6f}, {um_vol[2]:.6f}]"
            conf_lines[26] = f"um_bc = [{um_bc[0]:.6f}, {um_bc[1]:.6f}, {um_bc[2]:.6f}]"
            conf_lines[27] = f'downstream_bc = "{downstream_face}"'
            conf_lines[28] = f"downstream_bc_yaw = {yaw_angle:.2f}"
            conf_file.write_text("\n".join(conf_lines), encoding="utf-8")
            _log_info("Wrote mean velocity and downstream info to deck file")

        # close file
        ds.close()

        # summary
        _log_info(f"Lon range {lon_min:.6f} to {lon_max:.6f} Lat range {lat_min:.6f} to {lat_max:.6f}")
        _log_info(f"Target domain range X 0.000 to {si_x_range:.3f} m Y 0.000 to {si_y_range:.3f} m")
        _log_info(f"Grid spacing dx={dx:.3f} m dy={dy:.3f} m dz={dz:.3f} m")
        _log_info("buildBC_dev finished")

    except Exception as e:
        _log_error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build boundary conditions with DEM terrain support"
    )
    parser.add_argument(
        "conf_file",
        type=str,
        help="Path to configuration file (conf.luw)"
    )
    parser.add_argument(
        "--elevation-scale",
        type=float,
        default=1.0,
        help="Scale factor for elevation differences (for visualization/testing). Default: 1.0"
    )
    parser.add_argument(
        "--write-vtk",
        action="store_true",
        help="Enable VTK (*.vti) output. Disabled by default."
    )

    args = parser.parse_args()
    buildBC_dev(
        args.conf_file,
        elevation_scale=args.elevation_scale,
        write_vtk=args.write_vtk,
    )

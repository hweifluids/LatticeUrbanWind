# 1_buildBC_with_dem.py
# Modified version of 1_buildBC.py with DEM terrain support
from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, Optional
import math
import vtk
from vtk.util import numpy_support as nps
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
import shutil
import os
import sys
import re

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
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def _log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _load_dem_data(project_home: Path, work_crs: str, elevation_field: str = "elevation") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load DEM data from shapefile in DEM folder.
    Returns (points_xy, elevations) in work_crs or (None, None) if not found.
    Elevations are adjusted so minimum = 0.

    Args:
        project_home: Project directory containing DEM folder
        work_crs: Target CRS (e.g., "EPSG:32651")
        elevation_field: Name of elevation field in shapefile
    """
    dem_folder = project_home / "terrain_db"
    if not dem_folder.exists():
        _log_warn("Terrain terrain_db folder not found, proceeding without terrain data")
        return None, None

    shp_files = list(dem_folder.glob("*.shp"))
    if not shp_files:
        _log_warn("No shapefile found in terrain_db folder, proceeding without terrain data")
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


def _forward_fill_whole_layer(u: np.ndarray, v: np.ndarray) -> None:
    """
    Forward-fill entire vertical layers to eliminate NaNs.
    """
    nz = u.shape[0]
    first_valid = None
    for k in range(nz):
        if not np.isnan(u[k]).all():
            first_valid = k
            break

    if first_valid is None:
        raise ValueError("All vertical layers are NaN, invalid data")

    for k in range(first_valid):
        u[k] = u[first_valid]
        v[k] = v[first_valid]

    for k in range(first_valid + 1, nz):
        if np.isnan(u[k]).all():
            u[k] = u[k - 1]
            v[k] = v[k - 1]


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



def _crop_lonlat_by_conf(ds: xr.Dataset, conf_raw: str | None) -> xr.Dataset:
    """
    Optional lon/lat cropping according to conf cut_lon_manual and cut_lat_manual.
    Each bound is expanded by half a grid step to keep edge cells.
    """
    if not conf_raw:
        _log_info("No manual crop range from conf, skip lon/lat crop")
        return ds

    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", conf_raw)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", conf_raw)
    if not (m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip()):
        _log_info("cut_lon_manual or cut_lat_manual not provided, skip lon/lat crop")
        return ds

    lon_min_c, lon_max_c = [float(v) for v in m_lon.group(1).split(",")]
    lat_min_c, lat_max_c = [float(v) for v in m_lat.group(1).split(",")]
    lon_lo, lon_hi = sorted((lon_min_c, lon_max_c))
    lat_lo, lat_hi = sorted((lat_min_c, lat_max_c))

    dlon = float(abs(ds["lon"].diff("lon").mean().values)) if "lon" in ds.dims else 0.0
    dlat = float(abs(ds["lat"].diff("lat").mean().values)) if "lat" in ds.dims else 0.0

    _log_info(f"Crop lon to [{lon_lo}, {lon_hi}] and lat to [{lat_lo}, {lat_hi}], with half-grid padding")
    ds2 = ds.sel(
        lon=slice(lon_lo - 0.5 * dlon, lon_hi + 0.5 * dlon),
        lat=slice(lat_lo - 0.5 * dlat, lat_hi + 0.5 * dlat),
    )
    return ds2


def _get_fixed_utm_crs():
    """
    Return fixed UTM CRS to match main.ipynb configuration.
    Uses EPSG:32651 as specified in main.ipynb.
    """
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
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    Lon, Lat = np.meshgrid(lon, lat, indexing="xy")
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
                                  x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

    # build geometry once
    tri = Delaunay(pts)
    tree = cKDTree(pts)
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

    _log_info(f"Total vertical levels: {nz}")
    _log_info("Start horizontal interpolation onto uniform grid: precomputed barycentric weights with nearest fallback")
    last_percent = -1
    last_len = 0
    last_percent, last_len = _progress_draw(0, nz, last_percent, last_len)

    M = qpts.shape[0]

    for k in range(nz):
        uk = u[k].astype(np.float64).ravel()
        vk = v[k].astype(np.float64).ravel()

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

        last_percent, last_len = _progress_draw(k + 1, nz, last_percent, last_len)

    sys.stdout.write("\n")
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
        utm_crs = _get_fixed_utm_crs()

        # Load DEM data (will be reprojected to UTM)
        project_home = conf_file.parent
        dem_points_utm, dem_elevations = _load_dem_data(project_home, utm_crs)

        ds = xr.open_dataset(nc_path_resolved, chunks={"lon": 64, "lat": 64})

        # lon/lat crop
        ds = _crop_lonlat_by_conf(ds, conf.get("__raw__"))

        # vertical coordinate detection
        if "lev" in ds.coords:
            vert = "lev"
        elif "height_agl" in ds.coords:
            vert = "height_agl"
        elif "height" in ds.coords:
            vert = "height"
        elif "elevation" in ds.coords:
            vert = "elevation"
        else:
            raise ValueError("Vertical coord not found. Expect lev or height_agl")

        # upsample if too small
        target_n_map = {"lon": 200, "lat": 200, vert: 10}
        new_coords = {}
        for dim in ("lon", "lat", vert):
            target_n = target_n_map[dim]
            if ds.sizes[dim] < target_n:
                coords = ds[dim].values
                new_coords[dim] = np.linspace(coords[0], coords[-1], target_n, dtype=coords.dtype)
                _log_info(f"Dimension {dim} has {ds.sizes[dim]} points, upsample to {target_n}")

        if new_coords:
            _log_info("Start interpolation on lon, lat and vertical (Dense Compute)...")
            # Use different methods for single and multi-dim interpolation
            if len(new_coords) == 1:
                # Single-dimension interpolation can use PCHIP
                ds = ds.interp(new_coords, method="pchip")
            else:
                # Linear for horizontal, PCHIP for vertical
                horizontal_coords = {k: v for k, v in new_coords.items() if k in ["lon", "lat"]}
                vertical_coords = {k: v for k, v in new_coords.items() if k not in ["lon", "lat"]}

                if horizontal_coords:
                    _log_info("Interpolating horizontal dimensions (lon, lat) with linear method...")
                    ds = ds.interp(horizontal_coords, method="linear")

                if vertical_coords:
                    _log_info("Interpolating vertical dimension with pchip method...")
                    ds = ds.interp(vertical_coords, method="pchip")

            ds = ds.persist()
            _log_info("Finished lon/lat/vertical interpolation")

        # extract variables
        var_w = "w"
        u = ds[var_u].transpose(vert, "lat", "lon").astype(np.float32).values
        v = ds[var_v].transpose(vert, "lat", "lon").astype(np.float32).values
        w = ds[var_w].transpose(vert, "lat", "lon").astype(np.float32).values if var_w in ds.variables else np.zeros_like(u)
        lev = ds[vert].values
        lon = ds["lon"].values
        lat = ds["lat"].values

        # original lon/lat range
        orig_lon_min = float(ds["lon"].min())
        orig_lon_max = float(ds["lon"].max())
        orig_lat_min = float(ds["lat"].min())
        orig_lat_max = float(ds["lat"].max())

        nz, ny, nx = u.shape
        _log_info(f"Data shape nz={nz} ny={ny} nx={nx}")

        # fill NaNs
        if np.isnan(u).any() or np.isnan(v).any():
            _log_warn("NaNs found. Apply vertical forward fill")
            _forward_fill_whole_layer(u, v)

        # projection and rotation - use fixed UTM CRS to match main.ipynb
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        _log_info(f"Using fixed UTM CRS to match main.ipynb: {utm_crs}")

        pts_xy, center_xy, rotate_deg, pivot_xy = _project_bbox_and_rotation(lon_min, lon_max, lat_min, lat_max, utm_crs)
        _log_info(f"Estimated convergence angle = {rotate_deg:.6f} deg, use projected bbox centroid as pivot")

        # project each grid point and rotate
        x_rot, y_rot = _project_rotate_grid(lon, lat, utm_crs, rotate_deg, pivot_xy)

        # global min/max in meters after rotation, then shift to zero origin
        x_min, x_max = float(x_rot.min()), float(x_rot.max())
        y_min, y_max = float(y_rot.min()), float(y_rot.max())
        _log_info(f"Meter range after rotation X [{x_min:.3f}, {x_max:.3f}] Y [{y_min:.3f}, {y_max:.3f}]")

        x_src = x_rot - x_min
        y_src = y_rot - y_min

        # interpolate to uniform grid
        u_m, v_m, dx, dy = _interp_to_uniform_meter_grid(
            u, v, x_src, y_src, nx=nx, ny=ny, x_min=0.0, x_max=x_max - x_min, y_min=0.0, y_max=y_max - y_min
        )
        dz = float(lev[-1] - lev[0]) / (nz - 1) if nz > 1 else 0.0

        # Interpolate DEM to wind grid (if available)
        dem_grid = None
        dem_data_to_save = None
        if dem_points_utm is not None and dem_elevations is not None:
            _log_info("Interpolating DEM to wind field grid")
            x_grid = np.linspace(0.0, x_max - x_min, nx)
            y_grid = np.linspace(0.0, y_max - y_min, ny)
            # Shift DEM points to match wind grid origin
            dem_points_shifted = dem_points_utm - np.array([x_min, y_min])
            dem_grid = _interpolate_dem_to_grid(
                dem_points_shifted, dem_elevations,
                x_grid, y_grid,
                rotate_deg, pivot_xy,
                x_min, y_min
            )

            # Prepare DEM data for saving (will save later after proj_temp is created)
            # IMPORTANT: Always save UNSCALED data to pkl for reusability
            dem_data_to_save = {
                'dem_grid': dem_grid,  # (ny, nx) elevation difference array (UNSCALED)
                'x_grid': x_grid,      # 1D x coordinates
                'y_grid': y_grid,      # 1D y coordinates
                'dx': dx,
                'dy': dy,
                'base_height': 50.0,   # Fixed base height
                'rotate_deg': rotate_deg,
                'pivot_xy': pivot_xy,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            }
        else:
            _log_info("No DEM data, proceeding with flat terrain")

        # extend from z=0 to zmin by duplicating the first valid layer at zmin
        # build new arrays whose origin starts at z=0
        if dz > 0.0 and float(lev[0]) > 0.0:
            n_pad = int(round(float(lev[0]) / dz))
            if n_pad > 0:
                _log_info(f"Pad {n_pad} layers below zmin={float(lev[0]):.3f} so that vertical origin starts at 0")
                nz_new = nz + n_pad
                u_ext = np.empty((nz_new, ny, nx), dtype=np.float32)
                v_ext = np.empty((nz_new, ny, nx), dtype=np.float32)
                w_ext = np.empty((nz_new, ny, nx), dtype=np.float32)

                # fill 0 < z < zmin using values at zmin
                for k in range(n_pad):
                    u_ext[k] = u_m[0]
                    v_ext[k] = v_m[0]
                    w_ext[k] = w[0]

                # copy original layers upward
                u_ext[n_pad:n_pad + nz] = u_m
                v_ext[n_pad:n_pad + nz] = v_m
                w_ext[n_pad:n_pad + nz] = w

                u_m = u_ext
                v_m = v_ext
                w = w_ext
                nz = nz_new
            else:
                _log_info("No padding needed. zmin already near zero within dz")
        else:
            _log_info("No vertical padding applied. Either dz=0 or zmin<=0")

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
            conf_lines[15] = f"si_x_cfd = [0.000000, {(x_max - x_min):.6f}]"
            conf_lines[16] = f"si_y_cfd = [0.000000, {(y_max - y_min):.6f}]"
            # vertical range starts at 0
            z_max_new = float(lev[-1]) if float(lev[0]) <= 0 else float(lev[-1]) + float(lev[0])
            conf_lines[17] = f"si_z_cfd = [0.000000, {z_max_new:.6f}]"
            conf_lines[20:23] = [
                f"utm_crs = \"{utm_crs}\"",
                f"rotate_deg = {rotate_deg:.6f}",
                "origin_shift_applied = true",
            ]
            conf_file.write_text("\n".join(conf_lines), encoding="utf-8")
            _log_info("Wrote lon/lat and SI ranges to deck file")

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

        writer = vtk.vtkXMLUniformGridWriter() if hasattr(vtk, "vtkXMLUniformGridWriter") else vtk.vtkXMLImageDataWriter()
        project_home = conf_file.parent
        proj_temp = project_home / "proj_temp"
        proj_temp.mkdir(parents=True, exist_ok=True)
        vti_path = proj_temp / (Path(nc_path_resolved).with_suffix(".vti").name)
        writer.SetFileName(str(vti_path))
        writer.SetInputData(grid)
        writer.SetDataModeToBinary()
        writer.Write()
        mb = Path(vti_path).stat().st_size / 1e6
        _log_info(f"Wrote VTI: {vti_path} size ~ {mb:.3f} MB")

        # Save DEM data as pkl for 3_voxelization_with_dem.py
        if dem_data_to_save is not None:
            dem_pkl_path = proj_temp / "dem_grid.pkl"
            with open(dem_pkl_path, 'wb') as f:
                pickle.dump(dem_data_to_save, f)
            _log_info(f"Saved DEM grid data to: {dem_pkl_path}")

        # Prepare interpolation for terrain-adjusted wind field
        # Fixed top height: z_max_output = original_z_max + base_height
        base_height = 50.0  # Fixed base height
        z_max_output = float((nz - 1) * dz) + base_height  # Fixed top boundary
        z_original = np.arange(nz, dtype=np.float32) * dz  # Original z levels (0, dz, 2*dz, ...)

        if dem_grid is not None:
            _log_info(f"Applying terrain-following coordinates with fixed top at {z_max_output:.2f}m")
            _log_info(f"DEM elevation difference range: {dem_grid.min():.2f}m to {dem_grid.max():.2f}m")
            _log_info(f"Bottom elevation range: {base_height + dem_grid.min():.2f}m to {base_height + dem_grid.max():.2f}m")
        else:
            _log_info(f"No DEM data, applying flat base offset: Z = Z_grid + {base_height}m")

        def _interpolate_wind_at_z(u_col, v_col, w_col, z_target, z_src, dem_offset):
            """
            Interpolate wind components at target z using IDW.
            z_target: target z in output coordinates (e.g., 1150m)
            z_src: original z levels (0, dz, 2*dz, ...)
            dem_offset: base_height + dem_elevation
            """
            # Map target z back to original coordinate
            z_in_original = z_target - dem_offset

            # Clamp to valid range
            z_in_original = np.clip(z_in_original, z_src[0], z_src[-1])

            # Find two nearest levels for IDW
            if z_in_original <= z_src[0]:
                return u_col[0], v_col[0], w_col[0]
            elif z_in_original >= z_src[-1]:
                return u_col[-1], v_col[-1], w_col[-1]
            else:
                # Find bracketing indices
                k_upper = np.searchsorted(z_src, z_in_original)
                k_lower = k_upper - 1

                z_lower = z_src[k_lower]
                z_upper = z_src[k_upper]

                # IDW with power=1 (linear-like but using inverse distance)
                d_lower = abs(z_in_original - z_lower)
                d_upper = abs(z_in_original - z_upper)

                if d_lower < 1e-6:
                    return u_col[k_lower], v_col[k_lower], w_col[k_lower]
                elif d_upper < 1e-6:
                    return u_col[k_upper], v_col[k_upper], w_col[k_upper]
                else:
                    w_lower = 1.0 / d_lower
                    w_upper = 1.0 / d_upper
                    w_sum = w_lower + w_upper

                    u_interp = (w_lower * u_col[k_lower] + w_upper * u_col[k_upper]) / w_sum
                    v_interp = (w_lower * v_col[k_lower] + w_upper * v_col[k_upper]) / w_sum
                    w_interp = (w_lower * w_col[k_lower] + w_upper * w_col[k_upper]) / w_sum

                    return u_interp, v_interp, w_interp

        bc_sum = np.zeros(3, dtype=float)
        bc_cnt = 0
        csv_path = vti_path.parent / "SurfData_Latest.csv"

        # Log elevation scale if not 1.0
        if elevation_scale != 1.0:
            _log_info(f"Applying elevation scale {elevation_scale}x to CSV output")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("X,Y,Z,u,v,w\n")
            # bottom face (k=0)
            for j in range(ny):
                y = j * dy
                for i in range(nx):
                    x = i * dx
                    dem_diff = dem_grid[j, i] if dem_grid is not None else 0.0
                    # Apply elevation scale to CSV output
                    dem_diff_scaled = dem_diff * elevation_scale
                    z_output = base_height + dem_diff_scaled

                    # Use bottom layer data
                    uval = u_m[0, j, i]
                    vval = v_m[0, j, i]
                    wval = 0.0

                    f.write(f"{x:.3f},{y:.3f},{z_output:.3f},{uval},{vval},{wval}\n")
                    bc_sum += np.array([uval, vval, wval])
                    bc_cnt += 1

            # top face: interpolate at z_max_output for each (x,y)
            for j in range(ny):
                y = j * dy
                for i in range(nx):
                    x = i * dx
                    dem_diff = dem_grid[j, i] if dem_grid is not None else 0.0
                    # Apply elevation scale to CSV output
                    dem_diff_scaled = dem_diff * elevation_scale

                    # Map z_max_output back to original coordinate
                    z_in_original = z_max_output - base_height - dem_diff_scaled

                    # Interpolate wind at this z
                    u_col = u_m[:, j, i]
                    v_col = v_m[:, j, i]
                    w_col = w[:, j, i]

                    # Find two nearest levels for interpolation
                    if z_in_original <= z_original[0]:
                        uval, vval, wval = u_col[0], v_col[0], w_col[0]
                    elif z_in_original >= z_original[-1]:
                        uval, vval, wval = u_col[-1], v_col[-1], w_col[-1]
                    else:
                        k_upper = np.searchsorted(z_original, z_in_original)
                        k_lower = k_upper - 1

                        z_lower = z_original[k_lower]
                        z_upper = z_original[k_upper]

                        # IDW interpolation
                        d_lower = abs(z_in_original - z_lower)
                        d_upper = abs(z_in_original - z_upper)

                        if d_lower < 1e-6:
                            uval, vval, wval = u_col[k_lower], v_col[k_lower], w_col[k_lower]
                        elif d_upper < 1e-6:
                            uval, vval, wval = u_col[k_upper], v_col[k_upper], w_col[k_upper]
                        else:
                            w_lower = 1.0 / d_lower
                            w_upper = 1.0 / d_upper
                            w_sum = w_lower + w_upper

                            uval = (w_lower * u_col[k_lower] + w_upper * u_col[k_upper]) / w_sum
                            vval = (w_lower * v_col[k_lower] + w_upper * v_col[k_upper]) / w_sum
                            wval = (w_lower * w_col[k_lower] + w_upper * w_col[k_upper]) / w_sum

                    f.write(f"{x:.3f},{y:.3f},{z_max_output:.3f},{uval},{vval},0.0\n")
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1
            # south and north faces
            for j in (0, ny - 1):
                y = j * dy
                for k in range(1, nz - 1):
                    z_grid = float(k) * dz
                    for i in range(nx):
                        x = i * dx
                        dem_diff = dem_grid[j, i] if dem_grid is not None else 0.0
                        # Apply elevation scale to CSV output
                        dem_diff_scaled = dem_diff * elevation_scale

                        # Output z with DEM offset
                        z_output = z_grid + base_height + dem_diff_scaled

                        # Skip if exceeds max (remove超出部分)
                        if z_output > z_max_output:
                            continue

                        # Use original layer data (no interpolation for middle layers)
                        uval = u_m[k, j, i]
                        vval = v_m[k, j, i]
                        wval = 0.0

                        f.write(f"{x:.3f},{y:.3f},{z_output:.3f},{uval},{vval},{wval}\n")
                        bc_sum += np.array([uval, vval, wval])
                        bc_cnt += 1
            # west and east faces
            for i in (0, nx - 1):
                x = i * dx
                for k in range(1, nz - 1):
                    z_grid = float(k) * dz
                    for j in range(ny):
                        y = j * dy
                        dem_diff = dem_grid[j, i] if dem_grid is not None else 0.0
                        # Apply elevation scale to CSV output
                        dem_diff_scaled = dem_diff * elevation_scale

                        # Output z with DEM offset
                        z_output = z_grid + base_height + dem_diff_scaled

                        # Skip if exceeds max (remove超出部分)
                        if z_output > z_max_output:
                            continue

                        # Use original layer data (no interpolation for middle layers)
                        uval = u_m[k, j, i]
                        vval = v_m[k, j, i]
                        wval = 0.0

                        f.write(f"{x:.3f},{y:.3f},{z_output:.3f},{uval},{vval},{wval}\n")
                        bc_sum += np.array([uval, vval, wval])
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
            dst_path = vti_path.parent / f"SurfData_{dt_str}.csv"
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
        _log_info(f"Rotated SI range X 0.000 to {(x_max - x_min):.3f} m Y 0.000 to {(y_max - y_min):.3f} m")
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

    args = parser.parse_args()
    buildBC_dev(args.conf_file, elevation_scale=args.elevation_scale)


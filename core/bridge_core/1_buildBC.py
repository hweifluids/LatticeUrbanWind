# 1_buildBC.py
from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple
import math
import vtk
from vtk.util import numpy_support as nps
import geopandas as gpd
from shapely.geometry import Point, Polygon  # kept for compatibility if used elsewhere
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
        utm_crs = _get_fixed_utm_crs()
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

        # write boundary CSV and boundary mean
        bc_sum = np.zeros(3, dtype=float)
        bc_cnt = 0
        csv_path = vti_path.parent / "SurfData_Latest.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("X,Y,Z,u,v,w\n")
            # bottom and top faces
            for k in (0, nz - 1):
                z = float(k) * dz
                for j in range(ny):
                    y = j * dy
                    for i in range(nx):
                        x = i * dx
                        uval = u_m[k, j, i]
                        vval = v_m[k, j, i]
                        f.write(f"{x},{y},{z},{uval},{vval},0.0\n")
                        bc_sum += np.array([uval, vval, 0.0])
                        bc_cnt += 1
            # south and north faces
            for j in (0, ny - 1):
                y = j * dy
                for k in range(1, nz - 1):
                    z = float(k) * dz
                    for i in range(nx):
                        x = i * dx
                        uval = u_m[k, j, i]
                        vval = v_m[k, j, i]
                        f.write(f"{x},{y},{z},{uval},{vval},0.0\n")
                        bc_sum += np.array([uval, vval, 0.0])
                        bc_cnt += 1
            # west and east faces
            for i in (0, nx - 1):
                x = i * dx
                for k in range(1, nz - 1):
                    z = float(k) * dz
                    for j in range(ny):
                        y = j * dy
                        uval = u_m[k, j, i]
                        vval = v_m[k, j, i]
                        f.write(f"{x},{y},{z},{uval},{vval},0.0\n")
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
    if len(sys.argv) != 2:
        _log_error("Usage: python 1_buildBC.py <path-to-conf-file>")
        sys.exit(2)
    buildBC_dev(sys.argv[1])


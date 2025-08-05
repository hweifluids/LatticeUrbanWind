"""
buildBC.py
"""
from pathlib import Path
from typing import Union
from matplotlib import lines
import numpy as np
import xarray as xr
import math
import vtk
from vtk.util import numpy_support as nps
import geopandas as gpd
from shapely.geometry import Point
import shutil


## Filling the basement
def _forward_fill_whole_layer(u: np.ndarray, v: np.ndarray) -> None:
    """
    Building single-channel coupling BC from WRF to CFD
    """
    nz = u.shape[0]
    first_valid = None
    for k in range(nz):
        if not np.isnan(u[k]).all():
            first_valid = k
            break

    if first_valid is None:
        raise ValueError("All layers are filled with NaN. Data is invalid.")

    # Fill all NaN layers below the top-most valid layer
    for k in range(first_valid):
        u[k] = u[first_valid]
        v[k] = v[first_valid]

    # Regular forward-fill: start checking layer by layer from first_valid+1
    for k in range(first_valid + 1, nz):
        if np.isnan(u[k]).all():
            u[k] = u[k - 1]
            v[k] = v[k - 1]

## Building BC data
def buildBC(
    nc_path: Union[str, Path] = None,
    var_u: str = "u",
    var_v: str = "v",
) -> None:
    # -------- 1. read data --------
    conf_file = Path(__file__).parent.parent / "conf.txt"
    if conf_file.exists():
        import re
        txt = conf_file.read_text()
        casename = re.search(r"casename\s*=\s*([^\s]+)", txt).group(1)
        print(f"Loaded conf.txt, case={casename}")
        nc_path = Path("wrfInput")/casename/f"{casename}.nc"
    #    ds = xr.open_dataset(nc_path)
    else:
        print("No conf.txt found: use provided nc_path and no cropping")
    print(f"Opening: {nc_path}...")
    ds = xr.open_dataset(nc_path)

    # --- if conf.txt gives cut_lon / cut_lat, then perform cut ------------------
    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt)
    if m_lon and m_lat and m_lon.group(1).strip() and m_lat.group(1).strip():

        lon_min_c, lon_max_c = [float(v) for v in m_lon.group(1).split(",")]
        lat_min_c, lat_max_c = [float(v) for v in m_lat.group(1).split(",")]
        print(f"Crop to conf range: lon {lon_min_c}-{lon_max_c}, "
              f"lat {lat_min_c}-{lat_max_c}")
        ds = ds.sel(          
             lon=slice(min(lon_min_c, lon_max_c), max(lon_min_c, lon_max_c)),
             lat=slice(min(lat_min_c, lat_max_c), max(lat_min_c, lat_max_c)),
        )
        # lwg: 此处裁切后会导致网格减少最后一个格子，应该最大值增加半个格子
        # ds = ds.sel(          
        #     lon=slice(lon_min_c-0.01, lon_max_c+0.01),
        #     lat=slice(lat_min_c-0.01, lat_max_c+0.01),
        # )
    # -------------------------------------------------------------------------

    # --- lev or height_agl ---
    if "lev" in ds.coords:
        vert = "lev"
    elif "height_agl" in ds.coords:
        vert = "height_agl"
    else:
        raise ValueError("Cannot find height field 'lev' or 'height_agl'.")
    # read wind and height
    u   = ds[var_u].transpose(vert, "lat", "lon").astype(np.float32).values
    v   = ds[var_v].transpose(vert, "lat", "lon").astype(np.float32).values
    # try read vertical velocity; if absent, fill zeros
    var_w = "w"
    w = (ds[var_w].transpose(vert, "lat", "lon").astype(np.float32).values
         if var_w in ds.variables else np.zeros_like(u))


    lev = ds[vert].values                          # double precision
    lon = ds["lon"].values
    lat = ds["lat"].values
 
    # —— interpolation —— # To be revised - Huanxia
    target_n = 50
    new_coords = {}
    for dim in ("lon", "lat", vert):
        if ds.sizes[dim] < target_n:
            coords = ds[dim].values
            new_coords[dim] = np.linspace(coords[0], coords[-1],
                                         target_n, dtype=coords.dtype)
            # lwg: 此处插值坐标应与配置文件中目标网格一致 , 高度未考虑 
            # new_coords[dim] = np.linspace(eval(f"{dim}_min_c"), eval(f"{dim}_max_c"), target_n, dtype=coords.dtype)
            print(f"Dimension '{dim}' has only {ds.sizes[dim]} points, "
                f"interpolate to {target_n}.")
    
    # one order interpolation
    if new_coords:
        ds = ds.interp(new_coords,method="pchip")

        # re-extraction
        u   = ds[var_u].transpose(vert, "lat", "lon").astype(np.float32).values
        v   = ds[var_v].transpose(vert, "lat", "lon").astype(np.float32).values
        w   = (ds[var_w].transpose(vert, "lat", "lon").astype(np.float32).values
               if var_w in ds.variables else np.zeros_like(u))
        lev = ds[vert].values
        lon = ds["lon"].values
        lat = ds["lat"].values


    # -------- keep org range --------
    orig_lon_min = float(ds["lon"].min())
    orig_lon_max = float(ds["lon"].max())
    orig_lat_min = float(ds["lat"].min())
    orig_lat_max = float(ds["lat"].max())

    nz, ny, nx = u.shape
    ds.close()

    # -------- 2. filling --------
    if np.isnan(u).any() or np.isnan(v).any():
        _forward_fill_whole_layer(u, v)
    # volumetric mean velocity (u,v,w) over all cells
    um_vol = np.array([
        float(np.nanmean(u)),
        float(np.nanmean(v)),
        float(np.nanmean(w))
    ], dtype=float)

    # -------- 3. meshing --------
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # ——— project to UTM Zone 50N and obtain range ———
    corners = [
        Point(lon_min, lat_min), Point(lon_max, lat_min),
        Point(lon_max, lat_max), Point(lon_min, lat_max)
    ]
    gdf = gpd.GeoDataFrame(geometry=corners, crs="EPSG:4326") \
             .to_crs(epsg=32650)  
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # —— write the sliced and projected lon/lat range to deck —— 
    if conf_file.exists():
        lines = conf_file.read_text().splitlines()
        # fill enters to 50 lines
        while len(lines) < 49:
            lines.append('')
        # line 2-3
        lines[19]  = f"// WRF Data Range (raw data)"
        lines[20]  = f"cut_lon = [{orig_lon_min:.6f}, {orig_lon_max:.6f}]"
        lines[21]  = f"cut_lat = [{orig_lat_min:.6f}, {orig_lat_max:.6f}]"

        conf_file.write_text("\n".join(lines))

    p00 = Point(lon[0], lat[0])
    p10 = Point(lon[1], lat[0])
    p01 = Point(lon[0], lat[1])
    gdf2 = gpd.GeoDataFrame(geometry=[p00, p10, p01], crs="EPSG:4326") \
              .to_crs(epsg=32650)
    x00, x10, x01 = [pt.x for pt in gdf2.geometry]
    y00, y10, y01 = [pt.y for pt in gdf2.geometry]

    # lwg:等经纬度网格投影后计算出的网格距离直接做分辨率可能有问题？？
    dx = x10 - x00
    dy = y01 - y00
    dz = float(lev[-1] - lev[0]) / (nz - 1)

    if conf_file.exists():
        lines = conf_file.read_text().splitlines()
        while len(lines) < 49:
            lines.append('')

        # set origin
        x0, y0 = x00, y00

        # set box
        x1 = x0 + dx * (nx - 1)
        y1 = y0 + dy * (ny - 1)

        # sorting and writing
        x_min_glob, x_max_glob = sorted((x0, x1))
        y_min_glob, y_max_glob = sorted((y0, y1))

        lines[14] = "// WRF Data Projected SI Range (global / before shift)"
        lines[15] = f"si_x = [{x_min_glob:.6f}, {x_max_glob:.6f}]"
        lines[16] = f"si_y = [{y_min_glob:.6f}, {y_max_glob:.6f}]"

        # lwg：范围直接使用配置文件转换后的UTM 范围
        # lines[14] = "// WRF Data Projected SI Range (global / before shift)"
        # lines[15] = f"si_x = [{x_min:.6f}, {x_max:.6f}]"
        # lines[16] = f"si_y = [{y_min:.6f}, {y_max:.6f}]"

        z_min, z_max = sorted((float(lev[0]), float(lev[-1])))
        lines[17] = f"si_z_cfd = [{z_min:.6f}, {z_max:.6f}]"

        Path(conf_file).write_text("\n".join(lines))

    grid = vtk.vtkUniformGrid()
    # translate
    grid.SetOrigin(0.0, 0.0, float(lev[0]))
    grid.SetSpacing(dx, dy, dz)
    grid.SetDimensions(nx, ny, nz)

    # ——— print infomation ———
    print(f"Lon range: {lon_min:.4f} - {lon_max:.4f}, Lat range: {lat_min:.4f}-{lat_max:.4f}")
    print(f"Projected X range: {x_min:.2f}-{x_max:.2f} m "
        f"(span {(x_max - x_min):.2f} m), Y range: {y_min:.2f} - {y_max:.2f} m "
        f"(span {(y_max - y_min):.2f} m)")
    print(f"Computed grid spacing: dx = {dx:.2f} m, dy = {dy:.2f} m, dz = {dz:.2f} m.")

    # -------- 4. write data to vtk --------
    def _add(arr: np.ndarray, name: str):
        vtk_arr = nps.numpy_to_vtk(arr.ravel(order="C"), deep=True,
                                   array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName(name)
        grid.GetPointData().AddArray(vtk_arr)

    _add(u, "u")
    _add(v, "v")
    grid.GetPointData().SetActiveScalars("u")

    # -------- 5. writer of .vti --------
    writer = (vtk.vtkXMLUniformGridWriter()
              if hasattr(vtk, "vtkXMLUniformGridWriter")
              else vtk.vtkXMLImageDataWriter())
    vti_path = Path(nc_path).with_suffix(".vti")
    print(f"Write output: {vti_path}")
    writer.SetFileName(str(vti_path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    writer.Write()

    mb = Path(vti_path).stat().st_size / 1e6
    print(f"Finished: {vti_path}  (={mb:.6f} MB)")

    # # ===== debug mode =====
    # grid_abs = vtk.vtkUniformGrid()

    # grid_abs.SetOrigin(x00, y00, float(lev[0]))
    # grid_abs.SetSpacing(dx, dy, dz)
    # grid_abs.SetDimensions(nx, ny, nz)

    # for arr, name in ((u, "u"), (v, "v")):
    #     vtk_arr = nps.numpy_to_vtk(arr.ravel(order="C"), deep=True,
    #                                array_type=vtk.VTK_FLOAT)
    #     vtk_arr.SetName(name)
    #     grid_abs.GetPointData().AddArray(vtk_arr)
    # grid_abs.GetPointData().SetActiveScalars("u")

    # writer_abs = (vtk.vtkXMLUniformGridWriter()
    #               if hasattr(vtk, "vtkXMLUniformGridWriter")
    #               else vtk.vtkXMLImageDataWriter())

    # # xxx.vti: xxx_abs.vti
    # vti_path_abs = vti_path.with_stem(vti_path.stem + "_abs")

    # print(f"Write absolute-coords VTI: {vti_path_abs}")
    # writer_abs.SetFileName(str(vti_path_abs))
    # writer_abs.SetInputData(grid_abs)
    # writer_abs.SetDataModeToBinary()
    # writer_abs.Write()

    # -------- 6. write WRF data to CSV --------
    # accumulators for boundary-face mean (using exactly the same points as CSV)
    bc_sum = np.zeros(3, dtype=float)
    bc_cnt = 0

    csv_path = vti_path.parent / "SurfData_Latest.csv"
    with open(csv_path, "w") as csvfile:
        csvfile.write("X,Y,Z,u,v,w\n")

        # 1) base and top
        for k in (0, nz-1):
            z = float(lev[k])
            for j in range(ny):
                for i in range(nx):
                    x = i * dx
                    y = j * dy
                    uval = u[k, j, i]
                    vval = v[k, j, i]
                    csvfile.write(f"{x},{y},{z},{uval},{vval},0.0\n")
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1


        # 2) N-S
        for j in (0, ny-1):
            y = j * dy
            for k in range(1, nz-1):
                z = float(lev[k])
                for i in range(nx):
                    x = i * dx
                    uval = u[k, j, i]
                    vval = v[k, j, i]
                    csvfile.write(f"{x},{y},{z},{uval},{vval},0.0\n")
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1


        # 3) E-W
        for i in (0, nx-1):
            x = i * dx
            for k in range(1, nz-1):
                z = float(lev[k])
                for j in range(ny):
                    y = j * dy
                    uval = u[k, j, i]
                    vval = v[k, j, i]
                    csvfile.write(f"{x},{y},{z},{uval},{vval},0.0\n")
                    bc_sum += np.array([uval, vval, 0.0])
                    bc_cnt += 1


    print(f"Write surface data CSV: {csv_path}")


    # === copy SurfData_Latest.csv to bcData with timestamped name === csv_path = vti_path.parent / "SurfData_Latest.csv"
    if conf_file.exists():
        # 
        m_dt = re.search(r"datetime\s*=\s*([0-9]{14})", txt)
        if m_dt:
            dt_str = m_dt.group(1)
            # project_home = Path(__file__).parent.parent  # $ProjectHome$
            dst_path = vti_path.parent / f"SurfData_{dt_str}.csv"
            shutil.copyfile(csv_path, dst_path)
            print(f"Copied SurfData to: {dst_path}")
        else:
            print("Warning: 'datetime' not found in conf.txt; skip copy.")


    um_bc = bc_sum / bc_cnt if bc_cnt > 0 else np.zeros(3, dtype=float)
    # Determine downstream face using mean flow (u,v only)
    mean_u, mean_v = um_vol[0], um_vol[1]

    if abs(mean_u) >= abs(mean_v):
        downstream_face = "+x" if mean_u >= 0 else "-x"
        # normal vector
        n = np.array([1.0 if downstream_face == "+x" else -1.0, 0.0, 0.0])
        parallel = mean_u if downstream_face == "+x" else -mean_u
        perp = mean_v  # deviation toward +y / -y
        sign = 1.0 if perp >= 0 else -1.0
    else:
        downstream_face = "+y" if mean_v >= 0 else "-y"
        n = np.array([0.0, 1.0 if downstream_face == "+y" else -1.0, 0.0])
        parallel = mean_v if downstream_face == "+y" else -mean_v
        perp = mean_u  # deviation toward +x / -x
        sign = 1.0 if perp >= 0 else -1.0

    # Yaw angle between mean velocity vector and downstream face normal (signed)
    # When the downstream face is +x / −x: downstream_bc_yaw > 0 means the mean flow velocity is deflected toward +y, 
    #                                                                       and < 0 means it is deflected toward −y.
    # When the downstream face is +y / −y: downstream_bc_yaw > 0 means the mean flow velocity is deflected toward +x, 
    #                                                                       and < 0 means it is deflected toward −x.
    theta = math.degrees(math.atan2(abs(perp), abs(parallel))) if parallel != 0 else 90.0
    yaw_angle = sign * theta  

    # -------- 7. write mean velocities etc. to conf.txt --------
    if conf_file.exists():
        lines = conf_file.read_text().splitlines()
        while len(lines) < 49:
            lines.append('')

        # Lines are 1-based in your description: 28~32
        lines[27] = "// Mean velocity uvw (volumetric / boundary / downstream with angle)"
        lines[28] = f"um_vol = [{um_vol[0]:.6f}, {um_vol[1]:.6f}, {um_vol[2]:.6f}]"
        lines[29] = f"um_bc = [{um_bc[0]:.6f}, {um_bc[1]:.6f}, {um_bc[2]:.6f}]"
        lines[30] = f'downstream_bc = "{downstream_face}"'
        lines[31] = f"downstream_bc_yaw = {yaw_angle:.2f}"

        conf_file.write_text("\n".join(lines))


if __name__ == "__main__":
    buildBC()

# shp_to_stl.py
import argparse
from pathlib import Path
import math

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
import shapely
import numpy as np
import trimesh
from trimesh import boolean
import re

def _make_valid(geom):
    """
    Make geometry valid using shapely's make_valid or buffer(0) as fallback.
    """
    try:
        from shapely.validation import make_valid
        return make_valid(geom)
    except Exception:
        try:
            return geom.buffer(0)
        except Exception:
            return None


def _to_iter_polygons(geom):
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if isinstance(p, Polygon)]
    return []


def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def extrude_polygon_to_mesh(poly: Polygon, height: float) -> trimesh.Trimesh:
    # Ensure the polygon is in 2D (ignore Z if present)
    def _ensure_2d_coords(coords):
        """Ensure coordinates are 2D"""
        return [(x, y) for x, y, *_ in coords] if len(coords[0]) > 2 else coords

    # Process exterior ring
    exterior_coords = list(poly.exterior.coords)
    exterior_2d = _ensure_2d_coords(exterior_coords)

    # Process interior rings (holes)
    holes_2d = []
    for interior in poly.interiors:
        interior_coords = list(interior.coords)
        interior_2d = _ensure_2d_coords(interior_coords)
        holes_2d.append(interior_2d)

    # Create 2D polygon
    if holes_2d:
        poly_2d = Polygon(exterior_2d, holes_2d)
    else:
        poly_2d = Polygon(exterior_2d)

    try:
        print("[INFO] Trying to extrude polygon using 'earcut' engine")
        return trimesh.creation.extrude_polygon(poly_2d, height, engine="earcut")
    except Exception:
        try:
            print("[INFO] Trying to extrude polygon using 'triangle' engine")
            return trimesh.creation.extrude_polygon(poly_2d, height, engine="triangle")
        except Exception as e:
            raise RuntimeError("No triangulation engine available, please install mapbox_earcut or triangle") from e



def _build_bbox_polygon_from_conf(conf_path) -> Polygon:
    """
    从给定的 conf 文件读取 cut_lon_manual 与 cut_lat_manual，构造经纬度 bbox 多边形。
    """
    conf_file = Path(conf_path).expanduser().resolve()
    if not conf_file.exists():
        raise FileNotFoundError(f"[ERROR] Configuration file not found: {conf_file}")
    txt = conf_file.read_text(encoding="utf-8", errors="ignore")

    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt)
    if not (m_lon and m_lat):
        raise ValueError("conf 中未找到 cut_lon_manual/cut_lat_manual")

    lon_vals = [float(v.strip()) for v in m_lon.group(1).split(",")]
    lat_vals = [float(v.strip()) for v in m_lat.group(1).split(",")]
    minx, maxx = min(lon_vals), max(lon_vals)
    miny, maxy = min(lat_vals), max(lat_vals)

    ring = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return Polygon(ring)


def _project_geometry(geom: Polygon, target_crs) -> Polygon:
    """
    Project geometry to target CRS using GeoSeries's to_crs.
    """
    gs = gpd.GeoSeries([geom], crs="EPSG:4326")
    gs2 = gs.to_crs(target_crs)
    result = gs2.iloc[0]
    if isinstance(result, Polygon):
        return result
    else:
        raise RuntimeError(f"投影结果不是Polygon类型: {type(result)}")


def main():
    parser = argparse.ArgumentParser(
        description="读取建筑底面 Shapefile，依据高度字段挤出并导出整体 STL。包含轴向对齐与经纬度 bbox 底座。"
    )
    parser.add_argument(
        "conf",
        help="path to deck file"
    )
    parser.add_argument(
        "--height-field",
        default="auto",
        help="height field in shapefile, or 'auto' to detect (Height, Elevation, etc.). Default 'auto'"
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.0,
        help="minimum valid height in meters. Buildings with height less than this will be ignored. Default 0.0"
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="do not reproject shapefile to UTM, use original CRS"
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=50.0,
        help="base height in meters. Default 50.0"
    )

    args = parser.parse_args()
    print("[INFO] Parsed CLI arguments")

    conf_file = Path(args.conf).expanduser().resolve()
    if not conf_file.exists():
        raise FileNotFoundError(f"conf file not found: {conf_file}")
    project_home = conf_file.parent
    proj_temp = project_home / "proj_temp"
    proj_temp.mkdir(parents=True, exist_ok=True)

    txt_conf = conf_file.read_text(encoding="utf-8", errors="ignore")
    m_case = re.search(r"casename\s*=\s*([^\s]+)", txt_conf)
    if not m_case:
        raise RuntimeError("casename not found in conf")
    case_name = m_case.group(1)

    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt_conf)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt_conf)
    if not (m_lon and m_lat):
        raise RuntimeError("conf 缺少 cut_lon_manual/cut_lat_manual")

    lon_vals = [float(v.strip()) for v in m_lon.group(1).split(",")]
    lat_vals = [float(v.strip()) for v in m_lat.group(1).split(",")]

    primary_shp = proj_temp / f"cutted_shp/{case_name}.shp"
    fallback_shp = proj_temp / f"{case_name}.shp"

    if primary_shp.exists():
        shp_path = primary_shp
    elif fallback_shp.exists():
        print(f"[WARN] Primary shapefile not found: {primary_shp.name}. Fallback to {fallback_shp.name}")
        shp_path = fallback_shp
    else:
        raise FileNotFoundError(f"Shapefile not found: {primary_shp} or {fallback_shp}")

    print(f"[INFO] Input Shapefile: {shp_path}")
    print(f"[INFO] Height field: {args.height_field}")
    print(f"[INFO] Min valid height: {args.min_height} m")
    print(f"[INFO] Base height: {args.base_height} m")
    print(f"[INFO] Auto reproject to UTM: {'No' if args.no_reproject else 'Yes'}")

    combined_file = proj_temp / f"{case_name}.stl"

    print(f"[INFO] Reading features from file")
    gdf = gpd.read_file(shp_path)
    if len(gdf) == 0:
        raise RuntimeError("Shapefile has no features")
    print(f"[INFO] Read {len(gdf)} features")
    print(f"[INFO] Source CRS: {gdf.crs}")

    # Use fixed UTM CRS to match main.ipynb configuration (EPSG:32651)
    gdf_work = gdf.copy()
    if not args.no_reproject:
        try:
            if gdf_work.crs is not None:
                # Use fixed EPSG:32651 to match main.ipynb
                gdf_work = gdf_work.to_crs("EPSG:32651")
        except Exception:
            pass

    work_crs = gdf_work.crs
    print(f"[INFO] Working CRS (fixed to match main.ipynb): {work_crs}")

    # 自动检测高度字段
    if args.height_field == "auto":
        height_candidates = ["Height", "Elevation", "height", "elevation", "HEIGHT", "ELEVATION"]
        height_field = None
        for candidate in height_candidates:
            if candidate in gdf_work.columns:
                height_field = candidate
                print(f"[INFO] Auto-detected height field: {height_field}")
                break
        if height_field is None:
            raise RuntimeError(f"Cannot auto-detect height field. Available fields: {list(gdf_work.columns)}")
        args.height_field = height_field
    else:
        print(f"[INFO] Using specified height field: {args.height_field}")

    # 构造经纬度 bbox 的矩形，并投到工作坐标系，作为底座平面
    # 使用与2_buildBC_dev1.py相同的边界框定义方式
    bbox_ll = _build_bbox_polygon_from_conf(conf_file)
    print("[INFO] Built geographic bbox from conf.txt to match 2_buildBC_dev1.py")
    if bbox_ll is None:
        raise RuntimeError("Cannot build lat-lon bbox")
    try:
        base_poly_proj = _project_geometry(bbox_ll, work_crs) if work_crs is not None else bbox_ll
    except Exception:
        # 若投影失败则退化为在当前工作坐标下以当前 bounds 构建矩形
        minx, miny, maxx, maxy = gdf_work.total_bounds
        base_poly_proj = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    print("[INFO] Projected bbox polygon into working CRS for base")

    # 估计与 XY 轴的偏角。取底座矩形投影后的第一条边方向
    coords = list(base_poly_proj.exterior.coords)
    if len(coords) < 2:
        raise RuntimeError("[ERROR] Base rectangle geometry is abnormal")
    x0, y0 = coords[0][0], coords[0][1]
    x1, y1 = coords[1][0], coords[1][1]
    dx, dy = x1 - x0, y1 - y0
    angle_rad = math.atan2(dy, dx)
    # 为了让该边与 X 轴对齐，整体绕 Z 轴旋转负角
    rotate_deg = - math.degrees(angle_rad)

    # 以底座矩形的质心为旋转中心，以降低旋转后的平移漂移
    pivot = base_poly_proj.centroid
    pivot_xy = (pivot.x, pivot.y)
    print(f"[INFO] Estimated rotation in degrees: {rotate_deg:.6f}")
    print(f"[INFO] Rotation pivot at centroid: ({pivot_xy[0]:.3f}, {pivot_xy[1]:.3f})")

    # 对底座做旋转
    base_rot = affinity.rotate(base_poly_proj, rotate_deg, origin=pivot_xy, use_radians=False)
    print("[INFO] Rotated base rectangle to align with X axis")
    print("[INFO] Rotating building polygons and filtering by height")

    # 对建筑几何逐要素做旋转与有效性修复
    rotated_polygons = []
    heights = []
    for _, row in gdf_work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        h = _safe_float(row.get(args.height_field))
        if h is None or h <= args.min_height:
            continue
        geom_valid = _make_valid(geom)
        if geom_valid is None or geom_valid.is_empty:
            continue
        polys = _to_iter_polygons(geom_valid)
        if not polys:
            continue
        for poly in polys:
            try:
                poly_rot = affinity.rotate(poly, rotate_deg, origin=pivot_xy, use_radians=False)
                rotated_polygons.append(poly_rot)
                heights.append(h)
            except Exception:
                continue

    if not rotated_polygons:
        raise RuntimeError("[ERROR] No valid building polygons available. Please check height field and geometry validity.")

    # 保持使用conf.txt定义的边界框，不要用建筑物范围替换
    # 这样确保与2_buildBC_dev1.py生成的.vti文件边界完全一致
    print("[INFO] Using boundary box from conf.txt to match 2_buildBC_dev1.py exactly")

    # 裁剪超出边界框的建筑物部分
    clipped_polygons = []
    clipped_heights = []
    boundary_box = base_rot

    for poly, h in zip(rotated_polygons, heights):
        try:
            # 裁剪建筑物到边界框内
            clipped = poly.intersection(boundary_box)
            if not clipped.is_empty and clipped.area > 0:
                if isinstance(clipped, Polygon):
                    clipped_polygons.append(clipped)
                    clipped_heights.append(h)
                elif isinstance(clipped, MultiPolygon):
                    for sub_poly in clipped.geoms:
                        if isinstance(sub_poly, Polygon) and sub_poly.area > 0:
                            clipped_polygons.append(sub_poly)
                            clipped_heights.append(h)
        except Exception:
            # 如果裁剪失败，跳过这个建筑物
            continue

    rotated_polygons = clipped_polygons
    heights = clipped_heights
    print(f"[INFO] Clipped buildings to boundary box. Remaining polygons: {len(rotated_polygons)}")


    print(f"[INFO] Usable building polygons after validation and rotation: {len(rotated_polygons)}")
    if heights:
        print(f"[INFO] Building height statistics before extrusion: min {min(heights):.3f} m, max {max(heights):.3f} m")

    # 旋转后再统一计算向原点平移，使 minx 与 miny 为零附近
    # 只使用边界框的范围，确保与2_buildBC_dev1.py完全一致
    minx_all, miny_all = base_rot.bounds[0], base_rot.bounds[1]
    tx, ty = -minx_all, -miny_all

    base_final = affinity.translate(base_rot, xoff=tx, yoff=ty)
    print(f"[INFO] Translation offsets applied to set origin near minima: dx {tx:.3f}, dy {ty:.3f}")

    shifted_polygons = [affinity.translate(p, xoff=tx, yoff=ty) for p in rotated_polygons]

    # Generate base mesh from zero to base_height
    try:
        base_mesh = extrude_polygon_to_mesh(base_final, max(args.base_height, 0.0))
        print("[INFO] Base mesh extruded")
    except Exception as e:
        raise RuntimeError(f"Failed to extrude base mesh: {e}")

    # Generate building meshes by extruding at z=0 and then translating up by base_height
    building_meshes = []
    failed_count = 0
    for i, (poly, h) in enumerate(zip(shifted_polygons, heights)):
        if h <= 0:
            continue
        try:
            m = extrude_polygon_to_mesh(poly, h)
            # Translate building above the base
            m.apply_translation([0.0, 0.0, max(args.base_height, 0.0)])
            building_meshes.append(m)
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only show first 5 errors
                print(f"[WARN] Building {i} extrusion failed: {e}")
            continue

    if failed_count > 0:
        print(f"[WARN] Total {failed_count} buildings failed to extrude out of {len(shifted_polygons)}")

    print(f"[INFO] Extruded {len(building_meshes)} building meshes")

    if not building_meshes:
        print("[INFO] No building meshes, using base only")
        final_mesh = base_mesh
    else:
        try:
            print("[INFO] Performing boolean union of base and buildings")
            final_mesh = boolean.union([base_mesh] + building_meshes)
            print("[INFO] Boolean union succeeded")
        except Exception:
            print("[WARN] Boolean union failed, concatenating meshes instead")
            final_mesh = trimesh.util.concatenate([base_mesh] + building_meshes)

    final_mesh.export(combined_file, file_type="stl")
    print(f"[INFO] Exported combined STL with axis alignment and base: {combined_file.resolve()}")

    # Report four base-corner coordinates in meters and STL Z range in meters
    minx, miny, maxx, maxy = base_final.bounds
    corners = [
        (minx, miny),  # lower left
        (maxx, miny),  # lower right
        (maxx, maxy),  # upper right
        (minx, maxy)   # upper left
    ]
    print("[RESULT] Base rectangle corners in meters")
    print(f"[RESULT] Lower left: ({corners[0][0]:.3f}, {corners[0][1]:.3f})")
    print(f"[RESULT] Lower right: ({corners[1][0]:.3f}, {corners[1][1]:.3f})")
    print(f"[RESULT] Upper right: ({corners[2][0]:.3f}, {corners[2][1]:.3f})")
    print(f"[RESULT] Upper left: ({corners[3][0]:.3f}, {corners[3][1]:.3f})")

    min_bounds, max_bounds = final_mesh.bounds
    zmin = float(min_bounds[2])
    zmax = float(max_bounds[2])
    print(f"[RESULT] STL Z range in meters: min {zmin:.3f}, max {zmax:.3f}")
    print(f"[RESULT] Total height in meters: {(zmax - zmin):.3f}")


if __name__ == "__main__":
    main()

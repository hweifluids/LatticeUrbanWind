from __future__ import annotations

import ast
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List


def _parse_conf_bbox(conf_raw: str) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    """
    Parse required fields from conf raw text:
      casename = ...
      cut_lon_manual = [min, max]
      cut_lat_manual = [min, max]
    """
    def find_value(pattern: str) -> str:
        m = re.search(pattern, conf_raw, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Missing required line in conf: {pattern}")
        return m.group(1).strip()

    casename_raw = find_value(r"^\s*casename\s*=\s*(.+?)\s*$")
    casename = casename_raw.strip().strip('"').strip("'")

    cut_lon_str = find_value(r"^\s*cut_lon_manual\s*=\s*(\[[^\]]+\])\s*$")
    cut_lat_str = find_value(r"^\s*cut_lat_manual\s*=\s*(\[[^\]]+\])\s*$")

    try:
        cut_lon = ast.literal_eval(cut_lon_str)
        cut_lat = ast.literal_eval(cut_lat_str)
    except Exception as e:
        raise ValueError("Failed to parse cut_lon_manual / cut_lat_manual as Python lists like [min, max].") from e

    if not (isinstance(cut_lon, (list, tuple)) and len(cut_lon) == 2):
        raise ValueError("cut_lon_manual must be a list/tuple of length 2, e.g. [113.302, 113.342].")
    if not (isinstance(cut_lat, (list, tuple)) and len(cut_lat) == 2):
        raise ValueError("cut_lat_manual must be a list/tuple of length 2, e.g. [23.093, 23.133].")

    try:
        lon0 = float(cut_lon[0])
        lon1 = float(cut_lon[1])
        lat0 = float(cut_lat[0])
        lat1 = float(cut_lat[1])
    except Exception as e:
        raise ValueError("cut_lon_manual / cut_lat_manual values must be numeric.") from e

    lon_min, lon_max = (lon0, lon1) if lon0 <= lon1 else (lon1, lon0)
    lat_min, lat_max = (lat0, lat1) if lat0 <= lat1 else (lat1, lat0)

    if lon_min == lon_max or lat_min == lat_max:
        raise ValueError("Invalid bbox: lon/lat range has zero size.")

    return casename, (lon_min, lon_max), (lat_min, lat_max)


def _expand_bbox_20pct(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> Tuple[float, float, float, float]:
    """
    Expand bbox by 20% on each side -> width/height become 1.4x.
    """
    w = lon_max - lon_min
    h = lat_max - lat_min
    lon_min2 = lon_min - 0.2 * w
    lon_max2 = lon_max + 0.2 * w
    lat_min2 = lat_min - 0.2 * h
    lat_max2 = lat_max + 0.2 * h
    return lon_min2, lon_max2, lat_min2, lat_max2


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
                    # Discard special key code
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
        print(f"[WARN] {kind} bounds are slightly smaller than target (max miss {max_miss_pct:.4f}% < 0.1%). Continue without interruption.")
        return

    print(f"[WARN] {kind} bounds do not fully cover the target area.")
    print(f"[WARN] Target lon/lat bounds: {_format_lonlat_bounds(target_bounds)}")
    print(f"[WARN] Input  lon/lat bounds: {_format_lonlat_bounds(input_bounds)}")

    timeout_s = 5.0
    ans_raw = _timed_input_line(
        f"Continue anyway? (Y/N) [auto-continue in {int(timeout_s)}s]: ",
        timeout_s=timeout_s,
    )
    if ans_raw is None:
        print(f"[WARN] No input received (timeout {int(timeout_s)}s). Continuing by default.")
        return

    ans = ans_raw.strip().lower()

    if ans in ("n", "no"):
        print("[INFO] User canceled. Exiting.")
        sys.exit(1)

    if ans in ("y", "yes", ""):
        print("[WARN] Continuing despite bounds mismatch.")
        return

    print(f"[WARN] Invalid input '{ans_raw}'. Continuing by default.")
    return


def _resolve_install_database_dem_path() -> Path:
    # Assume repo root is two levels up from core/bridge_core
    return Path(__file__).resolve().parents[2] / "database" / "dem.tif"


def _collect_tif_candidates(project_home: Path) -> Tuple[List[Path], str]:
    """
    Priority:
      1) Any *.tif/*.tiff under project_home/terrain_db (name unrestricted).
      2) Install database dem.tif (fixed name).
    Returns (candidates, source_label).
    """
    terrain_dir = project_home / "terrain_db"
    candidates: List[Path] = []
    if terrain_dir.exists():
        tif_list = sorted(terrain_dir.glob("*.tif"))
        tiff_list = sorted(terrain_dir.glob("*.tiff"))
        candidates = tif_list + tiff_list
        if candidates:
            return candidates, "terrain_db"

    install_dem = _resolve_install_database_dem_path()
    if install_dem.exists():
        return [install_dem], "install_database"

    return [], ""


def _remove_existing_shapefile(output_shp: Path) -> None:
    """
    Remove existing shapefile sidecar files to avoid stale leftovers.
    """
    base, _ = os.path.splitext(str(output_shp))
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]
    for ext in exts:
        p = base + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def _polygonize_dem_cut(
    dem_path: Path,
    bbox_lonlat: Tuple[float, float, float, float],
    output_shp: Path,
    casename: str,
    target_lonlat: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    """
    Cut raster by bbox (lon/lat EPSG:4326), transform bbox to raster CRS if needed,
    polygonize valid cells, and export shapefile.
    Returns True if success, False otherwise.
    """
    try:
        import numpy as np
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
        from rasterio.features import shapes
        from rasterio.warp import transform_geom, transform_bounds
        from shapely.geometry import box, mapping
    except Exception as e:
        print(f"[WARN] GeoTIFF conversion requires rasterio/geopandas: {e}")
        return False

    lon_min, lon_max, lat_min, lat_max = bbox_lonlat
    bbox_geom_4326 = mapping(box(lon_min, lat_min, lon_max, lat_max))

    try:
        with rasterio.open(str(dem_path)) as src:
            src_crs = src.crs
            input_bounds = None
            try:
                b = src.bounds
                if src_crs is None:
                    print("[WARN] DEM has no CRS. Assuming raster bounds are EPSG:4326 for coverage check.")
                    lon_min_i, lon_max_i = sorted((b.left, b.right))
                    lat_min_i, lat_max_i = sorted((b.bottom, b.top))
                    input_bounds = (lon_min_i, lon_max_i, lat_min_i, lat_max_i)
                else:
                    if str(src_crs).upper() in ["EPSG:4326", "WGS84", "WGS 84"]:
                        lon_min_i, lon_max_i = sorted((b.left, b.right))
                        lat_min_i, lat_max_i = sorted((b.bottom, b.top))
                        input_bounds = (lon_min_i, lon_max_i, lat_min_i, lat_max_i)
                    else:
                        try:
                            lon_min_i, lat_min_i, lon_max_i, lat_max_i = transform_bounds(
                                src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
                            )
                            lon_min_i, lon_max_i = sorted((lon_min_i, lon_max_i))
                            lat_min_i, lat_max_i = sorted((lat_min_i, lat_max_i))
                            input_bounds = (lon_min_i, lon_max_i, lat_min_i, lat_max_i)
                        except Exception as e:
                            print(f"[WARN] Failed to transform DEM bounds to EPSG:4326 for coverage check: {e}")
            except Exception as e:
                print(f"[WARN] Failed to read DEM bounds for coverage check: {e}")

            if input_bounds is not None:
                _confirm_bbox_coverage("DEM GeoTIFF", target_lonlat or bbox_lonlat, input_bounds)

            if src_crs is None:
                print("[WARN] DEM has no CRS. Assuming bbox is already in raster CRS.")
                bbox_geom_src = bbox_geom_4326
            else:
                if str(src_crs).upper() in ["EPSG:4326", "WGS84", "WGS 84"]:
                    bbox_geom_src = bbox_geom_4326
                else:
                    try:
                        bbox_geom_src = transform_geom("EPSG:4326", src_crs, bbox_geom_4326, precision=6)
                    except Exception as e:
                        print(f"[WARN] Failed to transform bbox from EPSG:4326 to DEM CRS ({src_crs}): {e}")
                        return False

            print(f"[INFO] DEM path: {dem_path}")
            print(f"[INFO] DEM CRS: {src_crs}")
            print("[INFO] Cutting raster by expanded bbox ...")

            try:
                out_image, out_transform = mask(src, [bbox_geom_src], crop=True)
            except Exception as e:
                print(f"[WARN] GeoTIFF does not overlap target bounds: {e}")
                return False

            if out_image.ndim < 3 or out_image.shape[0] < 1:
                print("[WARN] Unexpected raster data shape after masking.")
                return False

            data = out_image[0]

            invalid = np.zeros(data.shape, dtype=bool)
            if src.nodata is not None:
                invalid |= (data == src.nodata)
            invalid |= (data == -9999)

            valid_mask = ~invalid
            valid_count = int(valid_mask.sum())
            if valid_count == 0:
                print("[WARN] No valid elevation cells found within expanded bbox.")
                return False

            print("[INFO] Polygonizing raster cells ...")

            features = []
            try:
                for geom, val in shapes(data, mask=valid_mask, transform=out_transform):
                    try:
                        elev = float(val)
                    except Exception:
                        continue
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {"elevation": elev, "casename": casename},
                            "geometry": geom,
                        }
                    )
            except Exception as e:
                print(f"[WARN] Polygonize failed: {e}")
                return False

            if len(features) == 0:
                print("[WARN] Polygonize produced no features (all cells may be nodata after masking).")
                return False

            gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
            gdf["elevation"] = gdf["elevation"].astype(float)

            _remove_existing_shapefile(output_shp)

            print(f"[INFO] Writing shapefile: {output_shp}")
            try:
                gdf.to_file(str(output_shp), driver="ESRI Shapefile", encoding="utf-8")
            except Exception as e:
                print(f"[WARN] Failed to write shapefile: {e}")
                return False

            elev_min = float(gdf["elevation"].min())
            elev_max = float(gdf["elevation"].max())
            elev_mean = float(gdf["elevation"].mean())

            print("[INFO] DEM GeoTIFF conversion done.")
            print(f"[INFO] Feature count: {len(gdf)}")
            print(f"[INFO] Elevation min: {elev_min:.3f}")
            print(f"[INFO] Elevation max: {elev_max:.3f}")
            print(f"[INFO] Elevation mean: {elev_mean:.3f}")
            print(f"[INFO] Elevation range: {(elev_max - elev_min):.3f}")

            return True

    except Exception as e:
        print(f"[WARN] Failed to process GeoTIFF: {e}")
        return False


def ensure_dem_shp_from_tif(conf_raw: str, project_home: Path) -> Optional[Path]:
    """
    Ensure a DEM shapefile exists under project_home/terrain_db by cutting a GeoTIFF.
    Priority:
      - Use any tif/tiff in terrain_db (name unrestricted).
      - If none, use install database dem.tif (fixed name).
    Returns the created shapefile path, or None on failure.
    """
    if not conf_raw:
        print("[WARN] No conf text provided; cannot derive DEM cut bounds.")
        return None

    try:
        casename, lon_pair, lat_pair = _parse_conf_bbox(conf_raw)
    except Exception as e:
        print(f"[WARN] Failed to parse DEM bounds from conf: {e}")
        return None

    lon_min, lon_max = lon_pair
    lat_min, lat_max = lat_pair
    lon_min2, lon_max2, lat_min2, lat_max2 = _expand_bbox_20pct(lon_min, lon_max, lat_min, lat_max)
    target_bounds = (lon_min, lon_max, lat_min, lat_max)

    terrain_dir = project_home / "terrain_db"
    if not terrain_dir.exists():
        try:
            terrain_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created terrain_db directory: {terrain_dir}")
        except Exception as e:
            print(f"[WARN] Failed to create terrain_db directory: {e}")
            return None

    output_shp = terrain_dir / f"{casename}.shp"
    if output_shp.exists():
        print("[INFO] DEM shapefile already exists; skip GeoTIFF conversion.")
        return output_shp

    candidates, source = _collect_tif_candidates(project_home)
    if not candidates:
        print("[INFO] No DEM GeoTIFF found in terrain_db or install database.")
        return None

    if source == "terrain_db":
        print("[INFO] Found DEM GeoTIFF(s) in terrain_db.")
    elif source == "install_database":
        print(f"[INFO] Using install database DEM: {candidates[0]}")

    print(
        "[INFO] DEM cut bbox (EPSG:4326) expanded 20%: "
        f"lon[{lon_min2}, {lon_max2}], lat[{lat_min2}, {lat_max2}]"
    )

    bbox = (lon_min2, lon_max2, lat_min2, lat_max2)
    for tif_path in candidates:
        print(f"[INFO] Trying DEM GeoTIFF: {tif_path}")
        ok = _polygonize_dem_cut(tif_path, bbox, output_shp, casename, target_lonlat=target_bounds)
        if ok:
            return output_shp

    print("[WARN] No GeoTIFF covers target bounds or contains valid cells.")
    return None

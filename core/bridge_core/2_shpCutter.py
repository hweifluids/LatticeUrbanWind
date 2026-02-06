# 2_shpCutter.py

from __future__ import annotations
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import math
from shapely.ops import transform, unary_union
from shapely import prepared, affinity
from pyproj import Transformer
import sys
import numpy as np
from auto_UTM import get_utm_epsg_from_lonlat, get_utm_crs_from_bounds
import time

# ----------------------- Helpers -----------------------

from typing import Tuple, Union, Optional
import re

def get_lonlat(conf_path: Union[str, Path] | None = None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if conf_path is None:
        conf_path = Path(__file__).parent.parent / "conf.txt"
    else:
        conf_path = Path(conf_path)

    if not conf_path.exists():
        raise FileNotFoundError(f"Cannot find deck file: {conf_path}")

    txt = conf_path.read_text(encoding="utf-8", errors="ignore")

    def _parse_pair(key: str):
        m = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", txt)
        if not m:
            return None
        arr = [s.strip() for s in m.group(1).split(",")]
        if len(arr) != 2:
            raise ValueError(f"{key} has wrong configuration format, expected two values")
        a, b = float(arr[0]), float(arr[1])
        lo, hi = (a, b) if a <= b else (b, a)
        return lo, hi

    lon_pair = _parse_pair("cut_lon_manual")
    lat_pair = _parse_pair("cut_lat_manual")
    if lon_pair is None or lat_pair is None:
        raise ValueError("conf.txt does not provide clipping range, missing cut_lon_manual/cut_lat_manual")

    return lon_pair, lat_pair

__all__ = ["get_lonlat"]

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

def select_height_column(gdf: gpd.GeoDataFrame):
    """Return the name of a height-like column if found, else None."""
    cols = [c for c in gdf.columns if isinstance(c, str)]
    lower_map = {c.lower(): c for c in cols}

    exact_priority = [
        "height",
        "height_m",
        "heightm",
        "measuredheight",
        "buildingheight",
        "bldgheight",
        "bldg_h",
        "bldghgt",
        "roof_height",
        "eave_height",
        "max_height",   
        "elevation"
    ]
    for key in exact_priority:
        if key in lower_map:
            return lower_map[key]

    for c in cols:
        lc = c.lower()
        if ("height" in lc) or lc.endswith("hgt") or lc.endswith("_h"):
            return c

    for key in ["levels", "level", "storeys", "stories", "floors", "floor"]:
        if key in lower_map:
            return lower_map[key]

    return None


def _area_mask_chunk(geoms, min_area):
    """Filter geometries by area in meters, with UTM CRS auto detection."""
    # Auto detect UTM CRS from a representative geometry
    first_valid = None
    for g in geoms:
        if g is not None and not g.is_empty:
            first_valid = g
            break
    if first_valid is None:
        return [False] * len(geoms)

    rep = first_valid.representative_point()
    utm_crs = get_utm_epsg_from_lonlat(rep.x, rep.y)

    transformer = Transformer.from_crs(4326, utm_crs, always_xy=True)

    def _proj(x, y, z=None):
        x2, y2 = transformer.transform(x, y)
        return x2, y2

    out = []
    for g in geoms:
        if g is None or g.is_empty:
            out.append(False)
            continue
        try:
            g_m = transform(_proj, g)
            out.append(g_m.area >= min_area)
        except Exception:
            out.append(False)
    return out


MIN_EXTRUDE_AREA = 20.0  # m^2, minimum polygon area to extrude
MIN_HOLE_AREA = 5.0  # m^2, minimum interior ring area to keep (smaller holes will be removed)


def _make_valid_safe(geom):
    """Attempt to return a valid polygonal geometry. Use make_valid if available, else buffer(0)."""
    if geom is None or geom.is_empty:
        return None
    try:
        # Shapely 2.0+
        from shapely.validation import make_valid  # type: ignore
        g2 = make_valid(geom)
    except Exception:
        try:
            g2 = geom.buffer(0)
        except Exception:
            return None
    if g2 is None or g2.is_empty:
        return None
    # Ensure polygonal
    if isinstance(g2, (Polygon, MultiPolygon)):
        return g2
    # Sometimes make_valid returns GeometryCollection. Extract polygonal parts.
    try:
        polys = []
        for part in getattr(g2, "geoms", []):
            if isinstance(part, (Polygon, MultiPolygon)):
                polys.extend(list(part.geoms) if isinstance(part, MultiPolygon) else [part])
        if len(polys) == 0:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    except Exception:
        return None


def clean_building_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop non-polygonal and invalid features, attempt to fix invalid ones, and ensure closed outlines."""
    print("Cleaning step started. Ensuring geometries are valid and polygonal.")
    before = len(gdf)
    print(f"Total geometries to validate: {before}")
    # Try to fix invalids and drop non-polygonal
    fixed = []

    import sys
    sys.stdout.write("Validation Progress (# is 5%): |")
    sys.stdout.flush()
    next_percent_idx = 1  # 1..20, each is 5%

    for idx, g in enumerate(gdf.geometry, 1):
        if g is None or g.is_empty:
            fixed.append(None)
        else:
            g2 = _make_valid_safe(g)
            fixed.append(g2)

        progress_ratio = idx / before if before > 0 else 1.0
        while next_percent_idx <= 20 and progress_ratio >= (next_percent_idx / 20.0):
            sys.stdout.write("#")
            if next_percent_idx % 4 == 0:
                sys.stdout.write(f"|{next_percent_idx * 5}%|")
            sys.stdout.flush()
            next_percent_idx += 1

    if next_percent_idx <= 20:
        while next_percent_idx <= 20:
            sys.stdout.write("#")
            if next_percent_idx % 4 == 0:
                sys.stdout.write(f"|{next_percent_idx * 5}%|")
            next_percent_idx += 1
        sys.stdout.flush()

    sys.stdout.write("|Finished|\n")
    sys.stdout.flush()

    gdf2 = gdf.copy()
    gdf2["geometry"] = fixed
    gdf2 = gdf2[gdf2.geometry.notna() & ~gdf2.geometry.is_empty]
    # Keep only polygonal
    gdf2 = gdf2[gdf2.geometry.type.isin(["Polygon", "MultiPolygon"])]
    after_fix = len(gdf2)
    dropped = before - after_fix
    print(f"Cleaning step completed. Dropped features that are non-polygonal or unrecoverable. Count dropped: {dropped}. Remaining: {after_fix}")
    return gdf2

def remove_small_interior_rings_projected(
    gdf: gpd.GeoDataFrame, min_hole_area_m2: float
) -> Tuple[gpd.GeoDataFrame, int]:
    """
    Remove interior rings (holes) with area smaller than min_hole_area_m2.
    Input GeoDataFrame must be in a projected CRS with meters as units.
    Returns (updated_gdf, removed_ring_count).
    """
    removed = 0
    new_geoms = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            new_geoms.append(geom)
            continue

        new_polys = []
        for poly in polys:
            if poly is None or poly.is_empty:
                continue

            keep_interiors = []
            for ring in getattr(poly, "interiors", []):
                try:
                    hole_area = float(Polygon(ring).area)
                except Exception:
                    hole_area = None

                if hole_area is not None and math.isfinite(hole_area) and hole_area < min_hole_area_m2:
                    removed += 1
                    continue

                keep_interiors.append(ring)

            try:
                new_poly = Polygon(poly.exterior, keep_interiors)
            except Exception:
                new_poly = poly

            new_polys.append(new_poly)

        if len(new_polys) == 0:
            new_geoms.append(geom)
        elif len(new_polys) == 1:
            new_geoms.append(new_polys[0])
        else:
            new_geoms.append(MultiPolygon(new_polys))

    out = gdf.copy()
    out["geometry"] = new_geoms
    return out, removed


def _to_float_or_nan(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def union_overlapping_buildings(gdf: gpd.GeoDataFrame, height_col: str | None) -> gpd.GeoDataFrame:
    """
    Merge buildings that have positive-area overlaps.
    Touching along edges without area overlap does not trigger merging.
    Attributes are taken from the first member of each cluster, while height column uses max if available.
    """
    print("Overlap union step started. Building overlap graph.")
    if gdf.empty:
        print("Overlap union skipped because there are no features.")
        return gdf

    gdf = gdf.reset_index(drop=True)
    sindex = gdf.sindex
    n = len(gdf)
    visited = np.zeros(n, dtype=bool)
    components = []
    comp_sizes = []

    # Pre-prepare geometries for faster intersection tests
    prepared_geoms = [prepared.prep(geom) for geom in gdf.geometry]

    import sys
    print(f"Total seeds to check for overlap components: {n}")
    sys.stdout.write("Component discovery progress (# is 5%): |")
    sys.stdout.flush()
    next_percent_idx = 1  # 1..20, each is 5 percent

    for i in range(n):
        if visited[i]:
            # still need to update progress
            pass
        else:
            # BFS to collect indices that positively overlap with i
            queue = [i]
            visited[i] = True
            comp = [i]
            while queue:
                a = queue.pop()
                ga = gdf.geometry.iloc[a]
                cand_idx = list(sindex.intersection(ga.bounds))
                for j in cand_idx:
                    if j == a or visited[j]:
                        continue
                    gj = gdf.geometry.iloc[j]
                    if not prepared_geoms[a].intersects(gj):
                        continue
                    try:
                        inter_area = ga.intersection(gj).area
                    except Exception:
                        inter_area = 0.0
                    if inter_area > 0.0:
                        visited[j] = True
                        queue.append(j)
                        comp.append(j)
            components.append(comp)
            comp_sizes.append(len(comp))

        progress_ratio = (i + 1) / n if n > 0 else 1.0
        while next_percent_idx <= 20 and progress_ratio >= (next_percent_idx / 20.0):
            sys.stdout.write("#")
            if next_percent_idx % 4 == 0:
                sys.stdout.write(f"|{next_percent_idx * 5}%|")
            sys.stdout.flush()
            next_percent_idx += 1

    if next_percent_idx <= 20:
        while next_percent_idx <= 20:
            sys.stdout.write("#")
            if next_percent_idx % 4 == 0:
                sys.stdout.write(f"|{next_percent_idx * 5}%|")
            next_percent_idx += 1
        sys.stdout.flush()
    sys.stdout.write("|Finished|\n")
    sys.stdout.flush()


    merged_rows = []
    merged_count = 0
    for idxs in components:
        geoms = list(gdf.geometry.iloc[idxs].values)
        if len(geoms) == 1:
            # Single, keep as is
            row = gdf.iloc[idxs[0]].copy()
            merged_rows.append(row)
            continue
        # Merge geometries in this overlapping cluster
        try:
            ug = unary_union(geoms)
        except Exception:
            # Fallback merge loop
            ug = geoms[0]
            for gg in geoms[1:]:
                try:
                    ug = ug.union(gg)
                except Exception:
                    pass

        # Build output row. Start from the first member attributes.
        base = gdf.iloc[idxs[0]].copy()
        base.geometry = ug

        # Aggregate height by max of numeric values if present
        if height_col is not None and height_col in gdf.columns:
            vals = pd.to_numeric(gdf.loc[idxs, height_col], errors="coerce")
            if vals.notna().any():
                base[height_col] = float(vals.max())
            else:
                base[height_col] = np.nan

        # Corner marker should be absent in merged real buildings
        if "corner_id" in base.index:
            base["corner_id"] = pd.NA

        merged_rows.append(base)
        merged_count += 1

    out = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=gdf.crs)
    print(f"Overlap union step completed. Components found: {len(components)}. Merged components with size greater than one: {merged_count}. Output features: {len(out)}")
    return out


def save_outline_preview(gdf: gpd.GeoDataFrame, case_name: str, shp_path: Path, output_dir: Path, bbox_wgs84) -> None:
    print("Preview step started. Preparing map outline and info panel.")
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    min_lon, min_lat, max_lon, max_lat = gdf_wgs84.total_bounds

    if (gdf.crs is None) or (not gdf.crs.is_projected):
        lon_center = 0.5 * (min_lon + max_lon)
        lat_center = 0.5 * (min_lat + max_lat)
        utm_crs = get_utm_epsg_from_lonlat(lon_center, lat_center)
        print(f"Preview step uses auto detected UTM CRS for plotting: {utm_crs}.")
        gdf_plot = gdf_wgs84.to_crs(utm_crs)
    else:
        print(f"Preview step uses projected CRS: {gdf.crs}")
        gdf_plot = gdf


    height_col = select_height_column(gdf_plot)
    if height_col is not None:
        print(f"Preview step detected height column: {height_col}")
    else:
        print("Preview step did not detect a height column. Only area threshold will be used for styling.")

    fig, (ax_map, ax_info) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [4, 1]}
    )

    line_width = 0.125
    areas = gdf_plot.geometry.area
    if height_col is not None:
        extruded_mask = (
            gdf_plot[height_col].notna()
            & (pd.to_numeric(gdf_plot[height_col], errors="coerce") > 0)
            & (areas >= MIN_EXTRUDE_AREA)
        )
    else:
        extruded_mask = areas >= MIN_EXTRUDE_AREA

    non_extruded = ~extruded_mask

    # Random color per polygon, alpha = 0.3
    if len(gdf_plot) > 0:
        rng = np.random.default_rng()
        rgb = rng.random((len(gdf_plot), 3))
        rgba = [(float(r), float(g), float(b), 0.3) for r, g, b in rgb]
        gdf_plot.plot(
            ax=ax_map,
            color=rgba,
            edgecolor="black",
            linewidth=line_width,
        )


    bbox_plot = gpd.GeoSeries([bbox_wgs84], crs="EPSG:4326").to_crs(gdf_plot.crs)
    bbox_plot.boundary.plot(ax=ax_map, linewidth=0.3, color="orange")

    if height_col is not None:
        for _, row in gdf_plot.iterrows():
            h_raw = row.get(height_col)
            try:
                h = float(h_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(h) or h <= 0:
                continue
            px, py = row.geometry.representative_point().coords[0]
            area = row.geometry.area
            ax_map.text(px, py, f"{h:.1f}", fontsize=0.1, ha="center", va="bottom", color="red", alpha=0.6)
            ax_map.text(px, py, f"{area:.1f}", fontsize=0.1, ha="center", va="top", color="green", alpha=0.6)
    # Hole (interior ring) validation and marking
    hole_areas = []
    hole_count = 0

    for geom in gdf_plot.geometry:
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            interiors = list(getattr(poly, "interiors", []))
            if len(interiors) == 0:
                continue

            for ring in interiors:
                try:
                    hole_poly = Polygon(ring)
                except Exception:
                    continue
                if hole_poly.is_empty:
                    continue

                try:
                    a = float(hole_poly.area)
                except Exception:
                    continue
                if not math.isfinite(a):
                    continue

                hole_areas.append(a)
                hole_count += 1

                try:
                    rp = hole_poly.representative_point()
                    ax_map.text(
                        rp.x,
                        rp.y,
                        f"H:{a:.1f}",
                        fontsize=0.1,
                        ha="center",
                        va="center",
                        color="blue",
                        alpha=0.6,
                    )
                except Exception:
                    pass

                try:
                    xs, ys = ring.xy
                    ax_map.plot(xs, ys, linewidth=0.05, color="blue", alpha=0.6)
                except Exception:
                    pass

    hole_area_sum = float(sum(hole_areas)) if len(hole_areas) > 0 else 0.0
    if hole_count > 0:
        hole_min = float(min(hole_areas))
        hole_max = float(max(hole_areas))
        print(
            "Preview step hole validation: "
            f"holes={hole_count}, area_sum={hole_area_sum:.3f}, area_min={hole_min:.3f}, area_max={hole_max:.3f} (square meters)"
        )
    else:
        print("Preview step hole validation: no interior rings detected.")

    ax_map.set_aspect("equal")
    if gdf_plot.empty:
        ax_map.text(0.5, 0.5, "No geometry in slice", transform=ax_map.transAxes, ha="center", va="center", fontsize=6)

    ax_map.axis("off")

    ax_info.axis("off")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        f"Time: {now}",
        f"caseName: {case_name}",
        f"lon range: [{min_lon:.6f}, {max_lon:.6f}]",
        f"lat range: [{min_lat:.6f}, {max_lat:.6f}]",
        f"holes: {hole_count}",
        f"hole area sum (m^2): {hole_area_sum:.1f}",
        f"shp: {shp_path.name}",
    ]
    ax_info.text(0.01, 0.99, "\n".join(info_lines), va="top", ha="left", fontsize=6)

    plt.tight_layout()
    out_img = output_dir / f"{case_name}_preview.jpg"
    fig.savefig(out_img, dpi=4800)
    plt.close(fig)
    print(f"Preview step completed. Saved preview image to {out_img}")

# ----------------------- Main -----------------------

def main():
    print("Program started.")
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_ts}")

    if len(sys.argv) < 2:
        print("Usage: python 2_shpCutter.py <path-to-conf-file> [--preview] [--validate]")
        sys.exit(2)
    conf_file = Path(sys.argv[1]).expanduser().resolve()
    save_preview = False
    do_double_validation = False
    for arg in sys.argv[2:]:
        a = arg.strip().lower()
        if a in ("--preview", "preview", "1", "true", "yes"):
            save_preview = True
        if a in ("--validate", "validate"):
            do_double_validation = True

    if not conf_file.exists():
        raise FileNotFoundError(f"conf file not found: {conf_file}")
    project_home = conf_file.parent

    # 读取 casename
    txt_conf = conf_file.read_text(encoding="utf-8", errors="ignore")
    m_case = re.search(r"casename\s*=\s*([^\s]+)", txt_conf)
    if not m_case:
        raise RuntimeError("casename not found in conf")
    case_name_base = m_case.group(1)

    # 读取裁剪范围，使用 manual 项
    cut_lon, cut_lat = get_lonlat(conf_file)
    lon_lo, lon_hi = min(cut_lon), max(cut_lon)
    lat_lo, lat_hi = min(cut_lat), max(cut_lat)

    # 输入 Shapefile 搜索。要求 ProjectHome/building_db 下唯一一个 .shp
    build_dir = project_home / "building_db"
    shp_list = sorted(build_dir.glob("*.shp"))
    if len(shp_list) != 1:
        raise FileNotFoundError(f"Expected exactly one .shp under {build_dir}, found {len(shp_list)}")
    in_path = shp_list[0]

    # 输出目录与文件名。全部写入 ProjectHome/proj_temp。基名为 casename_lat_lon
    out_dir = project_home / "proj_temp/cutted_shp"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"{case_name_base}"
    out_path = out_dir / f"{case_name}.shp"

    print(f"Input path: {in_path}")
    print(f"Output directory: {out_dir}")
    print(f"Output shapefile: {out_path}")
    print(f"Case name: {case_name}")

    cut_lon, cut_lat = get_lonlat(conf_file)
    print(f"Bbox lon range: {cut_lon}")
    print(f"Bbox lat range: {cut_lat}")

    print("Reading input shapefile.")
    gdf = gpd.read_file(in_path)
    print(f"Read completed. Feature count: {len(gdf)}")

    if gdf.crs is None:
        print("Notice: source SHP has no CRS. Assuming EPSG 4326 WGS84 geographic.")
        gdf = gdf.set_crs(epsg=4326)

    orig_crs = gdf.crs
    print(f"Original CRS: {orig_crs}")

    if orig_crs.to_epsg() != 4326:
        print("Reprojecting to EPSG 4326 for bbox based clipping.")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        print("Reprojection to EPSG 4326 completed.")
    else:
        gdf_wgs84 = gdf
        print("Data already in EPSG 4326. No reprojection needed.")

    # Validate coverage before clipping
    try:
        ib_minx, ib_miny, ib_maxx, ib_maxy = gdf_wgs84.total_bounds
        input_bounds = (float(ib_minx), float(ib_maxx), float(ib_miny), float(ib_maxy))
        target_bounds = (lon_lo, lon_hi, lat_lo, lat_hi)
        _confirm_bbox_coverage("Building SHP", target_bounds, input_bounds)
    except Exception as e:
        print(f"[WARN] Failed to validate building SHP bounds: {e}")

    minx, maxx = min(cut_lon), max(cut_lon)
    miny, maxy = min(cut_lat), max(cut_lat)
    bbox = box(minx, miny, maxx, maxy)
    print(f"Bbox constructed. MinLon: {minx}, MinLat: {miny}, MaxLon: {maxx}, MaxLat: {maxy}")

    utm_crs = get_utm_crs_from_bounds((minx, maxx), (miny, maxy))
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    x00, y00 = transformer.transform(minx, miny)
    x10, y10 = transformer.transform(maxx, miny)
    x11, y11 = transformer.transform(maxx, maxy)
    x01, y01 = transformer.transform(minx, maxy)

    cx = (x00 + x10 + x11 + x01) / 4.0
    cy = (y00 + y10 + y11 + y01) / 4.0

    dx0 = x10 - x00
    dy0 = y10 - y00
    angle_rad = math.atan2(dy0, dx0)
    rotate_deg = - math.degrees(angle_rad)

    def _rot_xy(x, y, deg, cx, cy):
        th = math.radians(deg)
        c = math.cos(th)
        s = math.sin(th)
        xr = c * (x - cx) - s * (y - cy) + cx
        yr = s * (x - cx) + c * (y - cy) + cy
        return xr, yr

    xr00, yr00 = _rot_xy(x00, y00, rotate_deg, cx, cy)
    xr10, yr10 = _rot_xy(x10, y10, rotate_deg, cx, cy)
    xr11, yr11 = _rot_xy(x11, y11, rotate_deg, cx, cy)
    xr01, yr01 = _rot_xy(x01, y01, rotate_deg, cx, cy)

    x_min_rot = min(xr00, xr10, xr11, xr01)
    x_max_rot = max(xr00, xr10, xr11, xr01)
    y_min_rot = min(yr00, yr10, yr11, yr01)
    y_max_rot = max(yr00, yr10, yr11, yr01)

    bbox_rot = box(x_min_rot, y_min_rot, x_max_rot, y_max_rot)
    bbox_utm = affinity.rotate(bbox_rot, -rotate_deg, origin=(cx, cy), use_radians=False)

    print(f"Clipping to rotated-rectangle bbox (four-corner projected). rotate_deg={rotate_deg:.6f}, pivot=({cx:.3f}, {cy:.3f})")
    gdf_wgs84_cleaned = clean_building_geometries(gdf_wgs84)
    gdf_utm = gdf_wgs84_cleaned.to_crs(utm_crs)
    clipped_utm = gpd.clip(gdf_utm, bbox_utm)
    clipped_wgs84 = clipped_utm.to_crs(epsg=4326)

    clipped_wgs84 = clipped_wgs84[clipped_wgs84.geometry.notna() & ~clipped_wgs84.geometry.is_empty]
    print(f"Clipping completed. Features after clip: {len(clipped_wgs84)}")

    # Data cleaning before area filtering
    if do_double_validation:
        print("Starting geometry cleaning before area filtering.")
        cleaned = clean_building_geometries(clipped_wgs84)
        print(f"Geometry cleaning finished. Features after cleaning: {len(cleaned)}")
    else:
        cleaned = clipped_wgs84
        print("Geometry cleaning skipped because --validate not provided.")


    # Overlap union
    height_col = select_height_column(cleaned) or "elevation"
    if height_col not in cleaned.columns:
        cleaned[height_col] = pd.NA
    print(f"Height column for processing: {height_col}")

    print("Starting overlap union on cleaned geometries.")
    merged = union_overlapping_buildings(cleaned, height_col=height_col)
    print(f"Overlap union finished. Features after union: {len(merged)}")

    # Area filter in meters
    print(f"Area filter step started. Minimum area: {MIN_EXTRUDE_AREA} square meters.")
    geoms = list(merged.geometry.values)
    if len(geoms) > 0:
        mask_area_list = _area_mask_chunk(geoms, MIN_EXTRUDE_AREA)
        mask_area = pd.Series(mask_area_list, index=merged.index)
        kept = int(mask_area.sum())
        print(f"Area filter step completed. Kept features: {kept}")
    else:
        mask_area = pd.Series([], dtype=bool, index=merged.index)
        print("Area filter skipped because there are no geometries after union.")
    merged = merged.loc[mask_area].copy()
    print(f"Features after area filter: {len(merged)}")
    print(f"Small interior rings removal started. Threshold: {MIN_HOLE_AREA} m^2.")
    removed_holes = 0
    if not merged.empty:
        merged_utm2 = merged.to_crs(utm_crs)
        merged_utm2, removed_holes = remove_small_interior_rings_projected(merged_utm2, MIN_HOLE_AREA)
        merged = merged_utm2.to_crs(epsg=4326)
    print(f"Small interior rings removed: {removed_holes}")
    print(f"Features after hole removal: {len(merged)}")

    print("Adding four 1m by 1m corner squares inside the bbox.")

    d = 1.0
    half = 0.5

    bbox_centroid = bbox_utm.centroid
    bbox_coords = list(bbox_utm.exterior.coords)
    if len(bbox_coords) >= 2 and bbox_coords[0] == bbox_coords[-1]:
        bbox_coords = bbox_coords[:-1]

    if len(bbox_coords) >= 4:
        corners = bbox_coords[:4]
    else:
        minx_m, miny_m, maxx_m, maxy_m = bbox_utm.bounds
        corners = [(minx_m, miny_m), (maxx_m, miny_m), (maxx_m, maxy_m), (minx_m, maxy_m)]

    corner_centers = []
    for (x, y) in corners:
        vx = bbox_centroid.x - x
        vy = bbox_centroid.y - y
        n = math.hypot(vx, vy)
        if n == 0.0:
            cx, cy = x, y
        else:
            cx = x + (vx / n) * d
            cy = y + (vy / n) * d
        corner_centers.append((cx, cy))

    corner_boxes_utm = [box(cx - half, cy - half, cx + half, cy + half) for cx, cy in corner_centers]

    corner_gdf_utm = gpd.GeoDataFrame(
        {height_col: [1.0, 1.0, 1.0, 1.0], "corner_id": [1, 2, 3, 4]},
        geometry=corner_boxes_utm,
        crs=utm_crs,
    )
    corner_gdf_wgs84 = corner_gdf_utm.to_crs(epsg=4326)
    print("Corner squares prepared and reprojected to EPSG 4326.")


    result_wgs84 = pd.concat([merged, corner_gdf_wgs84], ignore_index=True, sort=False)
    print(f"Corner squares appended. Total features now: {len(result_wgs84)}")

    # Back to original CRS and write
    if orig_crs.to_epsg() != 4326:
        print(f"Reprojecting result back to original CRS: {orig_crs}")
        subset = result_wgs84.to_crs(orig_crs)
        print("Reprojection to original CRS completed.")
    else:
        subset = result_wgs84
        print("Original CRS is EPSG 4326. No final reprojection needed.")

    print("Writing output shapefile.")
    subset.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
    print("Write completed.")

    print(f"Processing finished. Feature count written: {len(subset)}")
    print(f"Output file path: {out_path.resolve()}")
    if len(subset.columns) > 0:
        print("Output fields example:")
        print(list(subset.columns))

    if save_preview:
        print("Generating preview image.")
        save_outline_preview(
            gdf=subset,
            case_name=case_name,
            shp_path=in_path,
            output_dir=out_dir,
            bbox_wgs84=bbox,
        )
        print("Preview generation finished.")
    else:
        print("Preview generation skipped. Use --preview to enable.")


    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Program finished. End time: {end_ts}. Exiting now.")
    sys.exit(0)


if __name__ == "__main__":
    main()

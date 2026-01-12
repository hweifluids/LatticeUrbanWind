#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional



def utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def choose_metric_crs(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    if gdf.crs is None:
        raise ValueError("Input data has no CRS. Please set a CRS before measuring lengths.")

    try:
        is_geographic = bool(getattr(gdf.crs, "is_geographic", False))
    except Exception:
        is_geographic = False

    if is_geographic:
        centroid = gdf.unary_union.centroid
        epsg = utm_epsg_from_lon_lat(float(centroid.x), float(centroid.y))
        return gdf.to_crs(epsg=epsg), f"Reprojected from geographic CRS to EPSG:{epsg} for metric lengths."

    return gdf, "Using existing projected CRS for metric lengths."


def short_side_length_from_geom(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan

    gtype = getattr(geom, "geom_type", "")
    if gtype == "MultiPolygon":
        parts = list(getattr(geom, "geoms", []))
        if not parts:
            return np.nan
        geom = max(parts, key=lambda g: g.area)

    if getattr(geom, "geom_type", "") != "Polygon":
        return np.nan

    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 5:
        return np.nan

    edges = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        edges.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    return float(np.min(edges)) if edges else np.nan

def footprint_area_from_geom(geom) -> float:
    if geom is None or geom.is_empty:
        return np.nan

    gtype = getattr(geom, "geom_type", "")
    if gtype == "MultiPolygon":
        parts = list(getattr(geom, "geoms", []))
        if not parts:
            return np.nan
        geom = max(parts, key=lambda g: g.area)

    if getattr(geom, "geom_type", "") != "Polygon":
        return np.nan

    return float(geom.area)


def pick_height_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    cols = list(gdf.columns)

    exact_candidates = {
        "height",
        "hgt",
        "building_height",
        "bldg_height",
        "bldg_h",
        "bldgheight",
        "roof_height",
        "roof_h",
        "eave_height",
        "z",
    }
    for c in cols:
        if str(c).lower() in exact_candidates:
            s = pd.to_numeric(gdf[c], errors="coerce")
            if np.isfinite(s.to_numpy(dtype=float)).any():
                return c

    for c in cols:
        cl = str(c).lower()
        if ("height" in cl) or ("hgt" in cl):
            s = pd.to_numeric(gdf[c], errors="coerce")
            if np.isfinite(s.to_numpy(dtype=float)).any():
                return c

    return None


def plot_pdf_and_cdf(values: np.ndarray, weights: np.ndarray, out_path: Path) -> None:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    total_w = float(np.sum(weights))

    order_desc = np.argsort(values)[::-1]
    x_desc = values[order_desc]
    w_desc = weights[order_desc]
    cdf_y = np.cumsum(w_desc) / total_w

    bin_edges = np.histogram_bin_edges(values, bins="fd")
    w_hist, edges = np.histogram(values, bins=bin_edges, weights=weights, density=False)
    bin_widths = np.diff(edges)
    pdf = w_hist / (total_w * bin_widths)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig, ax_pdf = plt.subplots(figsize=(10, 3))
    ax_cdf = ax_pdf.twinx()

    ax_pdf.plot(
        centers,
        pdf,
        color="#2b78b1",
        label="Probability Density Function ",
    )
    ax_cdf.plot(
        x_desc,
        cdf_y,
        color="#f9812c",
        label="Cumulative Distribution Function ",
    )

    for v in (2.0, 5.0, 10.0, 20.0, 50.0):
        ax_pdf.axvline(v, color="#404040", linestyle="--", linewidth=0.8)

    ax_pdf.set_xlabel("Short-side length (m)")
    ax_pdf.set_ylabel("Probability Density Function ")
    ax_cdf.set_ylabel("Cumulative Distribution Function ")

    ax_pdf.set_xlim(float(np.max(values)), 0.0)

    ax_pdf.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    h1, l1 = ax_pdf.get_legend_handles_labels()
    h2, l2 = ax_cdf.get_legend_handles_labels()
    ax_pdf.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot empirical CDF of building short-side lengths from a shapefile."
    )
    parser.add_argument("shp_path", help="Path to the input .shp file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (PNG). Default: <shp_dir>/<shp_stem>_short_side_cdf.png",
    )
    args = parser.parse_args()

    shp_path = Path(args.shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Input shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(gdf) == 0:
        raise ValueError("No Polygon/MultiPolygon geometries found in the shapefile.")

    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass

    gdf, crs_msg = choose_metric_crs(gdf)
    print(crs_msg)

    lengths = np.array([short_side_length_from_geom(geom) for geom in gdf.geometry], dtype=float)
    areas = np.array([footprint_area_from_geom(geom) for geom in gdf.geometry], dtype=float)

    height_col = pick_height_column(gdf)
    if height_col is None:
        heights = np.ones(len(gdf), dtype=float)
        print("Height field: not found, using 1.0 for all features.")
    else:
        heights = pd.to_numeric(gdf[height_col], errors="coerce").to_numpy(dtype=float)
        print(f"Height field: {height_col}")

    weights = areas * heights

    mask = np.isfinite(lengths) & np.isfinite(weights) & (lengths > 0.0) & (weights > 0.0)
    lengths = lengths[mask]
    weights = weights[mask]

    if lengths.size == 0:
        raise ValueError("No valid weighted short-side lengths computed (check geometries, height field, and CRS).")


    out_path = Path(args.out) if args.out else (shp_path.parent / f"{shp_path.stem}_short_side_cdf.png")
    plot_pdf_and_cdf(lengths, weights, out_path)



    print(f"Output saved to: {out_path}")
    print(f"Count: {lengths.size}")
    print(f"Min/Median/Max (m): {lengths.min():.3f} / {np.median(lengths):.3f} / {lengths.max():.3f}")


if __name__ == "__main__":
    main()

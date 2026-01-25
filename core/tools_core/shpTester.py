#!/usr/bin/env python3
import sys
from collections import Counter

import geopandas as gpd
from shapely.validation import explain_validity


def _ring_is_closed(coords):
    if not coords:
        return False
    return coords[0] == coords[-1]


def _polygon_part_is_degenerate(poly):
    # poly: shapely Polygon
    if poly.is_empty:
        return True, "empty"
    try:
        ext = list(poly.exterior.coords) if poly.exterior is not None else []
    except Exception:
        ext = []
    if len(ext) < 4:
        return True, "too_few_points"
    if not _ring_is_closed(ext):
        return True, "ring_not_closed"
    if poly.area <= 0:
        return True, "zero_area"
    if not poly.is_valid:
        return True, f"invalid: {explain_validity(poly)}"
    return False, "ok"


def is_degenerate_geometry(geom):
    if geom is None:
        return True, "null"
    if geom.is_empty:
        return True, "empty"

    gtype = geom.geom_type

    if gtype == "Polygon":
        return _polygon_part_is_degenerate(geom)

    if gtype == "MultiPolygon":
        if len(geom.geoms) == 0:
            return True, "empty_multipolygon"
        for part in geom.geoms:
            bad, reason = _polygon_part_is_degenerate(part)
            if bad:
                return True, f"multipart: {reason}"
        if geom.area <= 0:
            return True, "zero_area"
        if not geom.is_valid:
            return True, f"invalid: {explain_validity(geom)}"
        return False, "ok"

    return True, f"not_polygon_type: {gtype}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_degenerate_shp.py <path_to_shp>", file=sys.stderr)
        sys.exit(2)

    shp_path = sys.argv[1]

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"Failed to read shapefile: {e}", file=sys.stderr)
        sys.exit(1)

    if "geometry" not in gdf.columns:
        print("No geometry column found.", file=sys.stderr)
        sys.exit(1)

    total = len(gdf)
    degenerate = 0
    reasons = Counter()

    for geom in gdf.geometry:
        bad, reason = is_degenerate_geometry(geom)
        if bad:
            degenerate += 1
            reasons[reason] += 1

    print(f"Total features: {total}")
    print(f"Degenerate features: {degenerate}")

    if degenerate > 0:
        for reason, cnt in reasons.most_common():
            print(f"{cnt}\t{reason}")

    sys.exit(0)


if __name__ == "__main__":
    main()

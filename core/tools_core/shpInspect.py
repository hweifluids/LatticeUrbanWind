"""
shpInspect.py
A self-contained CLI tool that inspects a target shapefile and reports CRS, bounds, and basic metadata.
"""

import os
import sys
import glob
import json
import warnings
import geopandas as gpd
from pyproj import CRS


def resolve_shp_path(input_path: str) -> str:
    """
    Resolve the actual shapefile path according to the rules described above.
    Returns the resolved .shp path if successful, otherwise raises RuntimeError.
    """
    # Normalize the input path to an absolute path for clarity in messages
    input_path = os.path.abspath(input_path)

    # Case A: the input is itself a .shp path
    if input_path.lower().endswith(".shp"):
        if os.path.isfile(input_path):
            return input_path
        raise RuntimeError(f"Input .shp does not exist: {input_path}")

    # Case B: the input has a non-.shp extension
    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")

    parent_dir = os.path.dirname(input_path)
    db_dir = os.path.join(parent_dir, "building_db")

    if not os.path.isdir(db_dir):
        raise RuntimeError(
            f"Derived folder 'building_db' does not exist: {db_dir}"
        )

    shp_candidates = sorted(glob.glob(os.path.join(db_dir, "*.shp")))
    if len(shp_candidates) == 0:
        raise RuntimeError(
            f"No .shp files found in folder: {db_dir}"
        )
    if len(shp_candidates) > 1:
        warnings.warn(
            "Multiple .shp files found under 'building_db'. "
            "The first shapefile in alphabetical order will be used."
        )
        print("All detected shapefiles:")
        for p in shp_candidates:
            print(f"  {os.path.basename(p)}")

    chosen = shp_candidates[0]
    print(f"Using shapefile: {chosen}")
    return chosen


def inspect_shapefile(shp_path: str) -> None:
    """
    Load the shapefile, ensure EPSG:4326, and print CRS details, bounds, and basic metadata.
    """
    print("========================================")
    print("Loading shapefile")
    print(f"Path: {shp_path}")
    print("========================================")

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read shapefile: {e}")

    print("\nCRS full details")
    print("========================================")
    if gdf.crs is None:
        print("CRS: None")
    else:
        try:
            crs = CRS.from_user_input(gdf.crs)
            print(json.dumps(crs.to_json_dict(), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Failed to parse CRS: {e}")

    # Harmonize to WGS 84
    if gdf.crs is None:
        print("\nShapefile has no CRS information. Assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)
    else:
        try:
            if gdf.crs.to_epsg() != 4326:
                print(f"\nReprojecting to EPSG:4326 from {gdf.crs}.")
                gdf = gdf.to_crs(epsg=4326)
        except Exception as e:
            print(f"\nCould not evaluate CRS EPSG code. Proceeding without reprojection. Details: {e}")

    # Bounds
    print("\nRange of longitude and latitude")
    print("========================================")
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        print(f"Longitude (X): {minx:.6f} to {maxx:.6f}")
        print(f"Latitude  (Y): {miny:.6f} to {maxy:.6f}")
    except Exception as e:
        print(f"Failed to compute bounds: {e}")

    # Basic metadata
    print("\nMeta data")
    print("========================================")
    try:
        print("Columns:")
        print(list(gdf.columns))
        print("\nDtypes:")
        print(gdf.dtypes)
        print("\nFirst 5 rows:")
        print(gdf.head())
    except Exception as e:
        print(f"Failed to print metadata: {e}")

def main():
    # Enforce exactly one argument which is the input file path
    if len(sys.argv) != 2:
        script = os.path.basename(sys.argv[0])
        print(f"Usage: python {script} <input_file_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    try:
        shp_path = resolve_shp_path(input_path)
        inspect_shapefile(shp_path)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

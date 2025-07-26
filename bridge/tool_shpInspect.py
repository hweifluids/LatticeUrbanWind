"""
tool_shpInspect.py
"""
import os
import glob
import sys
import geopandas as gpd

def main():
    # 1) Prompt user to input the case name
    if len(sys.argv) == 2:              # input parameter: python load_shp.py my_case
        case_name = sys.argv[1]
    elif len(sys.argv) == 1:            # no parameter, send to interaction
        case_name = input("Please enter the case name: ")
    else:                               # too many params, exit
        script = os.path.basename(sys.argv[0])
        print(f"Usage: {script} [case_name]")
        sys.exit(1)

    # 2) Construct the target folder path: sibling directory geoData/case_name
    base_dir = os.getcwd()
    shp_dir = os.path.join(base_dir, 'geoData', case_name)

    # 3) Search for .shp files
    shp_files = glob.glob(os.path.join(shp_dir, '*.shp'))

    # 4) Provide warnings or load based on the results
    if not os.path.isdir(shp_dir):
        print(f"Folder does not exist: {shp_dir}")
    elif len(shp_files) == 0:
        print(f"No .shp files found in folder: {shp_dir}")
    elif len(shp_files) > 1:
        print(f"Multiple .shp files found in folder: {shp_dir}")
        for shp in shp_files:
            print(f"  - {os.path.basename(shp)}")
        print("Please ensure there is exactly one .shp file in the folder.")
    else:
        shp_path = shp_files[0]
        print(f"Loading shapefile: {shp_path}")
        
        # Read the shapefile
        gdf = gpd.read_file(shp_path)

        # === keep WGS-84 (EPSG:4326) ============================
        if gdf.crs is None:
            print("  Shapefile has no CRS information; assuming EPSG:4326.")
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            print(f"Re-projecting from {gdf.crs} to EPSG:4326 (WGS-84)...")
            gdf = gdf.to_crs(epsg=4326)
        # ===============================================================

        # === bounds =============================================
        # total_bounds = [minx, miny, maxx, maxy]
        minx, miny, maxx, maxy = gdf.total_bounds

        print("\n-------  Range of longitude and latitude  -------")
        print(f"Longitude (X) : {minx:.6f} - {maxx:.6f}")
        print(f"Latitude  (Y) : {miny:.6f} - {maxy:.6f}")
        # ===============================================================
        print("\n-------  Meta data  -------")
        print("\nData list: ", list(gdf.columns))
        print("\nData type:")
        print(gdf.dtypes)
        print("\nFirst 5 lines:")
        print(gdf.head())

if __name__ == '__main__':
    main()
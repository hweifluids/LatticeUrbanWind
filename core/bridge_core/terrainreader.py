# terrainreader.py
# Single-file implementation for terrain processing in LatticeUrbanWind.
# Integrates logic from:
#   1. Original dem_tif_to_shp.py (config parsing, bbox expansion, search priority)
#   2. TerrainReader (SRTM tile search, merge, clip, polygonize)
#   3. Terrain Verification (metadata check, optional plot)

from __future__ import annotations

import os
import sys
import re
import ast
import time
import math
import glob
import shutil
import zipfile
import argparse
import traceback
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Optional, Tuple, List

# Third-party libraries
try:
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.io import MemoryFile
    from shapely.geometry import box
    from tqdm import tqdm
    # NOTE: matplotlib is NOT imported here to avoid main-thread GUI issues.
    # It is only imported in the subprocess verify mode.
except ImportError as e:
    print(f"[ERROR] Missing required library for terrain processing: {e}")
    # Don't exit immediately, as some features might still work or we can warn later
    # sys.exit(1)


# =============================================================================
# Part 1: TerrainReader Core Logic (SRTM Processing)
# =============================================================================

def _extract_tile_task(args):
    """
    Worker function for multiprocessing extraction of terrain tile files.

    Args:
        args (tuple): A tuple containing (zip_path, target_file, temp_dir).

    Returns:
        str or None: The absolute path to the extracted file if successful, else None.
    """
    zip_path, target_file, temp_dir = args
    expected_path = os.path.join(temp_dir, target_file)
    
    if os.path.exists(expected_path):
        return expected_path
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(target_file, temp_dir)
        return expected_path
    except Exception as e:
        print(f"[WARN] Error extracting {zip_path}: {e}")
        return None

def _get_tile_name(lat, lon):
    """
    Generate the standard SRTM tile name from latitude and longitude.

    Args:
        lat (float): Latitude value.
        lon (float): Longitude value.

    Returns:
        str: Tile name string (e.g., 'N30E120').
    """
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{ns}{abs(int(math.floor(lat))):02d}{ew}{abs(int(math.floor(lon))):03d}"

def _find_tile_zip(tile_name, search_dir):
    """
    Recursively search for a tile's zip file in the specified directory.
    Prioritizes .hgt.zip files (SRTM) if multiple matches are found.

    Args:
        tile_name (str): The name of the tile to search for (e.g., 'N30E120').
        search_dir (str): The root directory to search within.

    Returns:
        str or None: Path to the found zip file, or None if not found.
    """
    pattern = os.path.join(search_dir, "**", f"*{tile_name}*.zip")
    files = glob.glob(pattern, recursive=True)
    # Filter out Mac OS metadata files (._*)
    files = [f for f in files if not os.path.basename(f).startswith('._')]
    
    if not files:
        return None

    # Prioritize .hgt.zip files (elevation data) over others
    hgt_files = [f for f in files if '.hgt.zip' in f.lower()]
    if hgt_files:
        return hgt_files[0]
        
    return files[0]

def _format_coord_val(val, is_lat=True):
    """
    Format a coordinate value into a string with direction prefix (N/S/E/W).

    Args:
        val (float): The coordinate value.
        is_lat (bool): True if latitude, False if longitude.

    Returns:
        str: Formatted string (e.g., 'N30', 'E120.50').
    """
    direction = ''
    if is_lat:
        direction = 'N' if val >= 0 else 'S'
    else:
        direction = 'E' if val >= 0 else 'W'
    
    abs_val = abs(val)
    if abs_val.is_integer():
        return f"{direction}{int(abs_val)}"
    else:
        return f"{direction}{abs_val:.2f}"

def process_terrain(min_lon, max_lon, min_lat, max_lat, output_shp, input_file=None, search_dir=None, step=1, geom_type='polygon'):
    """
    Core engine to generate a terrain Shapefile from either a specific input file or by searching SRTM tiles.
    Performs extraction, merging, clipping, polygonization, and saving.

    Args:
        min_lon (float): Minimum longitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        min_lat (float): Minimum latitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        output_shp (str): Destination path for the generated Shapefile.
        input_file (str, optional): Specific input raster file (TIF/TIFF). If provided, tile search is skipped.
        search_dir (str, optional): Directory to search for SRTM tiles if input_file is not provided.
        step (int): Sampling step size for pixel reduction (default: 1).
        geom_type (str): Geometry type for output, 'polygon' or 'point' (default: 'polygon').

    Returns:
        bool: True if processing succeeded, False otherwise.
    """
    start_time = time.time()
    
    # Setup directories
    output_dir = os.path.dirname(output_shp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    temp_dir = os.path.join(output_dir, "temp_tiles")
    
    # Default search dir
    if not search_dir and not input_file:
        search_dir = os.path.join(os.getcwd(), "earthdata")
    
    src_files_to_mosaic = []
    extraction_tasks = []
    
    # 1. Identify Source Files
    if input_file:
        print(f"[INFO] Using input file: {input_file}")
        if not os.path.exists(input_file):
            print(f"[ERROR] File {input_file} not found.")
            return False
        src_files_to_mosaic = [input_file]
    else:
        print(f"[INFO] Searching for tiles in {search_dir}...")
        
        start_lat = math.floor(min_lat)
        end_lat = math.floor(max_lat)
        
        lat_range = range(start_lat, end_lat + 1)
        lon_range = range(math.floor(min_lon), math.floor(max_lon) + 1)
        
        for lat in lat_range:
            for lon in lon_range:
                tile_name = _get_tile_name(lat, lon)
                zip_path = _find_tile_zip(tile_name, search_dir)
                if zip_path:
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            all_files = z.namelist()
                            candidates = [f for f in all_files if f.endswith(('.hgt', '.tif'))]
                            candidates = [f for f in candidates if not os.path.basename(f).startswith('._')]
                            
                            num_files = [f for f in all_files if f.endswith('.num')]
                            if not candidates and num_files:
                                print(f"  [WARN] Found metadata file (.num) but NO elevation data in {zip_path}.")
                                continue

                            if candidates:
                                target_file = candidates[0]
                                extraction_tasks.append((zip_path, target_file, temp_dir))
                            else:
                                print(f"  [WARN] No valid data files found in zip: {zip_path}")
                    except Exception as e:
                        print(f"[WARN] Error reading zip {zip_path}: {e}")
                else:
                    print(f"[WARN] Tile {tile_name} not found.")

        if not extraction_tasks:
            print("[ERROR] No tiles found covering the area.")
            return False

    # 2. Process Files
    steps = ["Extracting", "Reading", "Merging", "Clipping", "Polygonizing", "Geometry", "Saving"]
    pbar = tqdm(total=len(steps), desc="Processing Terrain", unit="step", file=sys.stdout)
    
    try:
        out_image = None
        out_transform = None
        nodata_val = -32768 # Default for SRTM

        if input_file:
            pbar.update(1) # Skip Extracting
            pbar.set_description("Reading & Clipping")
            try:
                bbox = box(min_lon, min_lat, max_lon, max_lat)
                with rasterio.open(src_files_to_mosaic[0]) as src:
                    if src.nodata is not None:
                        nodata_val = src.nodata
                    out_image, out_transform = mask(src, [bbox], crop=True)
                    pbar.update(1) # Reading
                    pbar.update(1) # Merging (skipped)
                    pbar.update(1) # Clipping
            except Exception as e:
                print(f"[ERROR] Reading input file failed: {e}")
                pbar.close()
                return False
        else:
            # Step 1: Extraction
            pbar.set_description("Extracting Files")
            os.makedirs(temp_dir, exist_ok=True)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(_extract_tile_task, extraction_tasks))
            src_files_to_mosaic = [r for r in results if r is not None]
            
            if not src_files_to_mosaic:
                print("[ERROR] Extraction failed.")
                pbar.close()
                return False
            pbar.update(1)

            # Step 2: Reading
            pbar.set_description("Reading Files")
            src_datasets = [rasterio.open(f) for f in src_files_to_mosaic]
            if src_datasets and src_datasets[0].nodata is not None:
                nodata_val = src_datasets[0].nodata
            pbar.update(1)
            
            # Step 3: Merging
            pbar.set_description("Merging Tiles")
            mosaic, out_trans = merge(src_datasets)
            for ds in src_datasets: ds.close()
            pbar.update(1)
            
            # Step 4: Clipping
            pbar.set_description("Clipping Area")
            out_meta = {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": mosaic.shape[0],
                "dtype": mosaic.dtype
            }
            bbox = box(min_lon, min_lat, max_lon, max_lat)
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as dataset:
                    dataset.write(mosaic)
                    out_image, out_transform = mask(dataset, [bbox], crop=True)
            pbar.update(1)
        
        # Step 5: Convert to Points/Polygons
        pbar.set_description("Converting Data")
        data = out_image[0]
        
        if step > 1:
            data = data[::step, ::step]
            rows, cols = np.indices(data.shape)
            rows *= step
            cols *= step
        else:
            rows, cols = np.indices(data.shape)
            
        rows = rows.flatten()
        cols = cols.flatten()
        elevations_flat = data.flatten()
        
        valid_mask = elevations_flat != nodata_val
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        elevs = elevations_flat[valid_mask]
        
        if len(elevs) == 0:
            print("[WARN] No valid elevation points found.")
            pbar.close()
            return False

        pbar.update(1)
        
        # Step 6: Geometry Generation
        pbar.set_description("Generating Geometry")
        xs, ys = rasterio.transform.xy(out_transform, rows, cols, offset='center')
        
        if geom_type == 'polygon':
            pixel_width = out_transform[0] * step
            pixel_height = abs(out_transform[4]) * step
            half_w, half_h = pixel_width / 2, pixel_height / 2
            geometry = [box(x - half_w, y - half_h, x + half_w, y + half_h) for x, y in zip(xs, ys)]
        else:
            geometry = gpd.points_from_xy(xs, ys)
        
        name_str = f"{_format_coord_val(min_lat, True)}{_format_coord_val(min_lon, False)}"
        gdf = gpd.GeoDataFrame({'elevation': elevs, 'city': name_str}, geometry=geometry, crs="EPSG:4326")
        pbar.update(1)
        
        # Step 7: Saving
        pbar.set_description("Saving Shapefile")
        try:
            gdf.to_file(output_shp, engine="pyogrio")
        except Exception:
            gdf.to_file(output_shp) # Fallback to fiona if pyogrio fails
        
        pbar.update(1)
        pbar.close()
        
        print(f"[INFO] Done. Total time: {(time.time() - start_time):.2f}s")

        # Cleanup
        if os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir)
            except: pass
        
        return True

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Part 2: Verification Logic (Executed in Subprocess)
# =============================================================================

def _run_verify_process(shp_path: str):
    """
    Execute terrain verification and plotting in a standalone process.
    This isolation ensures Matplotlib runs with the 'Agg' backend without conflicting with the main thread's GUI event loop.

    Args:
        shp_path (str): Path to the shapefile to verify.
    """
    try:
        # Import Matplotlib ONLY here, inside the subprocess
        import matplotlib
        matplotlib.use('Agg') # Force non-interactive backend
        import matplotlib.pyplot as plt

        if not os.path.exists(shp_path):
            return

        print(f"[VERIFY] Loading {shp_path}...")
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            print(f"[VERIFY] Warning: {shp_path} is empty.")
            return

        # Basic stats
        if 'elevation' in gdf.columns:
            elev_min = gdf['elevation'].min()
            elev_max = gdf['elevation'].max()
            print(f"[VERIFY] Elev range: [{elev_min:.2f}, {elev_max:.2f}] m, Features: {len(gdf)}")
        else:
            print(f"[VERIFY] No elevation column found.")

        # Plotting (Low DPI for speed)
        try:
            output_dir = os.path.dirname(shp_path)
            output_png = os.path.join(output_dir, "terrain_preview.png")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(column='elevation', ax=ax, cmap='terrain', legend=True, markersize=1)
            ax.set_title(f"Terrain Preview: {os.path.basename(shp_path)}")
            plt.savefig(output_png, dpi=100, bbox_inches='tight') # High DPI
            plt.close(fig)
            print(f"[VERIFY] Saved preview image: {output_png}")
        except Exception as plot_err:
            print(f"[VERIFY] Plotting failed: {plot_err}")

    except Exception as e:
        print(f"[VERIFY] Verification process failed: {e}")
        traceback.print_exc()


# =============================================================================
# Part 3: Configuration & Integration Logic (LBM Specific)
# =============================================================================

def _parse_conf_bbox(conf_raw: str) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    """
    Parse the case name and manual cut coordinates from the configuration string.

    Args:
        conf_raw (str): Raw configuration text content.

    Returns:
        tuple: A tuple containing (casename, (lon_min, lon_max), (lat_min, lat_max)).

    Raises:
        ValueError: If required fields are missing or parsing fails.
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
        raise ValueError("Failed to parse cut_lon_manual / cut_lat_manual.") from e

    lon0, lon1 = float(cut_lon[0]), float(cut_lon[1])
    lat0, lat1 = float(cut_lat[0]), float(cut_lat[1])

    lon_min, lon_max = (lon0, lon1) if lon0 <= lon1 else (lon1, lon0)
    lat_min, lat_max = (lat0, lat1) if lat0 <= lat1 else (lat1, lat0)

    if lon_min == lon_max or lat_min == lat_max:
        raise ValueError("Invalid bbox: lon/lat range has zero size.")

    return casename, (lon_min, lon_max), (lat_min, lat_max)

def _expand_bbox_20pct(lon_min, lon_max, lat_min, lat_max):
    """
    Expand the given bounding box by 20% in all directions.

    Args:
        lon_min (float): Minimum longitude.
        lon_max (float): Maximum longitude.
        lat_min (float): Minimum latitude.
        lat_max (float): Maximum latitude.

    Returns:
        tuple: (new_lon_min, new_lon_max, new_lat_min, new_lat_max).
    """
    w = lon_max - lon_min
    h = lat_max - lat_min
    return (lon_min - 0.2 * w, lon_max + 0.2 * w, 
            lat_min - 0.2 * h, lat_max + 0.2 * h)

def _resolve_install_database_path():
    """
    Resolve the path to the installation database directory.
    Assumes a relative path structure from this script file.

    Returns:
        Path: Path object pointing to the database directory.
    """
    return Path(__file__).resolve().parents[2] / "database"

def ensure_dem_shp_from_tif(conf_raw: str, project_home: Path) -> Optional[Path]:
    """
    Main integration entry point. Ensures a valid DEM Shapefile exists for the given configuration.
    Handles logic for checking existing files, searching for TIF/SRTM sources, triggering generation,
    and launching background verification.

    Args:
        conf_raw (str): Raw configuration text.
        project_home (Path): Path to the project root directory.

    Returns:
        Optional[Path]: Path to the resulting Shapefile, or None if generation failed.
    """
    if not conf_raw:
        print("[WARN] No conf text provided.")
        return None

    try:
        casename, lon_pair, lat_pair = _parse_conf_bbox(conf_raw)
    except Exception as e:
        print(f"[WARN] Failed to parse DEM bounds: {e}")
        return None

    # 1. Expand BBox
    lon_min, lon_max = lon_pair
    lat_min, lat_max = lat_pair
    lon_min2, lon_max2, lat_min2, lat_max2 = _expand_bbox_20pct(lon_min, lon_max, lat_min, lat_max)
    
    terrain_dir = project_home / "terrain_db"
    if not terrain_dir.exists():
        try:
            terrain_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created terrain_db: {terrain_dir}")
        except Exception as e:
            print(f"[WARN] Failed to create terrain_db: {e}")
            return None

    output_shp = terrain_dir / f"{casename}.shp"
    
    # Define a helper to trigger verification in SUBPROCESS
    def _trigger_background_verify(shp_file):
        """
        Launch the verification process in the background using a subprocess.
        
        Args:
            shp_file (Path): Path to the shapefile to verify.
        """
        try:
            # Call this file as a script with --verify argument
            # This spawns a completely separate process, avoiding all Matplotlib threading issues
            cmd = [sys.executable, __file__, "--verify", str(shp_file)]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # print(f"[INFO] Triggered background verification for {os.path.basename(shp_file)}")
        except Exception as e:
            print(f"[WARN] Failed to trigger background verification: {e}")

    if output_shp.exists():
        print("[INFO] DEM shapefile exists; skipping generation.")
        # Trigger verification even if file exists (to ensure plot exists)
        _trigger_background_verify(output_shp)
        return output_shp

    print(
        f"[INFO] Target DEM bbox (expanded 20%): "
        f"Lon[{lon_min2:.4f}, {lon_max2:.4f}] Lat[{lat_min2:.4f}, {lat_max2:.4f}]"
    )

    # 2. Search Strategy
    input_tif_file = None
    search_dir_srtm = None
    
    # Check 1: TIF in project terrain_db
    tif_candidates = list(terrain_dir.glob("*.tif")) + list(terrain_dir.glob("*.tiff"))
    if tif_candidates:
        input_tif_file = tif_candidates[0]
        print(f"[INFO] Found TIF in project terrain_db: {input_tif_file}")
    
    # Check 2: TIF in Install Database
    if not input_tif_file:
        install_db = _resolve_install_database_path()
        if install_db.exists():
            db_candidates = list(install_db.glob("*.tif")) + list(install_db.glob("*.tiff"))
            if db_candidates:
                input_tif_file = db_candidates[0]
                print(f"[INFO] Found TIF in install database: {input_tif_file}")
            else:
                # Check 3: SRTM in Install Database
                search_dir_srtm = install_db
                print(f"[INFO] No TIF found. Will search for SRTM tiles in: {search_dir_srtm}")
        else:
            print(f"[WARN] Install database not found at {install_db}")

    # 3. Execute
    success = process_terrain(
        min_lon=lon_min2,
        max_lon=lon_max2,
        min_lat=lat_min2,
        max_lat=lat_max2,
        output_shp=str(output_shp),
        input_file=str(input_tif_file) if input_tif_file else None,
        search_dir=str(search_dir_srtm) if search_dir_srtm else None,
        step=1,
        geom_type='polygon'
    )

    if success and output_shp.exists():
        print(f"[INFO] Generated DEM shapefile: {output_shp}")
        # 4. Trigger Verification
        _trigger_background_verify(output_shp)
        return output_shp
    else:
        print("[ERROR] Failed to generate DEM shapefile.")
        return None

if __name__ == "__main__":
    # Support self-call for verification mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=str, help="Path to shapefile to verify and plot")
    args, unknown = parser.parse_known_args()
    
    if args.verify:
        _run_verify_process(args.verify)
    else:
        # Default behavior (can be extended for CLI usage of process_terrain)
        print("TerrainReader module. Usage: import and call ensure_dem_shp_from_tif()")

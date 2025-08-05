"""
voxelization.py
"""
import geopandas as gpd
import trimesh
from shapely.geometry import MultiPolygon
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

MIN_EXTRUDE_AREA = 20.0  # m^2, minimum polygon area to extrude

def stl_bounds(file_path: str | Path) -> Tuple[Tuple[float, float],
                                                       Tuple[float, float],
                                                       Tuple[float, float]]:
    mesh = trimesh.load(file_path, force='mesh')  # binary / ASCII
    bounds = mesh.bounds  # shape = (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)

def wobasement(input_file: str | Path, outputName: str | Path) -> None:
    """
    仅做与 basement() 相同的 XY 原点平移，不做 Z 方向抬升，也不做布尔并集。
    """
    mesh_buildings = trimesh.load(input_file, force='mesh')
    min_x, min_y, _ = mesh_buildings.bounds[0]
    mesh_buildings.apply_translation([-min_x, -min_y, 0.0])
    mesh_buildings.export(outputName)
    print(f"          Re-aligned wobase STL and overwrote: {outputName}")

def basement(baseheight, input_file, outputName, domain_width, domain_depth):
    # 1. load model
    mesh_buildings = trimesh.load(input_file, force='mesh')
    min_x, min_y, _ = mesh_buildings.bounds[0]
    mesh_buildings.apply_translation([-min_x, -min_y, 0.0])

    # print raw bounds
    orig_bounds = mesh_buildings.bounds
    print("          Original XYZ bounds:")
    print(f"            X: [{orig_bounds[0][0]:.3f}, {orig_bounds[1][0]:.3f}]")
    print(f"            Y: [{orig_bounds[0][1]:.3f}, {orig_bounds[1][1]:.3f}]")
    print(f"            Z: [{orig_bounds[0][2]:.3f}, {orig_bounds[1][2]:.3f}]")

    # 2. add z-axis basement
    translation_vec = [0, 0, baseheight]
    mesh_buildings.apply_translation(translation_vec)

    # print moved bounds
    trans_bounds = mesh_buildings.bounds
    print("          XYZ bounds after translation without basement:")
    print(f"            X: [{trans_bounds[0][0]:.3f}, {trans_bounds[1][0]:.3f}]")
    print(f"            Y: [{trans_bounds[0][1]:.3f}, {trans_bounds[1][1]:.3f}]")
    print(f"            Z: [{trans_bounds[0][2]:.3f}, {trans_bounds[1][2]:.3f}]")

    # 3. build basement box
    width  = domain_width
    depth  = domain_depth
    height = baseheight  # basement height

    box = trimesh.creation.box(extents=[width, depth, height])
    # 把盒子中心移到 (width/2, depth/2, height/2)
    box.apply_translation([width/2, depth/2, height/2])

    # 4. merge and export final STL
    combined = trimesh.util.concatenate([mesh_buildings, box])

    print(f"[9/10] Saving combined mesh...")
    combined.export(outputName)
    print(f"          Saved combined mesh to {outputName}...")


def save_outline_preview(gdf: gpd.GeoDataFrame, case_name: str, shp_path: Path, output_dir: Path) -> None:
    """Save a preview image of building outlines with height labels."""
    # compute geographic bounds after cutting
    min_lon, min_lat, max_lon, max_lat = gdf.to_crs(epsg=4326).total_bounds

    fig, (ax_map, ax_info) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [4, 1]}
    )

    line_width = 0.25
    if "height" in gdf.columns:
        extruded_mask = gdf["height"].notna() & (gdf["height"] > 0)
    else:
        extruded_mask = np.ones(len(gdf), dtype=bool)
    non_extruded = ~extruded_mask

    gdf[extruded_mask].boundary.plot(ax=ax_map, color="black", linewidth=line_width)
    if non_extruded.any():
        gdf[non_extruded].boundary.plot(
            ax=ax_map,
            color=(0.0, 0.0, 1.0, 0.5),
            linewidth=line_width,
        )

    for _, row in gdf.iterrows():
        h = row.get("height")
        if not (h == h) or h <= 0:
            continue
        px, py = row.geometry.representative_point().coords[0]
        area = row.geometry.area
        ax_map.text(px, py, f"{h:.1f}", fontsize=1.5, ha="center", va="bottom", color='red', alpha=0.5)
        ax_map.text(px, py, f"{area:.1f}", fontsize=1.5, ha="center", va="top", color='green', alpha=0.5)

    ax_map.set_aspect("equal")
    ax_map.axis("off")

    ax_info.axis("off")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        f"Time: {now}",
        f"caseName: {case_name}",
        f"lon range: [{min_lon:.6f}, {max_lon:.6f}]",
        f"lat range: [{min_lat:.6f}, {max_lat:.6f}]",
        f"shp: {shp_path.name}",
    ]
    ax_info.text(0.01, 0.99, "\n".join(info_lines), va="top", ha="left", fontsize=6)

    plt.tight_layout()
    out_path = output_dir / f"{case_name}_preview.jpg"
    fig.savefig(out_path, dpi=1200)
    plt.close(fig)
    print(f"          Saved preview image to {out_path}")


def main(caseName=None):
    script_dir = Path(__file__).resolve().parent
    conf_path = script_dir.parent / 'conf.txt'

    # 0) Load or create conf.txt for caseName
    if conf_path.exists():
        print("[!]   Loaded existing conf.txt")
        lines = conf_path.read_text(encoding='utf-8-sig').splitlines()
        first = lines[0].split('//')[0].strip()
        if first.startswith('casename') and '=' in first:
            caseName = first.split('=', 1)[1].strip()
            print(f"          Case name set from conf.txt: {caseName}")
        else:
            caseName = input("Enter case name: ")
            lines[0] = f"casename = {caseName}"
    else:
        print("[!]   conf.txt not found, creating new conf.txt")
        if caseName is None:
            caseName = input("Enter case name: ")
        lines = [f"casename = {caseName}"] + [''] * 10

    # # Ensure at least 11 lines
    while len(lines) < 49:
         lines.append('')

    # ——  resolve conf.txt towards geo range —— 
    import ast
    si_x = None
    si_y = None

    for ln in lines:
        txt = ln.split('//')[0].strip().replace(' ', '')
        # remove comments and whitespace, split into key=value
        if txt.lower().startswith('si_x') and '=' in txt:    
            si_x = ast.literal_eval(txt.split('=', 1)[1])
        elif txt.lower().startswith('si_y') and '=' in txt:   
            si_y = ast.literal_eval(txt.split('=', 1)[1])

    si_x_arr = np.asarray(si_x, dtype=float)
    si_y_arr = np.asarray(si_y, dtype=float)
    print(f"[!]   Cutting dataset! Range: x: {si_x_arr.min():.3f}-{si_x_arr.max():.3f} m, "
        f"y: {si_y_arr.min():.3f}-{si_y_arr.max():.3f} m.")

    # 1) Locate data folder
    data_folder = script_dir.parent / "geoData" / caseName

    # 2) Rename files
    shp_files = list(data_folder.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {data_folder}")
    orig_shp = shp_files[0]
    orig_base = orig_shp.stem
    for f in data_folder.iterdir():
        name = f.stem
        if orig_base in name:
            new_stem = name.replace(orig_base, caseName)
            f.rename(data_folder / f"{new_stem}{f.suffix}")
    shpName = data_folder / f"{caseName}.shp"

    print(f"[1/10] Reading shp: {shpName}...")
    gdf = gpd.read_file(shpName)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
        print(f"Set CRS to EPSG:4326...")

    orig_min_lon, orig_min_lat, orig_max_lon, orig_max_lat = (
        gdf.to_crs(epsg=4326).total_bounds
    )

    gdf = gdf.to_crs(epsg=32650)
    print("          Converted to EPSG:32650 for metric processing…")
    # Compute original geographic bounds and height range
    # orig_min_lon, orig_min_lat, orig_max_lon, orig_max_lat = gdf.total_bounds

    # —— Cutting dataset —— 
    if si_x is not None or si_y is not None:
        from shapely.geometry import box

        si_x = np.sort(np.array(si_x, dtype=float))
        si_y = np.sort(np.array(si_y, dtype=float))
        
        bbox_proj = box(si_x[0], si_y[0], si_x[1], si_y[1])   
        gdf = gpd.clip(gdf, bbox_proj)

        # if clipping results in empty GeoDataFrame
        if gdf.empty:
            raise RuntimeError(f"No features found within cut range x={si_x}, y={si_y}.")


    print(f"[2/10] Dimensionalizing to SI (UTM 50N)...")
    #gdf = gdf.to_crs(epsg=32650)

    # Continue original steps
    print(f"[3/10] Getting building height...")

    # 1) use absolute height difference
    if 'AbsZmax' in gdf.columns and 'AbsZmin' in gdf.columns:
        gdf['height'] = gdf['AbsZmax'] - gdf['AbsZmin']

    # 2) use Height field (case-insensitive)
    elif any(col in gdf.columns for col in ('Height', 'height', 'HEIGHT')):
        height_col = next(col for col in ('Height', 'height', 'HEIGHT') if col in gdf.columns)
        gdf['height'] = gpd.pd.to_numeric(gdf[height_col], errors='coerce')

    # 3) use floor count × average floor height
    else:

        if 'FLOOR' in gdf.columns:
            floor_col = 'FLOOR'
        elif 'Floor' in gdf.columns:
            floor_col = 'Floor'
        else:
            raise KeyError("No height or level information in SHP: AbsZmax/AbsZmin, Height, height, HEIGHT, FLOOR, Floor. Exiting...")

        gdf['floors'] = gpd.pd.to_numeric(gdf[floor_col], errors='coerce')
        AVG_FLOOR_HEIGHT = 3.0
        gdf['height'] = gdf['floors'] * AVG_FLOOR_HEIGHT


    print(f"[4/10] Generating building outline preview...")
    save_outline_preview(gdf, caseName, shpName, data_folder)

    print(f"[5/10] Stretching in Z-axis and BOOL combining...")
    meshes = []
    for _, row in gdf.iterrows():
        h = row['height']
        # skip NaN or non-positive heights
        if not (h == h) or h <= 0:
            continue
        polys = list(row.geometry.geoms) if isinstance(row.geometry, MultiPolygon) else [row.geometry]  # ★ fix: shapely-2

        for poly in polys:
            if poly.area < MIN_EXTRUDE_AREA:
                continue
            meshes.append(trimesh.creation.extrude_polygon(poly, h))

    if not meshes:
        raise RuntimeError(
            f"No building meshes generated after filtering; "
            f"check si_x={si_x} / si_y={si_y}"          
        )

    scene = trimesh.util.concatenate(meshes)


    print(f"[6/10] Translating projected origin to (0,0)…")
    # read conf.txt
    import ast
    lines = conf_path.read_text().splitlines()
    si_x = si_y = None
    for ln in lines:
        txt = ln.split('//')[0].strip().replace(' ', '')
        if txt.lower().startswith('si_x') and '=' in txt:
            si_x = ast.literal_eval(txt.split('=', 1)[1])
        elif txt.lower().startswith('si_y') and '=' in txt:
            si_y = ast.literal_eval(txt.split('=', 1)[1])
    if si_x is None or si_y is None:
        raise RuntimeError("Cannot find si_x/si_y in conf.txt")
    # si_x = [x_min, x_max], si_y = [y_min, y_max]
    x_min_proj, _ = si_x
    y_min_proj, _ = si_y

    scene.apply_translation([-x_min_proj, -y_min_proj, 0.0])
    print(f"          Translated projected origin ({x_min_proj:.2f}, {y_min_proj:.2f}) to zeros.")
    
    print(f"[7/10] Exporting buildings stereolithography stl...")
    file_wobase = f"{caseName}_wo_base.stl"
    basement_filename = f"{caseName}_with_base.stl"
    # build without basement
    scene.export(data_folder / file_wobase)
    print(f"          Saved combined mesh to {data_folder / file_wobase}.")
    bounds_wo = stl_bounds(data_folder / file_wobase)
    print(f"          Range verification: {bounds_wo}")

    print(f"[8/10] Adding basement for CFD voxelization...")
    # calculate domain width and depth from bounds of the no-basement STL
    bounds_wo = stl_bounds(data_folder / file_wobase)
    domain_width  = np.max(si_x) - np.min(si_x) 
    domain_depth  = np.max(si_y) - np.min(si_y)

    baseheight = 50

    bh_pat = re.compile(r'^\s*base_height_manual\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I)
    manual_defined = False
    for ln in lines:
        m = bh_pat.match(ln.split('//')[0])
        if m:
            baseheight = float(m.group(1))
            manual_defined = True
            break
    if manual_defined:
        print(f"          Basement height set from conf.txt (base_height_manual = {baseheight}).")
    else:
        print(f"          Basement height not specified, using default = {baseheight}.")

    basement(baseheight, 
        data_folder / file_wobase,
        data_folder / basement_filename,
        domain_width,
        domain_depth
    )

    wobasement(data_folder / file_wobase, data_folder / file_wobase)
    bounds_wo = stl_bounds(data_folder / file_wobase)
    print(f"          Re-aligned wobase STL bounds: {bounds_wo}")

    bounds_w = stl_bounds(data_folder / basement_filename)
    print(f"          Range verification: "
          f"X:[0.000, {domain_width:.3f}]  "
          f"Y:[0.000, {domain_depth:.3f}]  "
          f"Z:[{bounds_w[2][0]:.3f}, {bounds_w[2][1]:.3f}]")

    print("[10/10] Updating configuration file...")
        # Update conf.txt lines
    lines[3] = ""    
    lines[4] = "// Original SHP Range"
    lines[5] = f"orig_lon = [{orig_min_lon:.6f}, {orig_max_lon:.6f}]"
    lines[6] = f"orig_lat = [{orig_min_lat:.6f}, {orig_max_lat:.6f}]"
    lines[7] = f"orig_height = [{bounds_wo[2][0]:.6f}, {bounds_wo[2][1]:.6f}]"
    lines[8] = ""
    lines[9] = "// Projected SHP SI Range"
    lines[10] = f"si_x_cfd = [0.000000, {domain_width:.6f}]"
    lines[11] = f"si_y_cfd = [0.000000, {domain_depth:.6f}]"
    lines[12] = f"si_height = [{bounds_w[2][0]:.6f}, {bounds_w[2][1]:.6f}]"

    lines[23] = "// Basement Configurations for Model Voxelization"
    lines[24] = f"base_height = {baseheight}"
    if not any(bh_pat.match(l.split('//')[0]) for l in lines):
        lines[25] = "base_height_manual = 50"

    # Write back conf.txt
    conf_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"          Updated conf.txt at {conf_path}")

if __name__ == '__main__':
    main()

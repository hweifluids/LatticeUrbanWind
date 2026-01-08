#!/usr/bin/env python3
"""
dgPrepare.py
Prepare STL input for datagen core processing.
"""

import os
import sys
import re
import glob
import time
from typing import Tuple, Optional

try:
    import numpy as np
    import trimesh
except Exception as e:
    print("[FATAL] Missing dependencies. Please install numpy and trimesh.")
    print(f"[FATAL] Import error: {e}")
    sys.exit(2)


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def debug(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def fatal(msg: str, code: int = 1) -> None:
    print(f"[{ts()}] [FATAL] {msg}")
    sys.exit(code)


def find_input_luwdg(argv: list) -> str:
    """
    Find the input .luwdg file based on the rules:
    1) If an argument is provided, use it.
    2) Else look for conf.luwdg in current working directory.
    3) Else look for any *.luwdg in current working directory.
    4) If none found, error out.
    """
    cwd = os.getcwd()
    debug(f"Current working directory: {cwd}")

    if len(argv) >= 2:
        p = argv[1]
        debug(f"Input argument provided: {p}")
        if not os.path.isfile(p):
            fatal(f"Input path does not exist or is not a file: {p}", 1)
        if not p.lower().endswith(".luwdg"):
            fatal(f"Input file must end with .luwdg: {p}", 1)
        return os.path.abspath(p)

    conf_path = os.path.join(cwd, "conf.luwdg")
    debug("No input argument provided. Searching for conf.luwdg in current directory.")
    if os.path.isfile(conf_path):
        debug(f"Found conf.luwdg: {conf_path}")
        return os.path.abspath(conf_path)

    debug("conf.luwdg not found. Searching for any *.luwdg in current directory.")
    candidates = sorted(glob.glob(os.path.join(cwd, "*.luwdg")))
    if candidates:
        debug(f"Found candidate .luwdg files: {candidates}")
        debug(f"Using the first one (sorted): {candidates[0]}")
        return os.path.abspath(candidates[0])

    fatal("No .luwdg file found in current directory, and no input argument was provided.", 1)
    return ""


def parse_luwdg(path: str) -> Tuple[str, float, str]:
    """
    Parse casename and base_height from the luwdg file.
    Returns (casename, base_height, original_text).
    """
    debug(f"Reading luwdg file: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    casename = None
    base_height = None

    m_case = re.search(r'^\s*casename\s*=\s*([A-Za-z0-9_.\-]+)\s*$', text, flags=re.MULTILINE)
    if m_case:
        casename = m_case.group(1).strip()

    m_base = re.search(r'^\s*base_height\s*=\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+\-]?[0-9]+)?)\s*$', text, flags=re.MULTILINE)
    if m_base:
        base_height = float(m_base.group(1))

    if casename is None:
        fatal("casename not found in luwdg file.", 1)
    if base_height is None:
        fatal("base_height not found in luwdg file.", 1)
    if base_height <= 0:
        fatal(f"base_height must be positive. Got: {base_height}", 1)

    debug(f"Parsed casename: {casename}")
    debug(f"Parsed base_height (m): {base_height}")

    return casename, base_height, text


def pick_stl(building_db_dir: str) -> str:
    """
    Pick rawbuildings.stl if present, else fallback to any *.stl in the same directory.
    """
    preferred = os.path.join(building_db_dir, "rawbuildings.stl")
    debug(f"Looking for STL in: {building_db_dir}")
    if os.path.isfile(preferred):
        debug(f"Found preferred STL: {preferred}")
        return preferred

    debug("Preferred rawbuildings.stl not found. Falling back to any *.stl.")
    stls = sorted(glob.glob(os.path.join(building_db_dir, "*.stl")))
    if not stls:
        fatal(f"No STL file found in {building_db_dir}", 1)

    debug(f"Found STL candidates: {stls}")
    debug(f"Using the first one (sorted): {stls[0]}")
    return stls[0]


def load_mesh(stl_path: str) -> trimesh.Trimesh:
    debug(f"Loading STL: {stl_path}")
    mesh = trimesh.load(stl_path, force="mesh")
    if mesh is None:
        fatal("Failed to load STL mesh (returned None).", 1)

    if isinstance(mesh, trimesh.Scene):
        debug("Loaded a Scene. Attempting to merge geometry into a single mesh.")
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if not isinstance(mesh, trimesh.Trimesh):
        fatal(f"Loaded object is not a Trimesh. Type: {type(mesh)}", 1)

    debug(f"Mesh vertices: {len(mesh.vertices)}")
    debug(f"Mesh faces: {len(mesh.faces)}")

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        fatal("Mesh is empty.", 1)

    bounds = mesh.bounds
    debug(f"Mesh bounds before base: min={bounds[0].tolist()}, max={bounds[1].tolist()}")

    return mesh


def create_base_block(bounds: np.ndarray, base_height: float) -> trimesh.Trimesh:
    """
    Create a rectangular base block under the mesh:
    base thickness is base_height
    base x and y extents are 2x the mesh x and y extents
    """
    bmin = bounds[0]
    bmax = bounds[1]

    dx = float(bmax[0] - bmin[0])
    dy = float(bmax[1] - bmin[1])
    zmin = float(bmin[2])

    if dx <= 0 or dy <= 0:
        fatal(f"Invalid mesh XY extents. dx={dx}, dy={dy}", 1)

    base_dx = 2.0 * dx
    base_dy = 2.0 * dy
    base_dz = float(base_height)

    cx = float((bmin[0] + bmax[0]) * 0.5)
    cy = float((bmin[1] + bmax[1]) * 0.5)
    cz = float(zmin - base_dz * 0.5)

    debug(f"Creating base block with extents (m): x={base_dx}, y={base_dy}, z={base_dz}")
    debug(f"Base block center (m): x={cx}, y={cy}, z={cz}")

    base = trimesh.creation.box(extents=(base_dx, base_dy, base_dz))
    base.apply_translation((cx, cy, cz))

    bb = base.bounds
    debug(f"Base bounds: min={bb[0].tolist()}, max={bb[1].tolist()}")

    return base


def try_boolean_union(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, bool, str]:
    """
    Try to boolean union meshes. If it fails, fall back to concatenation.
    Returns (result_mesh, used_boolean, message).
    """
    debug("Attempting boolean union of mesh and base block.")

    used_boolean = False
    msg = ""

    try:
        result = trimesh.boolean.union([mesh_a, mesh_b])
        if result is None:
            raise RuntimeError("trimesh.boolean.union returned None")
        if isinstance(result, list):
            if len(result) == 0:
                raise RuntimeError("Boolean union returned an empty list")
            result = trimesh.util.concatenate(result)
        if isinstance(result, trimesh.Scene):
            result = trimesh.util.concatenate([g for g in result.geometry.values()])
        if not isinstance(result, trimesh.Trimesh):
            raise RuntimeError(f"Boolean union returned unexpected type: {type(result)}")

        used_boolean = True
        msg = "Boolean union succeeded."
        debug(msg)
        return result, used_boolean, msg

    except Exception as e:
        msg = f"Boolean union failed. Falling back to mesh concatenation. Error: {e}"
        debug(msg)

        merged = trimesh.util.concatenate([mesh_a, mesh_b])
        debug(f"Concatenated mesh vertices: {len(merged.vertices)}")
        debug(f"Concatenated mesh faces: {len(merged.faces)}")

        debug("Running basic cleanup on concatenated mesh.")
        try:
            merged.merge_vertices()
        except Exception as e2:
            debug(f"merge_vertices failed: {e2}")

        try:
            merged.remove_duplicate_faces()
        except Exception as e2:
            debug(f"remove_duplicate_faces failed: {e2}")

        try:
            merged.remove_degenerate_faces()
        except Exception as e2:
            debug(f"remove_degenerate_faces failed: {e2}")

        try:
            merged.remove_unreferenced_vertices()
        except Exception as e2:
            debug(f"remove_unreferenced_vertices failed: {e2}")

        debug(f"After cleanup vertices: {len(merged.vertices)}")
        debug(f"After cleanup faces: {len(merged.faces)}")

        return merged, False, msg


def translate_to_target_bounds(mesh: trimesh.Trimesh, base_height: float) -> trimesh.Trimesh:
    """
    Translate mesh so that x_min, y_min, z_min become [0, 0, -base_height].
    """
    bounds = mesh.bounds
    bmin = bounds[0]
    tx = float(0.0 - bmin[0])
    ty = float(0.0 - bmin[1])
    tz = float((-base_height) - bmin[2])

    debug(f"Translating mesh by (m): tx={tx}, ty={ty}, tz={tz}")
    mesh.apply_translation((tx, ty, tz))

    nb = mesh.bounds
    debug(f"Mesh bounds after translation: min={nb[0].tolist()}, max={nb[1].tolist()}")

    check_min = nb[0]
    debug(f"Target min should be close to [0, 0, {-base_height}]. Actual min is {check_min.tolist()}")

    return mesh


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        debug(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)


def export_mesh(mesh: trimesh.Trimesh, out_path: str) -> None:
    debug(f"Exporting STL to: {out_path}")
    try:
        mesh.export(out_path)
    except Exception as e:
        fatal(f"Failed to export STL. Error: {e}", 1)
    debug("Export completed.")


def format_range(a: float, b: float) -> str:
    return f"[{a:.6f}, {b:.6f}]"


def update_luwdg_ranges(luwdg_path: str, original_text: str, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """
    Replace si_x_cfd and si_y_cfd lines with new ranges in meters.
    If fields are missing, append them near the top (after datetime if present, else after casename).
    """
    new_x_line = f"si_x_cfd = {format_range(x_min, x_max)}"
    new_y_line = f"si_y_cfd = {format_range(y_min, y_max)}"

    debug(f"Updating luwdg si_x_cfd to: {new_x_line}")
    debug(f"Updating luwdg si_y_cfd to: {new_y_line}")

    text = original_text

    rx_x = re.compile(r'^\s*si_x_cfd\s*=\s*\[[^\]]*\]\s*$', flags=re.MULTILINE)
    rx_y = re.compile(r'^\s*si_y_cfd\s*=\s*\[[^\]]*\]\s*$', flags=re.MULTILINE)

    has_x = bool(rx_x.search(text))
    has_y = bool(rx_y.search(text))

    if has_x:
        text = rx_x.sub(new_x_line, text, count=1)
    if has_y:
        text = rx_y.sub(new_y_line, text, count=1)

    if not has_x or not has_y:
        debug("One or both of si_x_cfd and si_y_cfd were not found. Inserting missing fields.")

        insert_lines = []
        if not has_x:
            insert_lines.append(new_x_line)
        if not has_y:
            insert_lines.append(new_y_line)

        insert_block = "\n".join(insert_lines) + "\n"

        m_dt = re.search(r'^\s*datetime\s*=\s*.*$', text, flags=re.MULTILINE)
        m_case = re.search(r'^\s*casename\s*=\s*.*$', text, flags=re.MULTILINE)

        if m_dt:
            idx = m_dt.end()
            text = text[:idx] + "\n" + insert_block + text[idx:]
        elif m_case:
            idx = m_case.end()
            text = text[:idx] + "\n" + insert_block + text[idx:]
        else:
            text = insert_block + text

    backup_path = luwdg_path + ".bak"
    try:
        debug(f"Writing backup luwdg to: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original_text)
    except Exception as e:
        debug(f"Backup write failed (continuing). Error: {e}")

    debug(f"Writing updated luwdg to: {luwdg_path}")
    with open(luwdg_path, "w", encoding="utf-8") as f:
        f.write(text)

    debug("luwdg update completed.")


def main(argv: list) -> int:
    luwdg_path = find_input_luwdg(argv)
    project_dir = os.path.dirname(luwdg_path)
    debug(f"Resolved luwdg path: {luwdg_path}")
    debug(f"Project directory ($projectDir): {project_dir}")

    casename, base_height, luwdg_text = parse_luwdg(luwdg_path)

    building_db_dir = os.path.join(project_dir, "building_db")
    if not os.path.isdir(building_db_dir):
        fatal(f"Missing directory: {building_db_dir}", 1)

    stl_in = pick_stl(building_db_dir)
    mesh_raw = load_mesh(stl_in)

    base = create_base_block(mesh_raw.bounds, base_height)
    mesh_union, used_boolean, union_msg = try_boolean_union(mesh_raw, base)
    debug(f"Union method: {'boolean' if used_boolean else 'concatenation'}")
    debug(f"Union message: {union_msg}")

    mesh_final = translate_to_target_bounds(mesh_union, base_height)

    out_dir = os.path.join(project_dir, "proj_temp")
    ensure_dir(out_dir)

    out_stl = os.path.join(out_dir, f"{casename}_DG.stl")
    export_mesh(mesh_final, out_stl)

    fb = mesh_final.bounds
    x_min, x_max = float(fb[0][0]), float(fb[1][0])
    y_min, y_max = float(fb[0][1]), float(fb[1][1])

    debug(f"Final STL XY bounds (m): x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    update_luwdg_ranges(luwdg_path, luwdg_text, x_min, x_max, y_min, y_max)

    debug("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

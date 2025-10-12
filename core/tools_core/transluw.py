#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make post translate for VTK files so that the minimum bounds move to the origin.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pyvista as pv


# ------------------------------ Core geometry ops ------------------------------

def compute_translate_offset(mesh: pv.DataSet) -> Tuple[float, float, float]:
    """Compute the translation offset that moves the minimum bounds to the origin."""
    xmin, ymin, zmin = mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]
    return (-xmin, -ymin, -zmin)


def translate_mesh(mesh: pv.DataSet, offset: Tuple[float, float, float]) -> pv.DataSet:
    """Return a translated copy of mesh using the provided offset."""
    if offset == (0.0, 0.0, 0.0):
        return mesh
    return mesh.translate(offset, inplace=False)


# ------------------------------ File discovery ------------------------------

def find_latest_by_pattern(folder: Path, datetime_str: str, kind: str) -> Optional[Path]:
    """
    Find the file with the largest 9 digit index matching:
    {datetime}_raw_{kind}-#########.vtk
    kind must be 'u' or 'rho'.
    Return None when not found.
    """
    if kind not in {"u", "rho"}:
        raise ValueError("kind must be 'u' or 'rho'")
    if not folder.is_dir():
        return None

    pat = re.compile(rf"^{re.escape(datetime_str)}_raw_{kind}-(\d{{9}})\.vtk$", re.IGNORECASE)
    candidates: Dict[int, Path] = {}
    for f in folder.iterdir():
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if m:
            idx = int(m.group(1))
            candidates[idx] = f
    if not candidates:
        return None
    return candidates[max(candidates.keys())]


def derive_case_datetime_from_vtk(vtk_path: Path) -> Tuple[str, str, str]:
    """
    Derive casename, datetime and prefix from a single vtk filename.
    casename uses the parent folder name.
    datetime tries to parse from either:
      <datetime>_raw_u-#########.vtk  or  <datetime>_raw_rho-#########.vtk
      fall back to stem if no pattern matches.
    prefix is 'uvw' for u pattern, 'rho' for rho pattern, 'mesh' as a fallback.
    """
    casename = vtk_path.parent.name
    stem = vtk_path.stem
    m_u = re.match(r"^(.+?)_raw_u-\d{9}$", stem, flags=re.IGNORECASE)
    m_rho = re.match(r"^(.+?)_raw_rho-\d{9}$", stem, flags=re.IGNORECASE)
    if m_u:
        return casename, m_u.group(1), "uvw"
    if m_rho:
        return casename, m_rho.group(1), "rho"
    return casename, stem, "mesh"


# ------------------------------ Text config parsing ------------------------------

def parse_case_from_text(txt_path: Path) -> Tuple[str, str]:
    """
    Parse casename and datetime in a case insensitive manner from a text file.
    Expected patterns:
      casename = <value>
      datetime = <value>
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"Config file not found: {txt_path}")
    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    m_case = re.search(r"casename\s*=\s*([^\s]+)", content, flags=re.IGNORECASE)
    m_dt = re.search(r"datetime\s*=\s*([^\s]+)", content, flags=re.IGNORECASE)
    if not m_case or not m_dt:
        raise ValueError("Failed to parse 'casename' or 'datetime' from the config file")
    return m_case.group(1), m_dt.group(1)


# ------------------------------ Output path builder ------------------------------

def build_output_path(parent_dir: Path, prefix: str, casename: str, datetime_str: str, tailname: str) -> Path:
    """
    Build output path under parent_dir/RESULTS.
    Filename formats:
      <prefix>-<casename>_<datetime>.vtk
      <prefix>-<casename>_<datetime>_<tailname>.vtk when tailname is not empty.
    """
    results_dir = parent_dir / "RESULTS"
    results_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}-{casename}_{datetime_str}"
    if tailname:
        base = f"{base}_{tailname}"
    return results_dir / f"{base}.vtk"


# ------------------------------ Merge helpers ------------------------------

def copy_point_and_cell_data(src: pv.DataSet, dst: pv.DataSet, suffix: str = "") -> None:
    """
    Copy all point_data and cell_data arrays from src to dst.
    """
    for name, arr in src.point_data.items():
        out_name = name if name not in dst.point_data else f"{name}{suffix}"
        dst.point_data[out_name] = arr.copy()

    for name, arr in src.cell_data.items():
        out_name = name if name not in dst.cell_data else f"{name}{suffix}"
        dst.cell_data[out_name] = arr.copy()


def ensure_same_topology(a: pv.DataSet, b: pv.DataSet) -> None:
    """
    Ensure that two datasets have the same number of points and cells.
    Raise an informative error when not compatible for a safe merge.
    """
    if a.n_points != b.n_points:
        raise ValueError(f"Point count mismatch when merging: {a.n_points} vs {b.n_points}")
    if a.n_cells != b.n_cells:
        raise ValueError(f"Cell count mismatch when merging: {a.n_cells} vs {b.n_cells}")


# ------------------------------ Processing paths ------------------------------

def process_single_vtk(vtk_file: Path, tailname: str) -> Path:
    """
    Translate only this vtk and save the result.
    Output under the parent/RESULTS directory of the input vtk.
    Prefix derived from filename pattern.
    """
    if not vtk_file.is_file():
        raise FileNotFoundError(f"VTK file not found: {vtk_file}")
    casename, datetime_str, prefix = derive_case_datetime_from_vtk(vtk_file)
    mesh_in = pv.read(vtk_file)
    offset = compute_translate_offset(mesh_in)
    mesh_out = translate_mesh(mesh_in, offset)
    out_path = build_output_path(vtk_file.parent, prefix, casename, datetime_str, tailname)
    mesh_out.save(out_path, binary=True)
    print(f"[OK] Translated single VTK using offset {offset} and wrote: {out_path}")
    return out_path


def process_with_conf(conf_file: Path, tailname: str) -> Path:
    """
    Read casename and datetime from conf file, look under parent/proj_temp/vtk
    for latest u and rho files, apply the same translation, and save a single output file.
    Prefix rules
      uvwrho when both exist, uvw when only u exists, rho when only rho exists.
    """
    casename, datetime_str = parse_case_from_text(conf_file)
    parent_dir = conf_file.parent
    search_dir = parent_dir / "proj_temp" / "vtk"

    vtk_u = find_latest_by_pattern(search_dir, datetime_str, "u")
    vtk_rho = find_latest_by_pattern(search_dir, datetime_str, "rho")

    if vtk_u is None and vtk_rho is None:
        raise FileNotFoundError(f"No matching VTK files found under: {search_dir}")

    # Prefer using 'u' as base when both exist for a stable merge target
    base_path = vtk_u if vtk_u is not None else vtk_rho
    base_mesh_in = pv.read(base_path)
    offset = compute_translate_offset(base_mesh_in)
    base_mesh = translate_mesh(base_mesh_in, offset)

    prefix = "uvw" if vtk_rho is None else "uvwrho" if vtk_u is not None else "rho"

    if vtk_u is not None and base_path != vtk_u:
        mesh_u = translate_mesh(pv.read(vtk_u), offset)
        ensure_same_topology(base_mesh, mesh_u)
        copy_point_and_cell_data(mesh_u, base_mesh, suffix="_u")

    if vtk_rho is not None and base_path != vtk_rho:
        mesh_rho = translate_mesh(pv.read(vtk_rho), offset)
        ensure_same_topology(base_mesh, mesh_rho)
        copy_point_and_cell_data(mesh_rho, base_mesh, suffix="_rho")

    # When base is 'u', ensure arrays from base 'u' do not need suffixing
    # When base is 'rho', ensure arrays from base 'rho' do not need suffixing
    # Arrays from the other mesh are suffixed to avoid name collisions

    out_path = build_output_path(parent_dir, prefix, casename, datetime_str, tailname)
    base_mesh.save(out_path, binary=True)

    details = []
    if vtk_u is not None:
        details.append(f"u={vtk_u.name}")
    if vtk_rho is not None:
        details.append(f"rho={vtk_rho.name}")
    print(f"[OK] Translated with shared offset {offset}; merged {' and '.join(details)} into: {out_path}")
    return out_path


# ------------------------------ CLI ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate VTK datasets so that the minimum bounds move to the origin. "
                    "Accepts a single input path plus an optional --tailname."
    )
    parser.add_argument("input_path", help="Path to a .vtk file or a text config file containing casename and datetime")
    parser.add_argument("--tailname", default="", help="Optional tail string appended to the output filename")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    tailname = args.tailname.strip()

    if not input_path.exists():
        sys.exit(f"[ERROR] Input path does not exist: {input_path}")

    try:
        if input_path.suffix.lower() == ".vtk":
            process_single_vtk(input_path, tailname)
        else:
            process_with_conf(input_path, tailname)
    except Exception as e:
        sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate from LBM coordinate to SI coordinate.
"""

import os
import re
import sys
from pathlib import Path

import pyvista as pv


def ask_case_name() -> str:
    """Ask user for the case name (strip spaces, stop on empty)."""
    # while True:
    #     case = input("Input caseName: ").strip()
    #     if case:
    #         return case
    #     print("caseName cannot be empty！\n")
    conf_file = Path(__file__).parent.parent / "conf.txt"
    if conf_file.exists():
        txt = conf_file.read_text()
        casename = re.search(r"casename\s*=\s*([^\s]+)", txt).group(1)
        datetime = re.search(r"datetime\s*=\s*([^\s]+)", txt).group(1)
        print(f"Loaded conf.txt, case={casename}, datetime={datetime}, ")
        return casename, datetime
    else:
        print("No conf.txt found: use provided nc_path and no cropping")


def find_latest_vtk(case_dir: Path, datetime_str: str) -> Path:
    """Return the vtku-#########.vtk with the largest index in `case_dir`."""
    pattern = re.compile(rf"^{datetime_str}_raw_u-(\d{{9}})\.vtk$", re.IGNORECASE)
    candidates = []

    for file in case_dir.iterdir():
        m = pattern.match(file.name)
        if m:
            idx = int(m.group(1))
            candidates.append((idx, file))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find vtk file under {case_dir}."
        )

    # 最大编号
    latest_file = max(candidates, key=lambda t: t[0])[1]
    return latest_file


def translate_to_origin(vtk_path: Path) -> pv.DataSet:
    """Load VTK, translate so min coord to (0,0,0), return translated mesh."""
    mesh: pv.DataSet = pv.read(vtk_path)
    # min 每列坐标
    xmin, ymin, zmin = mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]
    # 若已在原点则直接返回
    if xmin == 0 and ymin == 0 and zmin == 0:
        return mesh
    # 平移
    mesh_translated = mesh.translate((-xmin, -ymin, -zmin), inplace=False)
    return mesh_translated


def main() -> None:
    #  case name
    case_name, datetime = ask_case_name()

    # find ./caseData/<caseName>
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    case_dir = parent_dir / "caseData" / case_name

    if not case_dir.is_dir():
        sys.exit(f"Dir not exists: {case_dir}")

    try:
        vtk_in = find_latest_vtk(case_dir, datetime)
    except FileNotFoundError as err:
        sys.exit(f"{err}")

    print(f"  Selected latest VTK: {vtk_in.name}")

    # read and translate
    mesh_out = translate_to_origin(vtk_in)

    # save
    out_path = vtk_in.with_name("uvw-"+case_name+"_"+datetime+".vtk")
    mesh_out.save(out_path, binary=True)  # PyVista

    print(f"  Saved translated file: {out_path}")

if __name__ == "__main__":
    main()

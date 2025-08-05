"""Validation script to check preprocessing outputs.

Functions:
1. Load voxelization output STL with basement and report XYZ ranges.
2. Load buildBC output CSV and report XYZ ranges.
3. Compare XY ranges between the two; if all relative errors for min/max/span are below 0.01%,
   validation passes. Otherwise print highlighted warning.
4. Ensure certain fields exist in conf.txt with default values. Write validation result to line 38.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import trimesh
import pandas as pd
import subprocess

TOL = 1e-4  # 0.01%

def default_memory_lbm() -> str:
    """Return default GPU memory (MiB) as 85% of GPU0 capacity.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
        total_mem_mib = int(subprocess.check_output(cmd).decode().split()[0])
        print(f"    GPU0 Memory is {total_mem_mib} MiB")
        return str(int(total_mem_mib * 0.85))
    except Exception:
        return "20000"

def stl_ranges(stl_path: Path) -> dict[str, tuple[float, float, float]]:
    mesh = trimesh.load(stl_path, force="mesh")
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    return {
        "x": (xmin, xmax, xmax - xmin),
        "y": (ymin, ymax, ymax - ymin),
        "z": (zmin, zmax, zmax - zmin),
    }

def csv_ranges(csv_path: Path) -> dict[str, tuple[float, float, float]]:
    data = pd.read_csv(csv_path, usecols=["X", "Y", "Z"])
    xmin, ymin, zmin = data.min()
    xmax, ymax, zmax = data.max()
    return {
        "x": (float(xmin), float(xmax), float(xmax - xmin)),
        "y": (float(ymin), float(ymax), float(ymax - ymin)),
        "z": (float(zmin), float(zmax), float(zmax - zmin)),
    }

def compare_xy(stl: dict, csv: dict) -> tuple[bool, dict]:
    res = {}
    max_err = 0.0
    for axis in ("x", "y"):
        smin, smax, sspan = stl[axis]
        cmin, cmax, cspan = csv[axis]
        err_min = abs(smin - cmin) / (abs(smin) if smin != 0 else 1.0)
        err_max = abs(smax - cmax) / (abs(smax) if smax != 0 else 1.0)
        err_span = abs(sspan - cspan) / (abs(sspan) if sspan != 0 else 1.0)
        res[axis] = {"min": err_min, "max": err_max, "span": err_span}
        max_err = max(max_err, err_min, err_max, err_span)
    return max_err < TOL, res

def ensure_conf_fields(conf_path: Path) -> list[str]:
    """Ensure required fields exist in conf.txt. Returns modified lines."""
    if not conf_path.exists():
        tmpl = conf_path.parent / "template-conf.txt"
        if tmpl.exists():
            conf_path.write_text(tmpl.read_text())
            print("[!] conf.txt not found. Created from template-conf.txt")
        else:
            conf_path.write_text("")
            print("[!] conf.txt not found. Created empty conf.txt")
    lines = conf_path.read_text().splitlines()
    # Ensure there are enough placeholder lines before inserting defaults so
    # that we can target specific line numbers later (34-37 in 1-based index).
    while len(lines) < 33:
        lines.append("")

    def has_key(key: str) -> int | None:
        for i, ln in enumerate(lines):
            raw = ln.split("//")[0].strip()
            if raw.split("//")[0].strip().startswith(key):
                parts = raw.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    return i

        return None

    # Ensure the marker and default fields appear exactly at lines 34-37.
    # Line numbers here are 1-based in comments for clarity.

    # Line 34: "// CFD control" marker
    if lines[33].strip() != "// CFD control":
        lines.insert(33, "// CFD control")

    # Lines 35-37: required configuration keys with defaults
    defaults: list[tuple[str, str | None]] = [
        ("n_gpu", "[1, 1, 1]"),
        ("datetime", "20260101120000"),
        ("memory_lbm", None),
    ]
    for i, (key, val) in enumerate(defaults):
        if has_key(key) is None:
            line_no = 35 + i  # desired 1-based line number for this key
            if key == "memory_lbm":
                val = default_memory_lbm()
            lines.insert(line_no - 1, f"{key} = {val}")
            print(f"[!] Field '{key}' missing. Set default {val} in conf.txt")

    # Pad to 38 lines so that validation result can be written at line 38 later
    while len(lines) < 38:
        lines.append("")

    return lines

def write_validation(conf_lines: list[str], conf_path: Path, passed: bool) -> None:
    while len(conf_lines) < 38:
        conf_lines.append("")
    conf_lines[37] = f"validation = {'pass' if passed else 'error'}"
    conf_path.write_text("\n".join(conf_lines) + "\n")

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    conf_path = root / "conf.txt"
    lines = ensure_conf_fields(conf_path)
    # obtain case name and datetime
    caseName = "example"
    dt_str = "20250723120000"
    for ln in lines:
        raw = ln.split("//")[0].strip()
        if raw.startswith("casename") and "=" in raw:
            caseName = raw.split("=",1)[1].strip()
        elif raw.startswith("datetime") and "=" in raw:
            dt_str = raw.split("=",1)[1].strip()
    stl_path = root / "geoData" / caseName / f"{caseName}_with_base.stl"
    csv_path = root / "wrfInput" / caseName / f"SurfData_{dt_str}.csv"
    if not csv_path.exists():
        alt = csv_path.with_name("SurfData_Latest.csv")
        if alt.exists():
            csv_path = alt
    try:
        stl = stl_ranges(stl_path)
        csv = csv_ranges(csv_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        write_validation(lines, conf_path, False)
        sys.exit(1)

    print("STL ranges:")
    for ax, (mn, mx, sp) in stl.items():
        print(f"    {ax.upper()}: min={mn:.3f}, max={mx:.3f}, span={sp:.3f}")
    print("CSV ranges:")
    for ax, (mn, mx, sp) in csv.items():
        print(f"    {ax.upper()}: min={mn:.3f}, max={mx:.3f}, span={sp:.3f}")

    passed, errs = compare_xy(stl, csv)
    if passed:
        print("Validation passed. Maximum XY relative error "
              f"{max(max(v.values()) for v in errs.values())*100:.6f}%")
    else:
        border = "=" * 60
        print(border)
        print("WARNING: XY range mismatch exceeds 0.01%!")
        for ax, e in errs.items():
            print(f"  Axis {ax}: min={e['min']*100:.6f}%, max={e['max']*100:.6f}%, span={e['span']*100:.6f}%")
        print(border)
    write_validation(lines, conf_path, passed)

if __name__ == "__main__":
    main()

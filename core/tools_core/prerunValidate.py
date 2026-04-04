"""Validation script to check preprocessing outputs.
"""
from __future__ import annotations

from pathlib import Path
import sys
import re
import numpy as np
import trimesh
import pandas as pd
import subprocess

_CORE_DIR = Path(__file__).resolve().parents[1]
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from deck_io import DeckDocument, load_deck, parse_deck_text

TOL = 1e-3  # 0.1%

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

SCALE_BACK = 1.0  # STL coordinates were saved at 1/100 scale


def stl_ranges(stl_path: Path) -> dict[str, tuple[float, float, float]]:
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.apply_scale(SCALE_BACK)
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
        # Use span-based normalization; min can be 0 for local-domain coordinates,
        # where abs(smin) would make the relative error meaningless.
        denom = abs(sspan) if sspan != 0 else (max(abs(smin), abs(smax), 1.0))
        err_min = abs(smin - cmin) / denom
        err_max = abs(smax - cmax) / denom
        err_span = abs(sspan - cspan) / denom
        res[axis] = {"min": err_min, "max": err_max, "span": err_span}
        max_err = max(max_err, err_min, err_max, err_span)
    return max_err < TOL, res

def ensure_conf_fields(conf_path: Path) -> DeckDocument:
    """Ensure required deck fields exist and save a canonical sectioned deck."""
    if conf_path.exists():
        deck = load_deck(conf_path)
    else:
        tmpl = conf_path.parent / "template-conf.txt"
        if tmpl.exists():
            deck = parse_deck_text(tmpl.read_text(encoding="utf-8", errors="ignore"))
            print("[!] deck file not found. Created from template-conf.txt")
        else:
            deck = parse_deck_text("")
            print("[!] deck file not found. Created empty deck")

    if not deck.get_text("datetime"):
        deck.set_text("datetime", "20990101120000")
        print("[!] Field 'datetime' missing. Set default.")

    if not deck.get_list("n_gpu"):
        deck.set_list("n_gpu", [1, 1, 1])
        print("[!] Field 'n_gpu' missing. Wrote default value.")

    mesh_control = (deck.get_text("mesh_control") or "").strip().lower()
    cell_size_raw = deck.get_raw("cell_size")
    has_cell_size_value = cell_size_raw is not None and cell_size_raw.strip() != ""
    if not mesh_control:
        deck.set_text("mesh_control", "gpu_memory", quoted=True)
        mesh_control = "gpu_memory"
        print("[!] Field 'mesh_control' missing. Wrote default value.")
    elif mesh_control == "cell_size" and not has_cell_size_value:
        deck.set_text("mesh_control", "gpu_memory", quoted=True)
        mesh_control = "gpu_memory"
        print("[!] 'mesh_control' set to 'gpu_memory' because 'cell_size' is missing")

    legacy_memory = deck.get_text("memory_lbm")
    if legacy_memory:
        print("[!] Removed legacy 'memory_lbm'")

    if mesh_control == "gpu_memory" and deck.get_int("gpu_memory") is None:
        migrated = None
        if legacy_memory:
            try:
                migrated = int(float(legacy_memory))
            except Exception:
                migrated = None
        deck.set_int("gpu_memory", migrated if migrated is not None else int(default_memory_lbm()))
        print("[!] Ensured 'gpu_memory'")

    if not deck.has("cell_size"):
        deck.set_raw("cell_size", "")
        print("[!] Inserted placeholder 'cell_size'")

    deck.remove("memory_lbm")
    deck.save(conf_path)
    return deck


def write_validation(deck: DeckDocument, conf_path: Path, passed: bool) -> DeckDocument:
    deck.set_text("validation", "pass" if passed else "error")
    if not deck.has("high_order"):
        deck.set_bool("high_order", True)
    if not deck.has("flux_correction"):
        deck.set_bool("flux_correction", True)
    deck.save(conf_path)
    return deck



def main() -> None:
    print("LUW Pre-run Validation Tool...")
    if len(sys.argv) != 2:
        print("Usage: python prerun_validation.py <path-to-deck-file>")
        sys.exit(2)

    conf_path = Path(sys.argv[1]).expanduser().resolve()
    project_home = conf_path.parent

    # 确保必要字段存在，并读取 casename 与 datetime
    deck = ensure_conf_fields(conf_path)
    caseName = deck.get_text("casename") or "example"
    dt_str = deck.get_text("datetime") or "20990101120000"

    # 解析手工裁剪范围，用于确定 STL 主文件名中的经纬度标签（微度整数，无小数点）
    if deck.get_pair("cut_lon_manual") is None or deck.get_pair("cut_lat_manual") is None:
        print("ERROR: conf 缺少 cut_lon_manual/cut_lat_manual")
        deck = write_validation(deck, conf_path, False)
        sys.exit(1)

    proj_temp = project_home / "proj_temp"
    stl_dem_path = proj_temp / f"{caseName}_DEM.stl"
    if stl_dem_path.exists():
        stl_path = stl_dem_path
    else:
        stl_path = proj_temp / f"{caseName}.stl"
    print(f"Using STL file: {stl_path}")

    # CSV 主备候选：优先 SurfData_{datetime}.csv，不存在则使用 SurfData_Latest.csv
    csv_path = proj_temp / f"SurfData_{dt_str}.csv"
    if not csv_path.exists():
        alt = proj_temp / "SurfData_Latest.csv"
        if alt.exists():
            csv_path = alt

    try:
        stl = stl_ranges(stl_path)
        csv = csv_ranges(csv_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        deck = write_validation(deck, conf_path, False)
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
        print("WARNING: XY range mismatch exceeds 0.1%!")
        for ax, e in errs.items():
            print(f"  Axis {ax}: min={e['min']*100:.6f}%, max={e['max']*100:.6f}%, span={e['span']*100:.6f}%")
        print(border)
    write_validation(deck, conf_path, passed)

if __name__ == "__main__":
    main()

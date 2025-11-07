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
        err_min = abs(smin - cmin) / (abs(smin) if smin != 0 else 1.0)
        err_max = abs(smax - cmax) / (abs(smax) if smax != 0 else 1.0)
        err_span = abs(sspan - cspan) / (abs(sspan) if sspan != 0 else 1.0)
        res[axis] = {"min": err_min, "max": err_max, "span": err_span}
        max_err = max(max_err, err_min, err_max, err_span)
    return max_err < TOL, res

def ensure_conf_fields(conf_path: Path) -> list[str]:
    """Ensure required fields exist in conf.txt, anchored by '// CFD control'.
    Returns modified lines.
    """
    if not conf_path.exists():
        tmpl = conf_path.parent / "template-conf.txt"
        if tmpl.exists():
            conf_path.write_text(tmpl.read_text())
            print("[!] conf.txt not found. Created from template-conf.txt")
        else:
            conf_path.write_text("")
            print("[!] conf.txt not found. Created empty conf.txt")

    lines = conf_path.read_text().splitlines()

    def ensure_len(n: int) -> None:
        while len(lines) < n:
            lines.append("")

    def find_key_idx(key: str) -> int | None:
        for i, ln in enumerate(lines):
            raw = ln.split("//")[0].strip()
            if raw.startswith(key):
                parts = raw.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    return i
        return None

    def get_value_from_line(ln: str) -> str | None:
        raw = ln.split("//")[0].strip()
        parts = raw.split("=", 1)
        if len(parts) == 2:
            return parts[1].strip().strip('"').strip("'")
        return None

    # datetime 缺失时，默认值写入第 4 行
    if find_key_idx("datetime") is None:
        ensure_len(4)
        lines.insert(3, "datetime = 20990101120000")
        print("[!] Field 'datetime' missing. Set default at line 4 in conf.txt")

    # 定位或补齐锚点“// CFD control”
    anchor_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == "// CFD control":
            anchor_idx = i
            break
    if anchor_idx is None:
        # 不存在则写入第 31 行
        ensure_len(31)
        lines.insert(30, "// CFD control")
        anchor_idx = 30
        print("[!] Inserted '// CFD control' at line 31")

    # 记录既有字段位置
    idx_ngpu = find_key_idx("n_gpu")
    idx_mesh = find_key_idx("mesh_control")
    idx_gpu_mem = find_key_idx("gpu_memory")
    idx_mem_lbm = find_key_idx("memory_lbm")
    idx_cell_size = find_key_idx("cell_size")

    # 若 mesh_control 已存在且为 "cell_size" 且不存在 cell_size 值，则回退到 gpu_memory 策略
    mesh_needs_gpu = False
    if idx_mesh is not None:
        mesh_val = get_value_from_line(lines[idx_mesh]) or ""
        if mesh_val.strip('"').strip("'") == "cell_size" and idx_cell_size is None:
            lines[idx_mesh] = 'mesh_control = "gpu_memory"'
            print("[!] 'mesh_control' set to 'gpu_memory' because 'cell_size' is missing")
            mesh_needs_gpu = True


    # 统一使用“写到指定行”的方式，避免 insert 产生空行和位移
    def set_at(pos: int, text: str) -> None:
        ensure_len(pos + 1)
        lines[pos] = text

    # 先删除 legacy memory_lbm，避免后续相对锚点的位置被位移
    if idx_mem_lbm is not None:
        for i, ln in enumerate(lines):
            if ln.split("//")[0].strip().startswith("memory_lbm"):
                del lines[i]
                print("[!] Removed legacy 'memory_lbm'")
                break
        # 删除后，刷新索引
        idx_ngpu = find_key_idx("n_gpu")
        idx_mesh = find_key_idx("mesh_control")
        idx_gpu_mem = find_key_idx("gpu_memory")
        idx_cell_size = find_key_idx("cell_size")

    # n_gpu 在锚点下一行，缺则写入
    if idx_ngpu is None:
        set_at(anchor_idx + 1, "n_gpu = [1, 1, 1]")
        print("[!] Field 'n_gpu' missing. Wrote at anchor + 1")

    # mesh_control 在锚点后两行，缺则写入默认值
    if idx_mesh is None:
        set_at(anchor_idx + 2, 'mesh_control = "gpu_memory"')
        idx_mesh = anchor_idx + 2
        print("[!] Field 'mesh_control' missing. Wrote at anchor + 2")

    # 若原先为 cell_size 且缺少 cell_size 值，已在上文改写为 gpu_memory，这里确保 idx_mesh 可用
    if mesh_needs_gpu and idx_mesh is None:
        idx_mesh = find_key_idx("mesh_control")

    # gpu_memory 在锚点后三行，按规则确保存在
    need_gpu_memory = (find_key_idx("gpu_memory") is None) or mesh_needs_gpu
    if need_gpu_memory:
        val_from_mem_lbm = None
        # 再次尝试读取 legacy memory_lbm 的值，用于迁移
        idx_mem_lbm2 = find_key_idx("memory_lbm")
        if idx_mem_lbm2 is not None:
            v = get_value_from_line(lines[idx_mem_lbm2])
            if v:
                val_from_mem_lbm = v
            # 同时删除遗留字段
            del lines[idx_mem_lbm2]
            print("[!] Removed legacy 'memory_lbm' during gpu_memory ensure")
        gpu_mem_val = val_from_mem_lbm or default_memory_lbm()
        set_at(anchor_idx + 3, f"gpu_memory = {gpu_mem_val}")
        print("[!] Ensured 'gpu_memory' at anchor + 3")

    # 若不存在 cell_size 字段，则在 gpu_memory 后一行补充占位
    # 先定位 gpu_memory 的实际行号
    gpu_idx = None
    for i, ln in enumerate(lines):
        if ln.split("//")[0].strip().startswith("gpu_memory"):
            gpu_idx = i
            break
    if gpu_idx is None:
        gpu_idx = anchor_idx + 3  # 回退到规范位置

    has_cell_size = any(ln.split("//")[0].strip().startswith("cell_size") for ln in lines)
    if not has_cell_size:
        set_at(gpu_idx + 1, "cell_size = ")
        print("[!] Inserted placeholder 'cell_size' after 'gpu_memory'")




    if idx_mem_lbm is not None:
        # 重新定位以防插入造成位移
        for i, ln in enumerate(lines):
            raw = ln.split("//")[0].strip()
            if raw.startswith("memory_lbm"):
                del lines[i]
                print("[!] Removed legacy 'memory_lbm'")
                break

    return lines

def write_validation(conf_lines: list[str], conf_path: Path, passed: bool) -> None:
    # 以“// CFD control”为锚，validation 写在其后第 5 行，同时在其下一行与下下一行按需补 high_order 与 flux_correction
    anchor_idx = None
    for i, ln in enumerate(conf_lines):
        if ln.strip() == "// CFD control":
            anchor_idx = i
            break

    if anchor_idx is None:
        # 极端情况未找到锚点时，把 validation 追加到文件末尾
        while len(conf_lines) < 1:
            conf_lines.append("")
        conf_lines.append(f"validation = {'pass' if passed else 'error'}")
        val_idx = len(conf_lines) - 1
    else:
        target = anchor_idx + 5
        while len(conf_lines) <= target:
            conf_lines.append("")
        conf_lines[target] = f"validation = {'pass' if passed else 'error'}"
        val_idx = target

    # 检查键是否已存在，忽略注释与空值
    def has_key(name: str) -> bool:
        for ln in conf_lines:
            raw = ln.split("//")[0].strip()
            if raw.startswith(name):
                parts = raw.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    return True
        return False

    # 若不存在 high_order 与 flux_correction 需要补写
    if not has_key("high_order"):
        insert_idx = val_idx + 1
        while insert_idx < len(conf_lines) and conf_lines[insert_idx].strip() != "":
            insert_idx += 1
        if insert_idx == len(conf_lines):
            conf_lines.append("high_order = true")
        else:
            conf_lines[insert_idx] = "high_order = true"

    if not has_key("flux_correction"):
        insert_idx = val_idx + 1
        while insert_idx < len(conf_lines) and conf_lines[insert_idx].strip() != "":
            insert_idx += 1
        if insert_idx == len(conf_lines):
            conf_lines.append("flux_correction = true")
        else:
            conf_lines[insert_idx] = "flux_correction = true"


    conf_path.write_text("\n".join(conf_lines) + "\n")



def main() -> None:
    print("LUW Pre-run Validation Tool...")
    if len(sys.argv) != 2:
        print("Usage: python prerun_validation.py <path-to-deck-file>")
        sys.exit(2)

    conf_path = Path(sys.argv[1]).expanduser().resolve()
    project_home = conf_path.parent

    # 确保必要字段存在，并读取 casename 与 datetime
    lines = ensure_conf_fields(conf_path)
    caseName = "example"
    dt_str = "20990101120000"
    for ln in lines:
        raw = ln.split("//")[0].strip()
        if raw.startswith("casename") and "=" in raw:
            caseName = raw.split("=", 1)[1].strip()
        elif raw.startswith("datetime") and "=" in raw:
            dt_str = raw.split("=", 1)[1].strip()

    # 解析手工裁剪范围，用于确定 STL 主文件名中的经纬度标签（微度整数，无小数点）
    txt = conf_path.read_text(encoding="utf-8", errors="ignore")
    m_lon = re.search(r"cut_lon_manual\s*=\s*\[([^\]]+)\]", txt)
    m_lat = re.search(r"cut_lat_manual\s*=\s*\[([^\]]+)\]", txt)
    if not (m_lon and m_lat):
        print("ERROR: conf 缺少 cut_lon_manual/cut_lat_manual")
        write_validation(lines, conf_path, False)
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

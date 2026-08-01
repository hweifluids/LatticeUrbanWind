#!/usr/bin/env python3
"""Run one LUW deck through a reference solver and CULBM, then compare them."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


NORMAL_SPEED_RE = re.compile(
    r"Normal benchmark\s*\|\s*normal Steps/s\s*=\s*([0-9]+(?:\.[0-9]+)?)"
)
STAGE_NORMAL_SPEED_RE = re.compile(
    r"(?:Mean-field stage|ETA model)\s*\|\s*entered\.\s*"
    r"normal Steps/s\s*=\s*([0-9]+(?:\.[0-9]+)?)"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
VTK_PAYLOAD_MARKER = b"LOOKUP_TABLE default\n"


@dataclass
class FieldComparison:
    name: str
    comparison_mode: str
    values: int
    exact_payload: bool
    finite_mismatches: int
    nonfinite_mismatches: int
    max_abs: float
    max_rel: float
    rms: float
    reference_rms: float | None
    candidate_rms: float | None
    relative_l2: float
    correlation: float
    max_ulp: int
    passed: bool


def parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    repo = here.parents[4]
    project = here.parents[1]
    executable_suffix = ".exe" if os.name == "nt" else ""
    p = argparse.ArgumentParser(
        description=(
            "Run the same multi-GPU LUW case with a reference solver and "
            "CULBM, then enforce numerical and speed gates."
        )
    )
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Explicit baseline executable used for the A/B regression.",
    )
    p.add_argument(
        "--cuda",
        type=Path,
        default=project / "bin" / f"CULBM{executable_suffix}",
    )
    p.add_argument(
        "--cuda-mpi-ranks",
        type=int,
        default=0,
        help=(
            "Launch the CUDA candidate with this many MPI ranks. Zero keeps "
            "the existing single-process local multi-GPU path."
        ),
    )
    p.add_argument(
        "--mpiexec",
        type=Path,
        default=Path(shutil.which("mpiexec") or "mpiexec"),
        help="MPI launcher used when --cuda-mpi-ranks is positive.",
    )
    p.add_argument(
        "--source-case",
        type=Path,
        default=repo / "examples" / "example_ProfileResearch_noDEM",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=project / "test-output",
        help="A unique run directory is created below this directory.",
    )
    p.add_argument("--cell-size", type=float, default=10.0)
    p.add_argument(
        "--domains-x",
        type=int,
        default=2,
        help="Cartesian reference/CUDA domain count in x (positive integer).",
    )
    p.add_argument(
        "--domains-y",
        type=int,
        default=1,
        help="Cartesian reference/CUDA domain count in y (positive integer).",
    )
    p.add_argument(
        "--domains-z",
        type=int,
        default=1,
        help="Cartesian reference/CUDA domain count in z (positive integer).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=1001,
        help=(
            "Keep this above 960 so the built-in 16-step normal benchmark "
            "does not include offline rendering."
        ),
    )
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--abs-tol-u", type=float, default=2.0e-5)
    p.add_argument("--rel-tol-u", type=float, default=2.0e-5)
    p.add_argument("--abs-tol-rho", type=float, default=2.0e-6)
    p.add_argument("--rel-tol-rho", type=float, default=2.0e-6)
    p.add_argument("--abs-tol-t", type=float, default=2.0e-6)
    p.add_argument("--rel-tol-t", type=float, default=2.0e-6)
    p.add_argument("--max-ulp-u", type=int, default=4)
    p.add_argument("--max-ulp-rho", type=int, default=4)
    p.add_argument("--max-ulp-t", type=int, default=4)
    p.add_argument(
        "--comparison-mode",
        choices=("pointwise", "statistical"),
        default="pointwise",
        help=(
            "pointwise enforces absolute, relative, and ULP limits; "
            "statistical is available for cross-architecture diagnosis."
        ),
    )
    p.add_argument("--stat-rel-l2-tol-u", type=float, default=1.0e-3)
    p.add_argument("--stat-rel-l2-tol-rho", type=float, default=5.0e-5)
    p.add_argument("--stat-rel-l2-tol-t", type=float, default=5.0e-5)
    p.add_argument("--stat-min-correlation-u", type=float, default=0.99999)
    p.add_argument("--stat-min-correlation-rho", type=float, default=0.999)
    p.add_argument("--stat-min-correlation-t", type=float, default=0.999)
    p.add_argument(
        "--min-speed-ratio",
        type=float,
        default=0.5,
        help="CUDA Steps/s divided by reference Steps/s.",
    )
    p.add_argument(
        "--enable-vk",
        action="store_true",
        help="Exercise the active Von Karman inlet kernels as well.",
    )
    p.add_argument(
        "--case-kind",
        choices=("profile", "standard"),
        default="profile",
        help=(
            "profile uses the production .luwpf path; standard synthesizes "
            "patch velocity/temperature SurfData and uses the .luw path."
        ),
    )
    p.add_argument(
        "--enable-forcing-models",
        action="store_true",
        help="Enable Coriolis, downstream nudging, top sponge, and buoyancy.",
    )
    p.add_argument(
        "--forcing-component",
        action="append",
        choices=("buoyancy", "coriolis", "buffer", "sponge"),
        default=[],
        help="Enable one forcing component; repeat to combine components.",
    )
    p.add_argument(
        "--enable-flux-correction",
        action="store_true",
        help="Exercise boundary mass-flux correction.",
    )
    p.add_argument(
        "--exercise-output",
        action="store_true",
        help="Also generate and byte-compare averaging, probe, and unsteady I/O.",
    )
    return p


def write_standard_surface_data(
    case_dir: Path,
    datetime: str,
    size_x: float,
    size_y: float,
    size_z: float,
) -> None:
    """Create deterministic six-patch velocity and temperature input."""

    rows = ["X,Y,Z,u,v,w,T,patch"]
    samples_per_axis = 5
    fractions = [
        index / (samples_per_axis - 1) for index in range(samples_per_axis)
    ]

    def append(
        x: float,
        y: float,
        z: float,
        patch: int,
        velocity_y: float,
    ) -> None:
        temperature = (
            288.0
            + 8.0 * (x / size_x)
            + 2.0 * (y / size_y)
            + 5.0 * (z / size_z)
        )
        rows.append(
            f"{x:.9g},{y:.9g},{z:.9g},1,{velocity_y:.9g},0,"
            f"{temperature:.9g},{patch}"
        )

    for fx in fractions:
        for fy in fractions:
            x, y = fx * size_x, fy * size_y
            append(x, y, 0.0, 0, 0.0)
            append(x, y, size_z, 1, 8.0)
    for fx in fractions:
        for fz in fractions:
            x, z = fx * size_x, fz * size_z
            speed = 8.0 + 2.0 * fz
            append(x, 0.0, z, 2, speed)
            append(x, size_y, z, 3, speed)
    for fy in fractions:
        for fz in fractions:
            y, z = fy * size_y, fz * size_z
            speed = 8.0 + 2.0 * fz
            append(0.0, y, z, 4, speed)
            append(size_x, y, z, 5, speed)

    csv_path = case_dir / "proj_temp" / f"SurfData_{datetime}.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def clean_console(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")


def make_deck(
    case_dir: Path,
    cell_size: float,
    steps: int,
    enable_vk: bool,
    forcing_components: set[str],
    enable_flux_correction: bool,
    exercise_output: bool,
    case_kind: str,
    domains_x: int,
    domains_y: int,
    domains_z: int,
) -> Path:
    size_x = 2022.500153
    size_y = 1996.500092
    size_z = 270.0
    datetime = "20990102000000"
    source_mesh = case_dir / "_source" / "proj_temp" / "CaseE_PF.stl"
    source_profile = case_dir / "_source" / "wind_bc" / "profile.dat"
    target_mesh = case_dir / "proj_temp" / "CaseE_PF.stl"
    target_profile = case_dir / "wind_bc" / "profile.dat"
    target_mesh.parent.mkdir(parents=True, exist_ok=True)
    target_profile.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mesh, target_mesh)
    shutil.copy2(source_profile, target_profile)

    vk_settings = (
        "vk_inlet_enable = true\n"
        "vk_inlet_nmodes = 32\n"
        "vk_inlet_update_stride = 1\n"
        "vk_inlet_ti = 0.08\n"
        "vk_inlet_l = 50\n"
        "vk_inlet_seed = 20260728\n"
        if enable_vk
        else "vk_inlet_enable = false\n"
    )
    buoyancy_enabled = (
        case_kind == "standard" or "buoyancy" in forcing_components
    )
    coriolis_enabled = "coriolis" in forcing_components
    buffer_enabled = "buffer" in forcing_components
    sponge_enabled = "sponge" in forcing_components
    forcing_settings = (
        f"buoyancy = {'true' if buoyancy_enabled else 'false'}\n"
        f"coriolis_term = {'true' if coriolis_enabled else 'false'}\n"
        "cut_lon_manual = [-0.01, 0.01]\n"
        "cut_lat_manual = [51.491, 51.509]\n"
        f"enable_buffer_nudging = {'true' if buffer_enabled else 'false'}\n"
        "buffer_thickness_m = 80\n"
        "buffer_tau_s = 60\n"
        "buffer_nudge_vertical = true\n"
        f"enable_top_sponge = {'true' if sponge_enabled else 'false'}\n"
        "sponge_thickness_m = 40\n"
        "sponge_tau_s = 60\n"
        "sponge_ref_mode = 0\n"
    )
    if case_kind == "standard":
        write_standard_surface_data(
            case_dir, datetime, size_x, size_y, size_z
        )
        mode_settings = 'downstream_bc = "+y"\n'
        deck = case_dir / "backend_compare.luw"
    else:
        mode_settings = "x_exp_rat = 5\ny_exp_rat = 5\nangle = [0]\n"
        deck = case_dir / "backend_compare.luwpf"
    if exercise_output:
        output_settings = (
            f"purge_avg = {steps}\n"
            f"unsteady_output = {max(1, steps // 4)}\n"
            f"probes_output = {steps}\n"
            "probes = center\n"
        )
    else:
        output_settings = "purge_avg = 0\nunsteady_output = 0\n"
    deck.write_text(
        f"""// Generated by tests/compare_backends.py.
casename = CaseE
datetime = {datetime}

si_x_cfd = [0.0, {size_x}]
si_y_cfd = [0.0, {size_y}]
si_z_cfd = [0.0, {size_z}]
base_height = 20.0
z_limit = 250

n_gpu = [{domains_x}, {domains_y}, {domains_z}]
mesh_control = "cell_size"
cell_size = {cell_size:.9g}
validation = pass
high_order = true
flux_correction = {"true" if enable_flux_correction else "false"}
run_nstep = {steps}
{output_settings}

{forcing_settings}
{vk_settings}
{mode_settings}
""",
        encoding="utf-8",
    )
    return deck


def prepare_case(root: Path, name: str, source_case: Path) -> tuple[Path, Path]:
    case_dir = root / name
    source_link = case_dir / "_source"
    (source_link / "proj_temp").mkdir(parents=True)
    (source_link / "wind_bc").mkdir(parents=True)
    shutil.copy2(
        source_case / "proj_temp" / "CaseE_PF.stl",
        source_link / "proj_temp" / "CaseE_PF.stl",
    )
    shutil.copy2(
        source_case / "wind_bc" / "profile.dat",
        source_link / "wind_bc" / "profile.dat",
    )
    return case_dir, source_link


def run_solver(
    label: str,
    executable: Path,
    deck: Path,
    log_path: Path,
    timeout: int,
    command_prefix: Sequence[str] = (),
) -> tuple[float, float]:
    started = time.perf_counter()
    command = [*command_prefix, str(executable), str(deck)]
    completed = subprocess.run(
        command,
        cwd=str(executable.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout.decode("utf-8", errors="replace")
    clean_output = clean_console(output)
    log_path.write_text(clean_output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} exited with {completed.returncode}; see {log_path}"
        )
    match = NORMAL_SPEED_RE.search(clean_output)
    if not match:
        match = STAGE_NORMAL_SPEED_RE.search(clean_output)
    if not match:
        raise RuntimeError(f"{label} normal Steps/s was not found in {log_path}")
    return float(match.group(1)), elapsed


def find_field(case_dir: Path, field: str, steps: int) -> Path:
    expected = (
        case_dir
        / "RESULTS"
        / "vtk"
        / f"20990102000000_raw_{field}-{steps:09d}.vtk"
    )
    if not expected.is_file():
        candidates = sorted((case_dir / "RESULTS" / "vtk").glob(f"*_{field}-*.vtk"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one {field} VTK under {case_dir}, found {candidates}"
            )
        return candidates[0]
    return expected


def vtk_payload(path: Path) -> tuple[bytes, int]:
    raw = path.read_bytes()
    marker_at = raw.find(VTK_PAYLOAD_MARKER)
    if marker_at < 0:
        raise RuntimeError(f"Unsupported VTK payload in {path}")
    payload_at = marker_at + len(VTK_PAYLOAD_MARKER)
    header = raw[:payload_at].decode("ascii", errors="strict")
    point_match = re.search(r"^POINT_DATA\s+(\d+)\s*$", header, re.MULTILINE)
    scalar_match = re.search(
        r"^SCALARS\s+\S+\s+float(?:\s+(\d+))?\s*$", header, re.MULTILINE
    )
    if not point_match or not scalar_match:
        raise RuntimeError(f"Missing POINT_DATA/SCALARS declaration in {path}")
    components = int(scalar_match.group(1) or "1")
    values = int(point_match.group(1)) * components
    payload = raw[payload_at : payload_at + 4 * values]
    if len(payload) != 4 * values:
        raise RuntimeError(
            f"Truncated VTK payload in {path}: {len(payload)} != {4 * values}"
        )
    return payload, values


def float_array_from_vtk(payload: bytes) -> array.array:
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder == "little":
        values.byteswap()
    return values


def ordered_float_bits(value: float) -> int:
    bits = struct.unpack("=I", struct.pack("=f", value))[0]
    return (0x80000000 - bits) if bits & 0x80000000 else (bits + 0x80000000)


def compare_field(
    name: str,
    reference_path: Path,
    candidate_path: Path,
    abs_tol: float,
    rel_tol: float,
    maximum_ulp: int,
    comparison_mode: str,
    statistical_relative_l2_tolerance: float,
    statistical_minimum_correlation: float,
) -> FieldComparison:
    reference_payload, reference_count = vtk_payload(reference_path)
    candidate_payload, candidate_count = vtk_payload(candidate_path)
    if reference_count != candidate_count:
        raise RuntimeError(
            f"{name} value count differs: {reference_count} != {candidate_count}"
        )
    exact = hashlib.sha256(reference_payload).digest() == hashlib.sha256(
        candidate_payload
    ).digest()
    if exact:
        return FieldComparison(
            name=name,
            comparison_mode=comparison_mode,
            values=reference_count,
            exact_payload=True,
            finite_mismatches=0,
            nonfinite_mismatches=0,
            max_abs=0.0,
            max_rel=0.0,
            rms=0.0,
            reference_rms=None,
            candidate_rms=None,
            relative_l2=0.0,
            correlation=1.0,
            max_ulp=0,
            passed=True,
        )

    reference = float_array_from_vtk(reference_payload)
    candidate = float_array_from_vtk(candidate_payload)
    finite_mismatches = 0
    nonfinite_mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    sum_squared = 0.0
    sum_reference = 0.0
    sum_candidate = 0.0
    sum_reference_squared = 0.0
    sum_candidate_squared = 0.0
    sum_cross = 0.0
    finite_values = 0
    max_ulp = 0
    passed = True
    for expected, actual in zip(reference, candidate):
        expected_finite = math.isfinite(expected)
        actual_finite = math.isfinite(actual)
        if not expected_finite or not actual_finite:
            same_nonfinite = (
                math.isnan(expected)
                and math.isnan(actual)
                or expected == actual
            )
            if not same_nonfinite:
                nonfinite_mismatches += 1
                passed = False
            continue
        difference = abs(actual - expected)
        scale = max(abs(expected), abs(actual))
        relative = difference / scale if scale > 0.0 else 0.0
        finite_values += 1
        sum_squared += difference * difference
        sum_reference += expected
        sum_candidate += actual
        sum_reference_squared += expected * expected
        sum_candidate_squared += actual * actual
        sum_cross += expected * actual
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, relative)
        max_ulp = max(
            max_ulp,
            abs(ordered_float_bits(actual) - ordered_float_bits(expected)),
        )
        if difference != 0.0:
            finite_mismatches += 1
        if difference > abs_tol + rel_tol * scale:
            if comparison_mode == "pointwise":
                passed = False
    reference_rms = (
        math.sqrt(sum_reference_squared / finite_values)
        if finite_values
        else 0.0
    )
    candidate_rms = (
        math.sqrt(sum_candidate_squared / finite_values)
        if finite_values
        else 0.0
    )
    relative_l2 = (
        math.sqrt(sum_squared / sum_reference_squared)
        if sum_reference_squared > 0.0
        else (0.0 if sum_squared == 0.0 else math.inf)
    )
    if finite_values:
        reference_mean = sum_reference / finite_values
        candidate_mean = sum_candidate / finite_values
        reference_variance = max(
            0.0,
            sum_reference_squared / finite_values
            - reference_mean * reference_mean,
        )
        candidate_variance = max(
            0.0,
            sum_candidate_squared / finite_values
            - candidate_mean * candidate_mean,
        )
        covariance = (
            sum_cross / finite_values - reference_mean * candidate_mean
        )
        variance_product = reference_variance * candidate_variance
        correlation = (
            covariance / math.sqrt(variance_product)
            if variance_product > 0.0
            else (1.0 if sum_squared == 0.0 else 0.0)
        )
        correlation = min(1.0, max(-1.0, correlation))
    else:
        correlation = 0.0
    if comparison_mode == "statistical":
        passed = (
            passed
            and nonfinite_mismatches == 0
            and relative_l2 <= statistical_relative_l2_tolerance
            and correlation >= statistical_minimum_correlation
        )
    else:
        passed = passed and max_ulp <= maximum_ulp
    return FieldComparison(
        name=name,
        comparison_mode=comparison_mode,
        values=reference_count,
        exact_payload=False,
        finite_mismatches=finite_mismatches,
        nonfinite_mismatches=nonfinite_mismatches,
        max_abs=max_abs,
        max_rel=max_rel,
        rms=math.sqrt(sum_squared / finite_values) if finite_values else 0.0,
        reference_rms=reference_rms,
        candidate_rms=candidate_rms,
        relative_l2=relative_l2,
        correlation=correlation,
        max_ulp=max_ulp,
        passed=passed,
    )


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) missing: " + ", ".join(missing))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_result_artifacts(
    reference_case: Path, candidate_case: Path
) -> dict[str, object]:
    reference_root = reference_case / "RESULTS"
    candidate_root = candidate_case / "RESULTS"

    def artifacts(root: Path) -> dict[str, Path]:
        return {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".vtk", ".csv"}
        }

    reference = artifacts(reference_root)
    candidate = artifacts(candidate_root)
    missing_from_cuda = sorted(set(reference) - set(candidate))
    extra_in_cuda = sorted(set(candidate) - set(reference))
    mismatched = sorted(
        relative
        for relative in set(reference) & set(candidate)
        if file_sha256(reference[relative]) != file_sha256(candidate[relative])
    )
    return {
        "reference_files": len(reference),
        "candidate_files": len(candidate),
        "missing_from_cuda": missing_from_cuda,
        "extra_in_cuda": extra_in_cuda,
        "byte_mismatches": mismatched,
        "passed": not missing_from_cuda
        and not extra_in_cuda
        and not mismatched,
    }


def main() -> int:
    args = parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.cell_size <= 0.0:
        raise ValueError("--cell-size must be positive")
    if min(args.domains_x, args.domains_y, args.domains_z) <= 0:
        raise ValueError("--domains-x/--domains-y/--domains-z must be positive")
    if args.cuda_mpi_ranks < 0:
        raise ValueError("--cuda-mpi-ranks must be non-negative")
    domain_count = args.domains_x * args.domains_y * args.domains_z
    if args.cuda_mpi_ranks and args.cuda_mpi_ranks != domain_count:
        raise ValueError(
            "--cuda-mpi-ranks must equal "
            "--domains-x*--domains-y*--domains-z"
        )
    if args.max_ulp_u < 0 or args.max_ulp_rho < 0 or args.max_ulp_t < 0:
        raise ValueError("--max-ulp-* values must be non-negative")
    comparison_mode = args.comparison_mode
    forcing_components = set(args.forcing_component)
    if args.enable_forcing_models:
        forcing_components.update(("buoyancy", "coriolis", "buffer", "sponge"))
    require_files(
        [
            args.reference,
            args.cuda,
            args.source_case / "proj_temp" / "CaseE_PF.stl",
            args.source_case / "wind_bc" / "profile.dat",
        ]
    )
    if args.cuda_mpi_ranks:
        require_files([args.mpiexec])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.resolve() / (
        f"{stamp}_cell{args.cell_size:g}_steps{args.steps}"
        + ("_standard" if args.case_kind == "standard" else "")
        + ("_vk" if args.enable_vk else "")
        + ("_forcing" if forcing_components else "")
        + ("_flux" if args.enable_flux_correction else "")
        + ("_io" if args.exercise_output else "")
    )
    suffix = 1
    unique_root = run_root
    while unique_root.exists():
        suffix += 1
        unique_root = Path(f"{run_root}_{suffix}")
    run_root = unique_root
    run_root.mkdir(parents=True)

    reference_case, _ = prepare_case(run_root, "reference", args.source_case.resolve())
    cuda_case, _ = prepare_case(run_root, "cuda", args.source_case.resolve())
    reference_deck = make_deck(
        reference_case,
        args.cell_size,
        args.steps,
        args.enable_vk,
        forcing_components,
        args.enable_flux_correction,
        args.exercise_output,
        args.case_kind,
        args.domains_x,
        args.domains_y,
        args.domains_z,
    )
    cuda_deck = make_deck(
        cuda_case,
        args.cell_size,
        args.steps,
        args.enable_vk,
        forcing_components,
        args.enable_flux_correction,
        args.exercise_output,
        args.case_kind,
        args.domains_x,
        args.domains_y,
        args.domains_z,
    )

    print(f"[1/4] Reference: {args.reference}")
    reference_speed, reference_wall = run_solver(
        "Reference",
        args.reference.resolve(),
        reference_deck.resolve(),
        run_root / "reference.log",
        args.timeout,
    )
    cuda_prefix = (
        [str(args.mpiexec.resolve()), "-n", str(args.cuda_mpi_ranks)]
        if args.cuda_mpi_ranks
        else []
    )
    launch_description = (
        " ".join([*cuda_prefix, str(args.cuda)])
        if cuda_prefix
        else str(args.cuda)
    )
    print(f"[2/4] CUDA:   {launch_description}")
    cuda_speed, cuda_wall = run_solver(
        "CUDA",
        args.cuda.resolve(),
        cuda_deck.resolve(),
        run_root / "cuda.log",
        args.timeout,
        cuda_prefix,
    )

    print("[3/4] Comparing final VTK field payloads")
    comparisons = [
        compare_field(
            "u",
            find_field(reference_case, "u", args.steps),
            find_field(cuda_case, "u", args.steps),
            args.abs_tol_u,
            args.rel_tol_u,
            args.max_ulp_u,
            comparison_mode,
            args.stat_rel_l2_tol_u,
            args.stat_min_correlation_u,
        ),
        compare_field(
            "rho",
            find_field(reference_case, "rho", args.steps),
            find_field(cuda_case, "rho", args.steps),
            args.abs_tol_rho,
            args.rel_tol_rho,
            args.max_ulp_rho,
            comparison_mode,
            args.stat_rel_l2_tol_rho,
            args.stat_min_correlation_rho,
        ),
    ]
    if args.case_kind == "standard":
        comparisons.append(
            compare_field(
                "T",
                find_field(reference_case, "T", args.steps),
                find_field(cuda_case, "T", args.steps),
                args.abs_tol_t,
                args.rel_tol_t,
                args.max_ulp_t,
                comparison_mode,
                args.stat_rel_l2_tol_t,
                args.stat_min_correlation_t,
            )
        )
    speed_ratio = cuda_speed / reference_speed
    io_comparison = (
        compare_result_artifacts(reference_case, cuda_case)
        if args.exercise_output
        else {
            "reference_files": 0,
            "candidate_files": 0,
            "missing_from_cuda": [],
            "extra_in_cuda": [],
            "byte_mismatches": [],
            "passed": True,
        }
    )
    numerical_passed = (
        all(item.passed for item in comparisons)
        and bool(io_comparison["passed"])
    )
    performance_passed = speed_ratio >= args.min_speed_ratio
    report = {
        "case": {
            "cell_size": args.cell_size,
            "steps": args.steps,
            "domain_decomposition": [
                args.domains_x,
                args.domains_y,
                args.domains_z,
            ],
            "cuda_mpi_ranks": args.cuda_mpi_ranks,
            "vk_enabled": args.enable_vk,
            "case_kind": args.case_kind,
            "forcing_components": sorted(forcing_components),
            "flux_correction_enabled": args.enable_flux_correction,
            "extended_io_enabled": args.exercise_output,
            "quantization": "FP16C",
            "comparison_mode": comparison_mode,
        },
        "performance": {
            "reference_normal_steps_per_second": reference_speed,
            "cuda_normal_steps_per_second": cuda_speed,
            "cuda_to_reference_ratio": speed_ratio,
            "minimum_ratio": args.min_speed_ratio,
            "reference_process_wall_seconds": reference_wall,
            "cuda_process_wall_seconds": cuda_wall,
            "passed": performance_passed,
        },
        "numerical_thresholds": {
            "pointwise": {
                "u": {
                    "absolute": args.abs_tol_u,
                    "relative": args.rel_tol_u,
                    "maximum_ulp": args.max_ulp_u,
                },
                "rho": {
                    "absolute": args.abs_tol_rho,
                    "relative": args.rel_tol_rho,
                    "maximum_ulp": args.max_ulp_rho,
                },
                "T": {
                    "absolute": args.abs_tol_t,
                    "relative": args.rel_tol_t,
                    "maximum_ulp": args.max_ulp_t,
                },
            },
            "statistical": {
                "u": {
                    "relative_l2": args.stat_rel_l2_tol_u,
                    "minimum_correlation": args.stat_min_correlation_u,
                },
                "rho": {
                    "relative_l2": args.stat_rel_l2_tol_rho,
                    "minimum_correlation": args.stat_min_correlation_rho,
                },
                "T": {
                    "relative_l2": args.stat_rel_l2_tol_t,
                    "minimum_correlation": args.stat_min_correlation_t,
                },
            },
        },
        "numerics": [asdict(item) for item in comparisons],
        "io_artifacts": io_comparison,
        "passed": numerical_passed and performance_passed,
    }
    report_path = run_root / "compare-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[4/4] Acceptance report")
    print(
        f"  speed: reference={reference_speed:.3f} Steps/s, "
        f"CUDA={cuda_speed:.3f} Steps/s, ratio={speed_ratio:.3f} "
        f"({'PASS' if performance_passed else 'FAIL'})"
    )
    for comparison in comparisons:
        print(
            f"  {comparison.name} [{comparison.comparison_mode}]: "
            f"exact={comparison.exact_payload}, "
            f"max_abs={comparison.max_abs:.9g}, "
            f"max_rel={comparison.max_rel:.9g}, "
            f"rel_l2={comparison.relative_l2:.9g}, "
            f"corr={comparison.correlation:.9g}, "
            f"max_ulp={comparison.max_ulp} "
            f"({'PASS' if comparison.passed else 'FAIL'})"
        )
    if args.exercise_output:
        print(
            "  I/O artifacts: "
            f"reference={io_comparison['reference_files']}, "
            f"CUDA={io_comparison['candidate_files']} "
            f"({'PASS' if io_comparison['passed'] else 'FAIL'})"
        )
    print(f"  report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

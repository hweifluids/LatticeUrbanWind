#!/usr/bin/env python3
"""Compare VTK fields at common physical coordinates, not raw payload offsets.

This tool is for regressions where an equivalent domain transformation (for
example, removal of permanently solid ground layers) changes grid extent or
origin. It matches samples by SI coordinate and reports field statistics plus
domain and per-plane QoIs. It deliberately does not require bytewise identity.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PAYLOAD_MARKER = b"LOOKUP_TABLE default\n"


@dataclass(frozen=True)
class VtkField:
    path: Path
    dimensions: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    scalar_type: str
    components: int
    values: array.array

    @property
    def points(self) -> int:
        nx, ny, nz = self.dimensions
        return nx * ny * nz


@dataclass
class PlaneQoi:
    z_si: float
    samples: int
    reference_mean: list[float]
    candidate_mean: list[float]
    max_abs_difference: float


@dataclass
class PhysicalComparison:
    reference: str
    candidate: str
    components: int
    matched_points: int
    matched_values: int
    skipped_bottom_planes: int
    masked_points: int
    mask_mismatches: int
    nonfinite_mismatches: int
    max_abs: float
    max_rel: float
    relative_l2: float
    correlation: float
    correlation_applicable: bool
    reference_mean: list[float]
    candidate_mean: list[float]
    qoi_max_relative_difference: float
    qoi_max_abs_difference: float
    pointwise_passed: bool
    qoi_passed: bool
    reference_mean_speed: float | None
    candidate_mean_speed: float | None
    plane_qoi: list[PlaneQoi]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--reference-mask",
        type=Path,
        help="Optional unsigned-char reference flags VTK; TYPE_S cells are excluded.",
    )
    parser.add_argument(
        "--candidate-mask",
        type=Path,
        help="Optional unsigned-char candidate flags VTK; TYPE_S cells are excluded.",
    )
    parser.add_argument("--abs-tol", type=float, default=2.0e-4)
    parser.add_argument("--rel-tol", type=float, default=2.0e-4)
    # Cross-grid comparisons retain physical QoIs and geometry masks, but do
    # not demand identical local rounding at every time step.
    parser.add_argument("--relative-l2-tol", type=float, default=1.0e-3)
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.999,
        help="Correlation gate for non-uniform fields; L2 and QoIs remain mandatory.",
    )
    parser.add_argument("--qoi-relative-tol", type=float, default=2.0e-4)
    parser.add_argument(
        "--qoi-abs-tol",
        type=float,
        default=2.0e-4,
        help="Absolute QoI tolerance, including means close to zero.",
    )
    parser.add_argument(
        "--skip-bottom-planes",
        type=int,
        default=0,
        help=(
            "Exclude this many lowest matched z planes, for example to omit "
            "a deliberately retained solid ground-boundary row."
        ),
    )
    parser.add_argument(
        "--require-pointwise",
        action="store_true",
        help="Also enforce pointwise absolute/relative tolerances.",
    )
    parser.add_argument(
        "--coordinate-tol",
        type=float,
        default=1.0e-5,
        help="Relative tolerance used to align physical lattice coordinates.",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def parse_vector(header: str, key: str, count: int) -> tuple[float, ...]:
    match = re.search(rf"^{key}\s+(.+?)\s*$", header, re.MULTILINE)
    if not match:
        raise RuntimeError(f"VTK header has no {key} declaration")
    values = tuple(float(value) for value in match.group(1).split())
    if len(values) != count:
        raise RuntimeError(f"VTK {key} has {len(values)} values; expected {count}")
    return values


def read_vtk(path: Path) -> VtkField:
    """Read the first scalar/vector field of a legacy binary VTK file."""

    raw = path.read_bytes()
    marker_offset = raw.find(PAYLOAD_MARKER)
    if marker_offset < 0:
        raise RuntimeError(f"Unsupported VTK payload: {path}")
    payload_offset = marker_offset + len(PAYLOAD_MARKER)
    header = raw[:payload_offset].decode("ascii", errors="strict")
    dimensions = tuple(int(value) for value in parse_vector(header, "DIMENSIONS", 3))
    origin = parse_vector(header, "ORIGIN", 3)
    spacing = parse_vector(header, "SPACING", 3)
    points_match = re.search(r"^POINT_DATA\s+(\d+)\s*$", header, re.MULTILINE)
    scalar_match = re.search(
        r"^SCALARS\s+\S+\s+(float|unsigned_char)(?:\s+(\d+))?\s*$",
        header,
        re.MULTILINE,
    )
    if not points_match or not scalar_match:
        raise RuntimeError(f"VTK data declaration is incomplete: {path}")
    points = int(points_match.group(1))
    if points != math.prod(dimensions):
        raise RuntimeError(f"VTK DIMENSIONS and POINT_DATA differ: {path}")
    scalar_type = scalar_match.group(1)
    components = int(scalar_match.group(2) or "1")
    bytes_per_value = 4 if scalar_type == "float" else 1
    payload = raw[
        payload_offset : payload_offset + bytes_per_value * points * components
    ]
    if len(payload) != bytes_per_value * points * components:
        raise RuntimeError(f"VTK payload is truncated: {path}")
    values = array.array("f" if scalar_type == "float" else "B")
    values.frombytes(payload)
    if scalar_type == "float" and sys.byteorder == "little":
        values.byteswap()
    return VtkField(
        path, dimensions, origin, spacing, scalar_type, components, values
    )


def matching_index(
    coordinate: float,
    origin: float,
    spacing: float,
    count: int,
    relative_tolerance: float,
) -> int | None:
    if not math.isfinite(coordinate) or spacing <= 0.0:
        return None
    raw_index = (coordinate - origin) / spacing
    index = int(round(raw_index))
    coordinate_tolerance = relative_tolerance * max(
        1.0, abs(spacing), abs(coordinate)
    )
    if index < 0 or index >= count:
        return None
    if abs(origin + index * spacing - coordinate) > coordinate_tolerance:
        return None
    return index


def relative_difference(reference: float, candidate: float) -> float:
    scale = max(abs(reference), abs(candidate))
    return abs(candidate - reference) / scale if scale > 0.0 else 0.0


def compare(
    reference: VtkField,
    candidate: VtkField,
    reference_mask: VtkField | None,
    candidate_mask: VtkField | None,
    abs_tolerance: float,
    rel_tolerance: float,
    relative_l2_tolerance: float,
    min_correlation: float,
    qoi_relative_tolerance: float,
    qoi_absolute_tolerance: float,
    coordinate_tolerance: float,
    skip_bottom_planes: int,
    require_pointwise: bool,
) -> PhysicalComparison:
    if reference.scalar_type != "float" or candidate.scalar_type != "float":
        raise RuntimeError("primary VTK fields must use float payloads")
    if reference.components != candidate.components:
        raise RuntimeError(
            f"VTK component count differs: {reference.components} != "
            f"{candidate.components}"
        )
    if skip_bottom_planes < 0:
        raise RuntimeError("--skip-bottom-planes must be non-negative")
    if (reference_mask is None) != (candidate_mask is None):
        raise RuntimeError("--reference-mask and --candidate-mask must be supplied together")

    def validate_mask(mask: VtkField, field: VtkField, label: str) -> None:
        if mask.scalar_type != "unsigned_char" or mask.components != 1:
            raise RuntimeError(f"{label} must be a scalar unsigned-char VTK")
        if mask.dimensions != field.dimensions:
            raise RuntimeError(f"{label} dimensions do not match its velocity VTK")
        for mask_value, field_value in zip(mask.origin + mask.spacing,
                                           field.origin + field.spacing):
            if not math.isclose(
                mask_value, field_value, rel_tol=coordinate_tolerance,
                abs_tol=coordinate_tolerance,
            ):
                raise RuntimeError(f"{label} coordinate transform does not match its velocity VTK")

    if reference_mask is not None:
        validate_mask(reference_mask, reference, "reference mask")
        validate_mask(candidate_mask, candidate, "candidate mask")

    components = reference.components
    ref_nx, ref_ny, ref_nz = reference.dimensions
    cand_nx, cand_ny, cand_nz = candidate.dimensions
    candidate_to_reference_x = [
        matching_index(
            candidate.origin[0] + index * candidate.spacing[0],
            reference.origin[0],
            reference.spacing[0],
            ref_nx,
            coordinate_tolerance,
        )
        for index in range(cand_nx)
    ]
    candidate_to_reference_y = [
        matching_index(
            candidate.origin[1] + index * candidate.spacing[1],
            reference.origin[1],
            reference.spacing[1],
            ref_ny,
            coordinate_tolerance,
        )
        for index in range(cand_ny)
    ]
    candidate_to_reference_z = [
        matching_index(
            candidate.origin[2] + index * candidate.spacing[2],
            reference.origin[2],
            reference.spacing[2],
            ref_nz,
            coordinate_tolerance,
        )
        for index in range(cand_nz)
    ]
    matched_z_pairs = [
        (candidate_z, reference_z)
        for candidate_z, reference_z in enumerate(candidate_to_reference_z)
        if reference_z is not None
    ]
    if skip_bottom_planes >= len(matched_z_pairs):
        raise RuntimeError("all matched z planes were excluded")

    matched_points = 0
    matched_values = 0
    masked_points = 0
    mask_mismatches = 0
    nonfinite_mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    sum_squared = 0.0
    sum_reference_squared = 0.0
    sum_reference = 0.0
    sum_candidate = 0.0
    sum_candidate_squared = 0.0
    sum_cross = 0.0
    pointwise_passed = True
    component_reference_sums = [0.0] * components
    component_candidate_sums = [0.0] * components
    reference_speed_sum = 0.0
    candidate_speed_sum = 0.0
    speed_samples = 0
    plane_sums: dict[int, tuple[int, list[float], list[float]]] = {}

    for candidate_z, reference_z in matched_z_pairs[skip_bottom_planes:]:
        for candidate_y, reference_y in enumerate(candidate_to_reference_y):
            if reference_y is None:
                continue
            for candidate_x, reference_x in enumerate(candidate_to_reference_x):
                if reference_x is None:
                    continue
                candidate_point = (
                    candidate_x + (candidate_y + candidate_z * cand_ny) * cand_nx
                )
                reference_point = (
                    reference_x + (reference_y + reference_z * ref_ny) * ref_nx
                )
                if reference_mask is not None and candidate_mask is not None:
                    reference_solid = (int(reference_mask.values[reference_point]) & 1) != 0
                    candidate_solid = (int(candidate_mask.values[candidate_point]) & 1) != 0
                    if reference_solid != candidate_solid:
                        mask_mismatches += 1
                    if reference_solid or candidate_solid:
                        masked_points += 1
                        continue
                matched_points += 1
                plane_count, plane_reference, plane_candidate = plane_sums.get(
                    reference_z, (0, [0.0] * components, [0.0] * components)
                )
                reference_vector: list[float] = []
                candidate_vector: list[float] = []
                for component in range(components):
                    expected = reference.values[reference_point * components + component]
                    actual = candidate.values[candidate_point * components + component]
                    reference_vector.append(expected)
                    candidate_vector.append(actual)
                    if not math.isfinite(expected) or not math.isfinite(actual):
                        if (
                            not (math.isnan(expected) and math.isnan(actual))
                            and expected != actual
                        ):
                            nonfinite_mismatches += 1
                            pointwise_passed = False
                        continue
                    difference = abs(actual - expected)
                    relative = relative_difference(expected, actual)
                    max_abs = max(max_abs, difference)
                    max_rel = max(max_rel, relative)
                    if difference > abs_tolerance + rel_tolerance * max(
                        abs(expected), abs(actual)
                    ):
                        pointwise_passed = False
                    sum_squared += difference * difference
                    sum_reference_squared += expected * expected
                    sum_reference += expected
                    sum_candidate += actual
                    sum_candidate_squared += actual * actual
                    sum_cross += expected * actual
                    component_reference_sums[component] += expected
                    component_candidate_sums[component] += actual
                    plane_reference[component] += expected
                    plane_candidate[component] += actual
                    matched_values += 1
                if components == 3 and all(
                    math.isfinite(value)
                    for value in reference_vector + candidate_vector
                ):
                    reference_speed_sum += math.sqrt(
                        sum(value * value for value in reference_vector)
                    )
                    candidate_speed_sum += math.sqrt(
                        sum(value * value for value in candidate_vector)
                    )
                    speed_samples += 1
                plane_sums[reference_z] = (
                    plane_count + 1,
                    plane_reference,
                    plane_candidate,
                )

    if matched_points == 0 or matched_values == 0:
        raise RuntimeError("VTK fields have no common physical lattice samples")
    relative_l2 = (
        math.sqrt(sum_squared / sum_reference_squared)
        if sum_reference_squared > 0.0
        else (0.0 if sum_squared == 0.0 else math.inf)
    )
    reference_mean = [value / matched_points for value in component_reference_sums]
    candidate_mean = [value / matched_points for value in component_candidate_sums]
    reference_mean_speed = (
        reference_speed_sum / speed_samples if speed_samples else None
    )
    candidate_mean_speed = (
        candidate_speed_sum / speed_samples if speed_samples else None
    )
    qoi_pairs = list(zip(reference_mean, candidate_mean))
    if reference_mean_speed is not None and candidate_mean_speed is not None:
        qoi_pairs.append((reference_mean_speed, candidate_mean_speed))
    qoi_differences = [
        relative_difference(expected, actual)
        for expected, actual in qoi_pairs
    ]
    qoi_absolute_differences = [
        abs(actual - expected) for expected, actual in qoi_pairs
    ]
    qoi_passed = all(
        difference <= qoi_absolute_tolerance
        + qoi_relative_tolerance * max(abs(expected), abs(actual))
        for (expected, actual), difference in zip(
            qoi_pairs, qoi_absolute_differences
        )
    )

    reference_mean_all = sum_reference / matched_values
    candidate_mean_all = sum_candidate / matched_values
    reference_variance = max(
        0.0, sum_reference_squared / matched_values - reference_mean_all**2
    )
    candidate_variance = max(
        0.0, sum_candidate_squared / matched_values - candidate_mean_all**2
    )
    covariance = sum_cross / matched_values - reference_mean_all * candidate_mean_all
    correlation = (
        covariance / math.sqrt(reference_variance * candidate_variance)
        if reference_variance > 0.0 and candidate_variance > 0.0
        else (1.0 if sum_squared == 0.0 else 0.0)
    )
    correlation = max(-1.0, min(1.0, correlation))
    # Correlation becomes ill-conditioned for an essentially uniform scalar
    # (for example density around one). In that regime absolute/relative L2
    # and the mean QoI are the meaningful physical checks.
    reference_stddev = math.sqrt(reference_variance)
    candidate_stddev = math.sqrt(candidate_variance)
    correlation_applicable = max(reference_stddev, candidate_stddev) >= 1.0e-4 * max(
        1.0, abs(reference_mean_all), abs(candidate_mean_all)
    )

    plane_qoi: list[PlaneQoi] = []
    for reference_z in sorted(plane_sums):
        count, reference_sum, candidate_sum = plane_sums[reference_z]
        reference_plane_mean = [value / count for value in reference_sum]
        candidate_plane_mean = [value / count for value in candidate_sum]
        plane_qoi.append(
            PlaneQoi(
                z_si=reference.origin[2] + reference_z * reference.spacing[2],
                samples=count,
                reference_mean=reference_plane_mean,
                candidate_mean=candidate_plane_mean,
                max_abs_difference=max(
                    abs(actual - expected)
                    for expected, actual in zip(
                        reference_plane_mean, candidate_plane_mean
                    )
                ),
            )
        )

    qoi_max_relative_difference = max(qoi_differences, default=0.0)
    qoi_max_abs_difference = max(qoi_absolute_differences, default=0.0)
    passed = (
        (not require_pointwise or pointwise_passed)
        and mask_mismatches == 0
        and nonfinite_mismatches == 0
        and relative_l2 <= relative_l2_tolerance
        and (not correlation_applicable or correlation >= min_correlation)
        and qoi_passed
    )
    return PhysicalComparison(
        reference=str(reference.path),
        candidate=str(candidate.path),
        components=components,
        matched_points=matched_points,
        matched_values=matched_values,
        skipped_bottom_planes=skip_bottom_planes,
        masked_points=masked_points,
        mask_mismatches=mask_mismatches,
        nonfinite_mismatches=nonfinite_mismatches,
        max_abs=max_abs,
        max_rel=max_rel,
        relative_l2=relative_l2,
        correlation=correlation,
        correlation_applicable=correlation_applicable,
        reference_mean=reference_mean,
        candidate_mean=candidate_mean,
        qoi_max_relative_difference=qoi_max_relative_difference,
        qoi_max_abs_difference=qoi_max_abs_difference,
        pointwise_passed=pointwise_passed,
        qoi_passed=qoi_passed,
        reference_mean_speed=reference_mean_speed,
        candidate_mean_speed=candidate_mean_speed,
        plane_qoi=plane_qoi,
        passed=passed,
    )


def main() -> int:
    args = parse_args()
    result = compare(
        read_vtk(args.reference),
        read_vtk(args.candidate),
        read_vtk(args.reference_mask) if args.reference_mask else None,
        read_vtk(args.candidate_mask) if args.candidate_mask else None,
        args.abs_tol,
        args.rel_tol,
        args.relative_l2_tol,
        args.min_correlation,
        args.qoi_relative_tol,
        args.qoi_abs_tol,
        args.coordinate_tol,
        args.skip_bottom_planes,
        args.require_pointwise,
    )
    report = asdict(result)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

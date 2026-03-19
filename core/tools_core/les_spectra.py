from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from scipy import fft as spfft


@dataclass
class StructuredPointsMeta:
    path: str
    dataset_type: str
    dimensions_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    point_data: int
    data_name: str
    dtype_name: str
    n_components: int
    data_offset: int


@dataclass
class LayerSpectrumRecord:
    target_height_m: float
    actual_height_m: float
    z_index: int
    valid_fraction: float
    valid_points: int
    output_png: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute horizontal kx-ky spectra every 50 m and a 3D isotropic "
            "energy spectrum for LES resolution checks from a legacy VTK file."
        )
    )
    parser.add_argument(
        "vtk_path",
        nargs="?",
        default="uvwrho-guangzhou4kmC3_20250910091000.vtk",
        help="Path to the legacy VTK file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where figures, CSVs and metadata are written.",
    )
    parser.add_argument(
        "--height-interval",
        type=float,
        default=50.0,
        help="Target height interval in meters for horizontal spectra.",
    )
    parser.add_argument(
        "--height-start",
        type=float,
        default=50.0,
        help="First target height in meters for horizontal spectra.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Worker count passed to scipy.fft routines (-1 uses all cores).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a smaller, faster validation pass before the full run.",
    )
    parser.add_argument(
        "--test-height-count",
        type=int,
        default=3,
        help="Number of heights used in quick-test mode.",
    )
    parser.add_argument(
        "--test-3d-layers",
        type=int,
        default=24,
        help="Number of vertical layers used for the 3D FFT in quick-test mode.",
    )
    return parser.parse_args()


def parse_legacy_structured_points(vtk_path: Path) -> StructuredPointsMeta:
    dataset_type = None
    dimensions = None
    spacing = None
    origin = None
    point_data = None
    data_name = None
    dtype_name = None
    n_components = 1
    data_offset = None

    with vtk_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            text = line.decode("latin1").strip()
            if text.startswith("DATASET"):
                _, dataset_type = text.split(None, 1)
            elif text.startswith("DIMENSIONS"):
                _, nx, ny, nz = text.split()
                dimensions = (int(nx), int(ny), int(nz))
            elif text.startswith("SPACING"):
                _, dx, dy, dz = text.split()
                spacing = (float(dx), float(dy), float(dz))
            elif text.startswith("ORIGIN"):
                _, ox, oy, oz = text.split()
                origin = (float(ox), float(oy), float(oz))
            elif text.startswith("POINT_DATA"):
                _, point_data = text.split()
                point_data = int(point_data)
            elif text.startswith("SCALARS"):
                parts = text.split()
                data_name = parts[1]
                dtype_name = parts[2]
                if len(parts) >= 4:
                    n_components = int(parts[3])
            elif text == "LOOKUP_TABLE default":
                data_offset = handle.tell()
                break

    if dataset_type != "STRUCTURED_POINTS":
        raise ValueError(f"Unsupported DATASET type: {dataset_type!r}")
    if dimensions is None or spacing is None or origin is None or point_data is None:
        raise ValueError("Incomplete VTK header; missing required grid metadata.")
    if dtype_name != "float":
        raise ValueError(f"Only float data is supported, got {dtype_name!r}.")
    if data_offset is None:
        raise ValueError("Could not locate LOOKUP_TABLE/data offset in VTK file.")

    return StructuredPointsMeta(
        path=str(vtk_path.resolve()),
        dataset_type=dataset_type,
        dimensions_xyz=dimensions,
        spacing_xyz=spacing,
        origin_xyz=origin,
        point_data=point_data,
        data_name=data_name or "data",
        dtype_name=dtype_name,
        n_components=n_components,
        data_offset=data_offset,
    )


def open_vtk_memmap(meta: StructuredPointsMeta) -> np.memmap:
    nx, ny, nz = meta.dimensions_xyz
    expected_points = nx * ny * nz
    if expected_points != meta.point_data:
        raise ValueError(
            f"POINT_DATA={meta.point_data} does not match DIMENSIONS product={expected_points}."
        )
    shape = (nz, ny, nx, meta.n_components)
    return np.memmap(
        meta.path,
        dtype=">f4",
        mode="r",
        offset=meta.data_offset,
        shape=shape,
        order="C",
    )


def native_float_array(view: np.ndarray) -> np.ndarray:
    return np.array(view, dtype=np.float32, copy=True)


def build_target_heights(
    origin_z: float,
    dz: float,
    nz: int,
    start_m: float,
    interval_m: float,
) -> list[tuple[float, int, float]]:
    top_height = origin_z + dz * (nz - 1)
    targets = np.arange(start_m, top_height + 1e-9, interval_m, dtype=float)
    indices = np.rint((targets - origin_z) / dz).astype(int)

    records: list[tuple[float, int, float]] = []
    seen_indices: set[int] = set()
    for target, z_index in zip(targets, indices):
        if z_index < 0 or z_index >= nz:
            continue
        if z_index in seen_indices:
            continue
        seen_indices.add(z_index)
        actual = origin_z + z_index * dz
        records.append((float(target), int(z_index), float(actual)))
    return records


def layer_valid_mask(layer: np.ndarray) -> np.ndarray:
    return np.any(layer != 0.0, axis=-1)


def compute_layer_coverages(data: np.memmap) -> np.ndarray:
    coverages = np.empty(data.shape[0], dtype=np.float64)
    for z_index in range(data.shape[0]):
        coverages[z_index] = layer_valid_mask(data[z_index]).mean()
    return coverages


def compute_horizontal_energy_spectrum(
    layer: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, float, int]:
    valid = layer_valid_mask(layer)
    valid_points = int(valid.sum())
    valid_fraction = float(valid.mean())
    ny, nx, _ = layer.shape
    total_points = nx * ny

    if valid_points == 0:
        return np.zeros((ny, nx), dtype=np.float32), valid_fraction, valid_points

    norm = float(total_points * valid_points)
    energy = np.zeros((ny, nx), dtype=np.float64)

    for component in range(min(layer.shape[-1], 3)):
        field = native_float_array(layer[..., component])
        mean_value = float(field[valid].mean(dtype=np.float64))
        field[valid] -= mean_value
        field[~valid] = 0.0
        coeff = spfft.fft2(field, workers=workers)
        energy += 0.5 * (np.abs(coeff) ** 2) / norm

    return np.fft.fftshift(energy.astype(np.float32)), valid_fraction, valid_points


def compute_axis_limits(k_values: np.ndarray) -> tuple[float, float]:
    return float(k_values.min()), float(k_values.max())


def robust_log_color_limits(log_arrays: list[np.ndarray]) -> tuple[float, float]:
    if not log_arrays:
        return -12.0, 0.0
    stacked = np.concatenate(log_arrays)
    vmin, vmax = np.percentile(stacked, [5.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -12.0, 0.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_layer_metadata_csv(records: list[LayerSpectrumRecord], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_height_m",
                "actual_height_m",
                "z_index",
                "valid_fraction",
                "valid_points",
                "output_png",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def plot_horizontal_spectrum(
    spectrum_log10: np.ndarray,
    actual_height_m: float,
    valid_fraction: float,
    output_path: Path,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.imshow(
        spectrum_log10,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"$k_x$ [rad m$^{-1}$]")
    ax.set_ylabel(r"$k_y$ [rad m$^{-1}$]")
    ax.set_title(f"z = {actual_height_m:.2f} m, valid = {valid_fraction:.1%}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}(E_{2D})$")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_horizontal_overview(
    spectra_log10: list[np.ndarray],
    heights_m: list[float],
    valid_fractions: list[float],
    output_path: Path,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    columns: int = 4,
    figure_size: tuple[float, float] | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 14.0,
    tick_fontsize: float = 10.5,
    colorbar_fontsize: float = 12.0,
    valid_inside: bool = False,
    valid_fontsize: float | None = None,
    height_as_integer: bool = False,
) -> None:
    plot_count = len(spectra_log10)
    rows = math.ceil(plot_count / columns)
    if figure_size is None:
        figure_size = (4.7 * columns, 4.0 * rows)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=figure_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    image = None

    for ax, spectrum, height_m, valid_fraction in zip(
        axes_array.flat, spectra_log10, heights_m, valid_fractions
    ):
        image = ax.imshow(
            spectrum,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        height_label = f"{int(round(height_m))} m" if height_as_integer else f"{height_m:.1f} m"
        if valid_inside:
            ax.set_title(height_label, fontsize=title_fontsize)
            valid_label_fontsize = tick_fontsize if valid_fontsize is None else valid_fontsize
            valid_text = ax.text(
                0.03,
                0.97,
                f"valid={valid_fraction:.1%}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=valid_label_fontsize,
            )
            valid_text.set_path_effects(
                [patheffects.withStroke(linewidth=2.0, foreground="black", alpha=0.7)]
            )
        else:
            ax.set_title(
                f"{height_label}\nvalid={valid_fraction:.1%}",
                fontsize=title_fontsize,
            )
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    for ax in axes_array.flat[plot_count:]:
        ax.axis("off")

    fig.supxlabel(r"$k_x$ [rad m$^{-1}$]", fontsize=label_fontsize)
    fig.supylabel(r"$k_y$ [rad m$^{-1}$]", fontsize=label_fontsize)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array, shrink=0.92)
        colorbar.set_label(r"$\log_{10}(E_{2D})$", fontsize=colorbar_fontsize)
        colorbar.ax.tick_params(labelsize=tick_fontsize)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_kx_weights(nx: int) -> np.ndarray:
    weights = np.ones(nx // 2 + 1, dtype=np.float64)
    if weights.size <= 1:
        return weights
    if nx % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def compute_isotropic_spectrum(
    data: np.memmap,
    meta: StructuredPointsMeta,
    coverages: np.ndarray,
    workers: int,
    quick_test_layers: int | None = None,
) -> dict[str, object]:
    nx, ny, nz = meta.dimensions_xyz
    dx, dy, dz = meta.spacing_xyz
    origin_z = meta.origin_xyz[2]

    full_layers = np.where(np.isclose(coverages, 1.0))[0]
    if full_layers.size == 0:
        raise RuntimeError("No fully valid horizontal layer exists for the 3D spectrum.")
    z_start = int(full_layers[0])
    z_stop = nz

    if quick_test_layers is not None:
        z_stop = min(nz, z_start + int(quick_test_layers))
        if z_stop - z_start < 4:
            raise RuntimeError("Quick-test 3D subvolume is too small to be meaningful.")

    nz_sub = z_stop - z_start
    total_points = nx * ny * nz_sub
    print(
        "3D spectrum subvolume: "
        f"z-index {z_start}:{z_stop} "
        f"({origin_z + z_start * dz:.2f} to {origin_z + (z_stop - 1) * dz:.2f} m)"
    )

    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz_sub, d=dz)

    dk = max(
        2.0 * np.pi / (nx * dx),
        2.0 * np.pi / (ny * dy),
        2.0 * np.pi / (nz_sub * dz),
    )
    k_max = math.sqrt(
        float(kx.max() ** 2 + np.max(np.abs(ky)) ** 2 + np.max(np.abs(kz)) ** 2)
    )
    bin_count = int(math.floor(k_max / dk)) + 1
    k_edges = np.arange(bin_count + 1, dtype=np.float64) * dk
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    kx_weights = compute_kx_weights(nx)
    shell_counts = np.zeros(bin_count, dtype=np.float64)
    shell_energy = np.zeros(bin_count, dtype=np.float64)

    for kz_value in kz:
        shell_index = np.floor(
            np.sqrt(kz_value * kz_value + ky[:, None] ** 2 + kx[None, :] ** 2) / dk
        ).astype(np.int32)
        shell_index = np.clip(shell_index, 0, bin_count - 1)
        shell_counts += np.bincount(
            shell_index.ravel(),
            weights=np.broadcast_to(kx_weights, shell_index.shape).ravel(),
            minlength=bin_count,
        )

    normalization = 0.5 / (float(total_points) ** 2)

    for component in range(min(data.shape[-1], 3)):
        print(f"3D FFT component {component + 1}/3")
        field = native_float_array(data[z_start:z_stop, :, :, component])
        field -= float(field.mean(dtype=np.float64))
        coeff = spfft.rfftn(field, workers=workers)
        del field
        gc.collect()

        for z_index, kz_value in enumerate(kz):
            shell_index = np.floor(
                np.sqrt(kz_value * kz_value + ky[:, None] ** 2 + kx[None, :] ** 2) / dk
            ).astype(np.int32)
            shell_index = np.clip(shell_index, 0, bin_count - 1)
            power = normalization * (np.abs(coeff[z_index]) ** 2) * kx_weights[None, :]
            shell_energy += np.bincount(
                shell_index.ravel(),
                weights=power.ravel(),
                minlength=bin_count,
            )

        del coeff
        gc.collect()

    isotropic_energy = np.zeros_like(shell_energy)
    nonzero = shell_counts > 0.0
    isotropic_energy[nonzero] = shell_energy[nonzero] / dk
    compensated_energy = np.zeros_like(isotropic_energy)
    positive_k = k_centers > 0.0
    compensated_energy[positive_k] = (
        isotropic_energy[positive_k] * (k_centers[positive_k] ** (5.0 / 3.0))
    )

    return {
        "z_start_index": z_start,
        "z_stop_index": z_stop,
        "z_start_height_m": origin_z + z_start * dz,
        "z_stop_height_m": origin_z + (z_stop - 1) * dz,
        "dk_rad_per_m": dk,
        "k_centers_rad_per_m": k_centers,
        "shell_energy": shell_energy,
        "shell_mode_weight": shell_counts,
        "E_k": isotropic_energy,
        "k_5_3_E_k": compensated_energy,
        "k_nyquist_rad_per_m": float(np.pi / max(dx, dy, dz)),
    }


def save_isotropic_csv(result: dict[str, object], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k_rad_per_m",
                "E_k",
                "k_5_3_E_k",
                "shell_energy",
                "shell_mode_weight",
            ]
        )
        for row in zip(
            result["k_centers_rad_per_m"],
            result["E_k"],
            result["k_5_3_E_k"],
            result["shell_energy"],
            result["shell_mode_weight"],
        ):
            writer.writerow(row)


def add_reference_slope(
    ax: plt.Axes,
    k_values: np.ndarray,
    e_values: np.ndarray,
    slope: float = -5.0 / 3.0,
    k1: float = 0.12,
    k2: float = 0.22,
    vertical_offset: float = 0.75,
) -> None:
    valid = (k_values > 0.0) & (e_values > 0.0)
    if np.count_nonzero(valid) < 2:
        return

    k_valid = k_values[valid]
    e_valid = e_values[valid]
    k1 = max(k1, float(k_valid.min()))
    k2 = min(k2, float(k_valid.max()))
    if not (k2 > k1):
        return

    log_k = np.log10(k_valid)
    log_e = np.log10(e_valid)
    anchor_log_e = np.interp(np.log10(k1), log_k, log_e)
    y1 = (10.0**anchor_log_e) * vertical_offset
    y2 = y1 * (k2 / k1) ** slope

    ax.loglog([k1, k2], [y1, y2], color="black", linewidth=1.3, solid_capstyle="butt")
    k_text = math.sqrt(k1 * k2)
    y_text = y1 * (k_text / k1) ** slope * 1.15
    ax.text(
        k_text,
        y_text,
        r"$-5/3$",
        fontsize=12,
        color="black",
        ha="left",
        va="bottom",
    )


def plot_isotropic_spectrum(result: dict[str, object], output_png: Path) -> None:
    k_values = np.asarray(result["k_centers_rad_per_m"])
    e_values = np.asarray(result["E_k"])
    compensated = np.asarray(result["k_5_3_E_k"])
    k_half_nyquist = 0.5 * float(result["k_nyquist_rad_per_m"])
    valid = (k_values > 0.0) & (e_values > 0.0)
    compensated_valid = (k_values > 0.0) & (compensated > 0.0)

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    line_original = ax.loglog(
        k_values[valid],
        e_values[valid],
        color="#1f77b4",
        linewidth=1.8,
        label=r"Original $E(k)$",
    )[0]
    line_nyquist = ax.axvline(
        result["k_nyquist_rad_per_m"],
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=r"$k_{Nyquist}=\pi/\Delta$",
    )
    line_half_nyquist = ax.axvline(
        k_half_nyquist,
        color="dimgray",
        linestyle=":",
        linewidth=1.2,
        label=r"$1/2\,k_{Nyquist}$",
    )
    ax2 = ax.twinx()
    ax2.set_xscale("log")
    line_compensated = ax2.plot(
        k_values[compensated_valid],
        compensated[compensated_valid],
        color="#ff7f0e",
        linewidth=1.6,
        linestyle="-.",
        label=r"Compensated $k^{5/3}E(k)$",
    )[0]
    ax.set_xlabel(r"Wavenumber $\mathit{k}$ [rad m$^{-1}$]")
    ax.set_ylabel(r"Energy spectrum $E(\mathit{k})$ [m$^3$s$^{-2}$]")
    ax2.set_ylabel(r"$k^{5/3}E(k)$")
    ax.set_title(
        "3D isotropic kinetic energy spectrum\n"
        f"z = {result['z_start_height_m']:.2f} to {result['z_stop_height_m']:.2f} m"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax2.yaxis.label.set_size(15)
    ax.title.set_size(17)
    ax.legend(
        [line_original, line_compensated, line_nyquist, line_half_nyquist],
        [
            line_original.get_label(),
            line_compensated.get_label(),
            line_nyquist.get_label(),
            line_half_nyquist.get_label(),
        ],
        loc="best",
        fontsize=12,
    )
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_isotropic_spectrum_plain(result: dict[str, object], output_png: Path) -> None:
    k_values = np.asarray(result["k_centers_rad_per_m"])
    e_values = np.asarray(result["E_k"])
    k_half_nyquist = 0.5 * float(result["k_nyquist_rad_per_m"])
    valid = (k_values > 0.0) & (e_values > 0.0)

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    line_original = ax.loglog(
        k_values[valid],
        e_values[valid],
        color="#1f77b4",
        linewidth=1.1,
        marker="o",
        markersize=3.8,
        markerfacecolor="none",
        markeredgewidth=0.9,
        label=r"$E(k)$",
    )[0]
    line_nyquist = ax.axvline(
        result["k_nyquist_rad_per_m"],
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=r"$k_{Nyquist}=\pi/\Delta$",
    )
    line_half_nyquist = ax.axvline(
        k_half_nyquist,
        color="dimgray",
        linestyle=":",
        linewidth=1.2,
        label=r"$1/2\,k_{Nyquist}$",
    )
    add_reference_slope(ax, k_values[valid], e_values[valid], slope=-5.0 / 3.0)
    ax.set_xlabel(r"Wavenumber $\mathit{k}$ [rad m$^{-1}$]")
    ax.set_ylabel(r"Energy spectrum $E(\mathit{k})$ [m$^3$s$^{-2}$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.legend(
        [line_original, line_nyquist, line_half_nyquist],
        [line_original.get_label(), line_nyquist.get_label(), line_half_nyquist.get_label()],
        loc="best",
        fontsize=12,
    )
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_time = time.time()

    vtk_path = Path(args.vtk_path).resolve()
    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK file not found: {vtk_path}")

    if args.output_dir is None:
        output_dir = vtk_path.with_name(f"{vtk_path.stem}_les_spectra")
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers_kxky"
    layer_dir.mkdir(exist_ok=True)

    meta = parse_legacy_structured_points(vtk_path)
    data = open_vtk_memmap(meta)
    nx, ny, nz = meta.dimensions_xyz
    dx, dy, dz = meta.spacing_xyz
    origin_z = meta.origin_xyz[2]

    target_layers = build_target_heights(
        origin_z=origin_z,
        dz=dz,
        nz=nz,
        start_m=args.height_start,
        interval_m=args.height_interval,
    )
    if args.quick_test:
        target_layers = target_layers[: args.test_height_count]

    print(f"Parsed grid: nx={nx}, ny={ny}, nz={nz}, dx={dx}, dy={dy}, dz={dz}")
    print(f"Selected {len(target_layers)} target height layers")

    coverages = compute_layer_coverages(data)

    kx_shift = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky_shift = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    x_limits = compute_axis_limits(kx_shift)
    y_limits = compute_axis_limits(ky_shift)
    extent = (x_limits[0], x_limits[1], y_limits[0], y_limits[1])

    layer_spectra_log10: list[np.ndarray] = []
    layer_heights_m: list[float] = []
    layer_valid_fractions: list[float] = []
    layer_valid_points: list[int] = []
    layer_records: list[LayerSpectrumRecord] = []
    raw_spectra: list[np.ndarray] = []

    for target_height_m, z_index, actual_height_m in target_layers:
        print(f"Computing horizontal spectrum at z-index {z_index} ({actual_height_m:.2f} m)")
        spectrum, valid_fraction, valid_points = compute_horizontal_energy_spectrum(
            data[z_index], workers=args.workers
        )
        raw_spectra.append(spectrum)
        layer_heights_m.append(actual_height_m)
        layer_valid_fractions.append(valid_fraction)
        layer_valid_points.append(valid_points)

    positive_logs = [
        np.log10(spectrum[spectrum > 0.0]).astype(np.float32, copy=False)
        for spectrum in raw_spectra
        if np.any(spectrum > 0.0)
    ]
    vmin, vmax = robust_log_color_limits(positive_logs)
    epsilon = 10.0**vmin
    layer_spectra_log10 = [
        np.log10(np.maximum(spectrum, epsilon)).astype(np.float32, copy=False)
        for spectrum in raw_spectra
    ]

    for (
        (target_height_m, z_index, actual_height_m),
        spectrum_log10,
        valid_fraction,
        valid_points,
    ) in zip(
        target_layers,
        layer_spectra_log10,
        layer_valid_fractions,
        layer_valid_points,
    ):
        output_png = layer_dir / f"kxky_z{int(round(actual_height_m)):04d}m.png"
        plot_horizontal_spectrum(
            spectrum_log10=spectrum_log10,
            actual_height_m=actual_height_m,
            valid_fraction=valid_fraction,
            output_path=output_png,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        layer_records.append(
            LayerSpectrumRecord(
                target_height_m=target_height_m,
                actual_height_m=actual_height_m,
                z_index=z_index,
                valid_fraction=valid_fraction,
                valid_points=valid_points,
                output_png=str(output_png.resolve()),
            )
        )

    overview_png = output_dir / "kxky_overview.png"
    plot_horizontal_overview(
        spectra_log10=layer_spectra_log10,
        heights_m=layer_heights_m,
        valid_fractions=layer_valid_fractions,
        output_path=overview_png,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    selected_targets = {100, 200, 300, 400, 500, 600}
    selected_indices = [
        idx
        for idx, record in enumerate(layer_records)
        if int(round(record.target_height_m)) in selected_targets
    ]
    selected_overview_png = (
        output_dir / "kxky_overview_100_200_300_400_500_600_largefont.png"
    )
    if selected_indices:
        plot_horizontal_overview(
            spectra_log10=[layer_spectra_log10[idx] for idx in selected_indices],
            heights_m=[layer_heights_m[idx] for idx in selected_indices],
            valid_fractions=[layer_valid_fractions[idx] for idx in selected_indices],
            output_path=selected_overview_png,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            x_limits=x_limits,
            y_limits=y_limits,
            columns=3,
            figure_size=(19.5, 12.0),
            title_fontsize=20.0,
            label_fontsize=22.0,
            tick_fontsize=18.0,
            colorbar_fontsize=22.0,
            valid_inside=True,
            valid_fontsize=17.5,
            height_as_integer=True,
        )
    save_layer_metadata_csv(layer_records, output_dir / "kxky_layer_metadata.csv")

    quick_test_layers = args.test_3d_layers if args.quick_test else None
    print("Computing 3D isotropic spectrum")
    isotropic_result = compute_isotropic_spectrum(
        data=data,
        meta=meta,
        coverages=coverages,
        workers=args.workers,
        quick_test_layers=quick_test_layers,
    )
    isotropic_png = output_dir / "isotropic_spectrum.png"
    isotropic_plain_png = output_dir / "isotropic_spectrum_plain.png"
    isotropic_csv = output_dir / "isotropic_spectrum.csv"
    plot_isotropic_spectrum(isotropic_result, isotropic_png)
    plot_isotropic_spectrum_plain(isotropic_result, isotropic_plain_png)
    save_isotropic_csv(isotropic_result, isotropic_csv)

    summary = {
        "source_vtk": str(vtk_path),
        "quick_test": bool(args.quick_test),
        "grid": asdict(meta),
        "coverages_by_layer": coverages.tolist(),
        "target_layers": [asdict(record) for record in layer_records],
        "kx_limits_rad_per_m": x_limits,
        "ky_limits_rad_per_m": y_limits,
        "color_limits_log10": [vmin, vmax],
        "isotropic_spectrum": {
            "z_start_index": isotropic_result["z_start_index"],
            "z_stop_index": isotropic_result["z_stop_index"],
            "z_start_height_m": isotropic_result["z_start_height_m"],
            "z_stop_height_m": isotropic_result["z_stop_height_m"],
            "dk_rad_per_m": isotropic_result["dk_rad_per_m"],
            "k_nyquist_rad_per_m": isotropic_result["k_nyquist_rad_per_m"],
            "plot_png": str(isotropic_png.resolve()),
            "plain_plot_png": str(isotropic_plain_png.resolve()),
            "csv": str(isotropic_csv.resolve()),
        },
        "overview_png": str(overview_png.resolve()),
        "selected_overview_png": (
            str(selected_overview_png.resolve()) if selected_indices else None
        ),
        "elapsed_seconds": time.time() - start_time,
    }

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    safe_output_dir = str(output_dir).encode("ascii", "backslashreplace").decode("ascii")
    print(f"Outputs written to: {safe_output_dir}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f} s")


if __name__ == "__main__":
    main()

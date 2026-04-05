from __future__ import annotations

import math
from typing import Any
import warnings

import numpy as np

from accelerator_runtime import apply_cuda_runtime_environment


MAX_KRIGING_GPU_NEIGHBORS = 16
MAX_KRIGING_GPU_SYSTEM = MAX_KRIGING_GPU_NEIGHBORS + 1
_CUDA_MODULE = None
_CUDA_FLOAT32 = None
_KRIGING_KERNEL = None
_WARMED_UP = False


class TerrainGpuKrigingUnavailable(RuntimeError):
    pass


def _load_cuda_kernel() -> tuple[Any, Any]:
    global _CUDA_MODULE
    global _CUDA_FLOAT32
    global _KRIGING_KERNEL

    if _KRIGING_KERNEL is not None and _CUDA_MODULE is not None and _CUDA_FLOAT32 is not None:
        return _CUDA_MODULE, _KRIGING_KERNEL

    runtime = apply_cuda_runtime_environment()
    if not runtime.get("configured"):
        raise TerrainGpuKrigingUnavailable(str(runtime.get("message") or "CUDA runtime is not configured."))

    try:
        from numba import cuda, float32
    except Exception as exc:
        raise TerrainGpuKrigingUnavailable(f"Failed to import numba.cuda: {exc}") from exc

    @cuda.jit(device=True)
    def gpu_variogram(distance: float, range_param: float, partial_sill: float, nugget: float) -> float:
        if distance <= 0.0:
            return 0.0
        return partial_sill * (1.0 - math.exp(-distance / range_param)) + nugget

    @cuda.jit(device=True)
    def gpu_idw_fallback(local_elevations, local_distances, neighbor_count: int, power: float) -> float:
        weight_sum = float32(0.0)
        value_sum = float32(0.0)
        for i in range(neighbor_count):
            dist = local_distances[i]
            if dist <= 1e-10:
                return local_elevations[i]
            weight = float32(1.0) / (dist ** power)
            weight_sum += weight
            value_sum += weight * local_elevations[i]
        if weight_sum <= 0.0:
            return local_elevations[0]
        return value_sum / weight_sum

    @cuda.jit
    def kriging_cuda_kernel(
        local_points,
        local_elevations,
        local_distances,
        range_param: float,
        partial_sill: float,
        nugget: float,
        regularization: float,
        fallback_power: float,
        neighbor_count: int,
        output,
    ):
        row = cuda.grid(1)
        if row >= output.size:
            return

        if neighbor_count > MAX_KRIGING_GPU_NEIGHBORS:
            output[row] = float32(0.0)
            return

        system = cuda.local.array((MAX_KRIGING_GPU_SYSTEM, MAX_KRIGING_GPU_SYSTEM), dtype=float32)
        rhs = cuda.local.array(MAX_KRIGING_GPU_SYSTEM, dtype=float32)
        weights = cuda.local.array(MAX_KRIGING_GPU_SYSTEM, dtype=float32)
        row_elevations = cuda.local.array(MAX_KRIGING_GPU_NEIGHBORS, dtype=float32)
        row_distances = cuda.local.array(MAX_KRIGING_GPU_NEIGHBORS, dtype=float32)
        row_points = cuda.local.array((MAX_KRIGING_GPU_NEIGHBORS, 2), dtype=float32)

        exact_index = -1
        local_min = local_elevations[row, 0]
        local_max = local_elevations[row, 0]
        first_value = local_elevations[row, 0]
        all_equal = True

        for i in range(neighbor_count):
            elev = local_elevations[row, i]
            dist = local_distances[row, i]
            row_elevations[i] = elev
            row_distances[i] = dist
            row_points[i, 0] = local_points[row, i, 0]
            row_points[i, 1] = local_points[row, i, 1]
            if dist <= 1e-10 and exact_index < 0:
                exact_index = i
            if elev < local_min:
                local_min = elev
            if elev > local_max:
                local_max = elev
            if math.fabs(elev - first_value) > 1e-5:
                all_equal = False

        if exact_index >= 0:
            output[row] = row_elevations[exact_index]
            return

        if neighbor_count == 1 or all_equal:
            output[row] = first_value
            return

        system_size = neighbor_count + 1
        for i in range(MAX_KRIGING_GPU_SYSTEM):
            rhs[i] = float32(0.0)
            weights[i] = float32(0.0)
            for j in range(MAX_KRIGING_GPU_SYSTEM):
                system[i, j] = float32(0.0)

        for i in range(neighbor_count):
            xi = row_points[i, 0]
            yi = row_points[i, 1]
            for j in range(neighbor_count):
                if i == j:
                    gamma = float32(0.0)
                else:
                    dx = xi - row_points[j, 0]
                    dy = yi - row_points[j, 1]
                    gamma = gpu_variogram(math.sqrt(dx * dx + dy * dy), range_param, partial_sill, nugget)
                system[i, j] = gamma
            system[i, i] += regularization
            system[i, neighbor_count] = float32(1.0)
            system[neighbor_count, i] = float32(1.0)
            rhs[i] = gpu_variogram(row_distances[i], range_param, partial_sill, nugget)

        system[neighbor_count, neighbor_count] = float32(0.0)
        rhs[neighbor_count] = float32(1.0)

        for pivot in range(system_size):
            pivot_row = pivot
            pivot_value = math.fabs(system[pivot, pivot])
            for candidate in range(pivot + 1, system_size):
                candidate_value = math.fabs(system[candidate, pivot])
                if candidate_value > pivot_value:
                    pivot_value = candidate_value
                    pivot_row = candidate

            if pivot_value < 1e-8:
                output[row] = gpu_idw_fallback(row_elevations, row_distances, neighbor_count, fallback_power)
                return

            if pivot_row != pivot:
                for col in range(system_size):
                    tmp = system[pivot, col]
                    system[pivot, col] = system[pivot_row, col]
                    system[pivot_row, col] = tmp
                tmp = rhs[pivot]
                rhs[pivot] = rhs[pivot_row]
                rhs[pivot_row] = tmp

            diagonal = system[pivot, pivot]
            for col in range(pivot + 1, system_size):
                system[pivot, col] = system[pivot, col] / diagonal
            rhs[pivot] = rhs[pivot] / diagonal
            system[pivot, pivot] = float32(1.0)

            for target_row in range(pivot + 1, system_size):
                factor = system[target_row, pivot]
                if math.fabs(factor) <= 1e-12:
                    system[target_row, pivot] = float32(0.0)
                    continue
                for col in range(pivot + 1, system_size):
                    system[target_row, col] = system[target_row, col] - factor * system[pivot, col]
                rhs[target_row] = rhs[target_row] - factor * rhs[pivot]
                system[target_row, pivot] = float32(0.0)

        for target_row in range(system_size - 1, -1, -1):
            acc = rhs[target_row]
            for col in range(target_row + 1, system_size):
                acc = acc - system[target_row, col] * weights[col]
            weights[target_row] = acc

        prediction = float32(0.0)
        for i in range(neighbor_count):
            prediction += weights[i] * row_elevations[i]

        if not math.isfinite(prediction):
            prediction = gpu_idw_fallback(row_elevations, row_distances, neighbor_count, fallback_power)

        if prediction < local_min:
            prediction = local_min
        elif prediction > local_max:
            prediction = local_max

        output[row] = prediction

    _CUDA_MODULE = cuda
    _CUDA_FLOAT32 = float32
    _KRIGING_KERNEL = kriging_cuda_kernel
    return cuda, kriging_cuda_kernel


def _warmup_cuda_kernel(neighbor_count: int) -> None:
    global _WARMED_UP
    if _WARMED_UP:
        return

    cuda, kernel = _load_cuda_kernel()
    warmup_rows = 256
    local_points = np.zeros((warmup_rows, neighbor_count, 2), dtype=np.float32)
    local_elevations = np.ones((warmup_rows, neighbor_count), dtype=np.float32)
    local_distances = np.ones((warmup_rows, neighbor_count), dtype=np.float32)
    dev_points = cuda.to_device(local_points)
    dev_elevations = cuda.to_device(local_elevations)
    dev_distances = cuda.to_device(local_distances)
    dev_output = cuda.device_array(warmup_rows, dtype=np.float32)
    threads_per_block = 128
    blocks_per_grid = max(1, math.ceil(warmup_rows / threads_per_block))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel[blocks_per_grid, threads_per_block](
            dev_points,
            dev_elevations,
            dev_distances,
            np.float32(10.0),
            np.float32(1.0),
            np.float32(0.02),
            np.float32(1e-6),
            np.float32(2.0),
            int(neighbor_count),
            dev_output,
        )
    cuda.synchronize()
    _WARMED_UP = True


def gpu_kriging_backend_summary() -> dict[str, object]:
    runtime = apply_cuda_runtime_environment()
    return {
        "configured": bool(runtime.get("configured")),
        "prepared": bool(runtime.get("prepared")),
        "provider": runtime.get("provider", ""),
        "runtime_root": runtime.get("runtime_root", ""),
        "cuda_home": runtime.get("cuda_home", ""),
        "max_neighbors": MAX_KRIGING_GPU_NEIGHBORS,
    }


def gpu_kriging_requires_warmup() -> bool:
    return not _WARMED_UP


def warmup_gpu_kriging_kernel(neighbor_count: int) -> None:
    if neighbor_count > MAX_KRIGING_GPU_NEIGHBORS:
        raise TerrainGpuKrigingUnavailable(
            f"Kriging (GPU) supports at most {MAX_KRIGING_GPU_NEIGHBORS} neighboring points, got {neighbor_count}."
        )
    _warmup_cuda_kernel(int(neighbor_count))


def interpolate_points_kriging_cuda_chunk(
    local_points: np.ndarray,
    local_elevations: np.ndarray,
    local_distances: np.ndarray,
    *,
    range_param: float,
    partial_sill: float,
    nugget: float,
    fallback_power: float,
) -> np.ndarray:
    neighbor_count = int(local_distances.shape[1]) if local_distances.ndim == 2 else 0
    if neighbor_count <= 0:
        return np.empty((0,), dtype=float)
    if neighbor_count > MAX_KRIGING_GPU_NEIGHBORS:
        raise TerrainGpuKrigingUnavailable(
            f"Kriging (GPU) supports at most {MAX_KRIGING_GPU_NEIGHBORS} neighboring points, got {neighbor_count}."
        )

    cuda, kernel = _load_cuda_kernel()
    try:
        cuda.current_context()
        cuda.get_current_device()
    except Exception as exc:
        raise TerrainGpuKrigingUnavailable(f"CUDA device is unavailable: {exc}") from exc

    _warmup_cuda_kernel(neighbor_count)

    local_points32 = np.asarray(local_points, dtype=np.float32)
    local_elevations32 = np.asarray(local_elevations, dtype=np.float32)
    local_distances32 = np.asarray(local_distances, dtype=np.float32)

    dev_points = cuda.to_device(local_points32)
    dev_elevations = cuda.to_device(local_elevations32)
    dev_distances = cuda.to_device(local_distances32)
    dev_output = cuda.device_array(local_distances32.shape[0], dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = max(1, math.ceil(local_distances32.shape[0] / threads_per_block))
    regularization = max(float(nugget) * 1e-6, 1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel[blocks_per_grid, threads_per_block](
            dev_points,
            dev_elevations,
            dev_distances,
            np.float32(range_param),
            np.float32(partial_sill),
            np.float32(nugget),
            np.float32(regularization),
            np.float32(fallback_power),
            int(neighbor_count),
            dev_output,
        )
    cuda.synchronize()
    return dev_output.copy_to_host().astype(np.float64, copy=False)

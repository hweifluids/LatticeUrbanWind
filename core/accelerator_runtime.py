from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import contextlib
import importlib
from importlib import metadata as importlib_metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import site
import sys
from typing import Iterable


IMPORT_NAME_MAP = {
    "cuda-python": "cuda",
    "netcdf4": "netCDF4",
}


@dataclass(frozen=True)
class PackageImportCheck:
    package: str
    module: str
    success: bool
    error: str = ""


@dataclass(frozen=True)
class AcceleratorDeviceInfo:
    name: str = ""
    vendor: str = ""
    version: str = ""
    driver_version: str = ""
    compute_capability: str = ""
    compute_units: int | None = None


def sanitize_requirement(line: str) -> str | None:
    raw = line.split("#", 1)[0].strip()
    if not raw:
        return None
    return re.split(r"[<>=!~;\[]", raw, maxsplit=1)[0].strip()


def candidate_modules(package_name: str) -> list[str]:
    normalized = package_name.replace("-", "_")
    mapped = IMPORT_NAME_MAP.get(package_name.lower(), normalized)
    candidates: list[str] = []
    for value in (mapped, normalized, package_name):
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def try_import(module_name: str) -> tuple[bool, str]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        details: list[str] = []
        std_out = stdout_buffer.getvalue().strip()
        std_err = stderr_buffer.getvalue().strip()
        if std_out:
            details.append(std_out)
        if std_err:
            details.append(std_err)
        details.append(str(exc))
        return False, " | ".join(part for part in details if part)


def has_distribution(distribution_name: str) -> bool:
    try:
        importlib_metadata.version(distribution_name)
        return True
    except importlib_metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


def run_requirements_import_check(requirements_path: str | Path) -> dict[str, object]:
    path = Path(requirements_path)
    checks: list[PackageImportCheck] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        package_name = sanitize_requirement(line)
        if not package_name:
            continue

        chosen_module = ""
        errors: list[str] = []
        success = False
        for module_name in candidate_modules(package_name):
            chosen_module = module_name
            ok, error_text = try_import(module_name)
            if ok:
                success = True
                errors = []
                break
            errors.append(f"{module_name}: {error_text}")

        if not success and has_distribution(package_name):
            success = True
            chosen_module = f"[distribution] {package_name}"
            errors = []

        checks.append(
            PackageImportCheck(
                package=package_name,
                module=chosen_module,
                success=success,
                error=" || ".join(errors),
            )
        )

    failures = [check for check in checks if not check.success]
    return {
        "checks": [asdict(check) for check in checks],
        "summary": {
            "total": len(checks),
            "passed": len(checks) - len(failures),
            "failed": len(failures),
            "text": (
                f"{len(checks) - len(failures)}/{len(checks)} packages imported successfully"
                if checks
                else "No requirements were listed for import checking."
            ),
        },
    }


def _site_directories() -> list[Path]:
    directories: list[Path] = []
    for value in site.getsitepackages():
        path = Path(value)
        if path.exists() and path not in directories:
            directories.append(path)
    try:
        user_site = Path(site.getusersitepackages())
    except Exception:
        user_site = None
    if user_site and user_site.exists() and user_site not in directories:
        directories.append(user_site)
    return directories


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _collect_files(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists() or not directory.is_dir():
        return []
    return [path for path in directory.iterdir() if path.is_file()]


def discover_cuda_wheel_layout() -> dict[str, object]:
    for site_dir in _site_directories():
        nvidia_dir = site_dir / "nvidia"
        if not nvidia_dir.exists():
            continue

        unified_root = _first_existing([nvidia_dir / "cu13", nvidia_dir / "cu12", nvidia_dir / "cuda"])
        if unified_root is not None:
            shared_bin = _first_existing(
                [
                    unified_root / "bin" / "x86_64",
                    unified_root / "bin",
                ]
            )
            static_lib = _first_existing(
                [
                    unified_root / "lib" / "x64",
                    unified_root / "lib64",
                    unified_root / "lib",
                ]
            )
            nvvm_bin = _first_existing(
                [
                    unified_root / "nvvm" / "bin",
                    unified_root / "nvvm" / "lib64",
                    unified_root / "nvvm" / "lib",
                ]
            )
            libdevice_dir = _first_existing([unified_root / "nvvm" / "libdevice"])
            if shared_bin and nvvm_bin and libdevice_dir:
                return {
                    "provider": "nvidia-pip-unified",
                    "root": str(unified_root),
                    "shared_lib_dir": str(shared_bin),
                    "static_lib_dir": str(static_lib) if static_lib else "",
                    "nvvm_dir": str(nvvm_bin),
                    "libdevice_dir": str(libdevice_dir),
                }

        shared_bin = _first_existing(
            [
                nvidia_dir / "cuda_runtime" / "bin",
                nvidia_dir / "cuda_runtime" / "lib",
                nvidia_dir / "cuda_runtime" / "lib64",
            ]
        )
        static_lib = _first_existing(
            [
                nvidia_dir / "cuda_runtime" / "lib" / "x64",
                nvidia_dir / "cuda_runtime" / "lib",
                nvidia_dir / "cuda_runtime" / "lib64",
            ]
        )
        nvvm_dir = _first_existing(
            [
                nvidia_dir / "nvvm" / "bin",
                nvidia_dir / "nvvm" / "lib64",
                nvidia_dir / "nvvm" / "lib",
                nvidia_dir / "cuda_nvcc" / "nvvm" / "bin",
            ]
        )
        libdevice_dir = _first_existing(
            [
                nvidia_dir / "nvvm" / "libdevice",
                nvidia_dir / "cuda_nvcc" / "nvvm" / "libdevice",
            ]
        )
        if shared_bin and nvvm_dir and libdevice_dir:
            return {
                "provider": "nvidia-pip-split",
                "root": str(nvidia_dir),
                "shared_lib_dir": str(shared_bin),
                "static_lib_dir": str(static_lib) if static_lib else "",
                "nvvm_dir": str(nvvm_dir),
                "libdevice_dir": str(libdevice_dir),
            }

    return {
        "provider": "",
        "root": "",
        "shared_lib_dir": "",
        "static_lib_dir": "",
        "nvvm_dir": "",
        "libdevice_dir": "",
    }


def _runtime_root(prefix: str | Path | None = None) -> Path:
    base = Path(prefix) if prefix else Path(sys.prefix)
    return base / "luw_cuda_runtime"


def _reset_numba_cuda_path_cache() -> None:
    try:
        import numba.cuda.cuda_paths as cuda_paths  # type: ignore
    except Exception:
        return
    if hasattr(cuda_paths.get_cuda_paths, "_cached_result"):
        delattr(cuda_paths.get_cuda_paths, "_cached_result")


def _ensure_file_copied(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if target.stat().st_size == source.stat().st_size:
                return
        except OSError:
            pass

    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return
    except Exception:
        pass
    shutil.copy2(source, target)


def prepare_cuda_runtime_layout(prefix: str | Path | None = None) -> dict[str, object]:
    layout = discover_cuda_wheel_layout()
    runtime_root = _runtime_root(prefix)
    result = {
        "prepared": False,
        "provider": layout["provider"],
        "source_root": layout["root"],
        "runtime_root": str(runtime_root),
        "message": "",
    }

    shared_lib_dir = Path(str(layout["shared_lib_dir"])) if layout["shared_lib_dir"] else None
    nvvm_dir = Path(str(layout["nvvm_dir"])) if layout["nvvm_dir"] else None
    libdevice_dir = Path(str(layout["libdevice_dir"])) if layout["libdevice_dir"] else None
    static_lib_dir = Path(str(layout["static_lib_dir"])) if layout["static_lib_dir"] else None

    if shared_lib_dir is None or nvvm_dir is None or libdevice_dir is None:
        result["message"] = "CUDA runtime wheel layout was not found in the current Python environment."
        return result

    runtime_root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shared_target_dir = runtime_root / "bin"
        nvvm_target_dir = runtime_root / "nvvm" / "bin"
        static_target_dir = runtime_root / "lib" / "x64"
    else:
        shared_target_dir = runtime_root / "lib64"
        nvvm_target_dir = runtime_root / "nvvm" / "lib64"
        static_target_dir = runtime_root / "lib64"
    libdevice_target_dir = runtime_root / "nvvm" / "libdevice"

    for source_file in _collect_files(shared_lib_dir):
        _ensure_file_copied(source_file, shared_target_dir / source_file.name)
        if os.name == "nt" and source_file.name.lower().startswith("nvvm"):
            _ensure_file_copied(source_file, nvvm_target_dir / source_file.name)
    for source_file in _collect_files(nvvm_dir):
        _ensure_file_copied(source_file, nvvm_target_dir / source_file.name)
    for source_file in _collect_files(libdevice_dir):
        _ensure_file_copied(source_file, libdevice_target_dir / source_file.name)
    for source_file in _collect_files(static_lib_dir):
        _ensure_file_copied(source_file, static_target_dir / source_file.name)

    manifest = {
        "provider": layout["provider"],
        "source_root": layout["root"],
        "shared_lib_dir": str(shared_lib_dir),
        "static_lib_dir": str(static_lib_dir) if static_lib_dir else "",
        "nvvm_dir": str(nvvm_dir),
        "libdevice_dir": str(libdevice_dir),
    }
    (runtime_root / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result["prepared"] = True
    result["message"] = f"Prepared CUDA runtime layout at {runtime_root}"
    return result


def apply_cuda_runtime_environment(prefix: str | Path | None = None, *, prepare: bool = True) -> dict[str, object]:
    runtime_root = _runtime_root(prefix)
    prepare_result = prepare_cuda_runtime_layout(prefix) if prepare else {
        "prepared": runtime_root.exists(),
        "provider": "",
        "source_root": "",
        "runtime_root": str(runtime_root),
        "message": "",
    }

    result = dict(prepare_result)
    if not runtime_root.exists():
        result["configured"] = False
        result["message"] = result.get("message") or "CUDA runtime layout is unavailable."
        return result

    os.environ["CUDA_HOME"] = str(runtime_root)
    os.environ["CUDA_PATH"] = str(runtime_root)

    path_entries: list[str] = []
    if os.name == "nt":
        path_entries.extend(
            [
                str(runtime_root / "bin"),
                str(runtime_root / "nvvm" / "bin"),
            ]
        )
    else:
        path_entries.extend(
            [
                str(runtime_root / "lib64"),
                str(runtime_root / "nvvm" / "lib64"),
            ]
        )

    existing_path = os.environ.get("PATH", "")
    path_parts = [part for part in existing_path.split(os.pathsep) if part]
    for entry in reversed(path_entries):
        if Path(entry).exists() and entry not in path_parts:
            path_parts.insert(0, entry)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    if os.name != "nt":
        for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            existing = [part for part in os.environ.get(key, "").split(os.pathsep) if part]
            for entry in reversed(path_entries):
                if Path(entry).exists() and entry not in existing:
                    existing.insert(0, entry)
            if existing:
                os.environ[key] = os.pathsep.join(existing)

    _reset_numba_cuda_path_cache()

    result["configured"] = True
    result["cuda_home"] = str(runtime_root)
    return result


def _format_version(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ".".join(str(part) for part in value)
    return str(value)


def _normalize_device_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def probe_cuda_environment() -> dict[str, object]:
    runtime = apply_cuda_runtime_environment()
    result = {
        "configured": bool(runtime.get("configured")),
        "prepared": bool(runtime.get("prepared")),
        "available": False,
        "warning": False,
        "summary": "",
        "error": "",
        "provider": runtime.get("provider", ""),
        "source_root": runtime.get("source_root", ""),
        "runtime_root": runtime.get("runtime_root", ""),
        "cuda_home": runtime.get("cuda_home", ""),
        "numba_version": "",
        "runtime_version": "",
        "driver_version": "",
        "devices": [],
    }

    try:
        import numba
        from numba import cuda
        from numba.cuda.cudadrv import runtime as cuda_runtime
        from numba.cuda.cudadrv import driver as cuda_driver
    except Exception as exc:
        result["error"] = str(exc)
        result["summary"] = f"CUDA probe unavailable: {exc}"
        return result

    result["numba_version"] = getattr(numba, "__version__", "")

    devices: list[AcceleratorDeviceInfo] = []
    try:
        gpu_list = list(cuda.gpus)
    except Exception as exc:
        result["error"] = str(exc)
        result["summary"] = f"CUDA driver is not available: {exc}"
        return result

    if not gpu_list:
        result["summary"] = "CUDA ready, but no supported NVIDIA GPU was detected."
        return result

    try:
        with gpu_list[0]:
            cuda.current_context()
            device = cuda.get_current_device()
            devices.append(
                AcceleratorDeviceInfo(
                    name=_normalize_device_name(device.name),
                    vendor="NVIDIA",
                    version="CUDA",
                    compute_capability=_format_version(getattr(device, "compute_capability", "")),
                )
            )
    except Exception as exc:
        result["error"] = str(exc)
        result["summary"] = f"CUDA device discovery failed: {exc}"
        return result

    for gpu in gpu_list[1:]:
        try:
            with gpu:
                device = cuda.get_current_device()
                devices.append(
                    AcceleratorDeviceInfo(
                        name=_normalize_device_name(device.name),
                        vendor="NVIDIA",
                        version="CUDA",
                        compute_capability=_format_version(getattr(device, "compute_capability", "")),
                    )
                )
        except Exception:
            continue

    try:
        result["runtime_version"] = _format_version(cuda_runtime.get_version())
    except Exception:
        result["runtime_version"] = ""

    try:
        result["driver_version"] = _format_version(cuda_driver.driver.get_version())
    except Exception:
        result["driver_version"] = ""

    result["devices"] = [asdict(device) for device in devices]
    result["available"] = bool(devices)
    primary = devices[0]
    version_bits = []
    if result["runtime_version"]:
        version_bits.append(f"runtime {result['runtime_version']}")
    if result["driver_version"]:
        version_bits.append(f"driver {result['driver_version']}")
    if primary.compute_capability:
        version_bits.append(f"cc {primary.compute_capability}")
    suffix = f" ({', '.join(version_bits)})" if version_bits else ""
    result["summary"] = f"CUDA available on {len(devices)} device(s); primary device: {primary.name}{suffix}"
    return result


def probe_opencl_environment() -> dict[str, object]:
    result = {
        "available": False,
        "warning": False,
        "summary": "",
        "error": "",
        "version": "",
        "platforms": [],
        "devices": [],
    }

    try:
        import pyopencl as cl
    except Exception as exc:
        result["warning"] = True
        result["error"] = str(exc)
        result["summary"] = f"OpenCL probe unavailable: {exc}"
        return result

    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        result["warning"] = True
        result["error"] = str(exc)
        result["summary"] = f"OpenCL platform discovery failed: {exc}"
        return result

    if not platforms:
        result["warning"] = True
        result["summary"] = "OpenCL is installed, but no OpenCL platform was detected."
        return result

    devices: list[AcceleratorDeviceInfo] = []
    platform_payloads: list[dict[str, object]] = []
    unique_versions: list[str] = []
    for platform_obj in platforms:
        platform_info = {
            "name": str(getattr(platform_obj, "name", "")).strip(),
            "vendor": str(getattr(platform_obj, "vendor", "")).strip(),
            "version": str(getattr(platform_obj, "version", "")).strip(),
            "device_count": 0,
        }
        if platform_info["version"] and str(platform_info["version"]) not in unique_versions:
            unique_versions.append(str(platform_info["version"]))
        try:
            platform_devices = platform_obj.get_devices()
        except Exception as exc:
            platform_info["error"] = str(exc)
            platform_payloads.append(platform_info)
            continue

        platform_info["device_count"] = len(platform_devices)
        for device in platform_devices:
            devices.append(
                AcceleratorDeviceInfo(
                    name=str(getattr(device, "name", "")).strip(),
                    vendor=str(getattr(device, "vendor", "")).strip(),
                    version=str(getattr(device, "version", "")).strip(),
                    driver_version=str(getattr(device, "driver_version", "")).strip(),
                    compute_units=int(getattr(device, "max_compute_units", 0) or 0),
                )
            )
        platform_payloads.append(platform_info)

    result["platforms"] = platform_payloads
    result["devices"] = [asdict(device) for device in devices]
    result["available"] = bool(devices)
    result["warning"] = not result["available"]
    result["version"] = " | ".join(unique_versions)

    if devices:
        primary = devices[0]
        result["summary"] = (
            f"OpenCL available on {len(platforms)} platform(s) and {len(devices)} device(s); "
            f"primary device: {primary.name}"
        )
    else:
        result["summary"] = "OpenCL platforms were detected, but no device could be enumerated."

    return result


def build_startup_report(requirements_path: str | Path) -> dict[str, object]:
    requirements = run_requirements_import_check(requirements_path)
    return {
        "host_platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "requirements_path": str(Path(requirements_path).resolve()),
        "checks": requirements["checks"],
        "requirements_summary": requirements["summary"],
        "cuda": probe_cuda_environment(),
        "opencl": probe_opencl_environment(),
    }


def _print_json(payload: dict[str, object]) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare accelerator runtimes and print startup diagnostics.")
    parser.add_argument("--prepare-cuda-runtime", action="store_true", help="Prepare the local CUDA runtime layout and print its summary.")
    parser.add_argument("--startup-report", type=str, default="", help="Requirements file used to build a startup diagnostics JSON report.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args(argv)

    if args.prepare_cuda_runtime:
        payload = apply_cuda_runtime_environment()
        return _print_json(payload) if args.json else 0

    if args.startup_report:
        payload = build_startup_report(args.startup_report)
        return _print_json(payload) if args.json else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

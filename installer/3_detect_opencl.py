# 3_detect_opencl.py
# Print OpenCL platform and device information.
# Usage: python 3_detect_opencl.py

import sys

def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    n = float(n)
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def main() -> int:
    try:
        import pyopencl as cl
    except Exception as e:
        print("pyopencl is not available. Install it in this environment: pip install pyopencl")
        print(f"ImportError: {e}")
        return 2

    try:
        try:
            platforms = cl.get_platforms()
        except cl.LogicError as e:
            print("No OpenCL platforms found.")
            print(repr(e))
            return 3

        if not platforms:
            print("No OpenCL platforms found.")
            return 3

        print("==== OpenCL Environment Report ====")
        for p in platforms:
            print(f"[Platform] {p.name} | {p.vendor} | {p.version}")
            try:
                devices = p.get_devices()
            except cl.LogicError as e:
                print("  Failed to enumerate devices on this platform.")
                print(f"  {repr(e)}")
                continue

            if not devices:
                print("  No devices found on this platform.")
                continue

            for d in devices:
                try:
                    dtype = cl.device_type.to_string(d.type)
                except Exception:
                    dtype = str(d.type)
                try:
                    gmem = human(d.global_mem_size)
                except Exception:
                    gmem = str(getattr(d, "global_mem_size", "n/a"))
                print(f"  [Device] {d.name.strip()}")
                print(f"           Type: {dtype}")
                print(f"           Vendor: {d.vendor.strip()}")
                print(f"           OpenCL: {d.version}")
                print(f"           Driver: {d.driver_version}")
                print(f"           Compute Units: {d.max_compute_units}")
                print(f"           Max Clock: {d.max_clock_frequency} MHz")
                print(f"           Global Memory: {gmem}")
        return 0
    except Exception as e:
        print("Unexpected error while querying OpenCL.")
        print(repr(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())

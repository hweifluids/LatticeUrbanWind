# ===== POSIX sh script: save as check_env.sh and run `sh check_env.sh` =====
# All prompts and comments are in English.

echo "============================================"
echo "[Python environment]"
echo "============================================"

PY=""
if command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
fi

if [ -n "$PY" ]; then
  echo "Python: FOUND"
  echo "Executable: $PY"
  "$PY" --version 2>&1 | awk '{print "Version: "$2}'
else
  echo "Python: NOT FOUND on PATH"
fi

echo
echo "============================================"
echo "[Pip environment]"
echo "============================================"

PIP=""
if command -v pip3 >/dev/null 2>&1; then
  PIP="$(command -v pip3)"
elif command -v pip >/dev/null 2>&1; then
  PIP="$(command -v pip)"
fi

if [ -n "$PIP" ]; then
  echo "Pip: FOUND"
  echo "Executable: $PIP"
  ver="$($PIP --version 2>/dev/null)"
  echo "$ver"
  echo "$ver" | awk '{print "Version: "$2}'
else
  echo "Pip: NOT FOUND as a standalone executable"
  if [ -n "$PY" ]; then
    if "$PY" -m pip --version >/dev/null 2>&1; then
      echo "Pip via python -m: FOUND"
      "$PY" -m pip --version
    fi
  fi
fi

echo
echo "============================================"
echo "[OpenCL runtime and tools]"
echo "============================================"

OS="$(uname -s)"
case "$OS" in
  Linux)
    if command -v ldconfig >/dev/null 2>&1; then
      LIB_PATH="$(ldconfig -p 2>/dev/null | grep -i 'libOpenCL.so' | head -n1 | awk -F'=> ' '{print $2}')"
      if [ -n "$LIB_PATH" ]; then
        echo "OpenCL runtime: FOUND"
        echo "Runtime library: $LIB_PATH"
      else
        echo "OpenCL runtime: not found via ldconfig"
      fi
    else
      echo "ldconfig not available to query shared libraries"
    fi
    ;;
  Darwin)
    if [ -e "/System/Library/Frameworks/OpenCL.framework/OpenCL" ]; then
      echo "OpenCL runtime: FOUND"
      echo "Runtime framework: /System/Library/Frameworks/OpenCL.framework/OpenCL"
    else
      echo "OpenCL runtime: not found in the default framework path"
    fi
    ;;
  *)
    echo "Runtime check: generic UNIX"
    if [ -e "/usr/lib/libOpenCL.so" ] || [ -e "/usr/local/lib/libOpenCL.so" ]; then
      echo "OpenCL runtime: FOUND"
    else
      echo "OpenCL runtime: not found in common locations"
    fi
    ;;
esac

if command -v clinfo >/dev/null 2>&1; then
  echo
  echo "clinfo tool: FOUND"
  echo "Executable: $(command -v clinfo)"
  clinfo --version 2>/dev/null || true
else
  echo "clinfo tool: NOT FOUND on PATH"
fi

echo
echo "============================================"
echo "[OpenCL device enumeration]"
echo "============================================"

if [ -n "$PY" ]; then
"$PY" - <<'PYEOF'
import ctypes, ctypes.util, sys

def load_opencl():
    candidates = []
    libname = ctypes.util.find_library("OpenCL")
    if libname:
        candidates.append(libname)
    if sys.platform.startswith("win"):
        candidates += ["OpenCL.dll"]
    elif sys.platform == "darwin":
        candidates += ["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
    else:
        candidates += ["libOpenCL.so.1", "libOpenCL.so"]
    last_err = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            return lib, name
        except OSError as e:
            last_err = e
    raise OSError(last_err or "OpenCL library not found")

def get_str_info_platform(cl, platform, param):
    size = ctypes.c_size_t()
    err = cl.clGetPlatformInfo(platform, param, 0, None, ctypes.byref(size))
    if err != 0:
        raise RuntimeError(f"OpenCL error {err}")
    buf = (ctypes.c_char * size.value)()
    err = cl.clGetPlatformInfo(platform, param, size.value, buf, None)
    if err != 0:
        raise RuntimeError(f"OpenCL error {err}")
    return buf.value.decode("utf-8").strip()

def get_str_info_device(cl, device, param):
    size = ctypes.c_size_t()
    err = cl.clGetDeviceInfo(device, param, 0, None, ctypes.byref(size))
    if err != 0:
        raise RuntimeError(f"OpenCL error {err}")
    buf = (ctypes.c_char * size.value)()
    err = cl.clGetDeviceInfo(device, param, size.value, buf, None)
    if err != 0:
        raise RuntimeError(f"OpenCL error {err}")
    return buf.value.decode("utf-8").strip()

def main():
    try:
        cl, loaded_name = load_opencl()
    except OSError as e:
        print("OpenCL: NOT FOUND")
        print(f"Reason: {e}")
        return

    print("OpenCL: FOUND")
    print(f"OpenCL library loaded from: {loaded_name}")

    cl_uint = ctypes.c_uint
    cl_int = ctypes.c_int
    cl_ulong = ctypes.c_ulonglong
    cl_platform_id = ctypes.c_void_p
    cl_device_id = ctypes.c_void_p

    CL_PLATFORM_NAME = 0x0902
    CL_PLATFORM_VERSION = 0x0901
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
    CL_DEVICE_NAME = 0x102B
    CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F

    cl.clGetPlatformIDs.restype = cl_int
    cl.clGetPlatformIDs.argtypes = [cl_uint, ctypes.POINTER(cl_platform_id), ctypes.POINTER(cl_uint)]

    cl.clGetPlatformInfo.restype = cl_int
    cl.clGetPlatformInfo.argtypes = [cl_platform_id, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    cl.clGetDeviceIDs.restype = cl_int
    cl.clGetDeviceIDs.argtypes = [cl_platform_id, ctypes.c_ulong, cl_uint, ctypes.POINTER(cl_device_id), ctypes.POINTER(cl_uint)]

    cl.clGetDeviceInfo.restype = cl_int
    cl.clGetDeviceInfo.argtypes = [cl_device_id, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    num_platforms = cl_uint(0)
    err = cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
    if err != 0:
        if err == -1001:
            print("No OpenCL platforms found.")
            return
        print(f"clGetPlatformIDs error: {err}")
        return

    print(f"OpenCL platforms: {num_platforms.value}")
    if num_platforms.value == 0:
        print("No OpenCL platforms found.")
        return

    platforms = (cl_platform_id * num_platforms.value)()
    err = cl.clGetPlatformIDs(num_platforms, platforms, None)
    if err != 0:
        print(f"clGetPlatformIDs error: {err}")
        return

    total_devices = 0
    for pi, plat in enumerate(platforms):
        try:
            name = get_str_info_platform(cl, plat, CL_PLATFORM_NAME)
        except Exception:
            name = "Unknown"
        try:
            version = get_str_info_platform(cl, plat, CL_PLATFORM_VERSION)
        except Exception:
            version = "Unknown"
        print(f"[Platform {pi}] {name} | {version}")

        num_devices = cl_uint(0)
        err = cl.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(num_devices))
        if err != 0 and err != -1:
            print(f"  clGetDeviceIDs error: {err}")
            continue
        print(f"  Devices: {num_devices.value}")
        total_devices += num_devices.value
        if num_devices.value == 0:
            continue

        devices = (cl_device_id * num_devices.value)()
        err = cl.clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, None)
        if err != 0:
            print(f"  clGetDeviceIDs error: {err}")
            continue

        for di, dev in enumerate(devices):
            try:
                dname = get_str_info_device(cl, dev, CL_DEVICE_NAME)
            except Exception:
                dname = "Unknown"

            mem = cl_ulong(0)
            size_ret = ctypes.c_size_t(0)
            err = cl.clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, ctypes.sizeof(cl_ulong), ctypes.byref(mem), ctypes.byref(size_ret))
            if err != 0:
                mem_bytes = 0
            else:
                mem_bytes = int(mem.value)

            gib = mem_bytes / (1024 ** 3)
            mib = mem_bytes / (1024 ** 2)
            print(f"    [Device {di}] {dname} | Global memory: {mib:.2f} MiB ({gib:.2f} GiB)")

    print(f"Total OpenCL devices across all platforms: {total_devices}")

if __name__ == "__main__":
    main()
PYEOF
else
  if command -v clinfo >/dev/null 2>&1; then
    echo "Python interpreter not available for API enumeration. Falling back to clinfo full report."
    clinfo
  else
    echo "Neither Python nor clinfo is available to enumerate devices."
  fi
fi

echo
echo "Done."
printf "Press Enter to exit..."
# Wait for user input before exiting
# shellcheck disable=SC2162
read _

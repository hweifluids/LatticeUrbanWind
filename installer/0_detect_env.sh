#!/usr/bin/env sh
set -u

echo "============================================"
echo "[Native build environment]"
echo "============================================"

RC=0
if command -v cmake >/dev/null 2>&1; then
  echo "CMake: FOUND"
  cmake --version
else
  echo "CMake: NOT FOUND on PATH"
  RC=2
fi

for compiler in c++ g++ clang++; do
  if command -v "$compiler" >/dev/null 2>&1; then
    echo "C++ compiler: $(command -v "$compiler")"
    "$compiler" --version 2>/dev/null | head -n 1
    break
  fi
done

echo
echo "============================================"
echo "[OpenCL environment]"
echo "============================================"
if command -v ldconfig >/dev/null 2>&1; then
  OPENCL_LIBRARY=$(ldconfig -p 2>/dev/null | grep -i 'libOpenCL.so' | head -n 1 | awk -F'=> ' '{print $2}')
  if [ -n "$OPENCL_LIBRARY" ]; then
    echo "OpenCL runtime: FOUND"
    echo "Runtime library: $OPENCL_LIBRARY"
  else
    echo "OpenCL runtime: not found via ldconfig"
  fi
elif [ -e /System/Library/Frameworks/OpenCL.framework/OpenCL ]; then
  echo "OpenCL runtime: FOUND"
  echo "Runtime framework: /System/Library/Frameworks/OpenCL.framework/OpenCL"
else
  echo "OpenCL runtime: not found in a standard location"
fi

if command -v clinfo >/dev/null 2>&1; then
  echo "clinfo tool: FOUND"
  clinfo --version 2>/dev/null || true
else
  echo "clinfo tool: NOT FOUND on PATH"
fi

exit "$RC"

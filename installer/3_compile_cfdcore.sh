#!/usr/bin/env sh
set -eu

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
LUW_ROOT=$(dirname "$SCRIPT_DIR")
CULBM_ROOT="$LUW_ROOT/core/cfd_core/CULBM"
CULBM_BUILD="$CULBM_ROOT/build-installer"
FLUIDX3D_BUILD="$LUW_ROOT/core/cfd_core/FluidX3D/make.sh"
LUW_CFD_BUILD_JOBS=${LUW_CFD_BUILD_JOBS:-${LUW_BUILD_JOBS:-2}}
LUW_CULBM_REQUIRE_MPI=${LUW_CULBM_REQUIRE_MPI:-OFF}
LUW_ENABLE_FLUIDX3D=${LUW_ENABLE_FLUIDX3D:-OFF}
case "$LUW_ENABLE_FLUIDX3D" in
  [Oo][Nn]) LUW_ENABLE_FLUIDX3D=ON ;;
  [Oo][Ff][Ff]) LUW_ENABLE_FLUIDX3D=OFF ;;
  *)
    echo "LUW_ENABLE_FLUIDX3D must be ON or OFF, got: $LUW_ENABLE_FLUIDX3D" >&2
    exit 2
    ;;
esac

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake was not found on PATH; CULBM is the default solver and cannot be built." >&2
  exit 2
fi
if [ ! -f "$CULBM_ROOT/CMakeLists.txt" ]; then
  echo "Cannot find CULBM source tree: $CULBM_ROOT" >&2
  exit 1
fi
echo "Configuring the default CULBM solver..."
cmake -S "$CULBM_ROOT" -B "$CULBM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCULBM_REQUIRE_MPI="$LUW_CULBM_REQUIRE_MPI"
cmake --build "$CULBM_BUILD" --config Release --target CULBM \
  --parallel "$LUW_CFD_BUILD_JOBS"
if [ ! -x "$CULBM_ROOT/bin/CULBM" ]; then
  echo "CULBM build completed without producing: $CULBM_ROOT/bin/CULBM" >&2
  exit 1
fi
echo "CULBM default solver build succeeded."

if [ "$LUW_ENABLE_FLUIDX3D" != "ON" ]; then
  echo "FluidX3D build skipped. Set LUW_ENABLE_FLUIDX3D=ON to build that separate optional backend."
  exit 0
fi
if [ ! -f "$FLUIDX3D_BUILD" ]; then
  echo "Cannot find FluidX3D build script: $FLUIDX3D_BUILD" >&2
  exit 1
fi

FLUIDX3D_ROOT=$(dirname "$FLUIDX3D_BUILD")
cd "$FLUIDX3D_ROOT"
export OMP_NUM_THREADS="$LUW_CFD_BUILD_JOBS"
export OMP_THREAD_LIMIT="$LUW_CFD_BUILD_JOBS"
case "$(uname -s)" in
  Darwin) FLUIDX3D_TARGET=macOS ;;
  *)
    if command -v xhost >/dev/null 2>&1 && xhost >/dev/null 2>&1; then
      FLUIDX3D_TARGET=Linux-X11
    else
      FLUIDX3D_TARGET=Linux
    fi
    ;;
esac
make clean
make "$FLUIDX3D_TARGET" -j"$LUW_CFD_BUILD_JOBS"
if [ ! -x "$FLUIDX3D_ROOT/bin/FluidX3D" ]; then
  echo "FluidX3D build completed without producing: $FLUIDX3D_ROOT/bin/FluidX3D" >&2
  exit 1
fi
echo "FluidX3D optional solver build succeeded."

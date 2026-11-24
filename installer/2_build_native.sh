#!/usr/bin/env sh
set -eu

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
LUW_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$LUW_ROOT/build"
LUW_BUILD_JOBS=${LUW_BUILD_JOBS:-2}
LUW_ENABLE_FLUIDX3D=${LUW_ENABLE_FLUIDX3D:-OFF}
case "$LUW_ENABLE_FLUIDX3D" in
  [Oo][Nn]) LUW_ENABLE_FLUIDX3D=ON ;;
  [Oo][Ff][Ff]) LUW_ENABLE_FLUIDX3D=OFF ;;
  *)
    echo "LUW_ENABLE_FLUIDX3D must be ON or OFF, got: $LUW_ENABLE_FLUIDX3D" >&2
    exit 2
    ;;
esac
VCPKG_MAX_CONCURRENCY=${VCPKG_MAX_CONCURRENCY:-$LUW_BUILD_JOBS}
export VCPKG_MAX_CONCURRENCY

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake was not found on PATH." >&2
  exit 2
fi

set -- -S "$LUW_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  "-DLUW_ENABLE_FLUIDX3D=$LUW_ENABLE_FLUIDX3D"
LUW_NATIVE_PREFIX_PATH=${LUW_CMAKE_PREFIX_PATH:-}
if [ -n "$LUW_NATIVE_PREFIX_PATH" ]; then
  echo "Native dependency prefixes: $LUW_NATIVE_PREFIX_PATH"
  set -- "$@" "-DCMAKE_PREFIX_PATH=$LUW_NATIVE_PREFIX_PATH"
elif [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
  echo "Native dependency prefixes from CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
fi
LUW_VCPKG_ROOT=${VCPKG_ROOT:-}
if [ -z "$LUW_VCPKG_ROOT" ] && [ -f "$LUW_ROOT/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ]; then
  LUW_VCPKG_ROOT="$LUW_ROOT/third_party/vcpkg"
fi
if [ -n "$LUW_VCPKG_ROOT" ]; then
  if [ ! -f "$LUW_VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
    echo "vcpkg toolchain was not found under: $LUW_VCPKG_ROOT" >&2
    exit 2
  fi
  set -- "$@" \
    "-DCMAKE_TOOLCHAIN_FILE=$LUW_VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    "-DVCPKG_OVERLAY_PORTS=$LUW_ROOT/third_party/ports"
else
  echo "vcpkg root was not detected. CMake will use native packages already visible in the configured prefixes."
  echo "Set VCPKG_ROOT to a bootstrapped vcpkg checkout if the manifest dependencies are not installed system-wide."
fi

cmake "$@"
cmake --build "$BUILD_DIR" --config Release --parallel "$LUW_BUILD_JOBS"
cmake --install "$BUILD_DIR" --config Release --prefix "$LUW_ROOT"

chmod +x "$LUW_ROOT/bin/luw" "$LUW_ROOT/bin/.luw-wrapper" \
  "$LUW_ROOT/bin/accelerator_report" "$LUW_ROOT/bin/cdfinspect" \
  "$LUW_ROOT/bin/detect_opencl" "$LUW_ROOT/bin/shpinspect" \
  "$LUW_ROOT/bin/shptester" "$LUW_ROOT/bin/visdem" \
  "$LUW_ROOT/bin/luwbc" "$LUW_ROOT/bin/luwcut" "$LUW_ROOT/bin/luwvox" \
  "$LUW_ROOT/bin/luwval" "$LUW_ROOT/bin/makeluw" "$LUW_ROOT/bin/runluw" \
  "$LUW_ROOT/bin/visluw" "$LUW_ROOT/bin/transluw" "$LUW_ROOT/bin/vtk2nc" \
  "$LUW_ROOT/bin/vtk2nc_new" "$LUW_ROOT/bin/cleanluw"
chmod +x "$LUW_ROOT/bin/dgprepare" "$LUW_ROOT/bin/season_average" \
  "$LUW_ROOT/bin/les_spectra" "$LUW_ROOT/bin/tke_viz" \
  "$LUW_ROOT/bin/cut_vis" "$LUW_ROOT/bin/buildingscale" \
  "$LUW_ROOT/bin/batch_tke_geo_viz" \
  "$LUW_ROOT/bin/vtk_avg_to_utm_asl_nc"

echo "Native LUW build installed to: $LUW_ROOT/bin"

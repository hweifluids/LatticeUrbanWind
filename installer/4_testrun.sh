#!/usr/bin/env sh
set -eu

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
LUW_ROOT=$(dirname "$SCRIPT_DIR")
LAUNCHER="$LUW_ROOT/bin/luw"
STUDIO="$LUW_ROOT/bin/luw_studio"
CULBM="$LUW_ROOT/core/cfd_core/CULBM/bin/CULBM"
FLUIDX3D="$LUW_ROOT/core/cfd_core/FluidX3D/bin/FluidX3D"
LUW_ENABLE_FLUIDX3D=${LUW_ENABLE_FLUIDX3D:-OFF}
case "$LUW_ENABLE_FLUIDX3D" in
  [Oo][Nn]) LUW_ENABLE_FLUIDX3D=ON ;;
  [Oo][Ff][Ff]) LUW_ENABLE_FLUIDX3D=OFF ;;
  *)
    echo "LUW_ENABLE_FLUIDX3D must be ON or OFF, got: $LUW_ENABLE_FLUIDX3D" >&2
    exit 2
    ;;
esac

if [ ! -f "$LAUNCHER" ]; then
  echo "Native launcher not found: $LAUNCHER" >&2
  exit 1
fi
if [ ! -x "$CULBM" ]; then
  echo "Default CULBM solver not found: $CULBM" >&2
  exit 1
fi
if [ "$LUW_ENABLE_FLUIDX3D" = "ON" ] && [ ! -x "$FLUIDX3D" ]; then
  echo "Optional FluidX3D solver not found: $FLUIDX3D" >&2
  exit 1
fi

export LUW_HOME="$LUW_ROOT"
export PATH="$LUW_ROOT/bin:/usr/bin:/bin"
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH
unset LD_LIBRARY_PATH DYLD_LIBRARY_PATH GDAL_DATA PROJ_DATA PROJ_LIB

"$LAUNCHER" --version
"$LAUNCHER" --help

if "$LAUNCHER" detect_opencl; then
  OPENCL_STATUS=0
else
  OPENCL_STATUS=$?
fi

if [ "$OPENCL_STATUS" -eq 3 ]; then
  echo "WARNING: No OpenCL platforms found; native CPU tools remain available." >&2
  OPENCL_STATUS=0
fi
[ "$OPENCL_STATUS" -ne 0 ] && exit "$OPENCL_STATUS"

if [ ! -f "$STUDIO" ]; then
  echo "Native GUI not found: $STUDIO" >&2
  exit 1
fi
"$STUDIO" -platform offscreen --runtime-self-check

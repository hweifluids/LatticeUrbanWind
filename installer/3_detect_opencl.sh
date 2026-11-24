#!/usr/bin/env sh
set -u

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
LUW_ROOT=$(dirname "$SCRIPT_DIR")
LAUNCHER="$LUW_ROOT/bin/luw"

if [ ! -f "$LAUNCHER" ]; then
  echo "Native launcher not found: $LAUNCHER" >&2
  exit 2
fi

echo "Probing OpenCL..."
LUW_HOME="$LUW_ROOT" "$LAUNCHER" detect_opencl
RC=$?

echo " --------------------------------------------------"
echo "   LatticeUrbanWind Installer: OpenCL device check"
echo "   ENJOY! Huanxia Wei - huanxia.wei@u.nus.edu"
echo " --------------------------------------------------"
exit "$RC"

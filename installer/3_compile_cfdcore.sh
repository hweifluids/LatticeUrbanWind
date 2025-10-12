#!/usr/bin/env sh
# run parent/core/cfd_core/FluidX3D/make.sh

set -eu

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

TARGET="$PARENT_DIR/core/cfd_core/FluidX3D/make.sh"

if [ ! -f "$TARGET" ]; then
  echo "Cannot find target script: $TARGET" >&2
  exit 1
fi

# Ensure the target script is executable
if [ ! -x "$TARGET" ]; then
  chmod +x "$TARGET"
fi

# Switch to the directory containing the target script to ensure relative paths work
cd "$(dirname "$TARGET")"

# Pass all arguments and execute
exec "$TARGET" "$@"

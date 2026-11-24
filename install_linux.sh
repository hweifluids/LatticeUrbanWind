#!/usr/bin/env bash
set -u -o pipefail
unset BASH_ENV

confirm_exit() {
  if [ -t 1 ] && [ -t 0 ]; then
    printf "Task completed, press Enter to exit..."
    read -r _
  fi
}
trap confirm_exit EXIT

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
INSTALLER_DIR="$SCRIPT_DIR/installer"

if [ ! -d "$INSTALLER_DIR" ]; then
  echo "Cannot find installer directory: $INSTALLER_DIR"
  exit 1
fi

mapfile -t SORTED < <(
  for file in "$INSTALLER_DIR"/*.sh; do
    [ -e "$file" ] || continue
    base=$(basename "$file")
    prefix=${base%%_*}
    if [[ "$prefix" =~ ^[0-9]+$ ]]; then
      printf "%s\t%s\n" "$prefix" "$file"
    fi
  done | sort -n -k1,1
)

if [ "${#SORTED[@]}" -eq 0 ]; then
  echo "Cannot find numbered installer scripts in: $INSTALLER_DIR"
  exit 1
fi

export LUW_INSTALL_ORCHESTRATED=1
SUCCESSFUL=()
FAILED=()
FAILURE_CODE=0

for line in "${SORTED[@]}"; do
  IFS=$'\t' read -r _ path <<<"$line"
  name=$(basename "$path")
  echo "Starting: $name"
  bash "$path"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "Completed: $name, status: success"
    SUCCESSFUL+=("$name")
  else
    echo "Completed: $name, status: failed, exit code: $code"
    FAILED+=("$name")
    FAILURE_CODE=$code
  fi
  echo
done

echo "========== Report =========="
if [ "${#SUCCESSFUL[@]}" -gt 0 ]; then
  echo "Successful scripts:"
  for name in "${SUCCESSFUL[@]}"; do echo "  $name"; done
else
  echo "No successful scripts."
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed scripts:"
  for name in "${FAILED[@]}"; do echo "  $name"; done
else
  echo "No failed scripts."
fi
echo "================================"

[ "${#FAILED[@]}" -gt 0 ] && exit "$FAILURE_CODE" || exit 0

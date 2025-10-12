#!/usr/bin/env bash
# Run all .sh installation scripts in the same-level installer directory as root, in order of numeric prefix
# Finally, output a report of successful and failed runs, and wait for user confirmation before exiting

set -u -o pipefail
unset BASH_ENV

confirm_exit() {
  if [ -t 1 ] && [ -t 0 ]; then
    printf "Task completed, press Enter to exit..."
    # shellcheck disable=SC2034
    read -r _
  fi
}
trap confirm_exit EXIT

# Parse directories
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
INSTALLER_DIR="$SCRIPT_DIR/installer"

if [ ! -d "$INSTALLER_DIR" ]; then
  echo "Cannot find installer directory: $INSTALLER_DIR"
  exit 1
fi

# Collect and sort by numeric prefix
mapfile -t SORTED < <(
  for f in "$INSTALLER_DIR"/*.sh; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    num=${base%%_*}
    if [[ "$num" =~ ^[0-9]+$ ]]; then
      printf "%s\t%s\n" "$num" "$f"
    else
      # No numeric prefix scripts are sorted last
      printf "%s\t%s\n" "999999999" "$f"
    fi
  done | sort -n -k1,1
)

if [ "${#SORTED[@]}" -eq 0 ]; then
  echo "Cannot find .sh files in installer directory: $INSTALLER_DIR"
  exit 1
fi

declare -a SUCC=()
declare -a FAIL=()

# Run scripts
for line in "${SORTED[@]}"; do
  IFS=$'\t' read -r _num path <<<"$line"
  name=$(basename "$path")
  echo "Starting: $name"
  chmod +x "$path" 2>/dev/null || true
  source ~/.bashrc
  bash -p "$path"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "Completed: $name, status: success"
    SUCC+=("$name")
  else
    echo "Completed: $name, status: failed, exit code: $rc"
    FAIL+=("$name")
  fi
  echo
done

# Report
echo "========== Report =========="
if [ "${#SUCC[@]}" -gt 0 ]; then
  echo "Successful scripts:"
  for s in "${SUCC[@]}"; do echo "  $s"; done
else
  echo "No successful scripts."
fi

if [ "${#FAIL[@]}" -gt 0 ]; then
  echo "Failed scripts:"
  for f in "${FAIL[@]}"; do echo "  $f"; done
else
  echo "No failed scripts."
fi
echo "================================"

[ "${#FAIL[@]}" -gt 0 ] && exit 2 || exit 0

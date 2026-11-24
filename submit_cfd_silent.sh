#!/usr/bin/env bash

set -euo pipefail

# CFD silent submit script for:
#   Solver: selected by cfd_solver in the deck (CULBM by default)
#   Config: /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/conf.luwpf
#
# Usage:
#   chmod +x ./submit_cfd_silent.sh
#   ./submit_cfd_silent.sh
#
# Monitor after submission:
#   tail -f /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/LatticeUrbanWind_latest.log
#   ps -fp "$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/LatticeUrbanWind.pid)"
#
# Stop the running job if needed:
#   kill -- "-$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/LatticeUrbanWind.pid)"
#
# Force kill only if the normal kill does not stop it:
#   kill -9 -- "-$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/LatticeUrbanWind.pid)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_PATH="$SCRIPT_DIR/bin/luw"
CONFIG_PATH="/home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/conf.luwpf"

CASE_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
RESULTS_DIR="$CASE_DIR/RESULTS"
PID_FILE="$RESULTS_DIR/LatticeUrbanWind.pid"
LATEST_LOG="$RESULTS_DIR/LatticeUrbanWind_latest.log"
STAMPED_LOG="$RESULTS_DIR/LatticeUrbanWind_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$LAUNCHER_PATH" ]]; then
    echo "ERROR: native launcher not found or not executable: $LAUNCHER_PATH" >&2
    exit 1
fi

if ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: setsid is required to manage the launcher and solver as one process group" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${OLD_PID:-}" ]] && kill -0 -- "-$OLD_PID" 2>/dev/null; then
        echo "ERROR: a LatticeUrbanWind job is already running with PID $OLD_PID" >&2
        echo "Check with: ps -fp $OLD_PID" >&2
        echo "Stop with : kill -- -$OLD_PID" >&2
        exit 1
    fi
fi

export LUW_HOME="$SCRIPT_DIR"
cd "$SCRIPT_DIR"

nohup setsid "$LAUNCHER_PATH" runluw "$CONFIG_PATH" >"$STAMPED_LOG" 2>&1 < /dev/null &
PID=$!

echo "$PID" >"$PID_FILE"
ln -sfn "$(basename "$STAMPED_LOG")" "$LATEST_LOG"

sleep 1
if ! kill -0 -- "-$PID" 2>/dev/null; then
    echo "ERROR: selected CFD backend exited immediately. Check log: $STAMPED_LOG" >&2
    exit 1
fi

echo "Submitted LatticeUrbanWind through runluw (CULBM unless the deck explicitly selects FluidX3D)."
echo "PID : $PID"
echo "LOG : $STAMPED_LOG"
echo "TAIL: tail -f $LATEST_LOG"

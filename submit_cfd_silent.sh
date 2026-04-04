#!/usr/bin/env bash

set -euo pipefail

# CFD silent submit script for:
#   Solver: ./core/cfd_core/FluidX3D/bin/FluidX3D
#   Config: /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/conf.luwpf
#
# Usage:
#   chmod +x ./submit_cfd_silent.sh
#   ./submit_cfd_silent.sh
#
# Monitor after submission:
#   tail -f /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/FluidX3D_latest.log
#   ps -fp "$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/FluidX3D.pid)"
#
# Stop the running job if needed:
#   kill "$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/FluidX3D.pid)"
#
# Force kill only if the normal kill does not stop it:
#   kill -9 "$(cat /home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/RESULTS/FluidX3D.pid)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_PATH="$SCRIPT_DIR/core/cfd_core/FluidX3D/bin/FluidX3D"
CONFIG_PATH="/home/featurize/WindFieldGen/AVGshanghai/ShanghaiAnnual/conf.luwpf"

CASE_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
RESULTS_DIR="$CASE_DIR/RESULTS"
PID_FILE="$RESULTS_DIR/FluidX3D.pid"
LATEST_LOG="$RESULTS_DIR/FluidX3D_latest.log"
STAMPED_LOG="$RESULTS_DIR/FluidX3D_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$SOLVER_PATH" ]]; then
    echo "ERROR: solver not found or not executable: $SOLVER_PATH" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: a FluidX3D job is already running with PID $OLD_PID" >&2
        echo "Check with: ps -fp $OLD_PID" >&2
        echo "Stop with : kill $OLD_PID" >&2
        exit 1
    fi
fi

cd "$(dirname "$SOLVER_PATH")"

nohup "$SOLVER_PATH" "$CONFIG_PATH" >"$STAMPED_LOG" 2>&1 < /dev/null &
PID=$!

echo "$PID" >"$PID_FILE"
ln -sfn "$(basename "$STAMPED_LOG")" "$LATEST_LOG"

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: FluidX3D exited immediately. Check log: $STAMPED_LOG" >&2
    exit 1
fi

echo "Submitted FluidX3D in background."
echo "PID : $PID"
echo "LOG : $STAMPED_LOG"
echo "TAIL: tail -f $LATEST_LOG"

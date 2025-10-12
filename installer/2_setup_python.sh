#!/usr/bin/env bash
set -euo pipefail

# Ensure LUW_HOME is defined
: "${LUW_HOME:?Environment variable LUW_HOME is not set}"

VENV_DIR="$LUW_HOME/.venv"
REQ_FILE="$LUW_HOME/installer/requirements.txt"

log() { printf '[setup-python] %s\n' "$*" >&2; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Ensure a Python interpreter with the venv module is available.
# Prefer python3. If missing or venv is unavailable, try to install via apt-get.
ensure_python_with_venv() {
  if have_cmd python3 && python3 -c "import venv" >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  if have_cmd python && python -c "import venv" >/dev/null 2>&1; then
    echo "python"
    return
  fi

  echo "Error: Python with the 'venv' module is not available. Please manually install 'python3-venv' or a compatible Python environment before running this script." >&2
  exit 1
}


# Validate requirements.txt
if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt not found at $REQ_FILE" >&2
  exit 1
fi

PY="$(ensure_python_with_venv)"

# Create or reuse the virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR. It will be reused."
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Ensure pip exists inside the venv
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  log "pip not detected in the virtual environment. Trying ensurepip inside the venv."
  "$VENV_PY" -m ensurepip --upgrade || true
fi

# Upgrade build tooling inside the venv
log "Upgrading pip, setuptools, and wheel inside the virtual environment"
"$VENV_PY" -m pip install --upgrade pip || true
"$VENV_PY" -m pip install --upgrade setuptools wheel

# Install project requirements
log "Installing requirements from $REQ_FILE"
"$VENV_PIP" install -r "$REQ_FILE"

# Final confirmation
log "Done. Virtual environment Python executable:"
"$VENV_PY" - <<'PY'
import sys, site
print(sys.executable)
print("site-packages:")
for p in site.getsitepackages():
    print("  " + p)
PY

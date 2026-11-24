#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
  DIRECTORY="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
  SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
  [[ "$SCRIPT_SOURCE" != /* ]] && SCRIPT_SOURCE="$DIRECTORY/$SCRIPT_SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
LUW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Detected parent directory:"
echo "  $LUW_ROOT"
read -r -p "Set this as LUW_HOME and ensure \$LUW_HOME/bin is on PATH? [y/N]: " reply
case "$reply" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "Cancelled"; exit 1 ;;
esac

ensure_luw_environment() {
  file="$1"
  [ -f "$file" ] || touch "$file"
  if grep -Eq '^[[:space:]]*export[[:space:]]+LUW_HOME=' "$file"; then
    sed -i -E "s|^[[:space:]]*export[[:space:]]+LUW_HOME=.*$|export LUW_HOME=\"$LUW_ROOT\"|g" "$file"
  else
    printf '\n# LatticeUrbanWind native environment\nexport LUW_HOME="%s"\n' "$LUW_ROOT" >> "$file"
  fi
  if ! grep -q 'LUW_HOME/bin' "$file"; then
    printf '%s\n' 'case ":$PATH:" in *":$LUW_HOME/bin:"*) ;; *) export PATH="$PATH:$LUW_HOME/bin" ;; esac' >> "$file"
  fi
}

ensure_luw_environment "$HOME/.bashrc"
ensure_luw_environment "$HOME/.zshrc"
ensure_luw_environment "$HOME/.profile"

echo "LUW_HOME set to: $LUW_ROOT"
echo "Shell startup files updated."

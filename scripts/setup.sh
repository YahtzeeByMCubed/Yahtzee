#!/usr/bin/env bash
# Brain-in-a-vat dev environment setup.
# Creates .venv with python3.10 and installs editable dev deps from pyproject.toml.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.10 (pyproject pins >=3.10,<3.11)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "Smoke test:          pytest"
echo
echo "Optional extras:"
echo "  pip install -e '.[vision]'           # YOLOv8 + RealSense (Linux x86_64 only)"
echo "  sudo apt install ros-humble-desktop  # ROS 2 (robot demo only)"

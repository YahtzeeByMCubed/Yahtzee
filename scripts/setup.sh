#!/usr/bin/env bash
# Bootstrap a Python 3.10 virtualenv and install brain-in-a-vat dev deps.
# Run from the repo root:  scripts/setup.sh
set -euo pipefail

PY=${PYTHON:-python3.10}
VENV=.venv

if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip wheel
pip install -e ".[dev]"

cat <<'EOF'

Done. Activate the venv with:
    source .venv/bin/activate

Optional extras (Linux/x86_64 robot host only):
    pip install -e '.[vision]'        # YOLOv8 + RealSense
    sudo apt install ros-humble-desktop   # ROS 2 (rclpy is NOT pip-installable)
EOF

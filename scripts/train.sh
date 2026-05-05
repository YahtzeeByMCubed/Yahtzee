#!/usr/bin/env bash
# Run a training session against the brain-in-a-vat simulator.
set -euo pipefail

# shellcheck disable=SC1091
source .venv/bin/activate

python main.py train "$@"

#!/bin/bash
# Thin wrapper around `python fep_runner.py analyse`.
# Reads all settings (system name, windows, replicas) from config.yaml.
#
# Usage:
#   bash analyse.sh              # use last 4000 dV/dλ records (default)
#   bash analyse.sh --tail 2000  # use last 2000 records
set -euo pipefail
python3 "$(dirname "${BASH_SOURCE[0]}")/fep_runner.py" analyse "$@"

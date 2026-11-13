#!/usr/bin/env bash
# Local counterpart of experiment.sh; activate the model's Python environment first.
# Examples: ./notes/experiment_local.sh tfv6 oracle 1,2,3
#           ./notes/experiment_local.sh hipad_depth_ttc 2 --dry-run
_F2D_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

run_experiment_local() (
    set -euo pipefail
    cd "$_F2D_LOCAL_ROOT"
    local python_bin="${F2D_LOCAL_PYTHON:-${F2D_EXPERIMENTS_PYTHON:-python3}}"
    "$python_bin" -u tools/run_local_experiment.py "$@"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_experiment_local "$@"
fi

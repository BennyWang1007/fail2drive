#!/usr/bin/env bash
# Execute: ./notes/experiment.sh hipad depth 2
# Multiple seeds: ./notes/experiment.sh tfv6 oracle 1,2,3
# One controller runs seeds sequentially; workers/CARLA remain separate per seed.
# Controllers default to cp17; override with F2D_MASTER_NODE if needed.
# Or source this file, then: run_experiment hipad depth 2
# Sourcing only defines functions; it never submits jobs.
_F2D_EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

_f2d_experiment_catalog() {
    cat <<'EOF'
hipad_oracle|1,2,3
hipad_depth_ttc|1,2,3
hipad_vlm|1,2,3
tfv6_oracle|1,2,3
tfv6_depth_ttc|1,2,3
tfv6_vlm|1,2,3
hipad_oracle_brake_only|3
hipad_oracle_brake_only_override|3
hipad_oracle_override|3
tfv6_oracle_brake_only|3
tfv6_oracle_brake_only_override|3
tfv6_oracle_override|3
EOF
}

run_experiment() (
    set -euo pipefail
    local name seed model method allowed="" entry seeds command requested_seed
    local -a requested_seeds
    if [[ $# -eq 1 && "$1" == --list ]]; then
        _f2d_experiment_catalog
        return
    fi
    if [[ $# -eq 1 && "$1" == all ]]; then
        while IFS='|' read -r entry seeds; do
            for seed in ${seeds//,/ }; do
                run_experiment "$entry" "$seed"
            done
        done < <(_f2d_experiment_catalog)
        return
    fi
    if [[ $# -eq 3 ]]; then
        model=$1
        method=$2
        seed=$3
        [[ "$method" != depth ]] || method=depth_ttc
        name="${model}_${method}"
    elif [[ $# -eq 2 ]]; then
        name=$1
        seed=$2
    else
        echo "Usage: run_experiment {hipad|tfv6} METHOD SEED[,SEED...]" >&2
        echo "       run_experiment TASK_NAME SEED[,SEED...] | --list | all" >&2
        echo "Example: ./notes/experiment.sh hipad depth 2" >&2
        return 2
    fi
    while IFS='|' read -r entry seeds; do
        if [[ "$entry" == "$name" ]]; then allowed=$seeds; break; fi
    done < <(_f2d_experiment_catalog)
    if [[ -z "$allowed" || ! "$seed" =~ ^[1-3](,[1-3])*$ ]]; then
        echo "Unsupported experiment/seed: $name $seed (use --list)" >&2
        return 2
    fi
    IFS=',' read -r -a requested_seeds <<< "$seed"
    # Validate the entire list before submitting anything.
    for requested_seed in "${requested_seeds[@]}"; do
        if [[ ",$allowed," != *",$requested_seed,"* ]]; then
            echo "Unsupported experiment/seed: $name $requested_seed (allowed: $allowed)" >&2
            return 2
        fi
    done

    local python_bin="${F2D_EXPERIMENTS_PYTHON:-/mnt/bapve/thome/wangyuhao/miniconda3/envs/fail2drive/bin/python}"
    local log_dir="$_F2D_EXPERIMENT_ROOT/logs"
    local master_node="${F2D_MASTER_NODE:-cp17}"
    export F2D_WORKER_EXCLUDE="${F2D_WORKER_EXCLUDE:-cp15,cp18}"
    mkdir -p "$log_dir"
    # Keep --wrap on one logical line and quote each argument to avoid the
    # truncated --max-job-per-task commands seen in manually pasted sbatches.
    printf -v command '%q ' "$python_bin" -u "$_F2D_EXPERIMENT_ROOT/run_all_evals.py" \
        --run-name "$name" --seed "$seed" --routes-per-job 200 \
        --total-max-jobs 60 --max-job-per-task 60
    echo "Submitting $name seeds=$seed routes/job=200 master=$master_node (sequential seeds; separate workers)"
    sbatch \
        --job-name="run_${name}_seed${seed//,/-}" \
        --partition=devq --qos=normal --time=2-00:00:00 \
        --nodelist="$master_node" \
        --nodes=1 --ntasks=1 --cpus-per-task="${F2D_MASTER_CPUS:-1}" --mem="${F2D_MASTER_MEM:-2G}" \
        --chdir="$_F2D_EXPERIMENT_ROOT" \
        --output="$log_dir/run_all_%j.out" \
        --error="$log_dir/run_all_%j.err" \
        --wrap="$command"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_experiment "$@"
fi

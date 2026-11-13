#!/usr/bin/env bash
# ./notes/reserved_experiments.sh plan       # print only (default)
# ./notes/reserved_experiments.sh submit     # four controllers, priority work
# ./notes/reserved_experiments.sh start      # screen + whole-node allocations
# F2D_INCLUDE_LOW=1 ./notes/reserved_experiments.sh submit  # append seed-1 work
# No jobs are cancelled here. Stop previous overlapping controllers AND workers
# under both accounts before submitting this replacement plan.
# Only wangyuhao may submit. Each lane has at most one worker at a time.
# Nodes: cp11 1 GPU/16 CPUs/31000 MiB; cp13 1 GPU/16 CPUs/64000 MiB;
#        cp16 2 GPUs/24 CPUs/61000 MiB; cp20 3 GPUs/32 CPUs/32000 MiB.
# CPU counts are Slurm schedulable units (88 total), not 88 physical cores.
# cp20's third GPU stays unused: 30000 MiB goes to its dual-GPU worker.
# Four 1-CPU/1-GiB controllers on cp11 + one 8-CPU/16-GiB oracle fit cp11.
# Workers: cp11/cp13 oracle 8 CPUs/16G/1 GPU; cp16 16 CPUs/32G/2 GPUs;
#          cp20 16 CPUs/30000M/2 GPUs. Total: 52 CPUs, 6 GPUs with controllers.
# cp20 memory is a scheduling allowance, not a measured peak; watch for OOM.
# Reservation ends 2026-11-16 21:06:04 Asia/Taipei; workers request at most 48h.
# Late in the reservation, --reservation may prevent long workers from starting.
# Completed results survive worker/controller expiry; rerun after checking queues.

set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/mnt/bapve/thome/wangyuhao/miniconda3/envs/fail2drive/bin/python"
RESERVATION="wangyuhao_shuwei_3day"
INCLUDE_LOW="${F2D_INCLUDE_LOW:-0}"
ROUTES_PER_JOB="${F2D_RESERVED_ROUTES_PER_JOB:-50}"

lane_tasks() {
    case "$1" in
        cp16)
            printf '%s\n' 'hipad_vlm 2' 'hipad_depth_ttc 2' \
                'hipad_depth_ttc 3' 'hipad_vlm 3'
            if [[ "$INCLUDE_LOW" == 1 ]]; then
                printf '%s\n' 'hipad_vlm 1' 'hipad_depth_ttc 1'
            fi
            ;;
        cp20)
            printf '%s\n' 'tfv6_vlm 1,2,3' 'tfv6_depth_ttc 2,3'
            if [[ "$INCLUDE_LOW" == 1 ]]; then
                printf '%s\n' 'tfv6_depth_ttc 1'
            fi
            ;;
        cp11)
            printf '%s\n' 'hipad_oracle 1' 'tfv6_oracle 1' \
                'tfv6_oracle_override 3'
            ;;
        cp13)
            printf '%s\n' 'hipad_oracle 2' 'tfv6_oracle_brake_only_override 3'
            ;;
        *) echo "Unknown lane: $1" >&2; return 2 ;;
    esac
}

clean_sbatch_environment() {
    local variable
    # Prevent inherited submission flags from overriding this plan's directives.
    for variable in ${!SBATCH_@}; do unset "$variable"; done
}

run_lane() {
    local node="$1" task seeds failed=0
    clean_sbatch_environment
    export F2D_WORKER_NODE="$node" F2D_RESERVATION="$RESERVATION"
    case "$node" in
        cp16) export F2D_WORKER_MEM=32G ;;
        cp20) export F2D_WORKER_MEM=30000M ;;
        cp11|cp13) export F2D_WORKER_MEM=16G ;;
        *) echo "Unknown lane: $node" >&2; return 2 ;;
    esac
    cd "$ROOT"
    while read -r task seeds; do
        echo "[$(date -Is)] lane=$node task=$task seeds=$seeds"
        if ! "$PYTHON_BIN" -u "$ROOT/run_all_evals.py" \
            --run-name "$task" --seed "$seeds" \
            --routes-per-job "$ROUTES_PER_JOB" \
            --total-max-jobs 60 --max-job-per-task 1; then
            echo "Unfinished/failed: $task seeds=$seeds; stopping lane" >&2
            return 1
        fi
    done < <(lane_tasks "$node")
    return "$failed"
}

monitor_resources() {
    local node="$1" log="$ROOT/logs/resource_${1}_${SLURM_JOB_ID}.log"
    while true; do
        {
            date -Is
            free -m
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
            # This process and the evaluation share the Slurm step's cgroup.
            local group
            group="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
            for metric in memory.current memory.peak memory.events; do
                if [[ -r "/sys/fs/cgroup${group}/$metric" ]]; then
                    echo "$metric"
                    cat "/sys/fs/cgroup${group}/$metric"
                fi
            done
        } >> "$log" 2>&1
        sleep 15
    done
}

allocated_lane() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; return 2; }
    export F2D_IN_ALLOCATION=1
    monitor_resources "$1" &
    local monitor_pid=$!
    trap "kill $monitor_pid 2>/dev/null || true" EXIT
    run_lane "$1"
}

main() {
    local action="${1:-plan}" node command
    case "$action" in
        plan)
            for node in cp16 cp20 cp11 cp13; do
                echo "=== $node (sequential) ==="
                lane_tasks "$node"
            done
            echo 'Completed seeds/routes are skipped using the existing evaluator logic.'
            echo 'Optional lower-priority seed 1: F2D_INCLUDE_LOW=1 ./notes/reserved_experiments.sh submit'
            ;;
        start)
            [[ "$(id -un)" == wangyuhao ]] || {
                echo 'Submit as wangyuhao only.' >&2; return 2;
            }
            clean_sbatch_environment
            mkdir -p "$ROOT/logs"
            local cpus gpus session
            local -a start_nodes=(cp20 cp16 cp11 cp13)
            if [[ $# -gt 1 ]]; then start_nodes=("${@:2}"); fi
            for node in "${start_nodes[@]}"; do
                case "$node" in
                    cp20) cpus=32; gpus=3 ;;
                    cp16) cpus=24; gpus=2 ;;
                    cp11|cp13) cpus=16; gpus=1 ;;
                    *) echo "Unknown node: $node" >&2; return 2 ;;
                esac
                session="f2d_${node}"
                if screen -ls | rg -q "[.]${session}[[:space:]]"; then
                    echo "Screen $session already exists; refusing duplicate launch" >&2
                    return 2
                fi
                screen -L -Logfile "$ROOT/logs/screen_${node}.log" -dmS "$session" \
                    env "F2D_INCLUDE_LOW=$INCLUDE_LOW" \
                    "F2D_RESERVED_ROUTES_PER_JOB=$ROUTES_PER_JOB" \
                    salloc --job-name="reserved_${node}" --partition=devq --qos=normal \
                    --reservation="$RESERVATION" --nodelist="$node" \
                    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --gres="gpu:$gpus" \
                    --mem=0 --time=2-00:00:00 --chdir="$ROOT" \
                    srun --ntasks=1 --cpus-per-task="$cpus" --gres="gpu:$gpus" \
                    bash "$ROOT/notes/reserved_experiments.sh" allocated "$node"
                echo "Started screen $session; log=$ROOT/logs/screen_${node}.log"
            done
            ;;
        submit)
            [[ "$(id -un)" == wangyuhao ]] || {
                echo 'Submit as wangyuhao only.' >&2; return 2;
            }
            clean_sbatch_environment
            mkdir -p "$ROOT/logs"
            for node in cp16 cp20 cp11 cp13; do
                printf -v command '%q ' env "F2D_INCLUDE_LOW=$INCLUDE_LOW" \
                    "F2D_RESERVED_ROUTES_PER_JOB=$ROUTES_PER_JOB" \
                    bash "$ROOT/notes/reserved_experiments.sh" lane "$node"
                sbatch --parsable --job-name="reserved_${node}" \
                    --partition=devq --qos=normal --reservation="$RESERVATION" \
                    --nodelist=cp11 --nodes=1 --ntasks=1 \
                    --cpus-per-task=1 --mem=1G --time=2-00:00:00 \
                    --chdir="$ROOT" --export=ALL \
                    --output="$ROOT/logs/reserved_${node}_%j.out" \
                    --error="$ROOT/logs/reserved_${node}_%j.err" --wrap="$command"
            done
            ;;
        lane) run_lane "${2:?Missing node}" ;;
        allocated) allocated_lane "${2:?Missing node}" ;;
        *) echo "Usage: $0 [plan|start|submit]" >&2; return 2 ;;
    esac
}

main "$@"

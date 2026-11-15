#!/bin/bash
# ==============================================================================
# f2d_ctl.sh — 100% Pure-Bash Launcher & Controller for login02 (ZERO Python on login node!)
#
# Solves two cluster constraints:
#   1. Login node (`login02`) cannot run Python directly.
#   2. `qos=debug` only lasts 30 minutes (`MaxWall=00:30:00`), whereas full evals take hours.
#
# Modes provided (all executed from login02 via `bash f2d_ctl.sh ...`):
#   - `bash f2d_ctl.sh start-server`  : Generates & submits VLM / Depth-TTC server .sbatch in pure Bash!
#                                       Supports elastic sharding (--gpus 8 --gpus-per-job 2) so you don't
#                                       wait for 8~10 4090s to be idle simultaneously!
#   - `bash f2d_ctl.sh submit-master` : Submits `run_all_evals2.py` itself as a `qos=normal` (48-hour walltime)
#                                       Slurm job on a compute node (no 30-min debug timeout, no login Python!).
#   - `bash f2d_ctl.sh one-shot`      : Uses a brief `srun` on a compute node to generate & submit the
#                                       batch jobs (20 jobs per seed for N=10, well under MaxSubmit=100) and
#                                       exits immediately — Slurm manages the queue for the next 48 hours!
#   - `bash f2d_ctl.sh status`        : Pure-Bash status check of squeue, active_server.json, and finished routes.
#   - `bash f2d_ctl.sh stop`          : Pure-Bash cleanup/scancel of server and/or worker jobs.
# ==============================================================================

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

# 所有 log / server state 統一放在這裡
LOG_ROOT="${F2D_LOG_ROOT:-$WORK_DIR/results_shome/logs}"
# One server provides this many concurrent CARLA batch slots (1 => 1:1).
export F2D_ROUTES_PER_SERVER="${F2D_ROUTES_PER_SERVER:-1}"
if ! [[ "$F2D_ROUTES_PER_SERVER" =~ ^[1-9][0-9]*$ ]]; then
    echo "F2D_ROUTES_PER_SERVER must be a positive integer (1 means 1:1)." >&2
    exit 1
fi
SERVER_STATE_ROOT="${F2D_SERVER_STATE_ROOT:-$LOG_ROOT}"
export F2D_SERVER_STATE_ROOT="$SERVER_STATE_ROOT"

CONDA_SH="/mnt/bapve/thome/wangyuhao/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi

usage() {
    cat <<'EOF'
Usage on login02 (Pure Bash — does NOT run Python on login02):

  1) Start Shared Server Pool (Pure Bash sbatch generation on login02):
     # Recommended: Elastic shards (--gpus 8 --gpus-per-job 2 -> 4 jobs of 2x 4090 each, starts as soon as 2 cards free up!)
     bash f2d_ctl.sh start-server --type vlm --gpus 8 --gpus-per-job 2
     # Or single monolithic job (waits until all 6 cards are free simultaneously):
     bash f2d_ctl.sh start-server --type vlm --gpus 6 --gpus-per-job 6
     # Start Depth-TTC shared server (4 GPUs of any type or 4090):
     bash f2d_ctl.sh start-server --type depth_ttc --gpus 4 --gpus-per-job 2 --gpu-type any
     # Health + registration deadline in seconds (default: 900):
     # Add --startup-timeout 1800 for a deliberately longer model startup.

  2) Launch Evaluations (Choose 2A or 2B — neither runs Python on login02!):
     # Shared capacity: 1 server : F2D_ROUTES_PER_SERVER concurrent CARLA batches (default 1).
     # export F2D_ROUTES_PER_SERVER=2   # 1:2; all users must use the same value
     # export F2D_SERVER_STATE_ROOT=/shared/group/f2d_pool  # same group-writable path for all users
     # Each batch submits a backup server, reserves capacity under a shared lock,
     # and cancels only its own new backup when existing capacity is available.
     # GPU types for both VLM and Depth-TTC: --server-gpu-type 4090 --route-gpu-type any (defaults).
     # Use --server-gpu-type any to request any GPU for newly submitted servers.
     # [2A] Submit Master Scheduler (`run_all_evals2.py`) as a 48-hour `qos=normal` Slurm job on a compute node:
     bash f2d_ctl.sh submit-master --run-name tfv6_vlm hipad_depth_ttc --seed 3 --routes-per-job 10 --no-wait-server

     # [2B] Run scheduler via a brief compute-node srun (5-minute allocation):
     bash f2d_ctl.sh one-shot --run-name tfv6_vlm --seed 3 --routes-per-job 10 --no-wait-server

  3) Check Status (Pure Bash on login02):
     bash f2d_ctl.sh status

  4) Stop Jobs (Pure Bash on login02):
     bash f2d_ctl.sh stop [servers|workers|all]
EOF
}

ACTION="${1:-status}"
shift || true

case "$ACTION" in
    start-server)
        STYPE="vlm"
        TOTAL_GPUS="8"
        GPUS_PER_JOB="2"
        GPU_TYPE="nvidia_geforce_rtx_4090"
        PARTITION="devq"
        QOS="normal"
        TIME_LIMIT="2-00:00:00"
        CONDA_ENV="fail2drive"
        STARTUP_TIMEOUT=900
        DRY_RUN=0
        STATE_DIR_OVERRIDE=""
        JOB_ID_FILE=""
        NO_LB=0
        HOLD_SERVER=0

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --type|--server-type) STYPE="$2"; shift 2 ;;
                --gpus|--num-gpus) TOTAL_GPUS="$2"; shift 2 ;;
                --gpus-per-job) GPUS_PER_JOB="$2"; shift 2 ;;
                --server-gpu-type|--gpu-type) GPU_TYPE="$2"; shift 2 ;;
                --partition) PARTITION="$2"; shift 2 ;;
                --qos) QOS="$2"; shift 2 ;;
                --time) TIME_LIMIT="$2"; shift 2 ;;
                --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
                --dry-run) DRY_RUN=1; shift 1 ;;
                --state-dir) STATE_DIR_OVERRIDE="$2"; shift 2 ;;
                --job-id-file) JOB_ID_FILE="$2"; shift 2 ;;
                --no-lb) NO_LB=1; shift ;;
                --hold) HOLD_SERVER=1; shift ;;
                *) echo "Unknown arg: $1"; usage; exit 1 ;;
            esac
        done

        if ! [[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
            echo "--startup-timeout must be a positive number of seconds" >&2
            exit 1
        fi
        if [[ "$GPU_TYPE" == "4090" ]]; then
            GPU_TYPE="nvidia_geforce_rtx_4090"
        fi

        if [[ "$STYPE" == "depth_ttc" ]]; then
            STATE_DIR="$SERVER_STATE_ROOT/depth_ttc_server"
            JOB_NAME="F2D_DepthTTC_Server"
            LB_PORT_START=41400
            PORT_OFFSET_BASE=40500
            ENV_MODS="common depth_ttc debug"
        else
            STATE_DIR="$SERVER_STATE_ROOT/alpamayo_server"
            JOB_NAME="F2D_Alpamayo_Server"
            LB_PORT_START=41200
            PORT_OFFSET_BASE=40100
            ENV_MODS="common vlm debug"
        fi

        STATE_DIR="${STATE_DIR_OVERRIDE:-$STATE_DIR}"
        # Shared group write permissions; directories inherit the pool's group.
        umask 0002
        mkdir -p "$STATE_DIR"
        STATE_DIR="$(cd "$STATE_DIR" && pwd)"
        if [[ -n "$JOB_ID_FILE" ]]; then
            : > "$JOB_ID_FILE"
        fi

        WORKERS_DIR="$STATE_DIR/workers"
        LOGS_DIR="$STATE_DIR/logs"
        mkdir -p "$WORKERS_DIR" "$LOGS_DIR"

        if [[ "$GPU_TYPE" == "any" ]]; then
            GRES_PREFIX="gpu"
            GPUS_PER_TASK_SPEC="1"
        else
            GRES_PREFIX="gpu:${GPU_TYPE}"
            GPUS_PER_TASK_SPEC="${GPU_TYPE}:1"
        fi

        # Write worker launcher script (carefully escaping all runtime variables for compute node execution)
        SUBMIT_DIR=$(mktemp -d "$STATE_DIR/submit_${STYPE}.XXXXXXXX")
        chmod 2775 "$SUBMIT_DIR"
        WORKER_LAUNCHER="$SUBMIT_DIR/launch_worker_${STYPE}.sh"
        cat > "$WORKER_LAUNCHER" <<EOF
#!/bin/bash
set -e
umask 0002
NODE_HOST=\$(hostname)
JOB_ID="\${SLURM_JOB_ID:-manual}"
JOB_NUM="\${SLURM_JOB_ID:-0}"
if ! [[ "\$JOB_NUM" =~ ^[0-9]+$ ]]; then
    JOB_NUM=0
fi
TASK_ID="\${SLURM_PROCID:-0}"
IFS=',' read -ra GPU_ARRAY <<< "\${CUDA_VISIBLE_DEVICES:-0}"
echo "[Worker-Node \$NODE_HOST] (${STYPE}) Job=\$JOB_ID Task=\$TASK_ID CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}"

PIDS=()
MONITOR_PIDS=()
MY_FILES=()
READY_DIR=\$(mktemp -d "\${SLURM_TMPDIR:-/tmp}/f2d-ready.XXXXXX")
cleanup_workers() {
    # Stop processes before touching shared storage, which may itself be stuck.
    for pid in "\${MONITOR_PIDS[@]}" "\${PIDS[@]}"; do kill "\$pid" 2>/dev/null || true; done
    sleep 1
    for pid in "\${MONITOR_PIDS[@]}" "\${PIDS[@]}"; do kill -KILL "\$pid" 2>/dev/null || true; done
    timeout -k 2s 5s rm -f "\${MY_FILES[@]}" 2>/dev/null || true
    rm -rf "\$READY_DIR"
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for idx in "\${!GPU_ARRAY[@]}"; do
    LOCAL_GPU="\${GPU_ARRAY[\$idx]}"
    WORKER_ID="\${NODE_HOST}_j\${JOB_ID}_t\${TASK_ID}_g\${LOCAL_GPU}"
    WORKER_SAVE_DIR="${LOGS_DIR}/\$WORKER_ID"
    mkdir -p "\$WORKER_SAVE_DIR/inputs"
    WORKER_LOG="\$WORKER_SAVE_DIR/server.log"
    WORKER_JSON="${WORKERS_DIR}/worker_\${WORKER_ID}.json"
    MY_FILES+=("\$WORKER_JSON" "\${WORKER_JSON}.tmp")

    PORT_BASE=\$(( ${PORT_OFFSET_BASE} + ((JOB_NUM % 50) * 6) + (TASK_ID * 15) + (idx * 3) ))
    WORKER_PORT=\$(comm -23 <(seq \$PORT_BASE \$((PORT_BASE + 350)) | sort) <(ss -Htan | awk '{print \$4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

    echo "[Worker \$WORKER_ID] Starting ${STYPE} server on \$NODE_HOST:\$WORKER_PORT (GPU=\$LOCAL_GPU)..."
    if [[ "${STYPE}" == "depth_ttc" ]]; then
        DEPTH_TTC_LOG_ARGS=()
        if [[ "\${DEPTH_TTC_INFERENCE_LOGGING:-0}" == "1" ]]; then
            export DEPTH_TTC_DEBUG_DIR="\$WORKER_SAVE_DIR/depth_ttc_debug"
            DEPTH_TTC_LOG_ARGS+=(--inference-logging)
        fi
        CUDA_VISIBLE_DEVICES="\$LOCAL_GPU" python -u "\$F2D/tools/depth_ttc_server.py" \\
            --host 0.0.0.0 --port "\$WORKER_PORT" --device cuda:0 \\
            --depth-anything-root "\$DEPTH_ROOT" --checkpoint "\$DEPTH_CHECKPOINT" \\
            "\${DEPTH_TTC_LOG_ARGS[@]}" \\
            >"\$WORKER_LOG" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="\$LOCAL_GPU" "\$ALPAMAYO_PY" -u "\$F2D/tools/alpamayo_vlm_server.py" \\
            --host 0.0.0.0 --port "\$WORKER_PORT" --device cuda:0 \\
            --model "\$ALPAMAYO_MODEL" --alpamayo-version "\$ALPAMAYO_VERSION" \\
            --max-new-tokens "\$ALPAMAYO_MAX_NEW_TOKENS" \\
            --input-save-dir "\$WORKER_SAVE_DIR/inputs" \\
            --raw-log-path "\$WORKER_SAVE_DIR/alpamayo_raw_responses.txt" \\
            >"\$WORKER_LOG" 2>&1 &
    fi
    WPID=\$!
    PIDS+=("\$WPID")

    (
        trap - EXIT INT TERM
        while kill -0 "\$WPID" 2>/dev/null; do
            if curl --noproxy '*' -sf --connect-timeout 2 --max-time 3 "http://127.0.0.1:\$WORKER_PORT/health" >/dev/null 2>&1; then
                cat > "\${WORKER_JSON}.tmp" <<JSONEOF
{
  "worker_id": "\$WORKER_ID",
  "slurm_job_id": "\$JOB_ID",
  "server_type": "${STYPE}",
  "host": "\$NODE_HOST",
  "port": \$WORKER_PORT,
  "gpu_id": "\$LOCAL_GPU",
  "base_url": "http://\$NODE_HOST:\$WORKER_PORT",
  "generate_url": "http://\$NODE_HOST:\$WORKER_PORT/generate",
  "score_url": "http://\$NODE_HOST:\$WORKER_PORT/score",
  "health_url": "http://\$NODE_HOST:\$WORKER_PORT/health"
}
JSONEOF
                mv "\${WORKER_JSON}.tmp" "\$WORKER_JSON"
                touch "\$READY_DIR/\$WPID"
                echo "[Worker \$WORKER_ID] READY at http://\$NODE_HOST:\$WORKER_PORT"
                exit 0
            fi
            if ! kill -0 "\$WPID" 2>/dev/null; then
                tail -50 "\$WORKER_LOG" >&2 || true
                exit 1
            fi
            sleep 5
        done
        exit 1
    ) &
    MONITOR_PIDS+=("\$!")
done

# Supervise readiness in the foreground. A stuck NFS operation in a background
# registration process cannot hide the startup deadline from this supervisor.
STARTED_AT=\$SECONDS
LAST_HEARTBEAT=\$SECONDS
while true; do
    for idx in "\${!PIDS[@]}"; do
        pid="\${PIDS[\$idx]}"
        if ! kill -0 "\$pid" 2>/dev/null; then
            echo "[Worker] ERROR: server PID \$pid exited; failing job." >&2
            exit 1
        fi
        if [[ ! -f "\$READY_DIR/\$pid" ]]; then
            if ! kill -0 "\${MONITOR_PIDS[\$idx]}" 2>/dev/null; then
                echo "[Worker] ERROR: registration for PID \$pid failed." >&2
                exit 1
            fi
            if (( SECONDS - STARTED_AT >= ${STARTUP_TIMEOUT} )); then
                echo "[Worker] ERROR: PID \$pid did not become healthy and register within ${STARTUP_TIMEOUT}s; failing job." >&2
                ps -o pid,stat,wchan:32,etime,args -p "\$pid" >&2 || true
                exit 1
            fi
        fi
    done
    if (( SECONDS - LAST_HEARTBEAT >= 30 )); then
        for file in "\${MY_FILES[@]}"; do
            [ ! -f "\$file" ] || touch "\$file"
        done
        LAST_HEARTBEAT=\$SECONDS
    fi
    sleep 2
done
EOF
        chmod +x "$WORKER_LAUNCHER"

        if [[ "$GPUS_PER_JOB" -le 0 || "$GPUS_PER_JOB" -ge "$TOTAL_GPUS" ]]; then
            GPUS_PER_JOB="$TOTAL_GPUS"
        fi

        REM="$TOTAL_GPUS"
        SHARD_IDX=0
        LB_PORT_END=$(( LB_PORT_START + 190 ))
        while [[ "$REM" -gt 0 ]]; do
            TAKE=$(( REM < GPUS_PER_JOB ? REM : GPUS_PER_JOB ))
            REM=$(( REM - TAKE ))
            SHARD_FILE="$SUBMIT_DIR/run_${STYPE}_shard${SHARD_IDX}.sbatch"

            if [[ "$TAKE" -le 2 ]]; then
                CPUS_REQ=$(( TAKE * 4 ))
                MEM_REQ=$(( TAKE * 16 ))
                SLURM_ALLOC_LINES="#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_REQ}
#SBATCH --mem=${MEM_REQ}G
#SBATCH --gres=${GRES_PREFIX}:${TAKE}"
            else
                SLURM_ALLOC_LINES="#SBATCH --ntasks=${TAKE}
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=${GPUS_PER_TASK_SPEC}"
            fi

            cat > "$SHARD_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH -o ${LOGS_DIR}/server_%j.out
#SBATCH -e ${LOGS_DIR}/server_%j.err
#SBATCH --time=${TIME_LIMIT}
${SLURM_ALLOC_LINES}

set -e
umask 0002
source "${CONDA_SH}"
set +u
conda activate ${CONDA_ENV}
set -u

export ROOT=/mnt/bapve/thome/wangyuhao/fail2drive
export FAIL2DRIVE_ROOT="\${FAIL2DRIVE_ROOT:-$WORK_DIR}"
export F2D="\${F2D:-$WORK_DIR}"
export LEAD_ROOT="\${LEAD_ROOT:-\$ROOT/lead-cvpr2026}"
export PYTHONPATH="$WORK_DIR/team_code:\$LEAD_ROOT:\${PYTHONPATH:-}"

if [ -f "$WORK_DIR/f2d_env_config.sh" ]; then
    source "$WORK_DIR/f2d_env_config.sh"
    load_f2d_env ${ENV_MODS}
fi

mkdir -p "${WORKERS_DIR}" "${LOGS_DIR}"

LB_PID=""
SRUN_PID=""
HEARTBEAT_PID=""
OWN_LB=0
# Keep this inode permanently: unlinking a flock file permits multiple owners.
# NFS read errors or HTTP 503 while workers initialize are not evidence that
# another job's lock can be removed. The kernel releases the lock on exit.
exec 9>>"${STATE_DIR}/lb.flock"
if [ "${NO_LB}" -eq 1 ]; then
    exec 9>&-
    echo "[Shard #${SHARD_IDX}] Direct-bound worker; no load balancer required."
elif flock -n -E 75 9; then
    OWN_LB=1
    LB_PORT=\$(comm -23 <(seq ${LB_PORT_START} ${LB_PORT_END} | sort) <(ss -Htan | awk '{print \$4}' | cut -d':' -f2 | sort -u) | head -n 1)
    echo "[Shard #${SHARD_IDX} on \$(hostname)] Starting ${STYPE} Load Balancer on port \$LB_PORT..."
    python3 -u "$WORK_DIR/slurm_alpamayo_server.py" _run_lb \\
        --state-dir "${STATE_DIR}" \\
        --server-type "${STYPE}" \\
        --lb-port "\$LB_PORT" \\
        >"${LOGS_DIR}/lb_\${SLURM_JOB_ID:-manual}.log" 2>&1 &
    LB_PID=\$!
else
    LOCK_RC=\$?
    if [ "\$LOCK_RC" -ne 75 ]; then
        echo "[Shard #${SHARD_IDX}] ERROR: cannot acquire LB filesystem lock (exit \$LOCK_RC)." >&2
        exit 1
    fi
    exec 9>&-
    echo "[Shard #${SHARD_IDX}] Load Balancer lock held by another shard; joining pool."
fi

cleanup() {
    if [ -n "\$HEARTBEAT_PID" ]; then
        kill "\$HEARTBEAT_PID" 2>/dev/null || true
        kill -KILL "\$HEARTBEAT_PID" 2>/dev/null || true
    fi
    [ -n "\$SRUN_PID" ] && kill "\$SRUN_PID" 2>/dev/null || true
    if [ "\$OWN_LB" -eq 1 ]; then
        [ -n "\$LB_PID" ] && kill "\$LB_PID" 2>/dev/null || true
    fi
    timeout -k 2s 5s rm -f "${WORKERS_DIR}"/worker_*_j\${SLURM_JOB_ID:-manual}_*.json \\
        "${WORKERS_DIR}"/worker_*_j\${SLURM_JOB_ID:-manual}_*.json.tmp 9>&- 2>/dev/null || true
    timeout -k 2s 10s python3 "$WORK_DIR/f2d_server_slots.py" server-end \\
        --state-dir "${STATE_DIR}" --server-type "${STYPE}" 9>&- || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 -u "$WORK_DIR/f2d_server_slots.py" server-watch \\
    --state-dir "${STATE_DIR}" --server-type "${STYPE}" --parent-pid "\$\$" 9>&- &
HEARTBEAT_PID=\$!
srun --kill-on-bad-exit=1 --ntasks=\${SLURM_NTASKS:-1} --cpu-bind=none "${WORKER_LAUNCHER}" 9>&- &
SRUN_PID=\$!
SUPERVISED_PIDS=("\$SRUN_PID" "\$HEARTBEAT_PID")
[ -z "\$LB_PID" ] || SUPERVISED_PIDS+=("\$LB_PID")
# A dead heartbeat publisher must fail the shard instead of leaving a server
# advertised indefinitely without lifecycle updates.
wait -n "\${SUPERVISED_PIDS[@]}" || true
echo "[Shard #${SHARD_IDX}] ERROR: heartbeat, load balancer, or worker step exited." >&2
exit 1
EOF
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[f2d_ctl Dry-Run] Generated ${SHARD_FILE} (${TAKE} GPUs)"
            else
                SBATCH_OPTIONS=()
                [ "$HOLD_SERVER" -eq 0 ] || SBATCH_OPTIONS+=(--hold)
                JID=$(sbatch --parsable "${SBATCH_OPTIONS[@]}" "$SHARD_FILE")
                JID="${JID%%;*}"
                if [[ -n "$JOB_ID_FILE" ]]; then
                    echo "$JID" >> "$JOB_ID_FILE"
                fi
                echo "[f2d_ctl] Submitted ${STYPE} Server Shard #${SHARD_IDX} (${TAKE} GPUs): JobID=${JID}"
            fi
            SHARD_IDX=$(( SHARD_IDX + 1 ))
        done
        ;;

    submit-master)
        # Submit `run_all_evals2.py` as a 48-hour `qos=normal` Slurm job on a compute node (NO Python on login02!)
        mkdir -p "$LOG_ROOT/multi_eval"
        MASTER_SBATCH=$(mktemp "$LOG_ROOT/multi_eval/run_master_scheduler.XXXXXXXX.sbatch")
        printf -v EXTRA_ARGS '%q ' "$@"
        cat > "$MASTER_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=F2D_Master_Sched
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH -o $LOG_ROOT/multi_eval/master_%j.out
#SBATCH -e $LOG_ROOT/multi_eval/master_%j.err

set -e
source "${CONDA_SH}"
set +u
conda activate fail2drive
set -u

cd "$WORK_DIR"
echo "[Master Scheduler] Running on compute node \$(hostname) (JobID=\$SLURM_JOB_ID)"
python -u run_all_evals2.py ${EXTRA_ARGS}
EOF
        JID=$(sbatch "$MASTER_SBATCH" | awk '{print $NF}')
        echo "[f2d_ctl] Submitted 48h Master Scheduler Job on qos=normal: JobID=${JID}"
        echo "          Monitor logs with: tail -f ${LOG_ROOT}/multi_eval/master_${JID}.out"
        ;;

    one-shot)
        # Run a brief submission on a compute node via srun
        echo "[f2d_ctl] Running one-shot batch job submission via compute node srun..."
        srun -p devq --qos=debug -N1 -n1 --cpus-per-task=2 --mem=4G --time=00:05:00 bash -c "
            source '${CONDA_SH}'
            conda activate fail2drive
            cd '$WORK_DIR'
            python -u run_all_evals2.py $*
        "
        ;;

    status)
        echo "========================================================================"
        echo "[1] Slurm Queue (squeue --me):"
        squeue --me -o "%.10i %.10P %.8q %.32j %.10T %.10M %.6D %R" || true
        echo "------------------------------------------------------------------------"
        for STYPE in vlm depth_ttc; do
            if [[ "$STYPE" == "vlm" ]]; then
                AFILE="$SERVER_STATE_ROOT/alpamayo_server/active_server.json"
                WDIR="$SERVER_STATE_ROOT/alpamayo_server/workers"
            else
                AFILE="$SERVER_STATE_ROOT/depth_ttc_server/active_server.json"
                WDIR="$SERVER_STATE_ROOT/depth_ttc_server/workers"
            fi
            WCOUNT=$(ls -1 "$WDIR"/worker_*.json 2>/dev/null | wc -l || echo 0)
            echo "[2] Shared ${STYPE^^} Server: ${WCOUNT} active GPU worker file(s)"
            if [[ -f "$AFILE" ]]; then
                grep -E '"(lb_base_url|vlm_server_url|depth_ttc_server_url|num_healthy_workers|num_total_workers)"' "$AFILE" || true
            else
                echo "    (No active_server.json yet)"
            fi
        done
        echo "------------------------------------------------------------------------"
        echo "[3] Evaluation Route Progress (results_shome/*/*/res/*_res.json):"
        if [[ -d "$WORK_DIR/results_shome" ]]; then
            for sdir in "$WORK_DIR"/results_shome/*/*; do
                [[ -d "$sdir/res" ]] || continue
                total_res=$(ls -1 "$sdir/res"/*_res.json 2>/dev/null | wc -l || echo 0)
                valid_res=$(grep -l '"global_record": *{' "$sdir/res"/*_res.json 2>/dev/null | wc -l || echo 0)
                rel_name="${sdir#$WORK_DIR/results_shome/}"
                printf "    %-45s : %3d / 200 valid (total files: %3d)\n" "$rel_name" "$valid_res" "$total_res"
            done
        fi
        echo "========================================================================"
        ;;

    stop)
        TARGET="${1:-all}"
        if [[ "$TARGET" == "servers" || "$TARGET" == "all" ]]; then
            squeue --me --noheader -o "%A %j" | awk '$2 ~ /F2D_(Alpamayo|DepthTTC)_Server/ {print $1}' | xargs -r scancel
            # Each server cleans up its own registration. Never erase another
            # user's shared registrations, reservations, or persistent lock inode.
            echo "[f2d_ctl] Stopped shared server jobs."
        fi
        if [[ "$TARGET" == "workers" || "$TARGET" == "all" ]]; then
            squeue --me --noheader -o "%A %j" | awk '$2 ~ /^(F2D_|Fail2Drive_)/ && $2 !~ /Server/ {print $1}' | xargs -r scancel
            echo "[f2d_ctl] Stopped CARLA worker & master scheduler jobs."
        fi
        ;;

    *)
        usage
        exit 1
        ;;
esac

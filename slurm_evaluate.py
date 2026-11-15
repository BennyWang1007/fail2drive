import os
import subprocess
import time
import json
import argparse
from tqdm import tqdm
import random
import re
import sys
import shlex
from collections import Counter

FAIL2DRIVE_JOB_PREFIX = "Fail2Drive_"
MAX_JOBS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_num_jobs.txt")
CLI_MAX_PARALLEL_JOBS = None
RESULTS_ROOT = ""
EXCLUDED_NODES = "cp15,cp18"
DEFAULT_ROUTES_PER_JOB = 50
WORKER_MAX_HOURS = 48  # normal QoS limit; completed route results survive job expiry.

# Applies to both HiPAD and TFv6. Later entries override earlier ones.
GRES_MAPPING = {
    "vlm": "gpu:2",
    "depth_ttc": "gpu:2",
    # "vlm": "gpu:nvidia_geforce_rtx_4090:2",
    # "depth_ttc": "gpu:nvidia_geforce_rtx_4090:2",
}
ROUTE_TIME_LIMIT = "4h"
CARLA_RPC_TIMEOUT = int(os.environ.get("F2D_CARLA_RPC_TIMEOUT", "12600"))  # Per-recovery override; route wall limit remains 4h.
CARLA_RPC_TIMEOUT_BY_NODE = {
    "cp16": 1200,
}

RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def job_name(job):
    return f"{FAIL2DRIVE_JOB_PREFIX}{job['seed']}_{job['route_id']}"


def batch_name(batch):
    seed = batch[0]["seed"] // 10000
    return f"{FAIL2DRIVE_JOB_PREFIX}{seed}_batch_{batch[0]['route_id']}_{batch[-1]['route_id']}"


def route_script(
    job,
    cfg,
    model_port_start=40000,
    fresh_carla=False,
):
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    env_var_cmd = cfg.get("env_var_cmd", "")
    eval_mode = cfg.get("eval_mode", "none")
    is_server = eval_mode in ("depth_ttc", "vlm")

    server_setup = ""
    carla_gpu_prefix = ""
    if is_server:
        carla_gpu_prefix = 'CUDA_VISIBLE_DEVICES="$EVAL_GPU" '
        if eval_mode == "depth_ttc":
            server_cmd = """export DEPTH_TTC_SERVER_URL="http://127.0.0.1:$FREE_MODEL_PORT/score"
export DEPTH_TTC_DEBUG_DIR="${DEPTH_TTC_DEBUG_DIR:-$VIZ_PATH/depth_ttc_debug}"
SERVER_LOG="$VIZ_PATH/depth_ttc_server_${SLURM_JOB_ID:-manual}.log"
if [ "${F2D_WORKER_QUIET:-0}" = "1" ]; then
    SERVER_LOG=/dev/null
    export DEPTH_TTC_DEBUG_DIR=""
fi
DEPTH_TTC_LOG_ARGS=()
if [ "${DEPTH_TTC_INFERENCE_LOGGING:-0}" = "1" ]; then
    DEPTH_TTC_LOG_ARGS+=(--inference-logging)
fi
CUDA_VISIBLE_DEVICES="$SERVER_GPU" setsid python -u "$F2D/tools/depth_ttc_server.py" \
    --host 127.0.0.1 \
    --port "$FREE_MODEL_PORT" \
    --device cuda:0 \
    --depth-anything-root "$DEPTH_ROOT" \
    --checkpoint "$DEPTH_CHECKPOINT" \
    "${DEPTH_TTC_LOG_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!"""
        else:  # vlm
            server_cmd = """export VLM_SERVER_URL="http://127.0.0.1:$FREE_MODEL_PORT/generate"
export ALPAMAYO_SERVER_URL="$VLM_SERVER_URL"
export ALPAMAYO_RAW_LOG_PATH="${ALPAMAYO_RAW_LOG_PATH:-$VIZ_PATH/alpamayo_raw_responses.txt}"
export ALPAMAYO_INPUT_SAVE_DIR="${ALPAMAYO_INPUT_SAVE_DIR:-$VIZ_PATH/alpamayo_inputs}"
SERVER_LOG="$VIZ_PATH/alpamayo_server_${SLURM_JOB_ID:-manual}.log"
if [ "${F2D_WORKER_QUIET:-0}" = "1" ]; then
    SERVER_LOG=/dev/null
    export ALPAMAYO_RAW_LOG_PATH="" ALPAMAYO_INPUT_SAVE_DIR=""
fi
CUDA_VISIBLE_DEVICES="$SERVER_GPU" setsid "$ALPAMAYO_PY" -u "$F2D/tools/alpamayo_vlm_server.py" \
    --host 127.0.0.1 \
    --port "$FREE_MODEL_PORT" \
    --device cuda:0 \
    --model "$ALPAMAYO_MODEL" \
    --alpamayo-version "$ALPAMAYO_VERSION" \
    --max-new-tokens "$ALPAMAYO_MAX_NEW_TOKENS" \
    --input-save-dir "$ALPAMAYO_INPUT_SAVE_DIR" \
    --raw-log-path "$ALPAMAYO_RAW_LOG_PATH" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!"""

        server_setup = f"""
IFS=',' read -r EVAL_GPU SERVER_GPU _ <<< "${{CUDA_VISIBLE_DEVICES:-0,1}}"
if [ -z "$EVAL_GPU" ] || [ -z "$SERVER_GPU" ] || [ "$EVAL_GPU" = "$SERVER_GPU" ]; then
    echo "VLM/Depth-TTC requires two distinct allocated GPUs." >&2
    exit 1
fi
echo "EVAL_GPU=$EVAL_GPU SERVER_GPU=$SERVER_GPU"

FREE_MODEL_PORT=`comm -23 \
    <(seq {model_port_start} {model_port_start + 49} | sort) \
    <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) \
    | shuf | head -n 1`
echo 'Model Server Port:' $FREE_MODEL_PORT

mkdir -p "$VIZ_PATH"
{server_cmd}
echo "Started {eval_mode} server pid=$SERVER_PID log=$SERVER_LOG"

sleep 60  # Fixed model startup delay; no readiness polling.
"""

    return f"""#!/bin/bash
echo 'HPC Node Name:' $(hostname)
echo JOB ID $SLURM_JOB_ID

# NOTE: You can use this in your agent to store visualization outputs
export VIZ_PATH={viz_path}
export SAVE_PATH={viz_path}
export VISUAL_OUTPUT_PATH={viz_path}
export BENCHMARK_ROUTE_ID="{route_id}"

{env_var_cmd}
if [ -n "$VIZ_PATH" ]; then mkdir -p "$VIZ_PATH" || exit 1; fi
printenv | sort
nvidia-smi -L
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

SERVER_PID=""
cleanup() {{
    if [ -n "$SERVER_PID" ]; then
        kill -TERM -- -"$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -- -"$SERVER_PID" 2>/dev/null || true
    fi
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{server_setup}

# Start a fresh episode on the existing CARLA process, even for the same map.
# This also removes actors/sensors left by a failed or timed-out prior route.
{carla_gpu_prefix}python -X faulthandler -u - {shlex.quote(route)} "$FREE_WORLD_PORT" "$TM_PORT" "$CARLA_RPC_TIMEOUT" <<'F2D_RESET_WORLD'
import faulthandler
import os
import sys
import time
import xml.etree.ElementTree as ET

def mark(message):
    print("[CARLA RESET]", time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)

faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(120, repeat=True)
mark("import carla: begin")
import carla
mark("import carla: done; module=" + str(carla.__file__))
mark("python=" + sys.executable + " pid=" + str(os.getpid()))
mark("route=" + sys.argv[1] + " rpc=" + sys.argv[2] + " tm=" + sys.argv[3])

route = ET.parse(sys.argv[1]).getroot().find("route")
if route is None or not route.get("town"):
    raise ValueError("Route XML has no town")
mark("Client(): begin")
client = carla.Client("127.0.0.1", int(sys.argv[2]))
mark("Client(): done; set_timeout(): begin")
client.set_timeout(int(sys.argv[4]))
mark("set_timeout(): done; client_version=" + client.get_client_version())
# Do not create a Traffic Manager in this short-lived reset process. The
# evaluator owns its TM and creates it after this process has loaded the map.
# TM asynchronous mode alone does not release a synchronous world waiting
# for ticks; explicitly restore world settings before requesting a reload.
if {fresh_carla!r}:
    mark("fresh CARLA: load requested map without waiting for the default world")
else:
    mark("get_world(): begin")
    previous_world = client.get_world()
    mark("get_world(): done; get_settings(): begin")
    settings = previous_world.get_settings()
    mark("get_settings(): done; synchronous_mode=" + str(settings.synchronous_mode))
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        mark("apply_settings(async): begin")
        previous_world.apply_settings(settings)
        mark("apply_settings(async): done")
    del previous_world
mark("load_world(" + route.get("town") + "): begin")
world = client.load_world(route.get("town"), reset_settings=True)
mark("load_world(): done; get_map(): begin")
print("[CARLA] Fresh world loaded:", world.get_map().name, flush=True)
mark("get_map(): done; reset complete; leaving interpreter")
faulthandler.cancel_dump_traceback_later()
F2D_RESET_WORLD
reset_exit=$?
if [ "$reset_exit" -ne 0 ]; then
    echo "[CARLA RESET] process failed exit=$reset_exit" >&2
    exit "$reset_exit"
fi

# NOTE: --track=MAP may have to be changed according to agent track
{carla_gpu_prefix}python -u {cfg["lb_script"]} \
--routes={route} \
--repetitions=1 \
--track=SENSORS \
--checkpoint={result_file} \
--timeout="$CARLA_RPC_TIMEOUT" \
--agent={cfg["agent_file"]} \
--agent-config={cfg["agent_config"]} \
--port=${{FREE_WORLD_PORT}} \
--traffic-manager-port=${{TM_PORT}} \
--traffic-manager-seed={seed}
"""


def carla_setup(cfg, log_dir, world_port, streaming_port, tm_port):
    """Launch one CARLA process for the entire sequential batch."""
    render_args = (
        '-RenderOffScreen -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-0}"'
        if cfg["rgb"] else "-nullrhi"
    )
    gpu_setup = ""
    gpu_prefix = ""
    timeout_cases = "\n".join(
        f"    {node}) export CARLA_RPC_TIMEOUT={seconds} ;;"
        for node, seconds in CARLA_RPC_TIMEOUT_BY_NODE.items()
    )
    if cfg["eval_mode"] in ("vlm", "depth_ttc"):
        gpu_setup = '''IFS=',' read -r EVAL_GPU SERVER_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [ -z "$EVAL_GPU" ] || [ -z "$SERVER_GPU" ] || [ "$EVAL_GPU" = "$SERVER_GPU" ]; then
    echo "VLM/Depth-TTC requires two distinct allocated GPUs." >&2
    exit 1
fi
export EVAL_GPU SERVER_GPU'''
        gpu_prefix = 'CUDA_VISIBLE_DEVICES="$EVAL_GPU" '
    return f"""
{cfg.get('env_var_cmd', '') or ':'}
env_exit=$?
if [ "$env_exit" -ne 0 ]; then exit "$env_exit"; fi
{gpu_setup}
node_name="${{SLURMD_NODENAME:-$(hostname -s)}}"
node_name="${{node_name%%.*}}"
export CARLA_RPC_TIMEOUT={CARLA_RPC_TIMEOUT}
case "$node_name" in
{timeout_cases}
esac
echo "[CARLA] node=$node_name RPC timeout=$CARLA_RPC_TIMEOUT seconds"
CARLA_PID=""
ROUTE_PID=""
cleanup_batch() {{
    if [ -n "$ROUTE_PID" ]; then
        kill -TERM "$ROUTE_PID" 2>/dev/null || true
    fi
    if [ -n "$CARLA_PID" ]; then
        kill -TERM -- -"$CARLA_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -- -"$CARLA_PID" 2>/dev/null || true
    fi
}}
trap cleanup_batch EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export FREE_WORLD_PORT=$(comm -23 \\
    <(seq {world_port} {world_port + 49} | sort) \\
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
FREE_STREAMING_PORT=$(comm -23 \\
    <(seq {streaming_port} {streaming_port + 49} | sort) \\
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export TM_PORT=$(comm -23 \\
    <(seq {tm_port} {tm_port + 49} | sort) \\
    <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

mkdir -p {shlex.quote(log_dir)} || exit 1
CARLA_LOG={shlex.quote(log_dir)}/carla_${{SLURM_JOB_ID:-manual}}.log
if [ "${{F2D_WORKER_QUIET:-0}}" = "1" ]; then CARLA_LOG=/dev/null; fi
{gpu_prefix}setsid "$CARLA_ROOT/CarlaUE4.sh" \\
    -carla-rpc-port="$FREE_WORLD_PORT" \\
    -nosound {render_args} \\
    -carla-primary-port=0 \\
    -carla-streaming-port="$FREE_STREAMING_PORT" \\
    >"$CARLA_LOG" 2>&1 &
CARLA_PID=$!
echo "[CARLA] Batch process pid=$CARLA_PID port=$FREE_WORLD_PORT log=$CARLA_LOG"
sleep 60
"""


def bash_file(batch, cfg, world_port, streaming_port, tm_port, model_port):
    """One allocation, sequential routes, each limited to four hours in total."""
    first = batch[0]
    base_dir = os.path.dirname(os.path.dirname(first["job_file"]))
    name = batch_name(batch)
    # Keep filenames short even when a batch contains 50 routes. The full route
    # list is stored in the script comment below for queue-to-route mapping.
    tag = f"{first['route_id']}_{batch[-1]['route_id']}"
    if os.environ.get("F2D_BATCH_NAMESPACE"):
        tag += "_" + os.environ["F2D_BATCH_NAMESPACE"]
    route_ids = ",".join(job["route_id"] for job in batch)
    job_file = os.path.join(base_dir, "run", f"eval_batch_{tag}.sh")
    log_file = os.path.join(base_dir, "out", f"batch_{tag}_out.log")
    err_file = os.path.join(base_dir, "err", f"batch_{tag}_err.log")
    hours = min(4 * len(batch), WORKER_MAX_HOURS)
    days, hours = divmod(hours, 24)
    time_limit = f"{days}-{hours:02d}:00:00"
    is_server = cfg["eval_mode"] in ("vlm", "depth_ttc")
    cpus = 16 if is_server else 8
    mem = "32G" if is_server else "16G"
    # Optional placement for notes/reserved_experiments.sh; normal submissions
    # retain their existing resource requests and node selection.
    mem = os.environ.get("F2D_WORKER_MEM", mem)
    excluded_nodes = os.environ.get("F2D_WORKER_EXCLUDE", EXCLUDED_NODES)
    placement = ""
    if os.environ.get("F2D_WORKER_NODE"):
        placement += f"#SBATCH --nodelist={os.environ['F2D_WORKER_NODE']}\n"
    if os.environ.get("F2D_RESERVATION"):
        placement += f"#SBATCH --reservation={os.environ['F2D_RESERVATION']}\n"
    gres = GRES_MAPPING.get(cfg["eval_mode"], "gpu:1")
    script = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH --exclude={excluded_nodes}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --gres={gres}
#SBATCH --output={log_file}
#SBATCH --error={err_file}
{placement}# F2D_ROUTE_IDS={route_ids}

echo "HPC Node Name: $(hostname)"
echo "SLURM JOB ID: $SLURM_JOB_ID | Batch: {name} ({len(batch)} routes)"
batch_exit=0
"""
    script += carla_setup(cfg, os.path.join(base_dir, "out", "carla_log"),
                          world_port, streaming_port, tm_port)
    for index, job in enumerate(batch, 1):
        body = route_script(job, cfg, model_port, fresh_carla=index == 1)
        quiet = os.environ.get("F2D_WORKER_QUIET") == "1"
        route_out = "/dev/null" if quiet else job["log_file"]
        route_err = "/dev/null" if quiet else job["err_file"]
        delimiter = f"F2D_ROUTE_{index}_SCRIPT"
        script += f"""
echo "[Batch] Starting route {index}/{len(batch)}: {job['route_id']}"
timeout --signal=TERM --kill-after=30s {ROUTE_TIME_LIMIT} bash >{shlex.quote(route_out)} 2>{shlex.quote(route_err)} <<'{delimiter}' &
{body}
{delimiter}
ROUTE_PID=$!
wait "$ROUTE_PID"
route_exit=$?
ROUTE_PID=""
echo "[Batch] Route {job['route_id']} exit=$route_exit"
if [ "$route_exit" -ne 0 ]; then batch_exit=1; fi
"""
    script += '\nexit "$batch_exit"\n'
    with open(job_file, "w", encoding="utf-8") as handle:
        handle.write(script)
    return job_file


def get_running_jobs():
    """{slurm job name: job id} of my jobs (running or pending) that use FAIL2DRIVE_JOB_PREFIX.
    The only place that calls squeue. If squeue fails we retry instead of returning an empty
    dict, because "nothing is running" would make the scheduler resubmit every route."""
    while True:
        try:
            out = subprocess.check_output(
                ["squeue", "--me", "--noheader", "--format", "%A|%j"], text=True
            )
            break
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[warn] Failed to query slurm: {exc}; retrying in 10s")
            time.sleep(10)

    running = {}
    if os.environ.get("F2D_IN_ALLOCATION") == "1":
        # An adopted worker retains its original batch job name. It must not
        # mistake its own allocation for another worker running these routes.
        own_job = os.environ.get("SLURM_JOB_ID", "")
        out = "\n".join(line for line in out.splitlines()
                        if line.partition("|")[0].strip() != own_job)
    # Exact prefix match, prevent a shorter prefix from matching jobs belonging to a longer prefix
    exact_job_pattern = re.compile(rf"^{re.escape(FAIL2DRIVE_JOB_PREFIX)}\d+_[^_]+$")
    for line in out.splitlines():
        job_id, _, name = line.partition("|")
        if exact_job_pattern.match(name.strip()):
            running[name.strip()] = job_id.strip()
    batch_pattern = re.compile(rf"^{re.escape(FAIL2DRIVE_JOB_PREFIX)}(\d+)_b([0-9-]+)$")
    compact_pattern = re.compile(
        rf"^{re.escape(FAIL2DRIVE_JOB_PREFIX)}(\d+)_batch_(\d+)_(\d+)$"
    )
    for line in out.splitlines():
        job_id, _, name = line.partition("|")
        compact = compact_pattern.fullmatch(name.strip())
        if compact:
            seed = int(compact.group(1))
            script = os.path.join(
                RESULTS_ROOT, str(seed), "run",
                f"eval_batch_{compact.group(2)}_{compact.group(3)}.sh",
            )
            route_ids = None
            with open(script, encoding="utf-8") as handle:
                for script_line in handle:
                    if script_line.startswith("# F2D_ROUTE_IDS="):
                        route_ids = script_line.strip().partition("=")[2].split(",")
                        break
            if not route_ids or any(not rid.isdigit() for rid in route_ids):
                raise RuntimeError(f"Cannot identify routes of running batch {job_id}: {script}")
            for route_id in route_ids:
                route_seed = seed * 10000 + int(route_id) % 1000
                running[f"{FAIL2DRIVE_JOB_PREFIX}{route_seed}_{route_id}"] = job_id.strip()
            continue
        match = batch_pattern.fullmatch(name.strip())
        if match:
            seed = int(match.group(1))
            for route_id in match.group(2).split("-"):
                route_seed = seed * 10000 + int(route_id) % 1000
                running[f"{FAIL2DRIVE_JOB_PREFIX}{route_seed}_{route_id}"] = job_id.strip()
    return running


def _summarize_result(evaluation_data):
    """(progress, [statuses]) for display only. Never raises."""
    try:
        checkpoint = evaluation_data.get("_checkpoint", {})
        progress = checkpoint.get("progress")
        statuses = [
            r.get("status") if isinstance(r, dict) else repr(r)
            for r in checkpoint.get("records", [])
        ]
        return progress, statuses
    except Exception:
        return None, []


def _retry_reason(evaluation_data):
    """None -> result is final (do NOT rerun). str -> why it must be rerun.
    Same rules as the old _is_retryable_result, but says *which* rule fired."""
    if not isinstance(evaluation_data, dict):
        return "json top level is not a dict"
    checkpoint = evaluation_data.get("_checkpoint")
    if not isinstance(checkpoint, dict):
        return "no _checkpoint"

    progress = checkpoint.get("progress")
    records = checkpoint.get("records")
    if not isinstance(progress, list) or not isinstance(records, list):
        return "progress/records malformed"
    if len(progress) < 2:
        return f"progress too short {progress}"
    if progress[0] < progress[1]:
        return f"progress incomplete {progress}"
    if len(records) == 0:
        return "records empty"

    for record in records:
        if not isinstance(record, dict):
            return "record is not a dict"
        scores = record.get("scores") or {}
        if scores.get("score_composed") is None or scores.get("score_route") is None:
            return "record scores missing"
        if record.get("status") in RETRYABLE_STATUSES:
            return f"status {record.get('status')!r} is in RETRYABLE_STATUSES"
    return None


# def _is_retryable_result(evaluation_data):
#     return _retry_reason(evaluation_data) is not None


def get_max_num_parallel_jobs():
    # [MODIFIED] Prioritize the CLI argument (--max-parallel-jobs) if provided
    if CLI_MAX_PARALLEL_JOBS is not None:
        return CLI_MAX_PARALLEL_JOBS

    try:
        with open(MAX_JOBS_FILE, "r", encoding="utf-8") as f:
            max_num_parallel_jobs = int(f.read().strip())
    except (OSError, ValueError) as exc:
        print(
            f"[warn] Failed to read max parallel jobs from '{MAX_JOBS_FILE}': {exc}. Falling back to 1."
        )
        return 1
    return max_num_parallel_jobs


def classify_job(job, running_jobs):
    """Single source of truth for 'does this job stay in the queue (= get run)?'.
    Used by BOTH filter_completed() and --dry-run, so they can never disagree.
    running_jobs is the {job name: job id} dict from get_running_jobs().

    Returns (keep, label, detail):
      RUNNING     in squeue                      -> keep, but not submitted again
      NO_RESULT   res file does not exist        -> run
      UNREADABLE  res file is not valid json     -> run
      RETRY       res file exists but fails _retry_reason() -> run
      DONE        res file is final              -> never run
    keep is additionally gated by job["tries"] > 0, exactly like before.
    """
    if job_name(job) in running_jobs:
        return True, "RUNNING", f"job_id {running_jobs[job_name(job)]} in squeue"

    tries_left = job["tries"] > 0
    result_file = job["result_file"]
    if not os.path.exists(result_file):
        return tries_left, "NO_RESULT", "result file missing"

    try:
        with open(result_file, "r") as f:
            evaluation_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return tries_left, "UNREADABLE", f"{type(exc).__name__}: {exc}"

    progress, statuses = _summarize_result(evaluation_data)
    reason = _retry_reason(evaluation_data)
    if reason is None:
        return False, "DONE", f"statuses={statuses} progress={progress}"
    return tries_left, "RETRY", f"{reason} | file has statuses={statuses} progress={progress}"


def filter_completed(jobs, running_jobs):
    filtered_jobs = []

    for job in jobs:
        keep, label, detail = classify_job(job, running_jobs)
        if label == "UNREADABLE":
            print(f"[warn] Could not read '{job['result_file']}': {detail}")
        if keep:
            filtered_jobs.append(job)
    return filtered_jobs


def dry_run_report(job_queue, verbose=False):
    """Print what the scheduler WOULD submit. Touches nothing (no sbatch, no rm, no file writes)."""
    running_jobs = get_running_jobs()

    rows = []  # (will_submit, label, detail, job)
    for job in job_queue:
        keep, label, detail = classify_job(job, running_jobs)
        # RUNNING jobs are kept in the queue but never submitted again.
        rows.append((keep and label != "RUNNING", label, detail, job))

    n_total = len(rows)
    submit = [r for r in rows if r[0]]
    by_label = Counter(r[1] for r in submit)
    rerun_with_result = [r for r in submit if r[1] in ("RETRY", "UNREADABLE")]

    print()
    print("=" * 78)
    print(f"DRY RUN  (nothing is submitted / deleted)   total jobs: {n_total}")
    print("=" * 78)
    print(f"would SUBMIT : {len(submit)}")
    print(f"   never run yet (no res file)      : {by_label['NO_RESULT']}")
    print(f"   res file EXISTS but will rerun   : {len(rerun_with_result)}   <-- check these")
    print(f"        RETRY (rule fired)          : {by_label['RETRY']}")
    print(f"        UNREADABLE (bad json)       : {by_label['UNREADABLE']}")
    print(f"skip, already RUNNING/PENDING : {sum(1 for r in rows if r[1] == 'RUNNING')}")
    print(f"skip, DONE (final result)     : {sum(1 for r in rows if r[1] == 'DONE')}")

    if rerun_with_result:
        print("\n-- why do jobs that already have a res file get rerun? --")
        why = Counter(re.sub(r" \| file has.*", "", r[2]) for r in rerun_with_result)
        for reason, n in why.most_common():
            print(f"  {n:5d}  {reason}")

        print("\n-- (statuses found in those res files, for the rerun jobs) --")
        seen = Counter(re.sub(r".*\| file has ", "", r[2]) for r in rerun_with_result if r[1] == "RETRY")
        for k, n in seen.most_common(15):
            print(f"  {n:5d}  {k}")

    print("\n-- jobs that would be submitted --")
    for will_submit, label, detail, job in rows:
        if will_submit:
            print(f"  [{label:10s}] seed_dir={os.path.basename(os.path.dirname(os.path.dirname(job['result_file'])))} "
                  f"route={job['route_id']}  {detail}")

    if verbose:
        print("\n-- jobs that would be skipped --")
        for will_submit, label, detail, job in rows:
            if not will_submit:
                print(f"  [{label:10s}] seed_dir={os.path.basename(os.path.dirname(os.path.dirname(job['result_file'])))} "
                      f"route={job['route_id']}  {detail}")
    print()


if __name__ == "__main__":

    default_mode = os.environ.get("ACTIVATION_POLICY", "none")
    if default_mode not in ("depth_ttc", "vlm"):
        default_mode = "none"

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", "--eval-mode", dest="eval_mode",
                        choices=("none", "depth_ttc", "vlm"), default=default_mode,
                        help="Evaluation mode: none (1 GPU), depth_ttc (2 GPUs), or vlm (2 GPUs)")
    parser.add_argument('--routes', type=str, default='fail2drive_split',
                        help='Path to folder containing the split route files')
    parser.add_argument('--out_root', '--out-root', dest='out_root', type=str, default='results/fail2drive',
                        help='Path where results should be stored')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='The seeds to evaluate')
    parser.add_argument('--retries', type=int, default=1,
                        help='Maximum number of retries per route')
    parser.add_argument('--lb_script', '--lb-script', dest='lb_script', type=str,
                        default='leaderboard/leaderboard/leaderboard_evaluator.py',
                        help='Path to leaderboard evaluator script')
    parser.add_argument('--agent_file', '--agent-file', dest='agent_file', type=str, default=None,
                        help='Path to agent entry file')
    parser.add_argument('--agent_config', '--agent-config', dest='agent_config', type=str, default=None,
                        help='Path to agent config/checkpoint')
    parser.add_argument('--no_rgb', action='store_true',
                        help='Disable RGB rendering and run with nullrhi')
    parser.add_argument('--no_viz', '--no-viz', dest='no_viz', action='store_true',
                        help='Disable VIZ_PATH output directory handling')
    parser.add_argument('--env_var_cmd', '--env-var-cmd', dest='env_var_cmd', type=str, default='',
                        help='Command to set environment variables for the job')
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be submitted without executing sbatch")
    parser.add_argument("--verbose", action="store_true",
                        help="With --dry-run also list the skipped (done/running) jobs")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Max passes over all routes (each pass gives every unfinished route "
                             "--retries fresh tries). Stops earlier once everything is done.")
    parser.add_argument("--job_prefix", "--job-prefix", dest="job_prefix", type=str, default=FAIL2DRIVE_JOB_PREFIX,
                        help="Prefix for slurm job names (default: Fail2Drive_)")
    # [MODIFIED] Add argument to override max parallel jobs per evaluator process
    parser.add_argument("--max_parallel_jobs", "--max-parallel-jobs", dest="max_parallel_jobs",
                        type=int, default=None,
                        help="Maximum number of concurrent Slurm jobs for this evaluator (overrides eval_num_jobs.txt)")

    parser.add_argument("--routes-per-job", "--routes_per_job", type=int, default=DEFAULT_ROUTES_PER_JOB,
                        help="Routes run sequentially per Slurm job (each at most 4h)")
    args = parser.parse_args()
    if args.routes_per_job < 1:
        parser.error("--routes-per-job must be >= 1")
    if not args.dry_run and (args.agent_file is None or args.agent_config is None):
        parser.error("--agent_file and --agent_config are required unless --dry-run is set")

    routes = sorted([x for x in os.listdir(args.routes) if x[-4:] == ".xml"])

    out_root = args.out_root
    RESULTS_ROOT = out_root
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)

    seeds = args.seeds
    retries = args.retries

    cfg = {
        "eval_mode": args.eval_mode,
        "lb_script": args.lb_script,
        "agent_file": args.agent_file,
        "agent_config": args.agent_config,
        # NOTE: If RGB is disabled here and the agent uses a camera, CARLA will crash
        "rgb": not args.no_rgb,
        "viz": not args.no_viz,
        "env_var_cmd": args.env_var_cmd,
    }
    FAIL2DRIVE_JOB_PREFIX = args.job_prefix
    # [MODIFIED] Set the global CLI override for max parallel jobs
    CLI_MAX_PARALLEL_JOBS = args.max_parallel_jobs

    # Set of route IDs currently executing to exclude from the queue
    skip_route_ids = {
        # "0004", "0020", "0015", "0021", "0022", "0081"
    }

    skip_existing_logs = True
    skip_existing_logs = False  # Set to True to skip jobs with existing logs

    # Filling the job queue
    job_queue = []
    for seed in seeds:

        base_dir = os.path.join(out_root, str(seed))
        if not args.dry_run:
            for sub_dir in ("run", "res", "out", "err"):
                os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

        for route in routes:
            route_id = route.split("_")[-1][:-4]

            # Only run a subset of routes for testing purposes
            # if route_id not in ("0000", "0001"):
            #     continue

            # Skip jobs that are currently running
            if route_id in skip_route_ids:
                continue

            # NOTE: Fail2Drive specific, pairs are route_id%1000
            route_seed = int(route_id) % 1000 + (10000 * seed)
            route = os.path.join(args.routes, route)

            viz_path = ""
            if cfg["viz"]:
                viz_path = os.path.join(base_dir, "viz", route_id)
                if os.environ.get("F2D_IN_ALLOCATION") == "1":
                    # Keep prior users' visualizations intact; new allocations
                    # write into a fresh directory with the current user's owner.
                    viz_path = os.path.join(base_dir, "viz",
                                            f"allocation_{os.environ['SLURM_JOB_ID']}", route_id)

            result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
            log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
            err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")

            # if skip_existing_logs and os.path.exists(log_file):
            #     continue

            job_file = os.path.join(base_dir, "run", f"eval_{route_id}.sh")

            job = {
                "route": route,
                "route_id": route_id,
                "seed": route_seed,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "viz_path": viz_path,
                "job_file": job_file,
                "env_var_cmd": args.env_var_cmd,
                "tries": retries,
            }

            job_queue.append(job)

    if args.dry_run:
        dry_run_report(job_queue, verbose=args.verbose)
        sys.exit(0)

    carla_world_ports = list(range(10000, 20000, 50))
    carla_streaming_ports = list(range(20000, 30000, 50))
    carla_tm_ports = list(range(30000, 40000, 50))
    model_server_ports = list(range(40000, 50000, 50))
    random.shuffle(carla_world_ports)
    random.shuffle(carla_streaming_ports)
    random.shuffle(carla_tm_ports)
    random.shuffle(model_server_ports)
    port_idx = 0

    # Submitting the jobs to slurm
    # One "round" = run the queue until it is empty. Then re-check ALL routes again:
    # finished ones drop out, ones lost to a CARLA segfault / connection failure come back
    # with fresh tries. Stops when every route is done (or after --max_rounds).
    all_jobs = job_queue
    round_idx = 0
    while True:
        job_queue = filter_completed([dict(job) for job in all_jobs], get_running_jobs())
        if not job_queue:
            print("All routes finished.")
            break
        if round_idx >= args.max_rounds:
            print(f"[warn] Giving up after {round_idx} rounds, still unfinished:")
            for job in job_queue:
                print(f"  {job_name(job)} route={job['route_id']}")
            sys.exit(1)
        round_idx += 1
        print(f"[round {round_idx}/{args.max_rounds}] {len(job_queue)} unfinished jobs")

        jobs = len(job_queue)
        progress = tqdm(total=jobs)
        while job_queue:
            running_jobs = get_running_jobs()  # the ONLY squeue call per iteration
            job_queue = filter_completed(job_queue, running_jobs)

            progress.update(jobs - len(job_queue) - progress.n)

            max_num_parallel_jobs = get_max_num_parallel_jobs()

            ready = [job for job in job_queue
                     if job["tries"] > 0 and job_name(job) not in running_jobs]
            for seed in seeds:
                seed_jobs = [job for job in ready if job["seed"] // 10000 == seed]
                for offset in range(0, len(seed_jobs), args.routes_per_job):
                    if len(set(running_jobs.values())) >= max_num_parallel_jobs:
                        break
                    batch = seed_jobs[offset:offset + args.routes_per_job]
                    print(f"[submit] Preparing script: seed={seed}, routes={len(batch)}", flush=True)
                    job_file = bash_file(
                        batch, cfg,
                        carla_world_ports[port_idx % len(carla_world_ports)],
                        carla_streaming_ports[port_idx % len(carla_streaming_ports)],
                        carla_tm_ports[port_idx % len(carla_tm_ports)],
                        model_server_ports[port_idx % len(model_server_ports)],
                    )
                    port_idx += 1
                    print(f"[submit] Script ready: {job_file}", flush=True)
                    try:
                        if os.environ.get("F2D_IN_ALLOCATION") == "1":
                            # Reuse the enclosing Slurm step. Running synchronously
                            # preserves result checks/retries without nested sbatch.
                            if not os.environ.get("SLURM_JOB_ID"):
                                raise RuntimeError("F2D_IN_ALLOCATION requires a Slurm allocation")
                            base_dir = os.path.dirname(os.path.dirname(job_file))
                            tag = f"batch_{batch[0]['route_id']}_{batch[-1]['route_id']}"
                            if os.environ.get("F2D_BATCH_NAMESPACE"):
                                tag += "_" + os.environ["F2D_BATCH_NAMESPACE"]
                            with open(os.path.join(base_dir, "out", tag + "_out.log"), "w") as batch_out, \
                                 open(os.path.join(base_dir, "err", tag + "_err.log"), "w") as batch_err:
                                result = subprocess.run(["bash", job_file], stdout=batch_out, stderr=batch_err)
                            job_id = os.environ["SLURM_JOB_ID"]
                            print(f"allocation batch finished: job={job_id} exit={result.returncode}", flush=True)
                        else:
                            print(f"[submit] Calling sbatch: {job_file}", flush=True)
                            job_id = subprocess.check_output(
                                ["sbatch", "--parsable", job_file], text=True
                            ).strip().split(";", 1)[0]
                    except (subprocess.SubprocessError, OSError) as exc:
                        print(f"[warn] Failed to submit {job_file}: {exc}")
                        break
                    for job in batch:
                        running_jobs[job_name(job)] = job_id
                        job["tries"] -= 1
                    print(f"submit {job_file} (JobID={job_id}, routes={[job['route_id'] for job in batch]})", flush=True)

            time.sleep(15)
        progress.close()

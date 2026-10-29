import os
import subprocess
import time
import json
import argparse
from tqdm import tqdm
import random
import shutil
import re
import sys
from collections import Counter

FAIL2DRIVE_JOB_PREFIX = "Fail2Drive_"
MAX_JOBS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_num_jobs.txt"
)
RETRYABLE_STATUSES = {
    "Failed - Agent couldn't be set up",
    # "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def job_name(job):
    return f"{FAIL2DRIVE_JOB_PREFIX}{job['seed']}_{job['route_id']}"


# You will likely have to customize this function a bit to work with your cluster partition names etc.
# NOTE: Make sure to run this python script with the correct conda env which automatically sets the env vars.
# If you have issues with the variables, such as WORK_DIR not being set, you can export them in the following bash file.
def bash_file(
    job, cfg, carla_world_port_start, carla_streaming_port_start, carla_tm_port_start
):
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    env_var_cmd = cfg.get("env_var_cmd", "")
    render_args = (
        '-RenderOffScreen -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-$EVALUATOR_GPU}"'
        if cfg["rgb"]
        else "-nullrhi"
    )

    with open(job_file, "w", encoding="utf-8") as rsh:
        rsh.write(f"""#!/bin/bash
#SBATCH --job-name={job_name(job)}
#SBATCH --partition=devq
#SBATCH --qos=normal
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#### SBATCH -w cp[14-15,17,20]
#### SBATCH -C "[nvidia_geforce_rtx_3090|nvidia_geforce_rtx_3090_ti|nvidia_geforce_rtx_4080_super|nvidia_geforce_rtx_4090]"
# NOTE: Partition and gres likely need to be updated for your cluster

# NOTE: Make sure that the time limit is enough for your model to fail the route!
# Example: Timeout may take 3-4 minutes in game time, if cluster time limit enables only 2:30 game time, the route will be resubmitted until it succeeds!

echo 'HPC Node Name:' $(hostname)
echo JOB ID $SLURM_JOB_ID

# NOTE: You can use this in your agent to store visualization outputs
export VIZ_PATH={viz_path}
export SAVE_PATH={viz_path}
export VISUAL_OUTPUT_PATH={viz_path}
export BENCHMARK_ROUTE_ID="{route_id}"

{env_var_cmd}
printenv | sort
nvidia-smi -L
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

FREE_WORLD_PORT=`comm -23 \
    <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) \
    <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) \
    | shuf | head -n 1`
echo 'World Port:' $FREE_WORLD_PORT

FREE_STREAMING_PORT=`comm -23 \
    <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} | sort) \
    <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) \
    | shuf | head -n 1`
echo 'Streaming Port:' $FREE_STREAMING_PORT

export TM_PORT=`comm -23 \
    <(seq {carla_tm_port_start} {carla_tm_port_start + 49} | sort) \
    <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) \
    | shuf | head -n 1`
echo 'TM Port:' $TM_PORT

# NOTE: Changing -graphicsadapter=0 can be useful on multi-gpu systems

CARLA_LOG_DIR="${{TEST_OUTPUT:-{viz_path}}}/carla_log"
mkdir -p "$CARLA_LOG_DIR"
CARLA_LOG="$CARLA_LOG_DIR/carla_log_${{SLURM_JOB_ID:-manual}}.log"

"$CARLA_ROOT/CarlaUE4.sh" \\
    -carla-rpc-port="$FREE_WORLD_PORT" \\
    -nosound \\
    {render_args} \\
    -carla-primary-port=0 \\
    -carla-streaming-port="$FREE_STREAMING_PORT" \\
    >"$CARLA_LOG" 2>&1 &

CARLA_PID=$!
echo "Started CARLA pid=$CARLA_PID log=$CARLA_LOG"

sleep 60  # Wait for CARLA to finish starting

# NOTE: --track=MAP may have to be changed according to agent track
python -u {cfg["lb_script"]} \
--routes={route} \
--repetitions=1 \
--track=SENSORS \
--checkpoint={result_file} \
--timeout=300 \
--agent={cfg["agent_file"]} \
--agent-config={cfg["agent_config"]} \
--port=${{FREE_WORLD_PORT}} \
--traffic-manager-port=${{TM_PORT}} \
--traffic-manager-seed={seed}
""")


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
    for line in out.splitlines():
        job_id, _, name = line.partition("|")
        if name.strip().startswith(FAIL2DRIVE_JOB_PREFIX):
            running[name.strip()] = job_id.strip()
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
        if record.get("status") in RETRYABLE_STATUSES:
            return f"status {record.get('status')!r} is in RETRYABLE_STATUSES"
    return None


# def _is_retryable_result(evaluation_data):
#     return _retry_reason(evaluation_data) is not None


def get_max_num_parallel_jobs():
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


def kill_dead_jobs(jobs, running_jobs):
    for job in jobs:
        job_id = running_jobs.get(job_name(job))
        if job_id is None:
            continue

        log = job["log_file"]
        if not os.path.exists(log):
            continue

        with open(log) as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue

        if (
            any(["Watchdog exception" in line for line in lines])
            or "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n"
            in lines
            or "[91mStopping the route, the agent has crashed:\n" in lines
            or "[91mError during the simulation:\n" in lines
        ):

            subprocess.Popen(f"scancel {job_id}", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str, default='fail2drive_split',
                        help='Path to folder containing the split route files')
    parser.add_argument('--out_root', type=str, default='results/fail2drive',
                        help='Path where results should be stored')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='The seeds to evaluate')
    parser.add_argument('--retries', type=int, default=2,
                        help='Maximum number of retries per route')
    parser.add_argument('--lb_script', type=str,
                        default='leaderboard/leaderboard/leaderboard_evaluator.py',
                        help='Path to leaderboard evaluator script')
    parser.add_argument('--agent_file', type=str, default=None,
                        help='Path to agent entry file')
    parser.add_argument('--agent_config', type=str, default=None,
                        help='Path to agent config/checkpoint')
    parser.add_argument('--no_rgb', action='store_true',
                        help='Disable RGB rendering and run with nullrhi')
    parser.add_argument('--no_viz', action='store_true',
                        help='Disable VIZ_PATH output directory handling')
    parser.add_argument('--env_var_cmd', type=str, default='',
                        help='Command to set environment variables for the job')
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be submitted without executing sbatch")
    parser.add_argument("--verbose", action="store_true",
                        help="With --dry-run also list the skipped (done/running) jobs")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Max passes over all routes (each pass gives every unfinished route "
                             "--retries fresh tries). Stops earlier once everything is done.")
    parser.add_argument("--job_prefix", type=str, default=FAIL2DRIVE_JOB_PREFIX,
                        help="Prefix for slurm job names (default: Fail2Drive_)")

    args = parser.parse_args()
    if not args.dry_run and (args.agent_file is None or args.agent_config is None):
        parser.error("--agent_file and --agent_config are required unless --dry-run is set")

    routes = sorted([x for x in os.listdir(args.routes) if x[-4:] == ".xml"])

    out_root = args.out_root
    if not args.dry_run:
        os.makedirs(out_root, exist_ok=True)

    seeds = args.seeds
    retries = args.retries

    cfg = {
        "lb_script": args.lb_script,
        "agent_file": args.agent_file,
        "agent_config": args.agent_config,
        # NOTE: If RGB is disabled here and the agent uses a camera, CARLA will crash
        "rgb": not args.no_rgb,
        "viz": not args.no_viz,
    }
    FAIL2DRIVE_JOB_PREFIX = args.job_prefix

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

            # Skip jobs that are currently running
            if route_id in skip_route_ids:
                continue

            # NOTE: Fail2Drive specific, pairs are route_id%1000
            route_seed = int(route_id) % 1000 + (10000 * seed)
            route = os.path.join(args.routes, route)

            viz_path = ""
            if cfg["viz"]:
                viz_path = os.path.join(base_dir, "viz", route_id)

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
    random.shuffle(carla_world_ports)
    random.shuffle(carla_streaming_ports)
    random.shuffle(carla_tm_ports)
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
            break
        round_idx += 1
        print(f"[round {round_idx}/{args.max_rounds}] {len(job_queue)} unfinished jobs")

        jobs = len(job_queue)
        progress = tqdm(total=jobs)
        while job_queue:
            running_jobs = get_running_jobs()  # the ONLY squeue call per iteration
            kill_dead_jobs(job_queue, running_jobs)
            job_queue = filter_completed(job_queue, running_jobs)

            progress.update(jobs - len(job_queue) - progress.n)

            max_num_parallel_jobs = get_max_num_parallel_jobs()

            # Fill all free slots in one pass (the old code submitted one job per squeue)
            for job in job_queue:
                if len(running_jobs) >= max_num_parallel_jobs:
                    break

                # Out of tries, or already in squeue (even if started by an earlier launch)
                if job["tries"] <= 0 or job_name(job) in running_jobs:
                    continue

                # Need to submit this job
                carla_world_port_start = carla_world_ports[port_idx]
                carla_streaming_port_start = carla_streaming_ports[port_idx]
                carla_tm_port_start = carla_tm_ports[port_idx]
                port_idx += 1
                port_idx %= 200

                # Make bash file:
                bash_file(
                    job,
                    cfg,
                    carla_world_port_start,
                    carla_streaming_port_start,
                    carla_tm_port_start,
                )

                # submit
                if cfg["viz"]:
                    if os.path.exists(job["viz_path"]):
                        shutil.rmtree(job["viz_path"])
                    os.makedirs(job["viz_path"], exist_ok=True)

                for file in [job["result_file"], job["err_file"], job["log_file"]]:
                    if os.path.exists(file):
                        os.remove(file)
                try:
                    job_id = (
                        subprocess.check_output(f'sbatch {job["job_file"]}', shell=True)
                        .decode("utf-8")
                        .strip()
                        .rsplit(" ", maxsplit=1)[-1]
                    )
                except (subprocess.SubprocessError, OSError) as exc:
                    print(
                        f"[warn] Failed to submit job '{job['job_file']}': {exc}")
                    continue

                running_jobs[job_name(job)] = job_id
                job["tries"] -= 1

                print(f'submit {job["job_file"]} {job_id}')
                print(f'Job queue size: {len(job_queue)}')

            time.sleep(15)
        progress.close()

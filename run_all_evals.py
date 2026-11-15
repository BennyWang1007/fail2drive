#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess
import shlex
from pathlib import Path

import json

# ==============================================================================
# 1. Global Settings & Model Defaults
# ==============================================================================
# Maximum total Slurm jobs allowed per user
TOTAL_MAX_SLURM_JOBS = 90

# Argument flag passed to slurm_evaluate.py to limit concurrent jobs per task
MAX_JOBS_ARG = "--max-parallel-jobs"

# Global dry-run switch (set to True to test without submitting sbatch jobs)
DRY_RUN = False
# DRY_RUN = True

# Global control for MAX_CARLA_FRAME:
# 0 disables the frame limit; export it explicitly to override inherited settings.
GLOBAL_MAX_CARLA_FRAME = 0

# Pre-check completion settings using classify_result
EXPECTED_ROUTES_PER_SEED = 200  # Set to exact number of routes in fail2drive_split
# ACCEPTABLE_CATEGORIES = {"Perfect", "Slow", "Not ideal"}  # "Fail" is considered incomplete

AUTO_CLEAN_FAILED_ROUTES = False  # If True, automatically move "Fail" routes to removed/ before launching

MODEL_DEFAULTS = {
    "hipad": {
        "conda_env": "hipad",
        "agent_file": "$F2D/team_code/hipad_f2d_agent.py",
        "agent_config": "$HIP/projects/configs/hipad_b2d_stage2.py+$HIP/ckpts/hipad_stage2.pth+hipad_f2d",
        "lb_script": "leaderboard/leaderboard/leaderboard_evaluator_local.py",
    },
    "tfv6": {
        "conda_env": "fail2drive",
        "agent_file": "$LEAD_ROOT/lead/inference/sensor_agent.py",
        "agent_config": "$LEAD_ROOT/outputs/checkpoints/tfv6_resnet34",
        "lb_script": "leaderboard/leaderboard/leaderboard_evaluator_local.py",
    }
}

# Mapping from profile name to STEERING_FEATURE_NAME_SHORT in f2d_env_config.sh
FEATURE_SHORT_MAP = {
    "tfv6": "fused",
    "tfv6_dec4": "decoder4",
    "tfv6_dec5": "decoder5",
    "tfv6_dec6": "decoder6",
}


# ==============================================================================
# 2. Helper Function to Simply & Uniformly Generate Tasks
# ==============================================================================
def make_task(
    model: str,
    mods: list[str],
    out_root: str,
    job_prefix: str,
    enabled: bool = True,
    eval_mode: str | None = None,
    extra_exports: list[str] | None = None,
    use_lb_script: bool | None = None,
    seed: int = 3,
) -> dict:
    """
    Uniformly construct a task configuration dictionary.
    - Ignores LIVE_VISU completely.
    - Keeps DEBUG_CHALLENGE at default (1 in env_common).
    - Applies GLOBAL_MAX_CARLA_FRAME uniformly across all tasks.
    - Sets VLM_MIN_SPEED_MPS=3.5 uniformly for all VLM tasks.
    """
    exports = []
    exports.append(f"export MAX_CARLA_FRAME={GLOBAL_MAX_CARLA_FRAME or 0}")
    if eval_mode == "vlm" or "vlm" in mods:
        exports.append("export VLM_MIN_SPEED_MPS=${VLM_MIN_SPEED_MPS:-3.5}")
    if extra_exports:
        exports.extend(extra_exports)

    # Default rule: TFv6 with special eval_mode (depth_ttc / vlm) doesn't pass --lb-script
    if use_lb_script is None:
        use_lb_script = not (model == "tfv6" and eval_mode in ("depth_ttc", "vlm"))

    # Infer static out_root for pre-checking results before bash evaluation
    resolved_out_root = out_root
    for mod in mods:
        if mod in FEATURE_SHORT_MAP:
            resolved_out_root = resolved_out_root.replace(
                "${STEERING_FEATURE_NAME_SHORT}", FEATURE_SHORT_MAP[mod]
            )

    cfg = {
        "enabled": enabled,
        "model": model,
        "env_mods": " ".join(["common"] + mods),
        "extra_exports": " && ".join(exports),
        "out_root": out_root,
        "resolved_out_root": resolved_out_root,
        "job_prefix": job_prefix,
        "use_lb_script": use_lb_script,
        "seed": seed,
    }
    if eval_mode:
        cfg["eval_mode"] = eval_mode
    return cfg


# ==============================================================================
# 3. Dictionary of Evaluation Tasks to Run
# ==============================================================================
TASKS = {
    # ==========================================================================
    # [A] ORACLE TASKS (8 tasks total; Oracle + Override set to enabled=False)
    # ==========================================================================
    # 1. TFv6 Override, brake-only (OFF - tuning left/right turns)
    "tfv6_oracle_brake_only_override": make_task(
        model="tfv6",
        mods=["tfv6", "oracle_brake_only", "override", "debug"],
        out_root="results_shome/tfv6_oracle_brake_only_override_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Oracle_Brake_Only_Override_",
        enabled=True,
    ),
    # 2. TFv6 Override, full action (OFF - tuning left/right turns)
    "tfv6_oracle_override": make_task(
        model="tfv6",
        mods=["tfv6", "oracle", "override", "debug"],
        out_root="results_shome/tfv6_oracle_override_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Fused_Oracle_Override_",
        enabled=False,
    ),
    # 3. TFv6 Steering, brake-only
    "tfv6_oracle_brake_only": make_task(
        model="tfv6",
        mods=["tfv6", "oracle_brake_only", "debug"],
        out_root="results_shome/tfv6_oracle_brake_only_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Oracle_Brake_Only_",
        enabled=True,
    ),
    # 4. TFv6 Steering, full action (also serves as TransfuserV6 + Fused + Oracle)
    "tfv6_oracle": make_task(
        model="tfv6",
        mods=["tfv6", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Fused_Oracle_",
        enabled=True,
    ),

    # 5. HipAD Override, brake-only (OFF - tuning left/right turns)
    "hipad_oracle_brake_only_override": make_task(
        model="hipad",
        mods=["hipad", "oracle_brake_only", "override", "debug"],
        out_root="results_shome/hipad_oracle_brake_only_override",
        job_prefix="F2D_Hipad_Oracle_Brake_Only_Override_",
        enabled=True,
    ),
    # 6. HipAD Override, full action (OFF - tuning left/right turns)
    "hipad_oracle_override": make_task(
        model="hipad",
        mods=["hipad", "oracle", "override", "debug"],
        out_root="results_shome/hipad_oracle_override",
        job_prefix="F2D_Hipad_Oracle_Override_",
        enabled=False,
    ),
    # 7. HipAD Steering, brake-only
    "hipad_oracle_brake_only": make_task(
        model="hipad",
        mods=["hipad", "oracle_brake_only", "debug"],
        out_root="results_shome/hipad_oracle_brake_only",
        job_prefix="F2D_Hipad_Oracle_Brake_Only_",
        enabled=True,
    ),
    # 8. HipAD Steering, full action (also serves as HipAD + PDM-Lite(Oracle) / _new)
    "hipad_oracle": make_task(
        model="hipad",
        mods=["hipad", "oracle", "debug"],
        out_root="results_shome/hipad_oracle",
        job_prefix="F2D_Hipad_Oracle_",
        enabled=True,
    ),

    # ==========================================================================
    # [B] OTHER HIPAD TASKS
    # ==========================================================================
    # HipAD (reproduce / no_policy)
    "hipad_no_policy": make_task(
        model="hipad",
        mods=["hipad", "no_policy", "debug"],
        out_root="results_shome/hipad_no_policy",
        job_prefix="F2D_Hipad_No_Policy_",
        enabled=True,
    ),
    # HipAD + Alpamayo-VQA (vlm)
    "hipad_vlm": make_task(
        model="hipad",
        mods=["hipad", "vlm", "debug"],
        eval_mode="vlm",
        out_root="results_shome/hipad_vlm",
        job_prefix="F2D_Hipad_VLM_",
        enabled=True,
    ),
    # HipAD + Occupancy (depth_ttc)
    "hipad_depth_ttc": make_task(
        model="hipad",
        mods=["hipad", "depth_ttc", "debug"],
        eval_mode="depth_ttc",
        out_root="results_shome/hipad_depth_ttc",
        job_prefix="F2D_Hipad_Depth_TTC_",
        enabled=True,
    ),

    # ==========================================================================
    # [C] OTHER TRANSFUSERV6 TASKS
    # ==========================================================================
    # TransfuserV6 (Reproduced / no_policy)
    "tfv6_no_policy": make_task(
        model="tfv6",
        mods=["tfv6", "no_policy", "debug"],
        out_root="results_shome/tfv6_${STEERING_FEATURE_NAME_SHORT}_no_policy",
        job_prefix="F2D_TFV6_No_Policy_",
        enabled=True,
        seed=2,
    ),
    # TransfuserV6 + Fused + Occupancy (depth_ttc)
    "tfv6_depth_ttc": make_task(
        model="tfv6",
        mods=["tfv6", "depth_ttc", "debug"],
        eval_mode="depth_ttc",
        out_root="results_shome/tfv6_depth_ttc_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFV6_Depth_TTC_",
        enabled=True,
    ),
    # TransfuserV6 + Fused + Alpamayo-VQA (vlm)
    "tfv6_vlm": make_task(
        model="tfv6",
        mods=["tfv6", "vlm", "debug"],
        eval_mode="vlm",
        out_root="results_shome/tfv6_vlm_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFV6_VLM_",
        enabled=True,
    ),
    # TransfuserV6 + Oracle + decoder4
    "tfv6_oracle_dec4": make_task(
        model="tfv6",
        mods=["tfv6_dec4", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec4_Oracle_",
        enabled=True,
    ),
    # TransfuserV6 + Oracle + decoder5
    "tfv6_oracle_dec5": make_task(
        model="tfv6",
        mods=["tfv6_dec5", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec5_Oracle_",
        enabled=True,
    ),
    # TransfuserV6 + Oracle + decoder6
    "tfv6_oracle_dec6": make_task(
        model="tfv6",
        mods=["tfv6_dec6", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec6_Oracle_",
        enabled=True,
    ),
}


# ==============================================================================
# 4. Pre-Check Completion via classify_result
# ==============================================================================

def has_valid_global_record(res_path: Path) -> bool:
    """Return True if the JSON exists and its _checkpoint.global_record is non-empty."""
    try:
        with res_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        global_record = data.get("_checkpoint", {}).get("global_record", {})
        return bool(global_record)
    except Exception:
        return False


def check_task_completion(task_name: str, cfg: dict) -> bool:
    """
    Check if a task is already completed across all seeds by verifying
    that global_record != {} in each *_res.json file.
    """
    out_root = Path(cfg.get("resolved_out_root", cfg["out_root"]))
    # num_seeds = cfg.get("seeds", 3)
    seed = cfg.get("seed", 3)

    if not out_root.is_dir():
        return False

    # seed_dirs = sorted([
    #     d for d in out_root.iterdir()
    #     if d.is_dir() and d.name != "removed" and ((d / "res").is_dir() or list(d.glob("*_res.json")))
    # ])

    seed_dir = out_root / str(seed)

    if not seed_dir.is_dir():
        return False

    # if len(seed_dirs) < num_seeds:
    #     return False

    all_seeds_complete = True
    # for s_dir in seed_dirs[:num_seeds]:
    # for s_dir in seed_dirs:
    for s_dir in [seed_dir]:
        res_dir = s_dir / "res" if (s_dir / "res").is_dir() else s_dir
        res_files = sorted(res_dir.glob("*_res.json"))

        valid_count = sum(1 for res_path in res_files if has_valid_global_record(res_path))
        # print(f"[Pre-Check] Task '{task_name}' ({s_dir.name}): {valid_count}/{EXPECTED_ROUTES_PER_SEED} valid routes finished.")

        if valid_count < EXPECTED_ROUTES_PER_SEED:
            print(
                f"[Pre-Check] Task '{task_name}' ({s_dir.name}): "
                f"{valid_count}/{EXPECTED_ROUTES_PER_SEED} valid routes finished."
            )
            all_seeds_complete = False

    return all_seeds_complete


# ==============================================================================
# 5. Process Spawning & Dynamic Quota Management
# ==============================================================================
def validate_unique_prefixes(tasks: dict):
    """Ensure every enabled task has a strictly unique job_prefix."""
    seen = {}
    for name, cfg in tasks.items():
        prefix = cfg["job_prefix"]
        if prefix in seen:
            raise ValueError(
                f"Duplicate job_prefix '{prefix}' found in tasks '{seen[prefix]}' and '{name}'."
            )
        seen[prefix] = name


def build_bash_command(task_name: str, cfg: dict, max_jobs: int) -> str:
    """Build the bash execution string for a single evaluation task."""
    m_def = MODEL_DEFAULTS[cfg["model"]]
    conda_env = cfg.get("conda_env", m_def["conda_env"])
    agent_file = cfg.get("agent_file", m_def["agent_file"])
    agent_config = cfg.get("agent_config", m_def["agent_config"])
    lb_script = cfg.get("lb_script", m_def["lb_script"])
    use_lb_script = cfg.get("use_lb_script", True)

    env_var_cmd = f"source $PWD/f2d_env_config.sh && load_f2d_env {cfg['env_mods']}"
    if cfg.get("extra_exports"):
        env_var_cmd += f" && {cfg['extra_exports']}"

    # The submitter only needs stdlib/tqdm, not the model's Conda environment.
    # Conda activation on the controller can block on NFS before any sbatch.
    # Activate explicitly: an inherited CONDA_PREFIX does not guarantee that
    # the worker's PATH contains that environment's Python executable.
    worker_env_var_cmd = (
        'source /mnt/bapve/thome/wangyuhao/miniconda3/etc/profile.d/conda.sh '
        f'&& conda activate {shlex.quote(conda_env)} && {env_var_cmd}'
    )

    py_args = [
        f"{shlex.quote(sys.executable)} -u slurm_evaluate.py",
        f"--routes {cfg.get('routes', 'fail2drive_split')}",
        f"--seeds {cfg.get('seed', 3)}",
        f"--retries {cfg.get('retries', 1)}",
        f'--out-root "{cfg["out_root"]}"',
        f'--agent-file "{agent_file}"',
        f'--agent-config "{agent_config}"',
        '--env-var-cmd "$worker_env_var_cmd"',
        f'--job-prefix "{cfg["job_prefix"]}"',
        f"{MAX_JOBS_ARG} {max_jobs}",
        "--verbose",
        f"--routes-per-job {cfg.get('routes_per_job', 50)}",
    ]
    if cfg.get("eval_mode", None):
        py_args.insert(1, f"--eval-mode {cfg['eval_mode']}")
    if use_lb_script and lb_script:
        py_args.append(f'--lb-script "{lb_script}"')
    if DRY_RUN or cfg.get("dry_run", False):
        py_args.append("--dry-run")

    py_cmd_str = """ \\
  """.join(py_args)

    bash_script = f"""
set -e
set -u

echo "[Submitter] Using {sys.executable}; model environment activation is deferred to workers"

export ROOT="${{ROOT:-$HOME}}"
export FAIL2DRIVE_ROOT="${{FAIL2DRIVE_ROOT:-$PWD}}"
export LEAD_ROOT="${{LEAD_ROOT:-$ROOT/lead-cvpr2026}}"
export HIP="${{HIP:-$FAIL2DRIVE_ROOT/../HiP-AD}}"
export PYTHONPATH="$PWD/team_code:$LEAD_ROOT:${{PYTHONPATH:-}}"

env_var_cmd="{env_var_cmd}"
worker_env_var_cmd={shlex.quote(worker_env_var_cmd)}
eval "$env_var_cmd"

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Task: {task_name}"
echo "Allocated Max Parallel Jobs: {max_jobs}"
echo "============================================================"

{py_cmd_str}
"""
    return bash_script


def main():
    parser = argparse.ArgumentParser(description="Run selected evaluation tasks sequentially.")
    parser.add_argument("--run-name", nargs="+")
    parser.add_argument("--seed", nargs="+")
    parser.add_argument("--routes-per-job", type=int, default=50)
    parser.add_argument("--total-max-jobs", type=int, default=TOTAL_MAX_SLURM_JOBS)
    parser.add_argument("--max-job-per-task", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    # Accepted for compatibility with older entry commands; GRES_MAPPING in
    # slurm_evaluate.py controls the allocation (VLM/Depth-TTC: 2 GPUs).
    parser.add_argument("--server-gpu-type", help=argparse.SUPPRESS)
    parser.add_argument("--route-gpu-type", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.routes_per_job, args.total_max_jobs, args.max_job_per_task) < 1:
        parser.error("route batch size and job limits must be positive")

    names = [name for value in args.run_name for name in value.split(",")] if args.run_name else None
    if names and any(name not in TASKS for name in names):
        parser.error(f"Unknown task; available tasks: {', '.join(TASKS)}")
    try:
        seeds = [int(seed) for value in args.seed for seed in value.split(",")] if args.seed else None
    except ValueError:
        parser.error("--seed must contain integer seeds")

    log_dir = Path("logs/multi_eval")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    quota = min(args.total_max_jobs, args.max_job_per_task)
    failed = []
    for name, task in TASKS.items():
        if (names is not None and name not in names) or (names is None and not task.get("enabled", True)):
            continue
        for seed in dict.fromkeys(seeds if seeds is not None else [task.get("seed", 3)]):
            cfg = dict(task, seed=seed, routes_per_job=args.routes_per_job)
            cfg["job_prefix"] = f"{task['job_prefix']}seed{seed}_"
            run_name = f"{name}_seed{seed}"
            if not args.dry_run and check_task_completion(run_name, cfg):
                print(f"[Scheduler] {run_name} already completed; skipping.", flush=True)
                continue
            command = build_bash_command(run_name, cfg, quota)
            if args.dry_run:
                print(command)
                continue
            print(f"[Scheduler] Running {run_name} (routes/job={args.routes_per_job}, max jobs={quota})", flush=True)
            with (log_dir / f"{run_name}.out").open("a", buffering=1) as out_log, \
                    (log_dir / f"{run_name}.err").open("a", buffering=1) as err_log:
                result = subprocess.run(["bash", "-c", command], stdout=out_log, stderr=err_log)
            if result.returncode:
                failed.append(run_name)
                print(f"[Scheduler] {run_name} failed (exit={result.returncode}).", flush=True)
            else:
                print(f"[Scheduler] {run_name} finished.", flush=True)
    if failed:
        sys.exit(f"Failed tasks: {', '.join(failed)}")


if __name__ == "__main__":
    main()

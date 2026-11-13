#!/usr/bin/env python3
import argparse
import os
import sys
import time
import signal
import shutil
import subprocess
from pathlib import Path
from f2d_server_slots import ServerSlots, DEFAULT_SERVER_INFO_FILES, normalize_gpu_type, gpu_gres

import json

# ==============================================================================
# 1. Global Settings & Model Defaults
# ==============================================================================
# CARLA jobs allowed in the queue (pending + running), excluding server/master
# jobs. Slurm enforces the actual execution and submission limits separately.
TOTAL_MAX_SLURM_JOBS = 29

# Argument flag passed to slurm_evaluate.py to limit concurrent jobs per task
MAX_JOBS_ARG = "--max-parallel-jobs"

# Default number of routes sequentially evaluated per single CARLA job
DEFAULT_ROUTES_PER_JOB = 1

# Default number of GPUs requested for the unified Alpamayo VLM & Depth-TTC server jobs
DEFAULT_VLM_SERVER_GPUS = "10"
DEFAULT_DEPTH_SERVER_GPUS = "4"

# Global dry-run switch (set to True to test without submitting sbatch jobs)
DRY_RUN = False

# Global control for MAX_CARLA_FRAME:
# Set to an integer (e.g., 2000) to enable for all tasks, or None / 0 to disable globally.
GLOBAL_MAX_CARLA_FRAME = None

# Pre-check completion settings
EXPECTED_ROUTES_PER_SEED = 200  # Exact number of routes in fail2drive_split

AUTO_CLEAN_FAILED_ROUTES = False

# Default cap on concurrent Slurm jobs per (task, seed) sub-task (override: --max-job-per-task)
DEFAULT_MAX_JOBS_PER_TASK = TOTAL_MAX_SLURM_JOBS

# Default seeds for every task (each seed becomes its own sub-task / progress bar)
DEFAULT_SEEDS = [3]

# Progress-bar scanning policy (keeps disk IO low)
MIN_SCAN_INTERVAL = 10
MAX_SCAN_INTERVAL = 60
LOOP_SLEEP = 3
BAR_WIDTH = 30

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
    seeds: list[int] | None = None,
) -> dict:
    exports = []
    if GLOBAL_MAX_CARLA_FRAME and GLOBAL_MAX_CARLA_FRAME > 0:
        exports.append(f"export MAX_CARLA_FRAME={GLOBAL_MAX_CARLA_FRAME}")
    if eval_mode == "vlm" or "vlm" in mods:
        exports.append("export VLM_MIN_SPEED_MPS=${VLM_MIN_SPEED_MPS:-3.5}")
    if extra_exports:
        exports.extend(extra_exports)

    if use_lb_script is None:
        use_lb_script = not (model == "tfv6" and eval_mode in ("depth_ttc", "vlm"))

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
        "seeds": list(seeds) if seeds else list(DEFAULT_SEEDS),
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
    "tfv6_oracle_brake_only_override": make_task(
        model="tfv6",
        mods=["tfv6", "oracle_brake_only", "override", "debug"],
        out_root="results_shome/tfv6_oracle_brake_only_override_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Oracle_Brake_Only_Override_",
        enabled=False,
        seeds=[3],
    ),
    "tfv6_oracle_override": make_task(
        model="tfv6",
        mods=["tfv6", "oracle", "override", "debug"],
        out_root="results_shome/tfv6_oracle_override_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Fused_Oracle_Override_",
        enabled=False,
        seeds=[3],
    ),
    "tfv6_oracle_brake_only": make_task(
        model="tfv6",
        mods=["tfv6", "oracle_brake_only", "debug"],
        out_root="results_shome/tfv6_oracle_brake_only_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Oracle_Brake_Only_",
        enabled=False,
        seeds=[3],
    ),
    "tfv6_oracle": make_task(
        model="tfv6",
        mods=["tfv6", "oracle", "new_oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Fused_Oracle_",
        enabled=False,
        seeds=[1, 2, 3],
    ),
    "hipad_oracle_brake_only_override": make_task(
        model="hipad",
        mods=["hipad", "oracle_brake_only", "override", "debug"],
        out_root="results_shome/hipad_oracle_brake_only_override",
        job_prefix="F2D_Hipad_Oracle_Brake_Only_Override_",
        enabled=False,
        seeds=[3],
    ),
    "hipad_oracle_override": make_task(
        model="hipad",
        mods=["hipad", "oracle", "override", "debug"],
        out_root="results_shome/hipad_oracle_override",
        job_prefix="F2D_Hipad_Oracle_Override_",
        enabled=False,
        seeds=[3],
    ),
    "hipad_oracle_brake_only": make_task(
        model="hipad",
        mods=["hipad", "oracle_brake_only", "debug"],
        out_root="results_shome/hipad_oracle_brake_only",
        job_prefix="F2D_Hipad_Oracle_Brake_Only_",
        enabled=False,
        seeds=[3],
    ),
    "hipad_oracle": make_task(
        model="hipad",
        mods=["hipad", "oracle", "debug"],
        out_root="results_shome/hipad_oracle",
        job_prefix="F2D_Hipad_Oracle_",
        enabled=False,
        seeds=[1, 2, 3],
    ),

    # ==========================================================================
    # [B] OTHER HIPAD TASKS
    # ==========================================================================
    "hipad_no_policy": make_task(
        model="hipad",
        mods=["hipad", "no_policy", "debug"],
        out_root="results_shome/hipad_no_policy",
        job_prefix="F2D_Hipad_No_Policy_",
        enabled=False,
        seeds=[1, 2, 3],
    ),
    "hipad_vlm": make_task(
        model="hipad",
        mods=["hipad", "vlm", "debug"],
        eval_mode="vlm",
        out_root="results_shome/hipad_vlm",
        job_prefix="F2D_Hipad_VLM_",
        enabled=False,
        seeds=[3],
    ),
    "hipad_depth_ttc": make_task(
        model="hipad",
        mods=["hipad", "depth_ttc", "debug"],
        eval_mode="depth_ttc",
        out_root="results_shome/hipad_depth_ttc",
        job_prefix="F2D_Hipad_Depth_TTC_",
        enabled=False,
        seeds=[2, 3],
    ),

    # ==========================================================================
    # [C] OTHER TRANSFUSERV6 TASKS
    # ==========================================================================
    "tfv6_no_policy": make_task(
        model="tfv6",
        mods=["tfv6", "no_policy", "debug"],
        out_root="results_shome/tfv6_${STEERING_FEATURE_NAME_SHORT}_no_policy",
        job_prefix="F2D_TFV6_No_Policy_",
        enabled=False,
        seeds=[1, 2, 3],
    ),
    "tfv6_depth_ttc": make_task(
        model="tfv6",
        mods=["tfv6", "depth_ttc", "debug"],
        eval_mode="depth_ttc",
        out_root="results_shome/tfv6_depth_ttc_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFV6_Depth_TTC_",
        enabled=False,
        seeds=[2, 3],
    ),
    "tfv6_vlm": make_task(
        model="tfv6",
        mods=["tfv6", "vlm", "debug"],
        eval_mode="vlm",
        out_root="results_shome/tfv6_vlm_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFV6_VLM_",
        enabled=True,
        seeds=[1, 2, 3],
    ),
    "tfv6_oracle_dec4": make_task(
        model="tfv6",
        mods=["tfv6_dec4", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec4_Oracle_",
        enabled=False,
        seeds=[3],
    ),
    "tfv6_oracle_dec5": make_task(
        model="tfv6",
        mods=["tfv6_dec5", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec5_Oracle_",
        enabled=False,
        seeds=[3],
    ),
    "tfv6_oracle_dec6": make_task(
        model="tfv6",
        mods=["tfv6_dec6", "oracle", "debug"],
        out_root="results_shome/tfv6_oracle_${STEERING_FEATURE_NAME_SHORT}",
        job_prefix="F2D_TFv6_Dec6_Oracle_",
        enabled=False,
        seeds=[3],
    ),
}


def expand_tasks(
    tasks: dict,
    run_names: list[str] | None = None,
    seeds_override: list[int] | None = None,
) -> dict:
    expanded = {}
    for name, cfg in tasks.items():
        if run_names is not None:
            if name not in run_names:
                continue
        elif not cfg.get("enabled", True):
            continue
        seeds = seeds_override or cfg.get("seeds", DEFAULT_SEEDS)
        for seed in seeds:
            sub = dict(cfg)
            sub["seed"] = seed
            sub["base_name"] = name
            sub["job_prefix"] = f"{cfg['job_prefix']}seed{seed}_"
            expanded[f"{name}_seed{seed}"] = sub
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Fail2Drive evaluations with shared cross-node Alpamayo VLM & Depth-TTC servers + multi-route CARLA batching."
    )
    parser.add_argument(
        "--run-name",
        nargs="+",
        default=None,
        help="Task name(s) in TASKS to run (comma or space separated), e.g. tfv6_vlm hipad_depth_ttc.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        default=None,
        help="Seed(s) to run (comma or space separated), e.g. 1,2,3.",
    )
    parser.add_argument(
        "--max-job-per-task",
        type=int,
        default=DEFAULT_MAX_JOBS_PER_TASK,
        help=f"Max concurrent Slurm jobs per (task, seed) sub-task (default: {DEFAULT_MAX_JOBS_PER_TASK}).",
    )
    parser.add_argument(
        "--total-max-jobs",
        type=int,
        default=None,
        help=f"Maximum queued CARLA jobs across tasks, pending + running (default: {TOTAL_MAX_SLURM_JOBS}; excludes server/master jobs).",
    )
    parser.add_argument(
        "--routes-per-job", "--routes-per-jobs",
        "-N",
        type=int,
        default=DEFAULT_ROUTES_PER_JOB,
        help=f"Number of routes sequentially evaluated per single CARLA Slurm job (default: {DEFAULT_ROUTES_PER_JOB}).",
    )
    parser.add_argument(
        "--vlm-gpus",
        default=DEFAULT_VLM_SERVER_GPUS,
        help="Legacy option; servers are now submitted per CARLA batch. Use f2d_ctl.sh start-server to prewarm a pool.",
    )
    parser.add_argument(
        "--depth-gpus",
        default=DEFAULT_DEPTH_SERVER_GPUS,
        help="Legacy option; servers are now submitted per CARLA batch.",
    )
    parser.add_argument(
        "--gpus-per-server-job",
        type=int,
        default=0,
        help="Legacy option; automatic servers now use one GPU per job.",
    )
    parser.add_argument(
        "--server-gpu-type", "--depth-gpu-type",
        dest="server_gpu_type",
        type=normalize_gpu_type,
        default="nvidia_geforce_rtx_4090",
        help="GPU type for VLM and Depth-TTC servers: 4090, any, or a Slurm type name (default: 4090).",
    )
    parser.add_argument(
        "--server-url",
        default="",
        help="Optional explicit URL for shared server (overrides active_server.json).",
    )
    parser.add_argument(
        "--vlm-server-info-file",
        "--server-info-file",
        dest="vlm_server_info_file",
        default=DEFAULT_SERVER_INFO_FILES["vlm"],
        help="Path to active_server.json for Alpamayo VLM server.",
    )
    parser.add_argument(
        "--depth-server-info-file",
        dest="depth_server_info_file",
        default=DEFAULT_SERVER_INFO_FILES["depth_ttc"],
        help="Path to active_server.json for Depth-TTC server.",
    )
    parser.add_argument(
        "--no-auto-server",
        action="store_true",
        help="Reserve existing server capacity only; do not submit backup servers.",
    )
    parser.add_argument(
        "--no-wait-server",
        action="store_true",
        help="Compatibility flag: server readiness is now always checked inside each CARLA job.",
    )
    parser.add_argument(
        "--use-local-server",
        action="store_true",
        help="Fallback to legacy 2-GPU-per-job local server mode instead of cross-node shared server.",
    )
    parser.add_argument(
        "--worker-gres",
        default=None,
        help="Legacy explicit CARLA GRES override; otherwise use --route-gpu-type.",
    )
    parser.add_argument("--route-gpu-type", type=normalize_gpu_type, default="any",
                        help="CARLA GPU type: any, 4090, or a Slurm type name (default: any).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print every (task, seed) run that would be submitted, then exit.",
    )
    args = parser.parse_args()

    def split(values):
        return [v.strip() for x in values for v in x.split(",") if v.strip()]

    if args.run_name is not None:
        args.run_name = list(dict.fromkeys(split(args.run_name)))
        unknown = [n for n in args.run_name if n not in TASKS]
        if unknown:
            parser.error(f"unknown --run-name {unknown}. Available: {sorted(TASKS)}")
    if args.seed is not None:
        try:
            args.seed = list(dict.fromkeys(int(v) for v in split(args.seed)))
        except ValueError:
            parser.error("--seed must be integers, e.g. --seed 1,2,3")
    if args.max_job_per_task < 1:
        parser.error("--max-job-per-task must be >= 1")
    if args.routes_per_job < 1:
        parser.error("--routes-per-job must be >= 1")
    return args


# ==============================================================================
# 4. Progress Tracking (low-IO, incremental)
# ==============================================================================
def has_valid_global_record(res_path: Path) -> bool:
    try:
        with res_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        global_record = data.get("_checkpoint", {}).get("global_record", {})
        return bool(global_record)
    except Exception:
        return False


class ProgressTracker:
    def __init__(self, cfg: dict):
        self.out_root = Path(cfg.get("resolved_out_root", cfg["out_root"]))
        self.seed = cfg.get("seed", 3)
        self.valid: set[str] = set()
        self._sig: dict[str, tuple] = {}
        self.count = 0

    def scan(self) -> int:
        seed_dir = self.out_root / str(self.seed)
        if not seed_dir.is_dir():
            self.valid.clear()
            self.count = 0
            return 0
        res_dir = seed_dir / "res" if (seed_dir / "res").is_dir() else seed_dir
        try:
            entries = [e for e in os.scandir(res_dir) if e.name.endswith("_res.json")]
        except OSError:
            return self.count

        names = {e.name for e in entries}
        self.valid &= names
        for e in entries:
            if e.name in self.valid:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if self._sig.get(e.name) == sig:
                continue
            self._sig[e.name] = sig
            if has_valid_global_record(Path(e.path)):
                self.valid.add(e.name)
        self.count = len(self.valid)
        return self.count

    @property
    def complete(self) -> bool:
        return self.count >= EXPECTED_ROUTES_PER_SEED


def check_task_completion(task_name: str, tracker: ProgressTracker) -> bool:
    n = tracker.scan()
    if n < EXPECTED_ROUTES_PER_SEED:
        print(f"[Pre-Check] Task '{task_name}': {n}/{EXPECTED_ROUTES_PER_SEED} valid routes finished.")
        return False
    return True


class ProgressBoard:
    def __init__(self, trackers: dict, log_dir: Path):
        self.trackers = trackers
        self.log_dir = log_dir
        self.active: list[str] = []
        self.is_tty = sys.stdout.isatty()
        self.lines_drawn = 0
        self._last_scan: dict[str, float] = {}
        self._log_sig: dict[str, tuple] = {}
        self._last_render_key = None

    def clear(self):
        if self.is_tty and self.lines_drawn:
            sys.stdout.write(f"\033[{self.lines_drawn}F\033[J")
            sys.stdout.flush()
        self.lines_drawn = 0

    def _format_lines(self) -> list[str]:
        if not self.active:
            return []
        width = max(len(n) for n in self.active)
        term_w = shutil.get_terminal_size((120, 20)).columns
        bar_w = max(10, min(BAR_WIDTH, term_w - width - 25))
        lines = []
        for name in self.active:
            cnt = self.trackers[name].count
            frac = min(1.0, cnt / EXPECTED_ROUTES_PER_SEED) if EXPECTED_ROUTES_PER_SEED else 0.0
            filled = int(round(bar_w * frac))
            bar = "█" * filled + "░" * (bar_w - filled)
            lines.append(
                f"  {name:<{width}} |{bar}| {cnt:>3}/{EXPECTED_ROUTES_PER_SEED} {frac * 100:5.1f}%"
            )
        return lines

    def render(self, force: bool = False):
        key = tuple((n, self.trackers[n].count) for n in self.active)
        if not force and key == self._last_render_key:
            return
        self._last_render_key = key
        lines = self._format_lines()
        if self.is_tty:
            self.clear()
            if lines:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
                self.lines_drawn = len(lines)
        elif lines:
            print("[Progress] " + " | ".join(l.strip() for l in lines), flush=True)

    def _log_changed(self, name: str) -> bool:
        sig = []
        for ext in ("out", "err"):
            try:
                st = os.stat(self.log_dir / f"{name}.{ext}")
                sig.append((st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append(None)
        sig = tuple(sig)
        changed = self._log_sig.get(name) != sig
        self._log_sig[name] = sig
        return changed

    def refresh(self, force: bool = False):
        now = time.time()
        for name in self.active:
            last = self._last_scan.get(name, 0.0)
            elapsed = now - last
            changed = self._log_changed(name)
            if force or elapsed >= MAX_SCAN_INTERVAL or (changed and elapsed >= MIN_SCAN_INTERVAL):
                self.trackers[name].scan()
                self._last_scan[name] = now
        self.render(force=force)


BOARD: ProgressBoard | None = None


def log(msg: str = ""):
    if BOARD is not None:
        BOARD.clear()
        print(msg, flush=True)
        BOARD.render(force=True)
    else:
        print(msg, flush=True)


# ==============================================================================
# 5. Process Spawning, Shared VLM & Depth-TTC Server Assurance & Quota Management
# ==============================================================================
def validate_unique_prefixes(tasks: dict):
    seen = {}
    for name, cfg in tasks.items():
        prefix = cfg["job_prefix"]
        if prefix in seen:
            raise ValueError(
                f"Duplicate job_prefix '{prefix}' found in tasks '{seen[prefix]}' and '{name}'."
            )
        seen[prefix] = name


def compute_effective_total_worker_jobs(active_tasks: dict, args: argparse.Namespace) -> int:
    if args.total_max_jobs is not None:
        return max(1, args.total_max_jobs)
    return max(1, TOTAL_MAX_SLURM_JOBS)


def ensure_shared_servers(active_tasks: dict, args: argparse.Namespace):
    if args.use_local_server:
        return
    if args.server_url:
        raise ValueError("Use --server-info-file for shared slots; --server-url bypasses reservations")
    print(f"[Scheduler] Server pool: server limit = running route jobs + {ServerSlots.SERVER_SPARE} "
          "across users (pending + running); ready workers prefer users with fewer active routes, FIFO on ties. "
          "Evaluators check every 15s to trim excess servers, replenish capacity, and release routes.")


def report_server_slots(active_tasks: dict, args: argparse.Namespace):
    """Expose the child controller's shared state in the master log."""
    if args.use_local_server:
        return
    for kind in sorted({cfg.get("eval_mode") for cfg in active_tasks.values()} & {"vlm", "depth_ttc"}):
        info = args.vlm_server_info_file if kind == "vlm" else args.depth_server_info_file
        path = Path(info).resolve().parent / "route_slots.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        own = [l for l in data["leases"].values() if l.get("owner_uid") == os.getuid()]
        held = sum(bool(l.get("held")) for l in own)
        cancelling = sum(bool(l.get("cancel_requested_at")) for l in own)
        servers = [s for s in data["managed_servers"].values() if s.get("owner_uid") == os.getuid()]
        pending = sum(s.get("state") in ("pending", "configuring") for s in servers)
        running = sum(s.get("state") == "running" and not s.get("cancel_requested_at") for s in servers)
        log(f"[Pool:{kind}] Own servers pending={pending}, running={running}; "
            f"route reservations={len(own)}, held={held}, cancelling={cancelling}. "
            "Controller actions: logs/multi_eval/<task>.out")


def get_task_server_info_file(cfg: dict, args: argparse.Namespace) -> str:
    eval_mode = cfg.get("eval_mode", "none")
    if eval_mode == "depth_ttc":
        return args.depth_server_info_file
    if eval_mode == "vlm":
        return args.vlm_server_info_file
    return ""


def build_bash_command(task_name: str, cfg: dict, max_jobs: int, args: argparse.Namespace) -> str:
    m_def = MODEL_DEFAULTS[cfg["model"]]
    conda_env = cfg.get("conda_env", m_def["conda_env"])
    agent_file = cfg.get("agent_file", m_def["agent_file"])
    agent_config = cfg.get("agent_config", m_def["agent_config"])
    lb_script = cfg.get("lb_script", m_def["lb_script"])
    use_lb_script = cfg.get("use_lb_script", True)

    env_var_cmd = f"source $PWD/f2d_env_config.sh && load_f2d_env {cfg['env_mods']}"
    if cfg.get("extra_exports"):
        env_var_cmd += f" && {cfg['extra_exports']}"

    info_file = get_task_server_info_file(cfg, args)

    py_args = [
        "python -u slurm_evaluate.py",
        f"--routes {cfg.get('routes', 'fail2drive_split')}",
        f"--seeds {cfg.get('seed', 3)}",
        f"--retries {cfg.get('retries', 1)}",
        f'--out-root "{cfg["out_root"]}"',
        f'--agent-file "{agent_file}"',
        f'--agent-config "{agent_config}"',
        '--env-var-cmd "$env_var_cmd"',
        f'--job-prefix "{cfg["job_prefix"]}"',
        f"{MAX_JOBS_ARG} {max_jobs}",
        f"--routes-per-job {args.routes_per_job}",
        f'--server-gpu-type "{args.server_gpu_type}"',
        f'--route-gpu-type "{args.route_gpu_type}"',
    ]
    if args.worker_gres:
        py_args.append(f'--worker-gres "{args.worker_gres}"')
    if info_file:
        py_args.append(f'--server-info-file "{info_file}"')
    py_args.append("--verbose")

    if cfg.get("eval_mode", None):
        py_args.insert(1, f"--eval-mode {cfg['eval_mode']}")
    if args.server_url:
        py_args.append(f'--server-url "{args.server_url}"')
    if args.use_local_server:
        py_args.append("--use-local-server")
    if args.no_auto_server:
        py_args.append("--no-auto-server")
    if use_lb_script and lb_script:
        py_args.append(f'--lb-script "{lb_script}"')
    if DRY_RUN or cfg.get("dry_run", False):
        py_args.append("--dry-run")

    py_cmd_str = """ \\
  """.join(py_args)

    bash_script = f"""
set -e
source /mnt/bapve/thome/wangyuhao/miniconda3/etc/profile.d/conda.sh
set +u
conda activate {conda_env}
set -u

export ROOT="${{ROOT:-$HOME}}"
export FAIL2DRIVE_ROOT="${{FAIL2DRIVE_ROOT:-$PWD}}"
export LEAD_ROOT="${{LEAD_ROOT:-$ROOT/lead-cvpr2026}}"
export PYTHONPATH="$PWD/team_code:$LEAD_ROOT:${{PYTHONPATH:-}}"

env_var_cmd="{env_var_cmd}"
eval "$env_var_cmd"

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Task: {task_name}"
echo "Allocated Max Parallel Jobs: {max_jobs} | Routes per CARLA Job: {args.routes_per_job}"
echo "============================================================"

{py_cmd_str}
"""
    return bash_script


def print_dry_run(enabled_tasks: dict, active_tasks: dict, completed_tasks: set,
                  trackers: dict, args: argparse.Namespace):
    num_active = len(active_tasks)
    eff_total_jobs = compute_effective_total_worker_jobs(active_tasks, args)
    quota = max(1, min(args.max_job_per_task, eff_total_jobs // num_active)) if num_active else 0
    needs_vlm = any(cfg.get("eval_mode") == "vlm" for cfg in active_tasks.values())
    needs_depth = any(cfg.get("eval_mode") == "depth_ttc" for cfg in active_tasks.values())

    print("\n" + "=" * 75)
    print("[Dry-Run] Nothing will be launched. Planned submissions:")
    print(f"[Dry-Run] Sub-tasks to submit: {num_active} | Already completed (skipped): {len(completed_tasks)}")
    print(f"[Dry-Run] Total Slurm worker job limit: {eff_total_jobs} | "
          f"Quota per sub-task ({MAX_JOBS_ARG}): {quota} (cap {args.max_job_per_task})")
    allocation = (f"local 2 GPUs, server type={args.server_gpu_type}, route type={args.route_gpu_type}"
                  if args.use_local_server else f"worker GRES: {args.worker_gres or gpu_gres(args.route_gpu_type)}")
    print(f"[Dry-Run] Batching: {args.routes_per_job} routes per CARLA job ({allocation})")
    if needs_vlm and not args.use_local_server:
        print(f"[Dry-Run] Alpamayo slots: 1:{os.environ.get('F2D_ROUTES_PER_SERVER', '1')}; "
              f"server GPU type {args.server_gpu_type} (state: {args.vlm_server_info_file})")
    if needs_depth and not args.use_local_server:
        print(f"[Dry-Run] Depth-TTC slots: 1:{os.environ.get('F2D_ROUTES_PER_SERVER', '1')}; "
              f"server GPU type {args.server_gpu_type} (state: {args.depth_server_info_file})")
    print("=" * 75)

    for i, (name, cfg) in enumerate(active_tasks.items(), 1):
        cnt = trackers[name].count
        pending = max(0, EXPECTED_ROUTES_PER_SEED - cnt)
        est_batches = (pending + args.routes_per_job - 1) // args.routes_per_job
        print(f"\n[{i}/{num_active}] {name}")
        print(f"    model        : {cfg['model']}")
        print(f"    seed         : {cfg['seed']}")
        print(f"    eval_mode    : {cfg.get('eval_mode', '-')}")
        print(f"    job_prefix   : {cfg['job_prefix']}")
        print(f"    out_root     : {cfg.get('resolved_out_root', cfg['out_root'])}")
        print(f"    routes done  : {cnt}/{EXPECTED_ROUTES_PER_SEED} "
              f"(pending: {pending} routes -> {est_batches} CARLA batch jobs)")
        script = build_bash_command(name, cfg, quota, args)
        idx = script.rfind("python -u slurm_evaluate.py")
        print("    command      :")
        for line in script[idx:].strip().splitlines():
            print(f"        {line}")

    if completed_tasks:
        print("\n[Dry-Run] Skipped (already 100% complete):")
        for name in sorted(completed_tasks):
            print(f"    - {name}")
    print("\n" + "=" * 75)


def start_task_process(task_name: str, cfg: dict, max_jobs: int, log_dir: Path, args: argparse.Namespace):
    cmd = build_bash_command(task_name, cfg, max_jobs, args)
    out_log = open(log_dir / f"{task_name}.out", "a", buffering=1)
    err_log = open(log_dir / f"{task_name}.err", "a", buffering=1)

    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=out_log,
        stderr=err_log,
        preexec_fn=os.setsid,
    )
    return proc, out_log, err_log


def terminate_all_running(running_procs: dict):
    if not running_procs:
        return

    log(f"[Scheduler] Reclaiming {len(running_procs)} running evaluator process(es) to rebalance job quota...")
    for name, (proc, _, _) in running_procs.items():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.time() + 10
    while time.time() < deadline:
        if all(p.poll() is not None for p, _, _ in running_procs.values()):
            break
        time.sleep(0.5)

    for name, (proc, out_f, err_f) in running_procs.items():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(1)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        out_f.close()
        err_f.close()

    running_procs.clear()


def main():
    global BOARD
    args = parse_args()

    log_dir = Path("logs/multi_eval")
    log_dir.mkdir(parents=True, exist_ok=True)

    enabled_tasks = expand_tasks(TASKS, args.run_name, args.seed)
    if not enabled_tasks:
        sys.exit("[Scheduler] No tasks selected (all tasks disabled or --seed empty).")
    validate_unique_prefixes(enabled_tasks)

    trackers = {name: ProgressTracker(cfg) for name, cfg in enabled_tasks.items()}
    BOARD = ProgressBoard(trackers, log_dir)

    completed_tasks = set()
    failed_tasks = {}
    running_procs = {}
    active_tasks = {}

    print("[Scheduler] Checking existing results (per task & seed)...")
    for name, cfg in enabled_tasks.items():
        if check_task_completion(name, trackers[name]):
            print(f"[Scheduler] Task '{name}' is already 100% completed! Skipping submission.")
            completed_tasks.add(name)
        else:
            active_tasks[name] = cfg

    if args.dry_run:
        print_dry_run(enabled_tasks, active_tasks, completed_tasks, trackers, args)
        return

    ensure_shared_servers(active_tasks, args)

    def handle_exit_signal(signum, frame):
        BOARD.clear()
        print(f"\n[Scheduler] Received signal ({signum}), terminating all evaluator subprocesses...")
        terminate_all_running(running_procs)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    eff_total_jobs = compute_effective_total_worker_jobs(active_tasks, args)
    print(
        f"[Scheduler] Tasks to run: {len(active_tasks)} | Already completed: {len(completed_tasks)} | "
        f"Total Slurm worker job limit: {eff_total_jobs} | Routes per CARLA job: {args.routes_per_job}"
    )
    print(f"[Scheduler] Task logs are saved to: {log_dir.resolve()}/")

    while active_tasks:
        num_active = len(active_tasks)
        eff_total_jobs = compute_effective_total_worker_jobs(active_tasks, args)
        # Keep minimum-one-per-task from exceeding the queued CARLA job limit.
        scheduled_tasks = list(active_tasks.items())[:eff_total_jobs]
        quota_per_task = max(1, min(args.max_job_per_task, eff_total_jobs // len(scheduled_tasks)))

        log("\n" + "=" * 75)
        log(f"[Scheduler] Starting scheduling round | Active tasks: {num_active} | "
            f"Worker Job Limit: {eff_total_jobs} | Quota per task ({MAX_JOBS_ARG}): {quota_per_task} (cap {args.max_job_per_task})")
        log(f"[Scheduler] Active task list: {list(active_tasks.keys())}")
        log("=" * 75)

        for name, cfg in scheduled_tasks:
            proc, out_f, err_f = start_task_process(name, cfg, quota_per_task, log_dir, args)
            running_procs[name] = (proc, out_f, err_f)

        BOARD.active = [name for name, _ in scheduled_tasks]
        BOARD.refresh(force=True)

        rebalance_triggered = False
        last_pool_report = 0
        while not rebalance_triggered:
            time.sleep(LOOP_SLEEP)
            if time.monotonic() - last_pool_report >= 60:
                report_server_slots(active_tasks, args)
                last_pool_report = time.monotonic()
            for name, (proc, out_f, err_f) in list(running_procs.items()):
                ret = proc.poll()
                if ret is not None:
                    out_f.close()
                    err_f.close()
                    del running_procs[name]
                    del active_tasks[name]

                    trackers[name].scan()
                    if ret == 0:
                        log(f"[Scheduler] Task '{name}' finished completely (Return Code: 0).")
                        completed_tasks.add(name)
                    else:
                        log(f"[Scheduler] Task '{name}' exited with error (Return Code: {ret}). Check logs/multi_eval/{name}.err")
                        failed_tasks[name] = ret

                    rebalance_triggered = True

            if rebalance_triggered:
                BOARD.active = [n for n in BOARD.active if n in active_tasks]
                if active_tasks:
                    terminate_all_running(running_procs)
                    time.sleep(2)
            else:
                BOARD.refresh()

    BOARD.active = []
    BOARD.clear()
    print("\n" + "=" * 75)
    print("[Scheduler] All evaluation tasks have finished processing!")
    print(f"  - Completed ({len(completed_tasks)}): {sorted(completed_tasks)}")
    if failed_tasks:
        print(f"  - Failed ({len(failed_tasks)}): {failed_tasks}")
    print("=" * 75)


if __name__ == "__main__":
    main()

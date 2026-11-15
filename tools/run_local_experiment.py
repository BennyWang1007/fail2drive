#!/usr/bin/env python3
"""Run the experiment.sh catalog locally, without importing or invoking Slurm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from string import Template
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import local_evaluate
from run_all_evals import MODEL_DEFAULTS, TASKS


def model_python(task):
    model = task["model"]
    override = os.environ.get(f"F2D_LOCAL_PYTHON_{model.upper()}")
    if override:
        return override
    base = os.environ.get("F2D_CONDA_BASE")
    if not base and shutil.which("conda"):
        base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
    candidate = Path(base or "") / "envs" / MODEL_DEFAULTS[model]["conda_env"] / "bin/python"
    if base and candidate.is_file():
        return str(candidate)
    raise RuntimeError(f"python for {model} not found; set F2D_LOCAL_PYTHON_{model.upper()}")


def catalog():
    # Sourcing defines functions only; share the original allowed names/seeds.
    output = subprocess.check_output(
        ["bash", "-c", 'source "$1"; _f2d_experiment_catalog',
         "bash", str(ROOT / "notes/experiment.sh")], text=True)
    return {name: seeds for name, seeds in
            (line.split("|") for line in output.splitlines())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Local experiment runner. Activate the model environment first.",
        epilog="Paths: CARLA_ROOT, LEAD_ROOT, HIP, DEPTH_ROOT, ALPAMAYO_ROOT. "
               "GPUs: F2D_LOCAL_EVAL_GPU and F2D_LOCAL_SERVER_GPU. "
               "Results default to results/<experiment>/<seed>.")
    parser.add_argument("experiment", nargs="*", metavar="MODEL METHOD SEED")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--routes", default="fail2drive_split")
    parser.add_argument("--out-root", help="Override output directory for one experiment")
    parser.add_argument("--route-filter", default="")
    parser.add_argument("--route-limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--evaluator-wall-timeout", type=float, default=14400)
    parser.add_argument("--carla-startup-wait", type=float, default=60)
    parser.add_argument("--existing-carla", action="store_true",
                        help="Use an already running CARLA instead of managing it")
    parser.add_argument("--reuse-carla", action="store_true",
                        help="Start CARLA once and reuse it for all routes (world reset per route)")
    parser.add_argument("--carla-recycle-every", type=int, default=0,
                        help="With --reuse-carla, restart CARLA every N routes (0 = never)")
    parser.add_argument("--graphics-adapter", default=os.environ.get("CARLA_GRAPHICS_ADAPTER", "0"))
    parser.add_argument("--server-startup-timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if args.reuse_carla and args.existing_carla:
        parser.error("--reuse-carla and --existing-carla are mutually exclusive")
    if args.carla_recycle_every < 0:
        parser.error("--carla-recycle-every must be >= 0")
    choices = catalog()
    if args.list:
        if args.experiment:
            parser.error("--list does not take an experiment")
        return args, choices, []
    if args.experiment == ["all"]:
        if args.out_root:
            parser.error("--out-root requires a single experiment")
        selected = [(name, [int(s) for s in seeds.split(",")])
                    for name, seeds in choices.items()]
    else:
        if len(args.experiment) == 3:
            model, method, seeds = args.experiment
            name = f"{model}_{'depth_ttc' if method == 'depth' else method}"
        elif len(args.experiment) == 2:
            name, seeds = args.experiment
        else:
            parser.error("use MODEL METHOD SEED[,SEED...] or TASK_NAME SEED[,SEED...] or all")
        if name not in choices or not seeds or any(s not in choices[name].split(",") for s in seeds.split(",")):
            parser.error("unsupported experiment/seed (use --list)")
        selected = [(name, list(dict.fromkeys(int(s) for s in seeds.split(","))))]
    if (args.retries < 1 or args.route_limit < 0 or args.timeout <= 0
            or args.server_startup_timeout <= 0 or args.carla_startup_wait < 0
            or args.evaluator_wall_timeout < 0):
        parser.error("invalid retry, route limit, or timeout value")
    if not (1 <= args.port <= 65535 and 1 <= args.traffic_manager_port <= 65535):
        parser.error("ports must be between 1 and 65535")
    if args.port == args.traffic_manager_port:
        parser.error("CARLA and Traffic Manager ports must differ")
    return args, choices, selected


def task_environment(task):
    inherited = os.environ.copy()
    defaults = {
        "CARLA_ROOT": str(ROOT / "f2d_carla"),
        "LEAD_ROOT": str(ROOT.parent / "lead-cvpr2026"),
        "HIP": str(ROOT.parent / "HiP-AD"),
        "DEPTH_ROOT": str(ROOT.parent / "Depth-Anything-V2"),
        "ALPAMAYO_ROOT": str(ROOT.parent / "alpamayo1.5"),
    }
    paths = {key: inherited.get(key, value) for key, value in defaults.items()}
    paths.update({
        "ROOT": inherited.get("ROOT", str(ROOT.parent)),
        "FAIL2DRIVE_ROOT": str(ROOT),
        "ALPAMAYO_REPO": inherited.get("ALPAMAYO_REPO", paths["ALPAMAYO_ROOT"]),
        "ALPAMAYO_PY": inherited.get("ALPAMAYO_PY", str(Path(paths["ALPAMAYO_ROOT"]) / "a1_5_venv/bin/python")),
    })
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in paths.items())
    # Restore local paths after the shared profiles' HPC defaults. Model/policy
    # profiles and per-task exports stay identical to run_all_evals.py.
    script = (f"set -e\n{exports}\nsource {shlex.quote(str(ROOT / 'f2d_env_config.sh'))}\n"
              f"load_f2d_env {task['env_mods']}\n{exports}\n"
              f"{task.get('extra_exports') or ':'}\n"
              f"exec {shlex.quote(sys.executable)} -c "
              + shlex.quote("import json, os; print(json.dumps(dict(os.environ)))"))
    env = json.loads(subprocess.check_output(["bash", "-c", script], cwd=ROOT,
                                            env={**inherited, **paths}, text=True))
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "team_code"), env["LEAD_ROOT"], str(ROOT / "leaderboard"),
        str(ROOT / "scenario_runner"), str(Path(env["CARLA_ROOT"]) / "PythonAPI"),
        str(Path(env["CARLA_ROOT"]) / "PythonAPI/carla"), env.get("PYTHONPATH", "")])
    env.setdefault("LEAD_PROJECT_ROOT", env["LEAD_ROOT"])
    env.setdefault("IS_BENCH2DRIVE", "0")
    env.setdefault("PLANNER_TYPE", "only_traj")
    visible = [v.strip() for v in env.get("CUDA_VISIBLE_DEVICES", "0").split(",") if v.strip()]
    env["CUDA_VISIBLE_DEVICES"] = inherited.get("F2D_LOCAL_EVAL_GPU", visible[0] if visible else "0")
    server_gpu = inherited.get("F2D_LOCAL_SERVER_GPU", visible[1] if len(visible) > 1 else env["CUDA_VISIBLE_DEVICES"])
    return env, server_gpu


def evaluator_args(name, task, seeds, args, env):
    model = MODEL_DEFAULTS[task["model"]]
    expand = lambda value: Template(value).substitute(env)
    output = args.out_root or str(Path("results") / Path(task["resolved_out_root"]).name)
    lb = task.get("lb_script", model["lb_script"])
    if not task.get("use_lb_script", True):
        lb = "leaderboard/leaderboard/leaderboard_evaluator.py"
    command = ["--routes", args.routes, "--out_root", output,
               "--seeds", *map(str, seeds), "--retries", str(args.retries),
               "--agent_file", expand(task.get("agent_file", model["agent_file"])),
               "--agent_config", expand(task.get("agent_config", model["agent_config"])),
               "--lb_script", lb, "--port", str(args.port),
               "--traffic-manager-port", str(args.traffic_manager_port),
               "--timeout", str(args.timeout),
               "--evaluator-wall-timeout", str(args.evaluator_wall_timeout),
               "--carla-startup-wait", str(args.carla_startup_wait),
               "--carla-root", env["CARLA_ROOT"],
               "--graphics-adapter", args.graphics_adapter,
               "--route-filter", args.route_filter, "--route-limit", str(args.route_limit)]
    if args.reuse_carla:
        command += ["--reuse-carla", "--carla-recycle-every", str(args.carla_recycle_every)]
    elif not args.existing_carla:
        command.append("--restart-carla")
    if args.force:
        command.append("--force")
    return command


def stop_process(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass


def wait_ready(url, timeout, process=None):
    parts = urlsplit(url)
    health = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"model server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"model server readiness timeout: {health}")


def run_task(name, task, seeds, args):
    if args.dry_run:
        # No profile sourcing, cache directories, server processes, or model imports.
        print(f"[dry-run] {name} seeds={','.join(map(str, seeds))} profiles={task['env_mods']}")
        print(f"  Python={model_python(task)}; CARLA={'existing' if args.existing_carla else 'reused' if args.reuse_carla else 'restart per route'}; retries={args.retries}")
        print(f"  results={args.out_root or 'results/' + Path(task['resolved_out_root']).name}; mode={task.get('eval_mode', 'none')}")
        return 0
    env, server_gpu = task_environment(task)
    py = model_python(task)
    env["PATH"] = str(Path(py).parent) + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(Path(py).parent.parent)
    fallback = os.environ.get("F2D_LOCAL_FALLBACK_SITE")   # Optional fallback for site-packages (e.g., for CARLA PythonAPI).
    if fallback:
        env["PYTHONPATH"] += os.pathsep + fallback
    command = evaluator_args(name, task, seeds, args, env)
    # Validate route selection before loading any models.
    routes = local_evaluate.route_files(Path(args.routes), args.route_filter, args.route_limit)
    output = Path(command[command.index("--out_root") + 1])
    if not args.force and all(
        local_evaluate.result_is_done(output / str(seed) / "res" / f"{route.stem.rsplit('_', 1)[-1]}_res.json")
        for seed in seeds for route in routes
    ):
        print(f"[local] {name}: selected routes/seeds already completed; skipping.", flush=True)
        return 0
    if not args.existing_carla and not (Path(env["CARLA_ROOT"]) / "CarlaUE4.sh").is_file():
        raise RuntimeError(f"CARLA executable not found: {env['CARLA_ROOT']}/CarlaUE4.sh")
    mode = task.get("eval_mode", "none")
    log_dir = ROOT / "logs/local_eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    server = evaluator = None
    server_log = None
    try:
        if mode in ("depth_ttc", "vlm"):
            key = "DEPTH_TTC_SERVER_URL" if mode == "depth_ttc" else "VLM_SERVER_URL"
            external = os.environ.get(key)
            if mode == "vlm":
                external = external or os.environ.get("ALPAMAYO_SERVER_URL")
            if external:
                env[key] = external
            else:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                env[key] = f"http://127.0.0.1:{port}/{'score' if mode == 'depth_ttc' else 'generate'}"
                if mode == "depth_ttc":
                    server_command = [sys.executable, "-u", "tools/depth_ttc_server.py",
                                      "--depth-anything-root", env["DEPTH_ROOT"],
                                      "--checkpoint", env["DEPTH_CHECKPOINT"]]
                else:
                    server_command = [env["ALPAMAYO_PY"], "-u", "tools/alpamayo_vlm_server.py",
                                      "--model", env["ALPAMAYO_MODEL"],
                                      "--alpamayo-version", env["ALPAMAYO_VERSION"],
                                      "--max-new-tokens", env["ALPAMAYO_MAX_NEW_TOKENS"]]
                server_command += ["--host", "127.0.0.1", "--port", str(port), "--device", "cuda:0"]
                run_id = time.time_ns()
                if mode == "depth_ttc" and env.get("DEPTH_TTC_INFERENCE_LOGGING", "0").lower() in ("1", "true", "yes", "on"):
                    env["DEPTH_TTC_DEBUG_DIR"] = env.get("DEPTH_TTC_DEBUG_DIR") or str(log_dir / f"{name}_debug_{run_id}")
                    server_command += ["--inference-logging", "--debug-dir", env["DEPTH_TTC_DEBUG_DIR"]]
                server_log = (log_dir / f"{name}_server_{run_id}.log").open("w")
                print(f"[local] Starting {mode} on GPU {server_gpu}; log={server_log.name}", flush=True)
                server = subprocess.Popen(server_command, cwd=ROOT,
                                          env={**env, "CUDA_VISIBLE_DEVICES": server_gpu},
                                          stdout=server_log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            if mode == "vlm":
                env["ALPAMAYO_SERVER_URL"] = env[key]
            wait_ready(env[key], args.server_startup_timeout, server)
        print(f"[local] {name} seeds={seeds} eval GPU={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        evaluator = subprocess.Popen([py, "-u", "local_evaluate.py", *command],
                                     cwd=ROOT, env=env, start_new_session=True)
        while evaluator.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError("model server exited during evaluation; see server log")
            time.sleep(1)
        return evaluator.returncode
    finally:
        stop_process(evaluator)
        stop_process(server)
        if server_log:
            server_log.close()


def main(argv=None):
    args, choices, selected = parse_args(argv)
    if args.list:
        for name, seeds in choices.items():
            print(f"{name}|{seeds}")
        return 0
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    failed = []
    for name, seeds in selected:
        if run_task(name, TASKS[name], seeds, args):
            failed.append(name)
    if failed:
        print("Failed experiments: " + ", ".join(failed), file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[local] {exc}", file=sys.stderr)
        raise SystemExit(1)

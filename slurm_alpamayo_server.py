#!/usr/bin/env python3
"""
Unified Multi-GPU / Cross-Node Model Server Manager for Fail2Drive (Alpamayo VLM & Depth-TTC).

Key Features:
1. Supports BOTH `--server-type vlm` (Alpamayo) and `--server-type depth_ttc` (Depth-Anything TTC),
   each with its own isolated state directory and Slurm job name so both can run simultaneously:
   - vlm       -> results_shome/logs/alpamayo_server/active_server.json  (Job: F2D_Alpamayo_Server)
   - depth_ttc -> results_shome/logs/depth_ttc_server/active_server.json (Job: F2D_DepthTTC_Server)
2. Supports BOTH:
   - Single Monolithic Job (`--num-gpus 6` or `10`, `--gpus-per-job 0`), AND
   - Elastic Sharded Server Pool (`--num-gpus 8 --gpus-per-job 2` -> submits 4 jobs of 2 GPUs each):
     Whichever shard gets allocated first automatically starts the Load Balancer, and as later shards
     get allocated, their GPUs dynamically join `workers/worker_*.json` with zero wait!
3. Queue-aware `--wait`: Never times out while the server job is `PENDING` in Slurm queue waiting for
   4090s; the startup timeout only applies AFTER at least one server job enters `RUNNING` state.
"""

import argparse
import http.client
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SERVER_TYPE_DEFAULTS = {
    "vlm": {
        "state_dir": "results_shome/logs/alpamayo_server",
        "job_name": "F2D_Alpamayo_Server",
        "default_gpus": "10",
        "lb_port_start": 41200,
    },
    "depth_ttc": {
        "state_dir": "results_shome/logs/depth_ttc_server",
        "job_name": "F2D_DepthTTC_Server",
        "default_gpus": "4",
        "lb_port_start": 41400,
    },
}
DEFAULT_GPU_TYPE = "nvidia_geforce_rtx_4090"


# ==============================================================================
# 1. Built-in Least-Connections HTTP Load Balancer (runs inside the Slurm Job)
# ==============================================================================
class BackendPool:
    """Thread-safe pool of backend GPU worker servers with least-in-flight routing."""

    def __init__(self, state_dir: Path, server_type: str, lb_host: str, lb_port: int):
        self.state_dir = state_dir
        self.workers_dir = state_dir / "workers"
        self.active_file = state_dir / "active_server.json"
        self.server_type = server_type
        self.lb_host = lb_host
        self.lb_port = lb_port

        self._lock = threading.Lock()
        self._rr_counter = 0
        self.backends: dict[str, dict] = {}
        self.stop_event = threading.Event()

    def discover_and_healthcheck(self):
        """Scan workers_dir for registered worker JSON files and check their /health endpoint."""
        self.workers_dir.mkdir(parents=True, exist_ok=True)
        discovered = {}
        for p in sorted(self.workers_dir.glob("worker_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                base_url = data.get("base_url", "").rstrip("/")
                if base_url:
                    discovered[base_url] = data
            except Exception:
                continue

        health_results = {}
        for base_url in discovered:
            healthy = False
            try:
                req = urllib.request.Request(f"{base_url}/health", method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    healthy = (resp.status == 200)
            except Exception:
                healthy = False
            health_results[base_url] = healthy

        with self._lock:
            for old_url in list(self.backends.keys()):
                if old_url not in discovered:
                    del self.backends[old_url]

            for base_url, meta in discovered.items():
                is_healthy = health_results.get(base_url, False)
                if base_url not in self.backends:
                    self.backends[base_url] = {
                        "base_url": base_url,
                        "host": meta.get("host", ""),
                        "port": meta.get("port", 0),
                        "gpu_id": meta.get("gpu_id", ""),
                        "worker_id": meta.get("worker_id", ""),
                        "slurm_job_id": meta.get("slurm_job_id", ""),
                        "healthy": is_healthy,
                        "in_flight": 0,
                        "total_requests": 0,
                        "total_errors": 0,
                        "total_latency_sec": 0.0,
                        "last_latency_sec": 0.0,
                    }
                else:
                    self.backends[base_url]["healthy"] = is_healthy

            self._write_active_file_locked()

    def _write_active_file_locked(self):
        healthy_urls = [
            b["base_url"] for b in self.backends.values() if b["healthy"]
        ]
        all_urls = [b["base_url"] for b in self.backends.values()]
        lb_base = f"http://{self.lb_host}:{self.lb_port}"
        payload = {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "server_type": self.server_type,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lb_host": self.lb_host,
            "lb_port": self.lb_port,
            "lb_base_url": lb_base,
            "vlm_server_url": f"{lb_base}/generate",
            "alpamayo_server_url": f"{lb_base}/generate",
            "depth_ttc_server_url": f"{lb_base}/score",
            "health_url": f"{lb_base}/health",
            "stats_url": f"{lb_base}/stats",
            "num_total_workers": len(all_urls),
            "num_healthy_workers": len(healthy_urls),
            "workers": [
                {
                    "worker_id": b["worker_id"],
                    "slurm_job_id": b.get("slurm_job_id", ""),
                    "base_url": b["base_url"],
                    "generate_url": f"{b['base_url']}/generate",
                    "score_url": f"{b['base_url']}/score",
                    "healthy": b["healthy"],
                    "in_flight": b["in_flight"],
                    "total_requests": b["total_requests"],
                }
                for b in self.backends.values()
            ],
        }
        tmp_path = self.active_file.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.active_file)
        except OSError:
            pass

    def acquire_backend(self, exclude: set[str] | None = None) -> dict | None:
        exclude = exclude or set()
        with self._lock:
            candidates = [
                b for b in self.backends.values()
                if b["healthy"] and b["base_url"] not in exclude
            ]
            if not candidates:
                candidates = [
                    b for b in self.backends.values()
                    if b["base_url"] not in exclude
                ]
            if not candidates:
                return None

            self._rr_counter = (self._rr_counter + 1) % max(1, len(candidates))
            n = len(candidates)
            best = min(
                range(n),
                key=lambda idx: (
                    candidates[idx]["in_flight"],
                    (idx - self._rr_counter) % n,
                ),
            )
            chosen = candidates[best]
            chosen["in_flight"] += 1
            return chosen

    def release_backend(self, base_url: str, latency_sec: float, ok: bool):
        with self._lock:
            b = self.backends.get(base_url)
            if not b:
                return
            b["in_flight"] = max(0, b["in_flight"] - 1)
            b["total_requests"] += 1
            if ok:
                b["total_latency_sec"] += latency_sec
                b["last_latency_sec"] = round(latency_sec, 3)
            else:
                b["total_errors"] += 1

    def snapshot_stats(self) -> dict:
        with self._lock:
            workers_info = []
            for b in self.backends.values():
                ok_reqs = max(0, b["total_requests"] - b["total_errors"])
                avg_lat = (b["total_latency_sec"] / ok_reqs) if ok_reqs > 0 else 0.0
                workers_info.append({
                    "worker_id": b["worker_id"],
                    "slurm_job_id": b.get("slurm_job_id", ""),
                    "base_url": b["base_url"],
                    "gpu_id": b["gpu_id"],
                    "healthy": b["healthy"],
                    "in_flight": b["in_flight"],
                    "total_requests": b["total_requests"],
                    "total_errors": b["total_errors"],
                    "avg_latency_sec": round(avg_lat, 3),
                    "last_latency_sec": b["last_latency_sec"],
                })
            healthy_cnt = sum(1 for w in workers_info if w["healthy"])
            return {
                "status": "ok" if healthy_cnt > 0 else "starting",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
                "server_type": self.server_type,
                "lb_url": f"http://{self.lb_host}:{self.lb_port}",
                "healthy_workers": healthy_cnt,
                "total_workers": len(workers_info),
                "workers": workers_info,
            }


def run_load_balancer(state_dir: str, server_type: str, port: int):
    state_path = Path(state_dir).resolve()
    lb_host = socket.gethostname()
    pool = BackendPool(state_path, server_type, lb_host, port)

    def monitor_loop():
        while not pool.stop_event.is_set():
            try:
                pool.discover_and_healthcheck()
            except Exception as exc:
                print(f"[LB-Monitor] Error during health check: {exc}", flush=True)
            pool.stop_event.wait(3.0)

    mon_thread = threading.Thread(target=monitor_loop, daemon=True)
    mon_thread.start()

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code: int, payload: dict):
            raw = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.startswith("/health"):
                stats = pool.snapshot_stats()
                code = 200 if stats["healthy_workers"] > 0 else 503
                self._send_json(code, stats)
                return
            if self.path.startswith("/stats"):
                self._send_json(200, pool.snapshot_stats())
                return
            self._proxy_request("GET")

        def do_POST(self):
            self._proxy_request("POST")

        def _proxy_request(self, method: str):
            content_len = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_len) if content_len > 0 else b""

            tried: set[str] = set()
            last_err = "No healthy GPU backend workers available yet"

            for _attempt in range(3):
                backend = pool.acquire_backend(exclude=tried)
                if backend is None:
                    break
                base_url = backend["base_url"]
                tried.add(base_url)
                t0 = time.time()
                ok = False
                try:
                    conn = http.client.HTTPConnection(
                        backend["host"], int(backend["port"]), timeout=300
                    )
                    fwd_headers = {
                        k: v for k, v in self.headers.items()
                        if k.lower() not in ("host", "connection", "transfer-encoding")
                    }
                    fwd_headers["Connection"] = "close"
                    conn.request(method, self.path, body=body, headers=fwd_headers)
                    resp = conn.getresponse()
                    resp_body = resp.read()
                    status = resp.status
                    resp_headers = resp.getheaders()
                    conn.close()

                    ok = (status < 500)
                    pool.release_backend(base_url, time.time() - t0, ok=ok)

                    self.send_response(status)
                    for hk, hv in resp_headers:
                        if hk.lower() not in ("transfer-encoding", "connection", "content-length"):
                            self.send_header(hk, hv)
                    self.send_header("X-Backend-Worker", backend["worker_id"])
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
                except Exception as exc:
                    last_err = f"Backend {base_url} error: {exc}"
                    pool.release_backend(base_url, time.time() - t0, ok=False)

            self._send_json(503, {"error": last_err, "tried_backends": list(tried)})

    class ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ReusableThreadingHTTPServer(("0.0.0.0", port), ProxyHandler)

    def _shutdown_handler(signum, frame):
        pool.stop_event.set()
        try:
            if pool.active_file.exists():
                pool.active_file.unlink()
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print(
        f"[LB] Started Least-Connections Load Balancer ({server_type}) on http://{lb_host}:{port} "
        f"(watching {state_path / 'workers'})",
        flush=True,
    )
    server.serve_forever()


# ==============================================================================
# 2. Cluster GPU Probing (`scontrol show node`)
# ==============================================================================
def probe_cluster_free_gpus(gpu_type: str = DEFAULT_GPU_TYPE, partition: str = "devq") -> list[dict]:
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "node", "-o"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []

    nodes = []
    is_any = (not gpu_type) or (gpu_type.lower() == "any")

    for line in out.splitlines():
        name_m = re.search(r"\bNodeName=(\S+)", line)
        state_m = re.search(r"\bState=(\S+)", line)
        part_m = re.search(r"\bPartitions=(\S+)", line)
        cfg_m = re.search(r"\bCfgTRES=(\S+)", line)
        alloc_m = re.search(r"\bAllocTRES=(\S+)", line)
        gres_m = re.search(r"\bGres=(\S+)", line)

        if not name_m or not state_m:
            continue
        node_name = name_m.group(1)
        state = state_m.group(1).upper()
        if any(bad in state for bad in ("DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPONDING")):
            continue
        if partition and part_m:
            parts = part_m.group(1).split(",")
            if partition not in parts:
                continue

        cfg_str = cfg_m.group(1) if cfg_m else ""
        alloc_str = alloc_m.group(1) if alloc_m else ""
        gres_str = gres_m.group(1) if gres_m else ""

        total_gpu = 0
        alloc_gpu = 0
        if is_any:
            m_tot = re.search(r"\bgres/gpu=(\d+)", cfg_str)
            if m_tot:
                total_gpu = int(m_tot.group(1))
            m_alloc = re.search(r"\bgres/gpu=(\d+)", alloc_str)
            if m_alloc:
                alloc_gpu = int(m_alloc.group(1))
        else:
            m_tot = re.search(rf"gres/gpu:{re.escape(gpu_type)}=(\d+)", cfg_str)
            if m_tot:
                total_gpu = int(m_tot.group(1))
            else:
                m_gres = re.search(rf"gpu:{re.escape(gpu_type)}:(\d+)", gres_str)
                if m_gres:
                    total_gpu = int(m_gres.group(1))
            m_alloc = re.search(rf"gres/gpu:{re.escape(gpu_type)}=(\d+)", alloc_str)
            if m_alloc:
                alloc_gpu = int(m_alloc.group(1))

        if total_gpu <= 0:
            continue

        if "IDLE" in state:
            alloc_gpu = 0
        elif "ALLOCATED" in state and not alloc_str:
            alloc_gpu = total_gpu

        free_gpu = max(0, total_gpu - alloc_gpu)
        nodes.append({
            "node": node_name,
            "total": total_gpu,
            "alloc": alloc_gpu,
            "free": free_gpu,
            "state": state,
        })

    nodes.sort(key=lambda x: (x["free"], x["total"]), reverse=True)
    return nodes


# ==============================================================================
# 3. Slurm Script Generation (Monolithic or Elastic Shards)
# ==============================================================================
def resolve_server_defaults(args: argparse.Namespace, server_type: str) -> tuple[Path, str, str, int]:
    d = SERVER_TYPE_DEFAULTS[server_type]
    state_dir = Path(args.state_dir if args.state_dir else d["state_dir"]).resolve()
    job_name = args.job_name if args.job_name else d["job_name"]
    num_gpus_str = str(args.num_gpus) if args.num_gpus is not None else d["default_gpus"]
    lb_port_start = args.lb_port_start if args.lb_port_start is not None else d["lb_port_start"]
    return state_dir, job_name, num_gpus_str, lb_port_start


def build_server_sbatch_script(
    args: argparse.Namespace,
    server_type: str,
    state_dir: Path,
    job_name: str,
    shard_gpus: int,
    lb_port_start: int,
    shard_idx: int = 0,
) -> str:
    state_dir = state_dir.resolve()
    workers_dir = state_dir / "workers"
    logs_dir = state_dir / "logs"
    script_self = Path(__file__).resolve()

    probed_nodes = probe_cluster_free_gpus(args.gpu_type, args.partition)

    alloc_mode = args.alloc_mode
    if alloc_mode == "auto":
        if args.gpus_per_node and args.gpus_per_node > 0:
            alloc_mode = "gres-per-node"
        elif shard_gpus <= 2:
            alloc_mode = "single-node-gres"
        else:
            alloc_mode = "gpus-per-task"

    is_any_gpu = (not args.gpu_type) or (args.gpu_type.lower() == "any")
    gres_prefix = "gpu" if is_any_gpu else f"gpu:{args.gpu_type}"
    gpus_per_task_spec = "1" if is_any_gpu else f"{args.gpu_type}:1"

    sbatch_directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH -o {logs_dir}/server_%j.out",
        f"#SBATCH -e {logs_dir}/server_%j.err",
        f"#SBATCH --time={args.time_limit}",
    ]

    is_hetjob = False
    if alloc_mode == "single-node-gres":
        sbatch_directives.extend([
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={max(4, shard_gpus * args.cpus_per_gpu)}",
            f"#SBATCH --mem={max(16, shard_gpus * args.mem_per_gpu_gb)}G",
            f"#SBATCH --gres={gres_prefix}:{shard_gpus}",
        ])
    elif alloc_mode == "hetjob" and probed_nodes:
        remaining = shard_gpus
        components = []
        for ninfo in probed_nodes:
            if remaining <= 0:
                break
            take = min(remaining, ninfo["free"] if ninfo["free"] > 0 else ninfo["total"])
            if take <= 0:
                continue
            components.append(take)
            remaining -= take
        while remaining > 0:
            take = min(remaining, 2)
            components.append(take)
            remaining -= take

        is_hetjob = len(components) > 1
        for idx, g_cnt in enumerate(components):
            if idx > 0:
                sbatch_directives.append("#SBATCH hetjob")
            sbatch_directives.extend([
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                f"#SBATCH --cpus-per-task={max(4, g_cnt * args.cpus_per_gpu)}",
                f"#SBATCH --mem={max(16, g_cnt * args.mem_per_gpu_gb)}G",
                f"#SBATCH --gres={gres_prefix}:{g_cnt}",
            ])
    elif alloc_mode == "gres-per-node":
        g_per_node = args.gpus_per_node if (args.gpus_per_node and args.gpus_per_node > 0) else min(2, shard_gpus)
        n_nodes = max(1, math.ceil(shard_gpus / g_per_node))
        sbatch_directives.extend([
            f"#SBATCH --nodes={n_nodes}",
            f"#SBATCH --ntasks={n_nodes}",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={max(4, g_per_node * args.cpus_per_gpu)}",
            f"#SBATCH --mem={max(16, g_per_node * args.mem_per_gpu_gb)}G",
            f"#SBATCH --gres={gres_prefix}:{g_per_node}",
        ])
    else:
        sbatch_directives.extend([
            f"#SBATCH --ntasks={shard_gpus}",
            f"#SBATCH --cpus-per-task={args.cpus_per_gpu}",
            f"#SBATCH --mem-per-cpu={max(2, args.mem_per_gpu_gb // max(1, args.cpus_per_gpu))}G",
            f"#SBATCH --gpus-per-task={gpus_per_task_spec}",
        ])

    directives_str = "\n".join(sbatch_directives)

    if server_type == "depth_ttc":
        model_cmd = """export DEPTH_TTC_DEBUG_DIR="${DEPTH_TTC_DEBUG_DIR:-$WORKER_SAVE_DIR/depth_ttc_debug}"
    CUDA_VISIBLE_DEVICES="$LOCAL_GPU" python -u "$F2D/tools/depth_ttc_server.py" \\
        --host 0.0.0.0 \\
        --port "$WORKER_PORT" \\
        --device cuda:0 \\
        --depth-anything-root "$DEPTH_ROOT" \\
        --checkpoint "$DEPTH_CHECKPOINT" \\
        >"$WORKER_LOG" 2>&1 &"""
        port_offset_base = 40500
    else:
        model_cmd = """CUDA_VISIBLE_DEVICES="$LOCAL_GPU" "$ALPAMAYO_PY" -u "$F2D/tools/alpamayo_vlm_server.py" \\
        --host 0.0.0.0 \\
        --port "$WORKER_PORT" \\
        --device cuda:0 \\
        --model "$ALPAMAYO_MODEL" \\
        --alpamayo-version "$ALPAMAYO_VERSION" \\
        --max-new-tokens "$ALPAMAYO_MAX_NEW_TOKENS" \\
        --input-save-dir "$WORKER_SAVE_DIR/inputs" \\
        --raw-log-path "$WORKER_SAVE_DIR/alpamayo_raw_responses.txt" \\
        >"$WORKER_LOG" 2>&1 &"""
        port_offset_base = 40100

    worker_launcher_path = state_dir / f"launch_worker_{server_type}.sh"
    worker_launcher_content = f"""#!/bin/bash
set -e
NODE_HOST=$(hostname)
JOB_ID="${{SLURM_JOB_ID:-manual}}"
JOB_NUM="${{SLURM_JOB_ID:-0}}"
if ! [[ "$JOB_NUM" =~ ^[0-9]+$ ]]; then
    JOB_NUM=0
fi
TASK_ID="${{SLURM_PROCID:-0}}"
HET_OFFSET="${{SLURM_HET_OFFSET:-0}}"
GLOBAL_TASK_ID=$((TASK_ID + HET_OFFSET))

IFS=',' read -ra GPU_ARRAY <<< "${{CUDA_VISIBLE_DEVICES:-0}}"
echo "[Worker-Node $NODE_HOST] ({server_type}) Job=$JOB_ID Task=$GLOBAL_TASK_ID CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}"

PIDS=()
MY_FILES=()
cleanup_workers() {{
    for f in "${{MY_FILES[@]}}"; do
        rm -f "$f" 2>/dev/null || true
    done
    for pid in "${{PIDS[@]}}"; do
        kill "$pid" 2>/dev/null || true
    done
}}
trap cleanup_workers EXIT INT TERM

for idx in "${{!GPU_ARRAY[@]}}"; do
    LOCAL_GPU="${{GPU_ARRAY[$idx]}}"
    WORKER_ID="${{NODE_HOST}}_j${{JOB_ID}}_t${{GLOBAL_TASK_ID}}_g${{LOCAL_GPU}}"
    WORKER_SAVE_DIR="{logs_dir}/$WORKER_ID"
    mkdir -p "$WORKER_SAVE_DIR/inputs"
    WORKER_LOG="$WORKER_SAVE_DIR/server.log"
    WORKER_JSON="{workers_dir}/worker_${{WORKER_ID}}.json"
    MY_FILES+=("$WORKER_JSON")

    PORT_BASE=$(({port_offset_base} + ((JOB_NUM % 50) * 6) + (GLOBAL_TASK_ID * 15) + (idx * 3)))
    WORKER_PORT=$(comm -23 \\
        <(seq $PORT_BASE $((PORT_BASE + 350)) | sort) \\
        <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) \\
        | shuf | head -n 1)

    echo "[Worker $WORKER_ID] Starting {server_type} server on $NODE_HOST:$WORKER_PORT (GPU=$LOCAL_GPU)..."
    {model_cmd}
    WPID=$!
    PIDS+=("$WPID")

    (
        for _ in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:$WORKER_PORT/health" >/dev/null 2>&1; then
                cat > "${{WORKER_JSON}}.tmp" <<EOF
{{
  "worker_id": "$WORKER_ID",
  "slurm_job_id": "$JOB_ID",
  "server_type": "{server_type}",
  "host": "$NODE_HOST",
  "port": $WORKER_PORT,
  "gpu_id": "$LOCAL_GPU",
  "base_url": "http://$NODE_HOST:$WORKER_PORT",
  "generate_url": "http://$NODE_HOST:$WORKER_PORT/generate",
  "score_url": "http://$NODE_HOST:$WORKER_PORT/score",
  "health_url": "http://$NODE_HOST:$WORKER_PORT/health"
}}
EOF
                mv "${{WORKER_JSON}}.tmp" "$WORKER_JSON"
                echo "[Worker $WORKER_ID] READY at http://$NODE_HOST:$WORKER_PORT"
                exit 0
            fi
            if ! kill -0 "$WPID" 2>/dev/null; then
                echo "[Worker $WORKER_ID] ERROR: Process exited during startup. Last 50 log lines:" >&2
                tail -50 "$WORKER_LOG" >&2 || true
                exit 1
            fi
            sleep 5
        done
        echo "[Worker $WORKER_ID] ERROR: Timed out waiting for /health" >&2
        exit 1
    ) &
done

wait "${{PIDS[@]}}"
"""
    worker_launcher_path.parent.mkdir(parents=True, exist_ok=True)
    worker_launcher_path.write_text(worker_launcher_content, encoding="utf-8")
    os.chmod(worker_launcher_path, 0o755)

    if is_hetjob:
        num_comp = sum(1 for d in sbatch_directives if d.strip() == "#SBATCH hetjob") + 1
        het_group_arg = f"--het-group=0-{num_comp - 1}"
        srun_cmd = f'srun {het_group_arg} "{worker_launcher_path}" &'
    else:
        srun_cmd = f'srun --ntasks=${{SLURM_NTASKS:-1}} "{worker_launcher_path}" &'

    env_mods = "common vlm debug" if server_type == "vlm" else "common depth_ttc debug"
    sbatch_script = f"""#!/bin/bash
{directives_str}

set -e
echo "============================================================"
echo "Slurm Unified {server_type.upper()} Server Job ID: ${{SLURM_JOB_ID:-manual}} (Shard #{shard_idx})"
echo "Head Node: $(hostname)"
echo "Allocated Nodes: ${{SLURM_JOB_NODELIST:-$(hostname)}}"
echo "============================================================"

source /mnt/bapve/thome/wangyuhao/miniconda3/etc/profile.d/conda.sh
set +u
conda activate {args.conda_env}
set -u

export ROOT="${{ROOT:-$HOME}}"
export FAIL2DRIVE_ROOT="${{FAIL2DRIVE_ROOT:-$PWD}}"
export F2D="${{F2D:-$PWD}}"
export LEAD_ROOT="${{LEAD_ROOT:-$ROOT/lead-cvpr2026}}"
export PYTHONPATH="$PWD/team_code:$LEAD_ROOT:${{PYTHONPATH:-}}"

if [ -f "$PWD/f2d_env_config.sh" ]; then
    source "$PWD/f2d_env_config.sh"
    load_f2d_env {env_mods}
fi

mkdir -p "{workers_dir}" "{logs_dir}"

LB_PID=""
OWN_LB=0
ensure_load_balancer() {{
    # Check if an existing Load Balancer from another running shard is already alive
    if [ -f "{state_dir}/active_server.json" ]; then
        EXISTING_HEALTH=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("health_url",""))' "{state_dir}/active_server.json" 2>/dev/null || true)
        if [ -n "$EXISTING_HEALTH" ] && curl -sf --max-time 3 "$EXISTING_HEALTH" >/dev/null 2>&1; then
            echo "[Head Node $(hostname)] Existing Load Balancer is active at $EXISTING_HEALTH; joining pool as worker shard."
            return 0
        fi
    fi

    # Claim Load Balancer role via atomic mkdir lock
    if mkdir "{state_dir}/lb.lock" 2>/dev/null; then
        OWN_LB=1
        LB_PORT=$(comm -23 \\
            <(seq {lb_port_start} $(( {lb_port_start} + 190 )) | sort) \\
            <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) \\
            | head -n 1)
        echo "[Head Node $(hostname)] Starting {server_type.upper()} Load Balancer on port $LB_PORT..."
        python3 -u "{script_self}" _run_lb \\
            --state-dir "{state_dir}" \\
            --server-type "{server_type}" \\
            --lb-port "$LB_PORT" \\
            >"{logs_dir}/lb_${{SLURM_JOB_ID:-manual}}.log" 2>&1 &
        LB_PID=$!
    fi
}}

ensure_load_balancer

cleanup() {{
    echo "[Head Node] Cleaning up shard ${{SLURM_JOB_ID:-manual}}..."
    rm -f "{workers_dir}"/worker_*_j${{SLURM_JOB_ID:-manual}}_*.json 2>/dev/null || true
    if [ "$OWN_LB" -eq 1 ]; then
        rm -rf "{state_dir}/lb.lock" "{state_dir}/active_server.json" 2>/dev/null || true
        [ -n "$LB_PID" ] && kill "$LB_PID" 2>/dev/null || true
    fi
    [ -n "$SRUN_PID" ] && kill "$SRUN_PID" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

{srun_cmd}
SRUN_PID=$!

wait "$SRUN_PID"
"""
    return sbatch_script


# ==============================================================================
# 4. CLI Management Actions: start / status / stop
# ==============================================================================
def get_existing_server_jobs(job_name: str) -> list[tuple[str, str, str]]:
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "-n", job_name, "--format", "%A|%T|%N"],
            text=True,
        )
    except Exception:
        return []
    jobs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0]:
            jobs.append((parts[0], parts[1], parts[2]))
    return jobs


def _target_server_types(args: argparse.Namespace) -> list[str]:
    if args.server_type == "all":
        return ["vlm", "depth_ttc"]
    return [args.server_type]


def cmd_status(args: argparse.Namespace):
    probed = probe_cluster_free_gpus(args.gpu_type, args.partition)
    if probed:
        total_free = sum(n["free"] for n in probed)
        total_cfg = sum(n["total"] for n in probed)
        print(f"[Cluster Probe] GPU='{args.gpu_type}' on partition '{args.partition}': "
              f"{total_free}/{total_cfg} free across {len(probed)} nodes")
        for n in probed:
            if n["free"] > 0 or args.verbose:
                print(f"  - {n['node']:8s}: {n['free']}/{n['total']} free ({n['state']})")

    for stype in _target_server_types(args):
        state_dir, job_name, _, _ = resolve_server_defaults(args, stype)
        active_file = state_dir / "active_server.json"
        print("\n" + "-" * 72)
        print(f"[{stype.upper()} Server] JobName='{job_name}' | StateDir='{state_dir}'")

        jobs = get_existing_server_jobs(job_name)
        print(f"  Slurm Queue : {len(jobs)} job(s)")
        for jid, st, nodes in jobs:
            print(f"    - JobID={jid} State={st} Nodes={nodes}")

        if not active_file.exists():
            print(f"  Server State: No active_server.json at {active_file}")
            continue

        try:
            info = json.loads(active_file.read_text(encoding="utf-8"))
            endpoint_key = "vlm_server_url" if stype == "vlm" else "depth_ttc_server_url"
            print(f"  Load Balancer URL : {info.get('lb_base_url')}")
            print(f"  Target Endpoint   : {info.get(endpoint_key)}")
            print(f"  Workers Ready     : {info.get('num_healthy_workers')}/{info.get('num_total_workers')}")
            stats_url = info.get("stats_url")
            if stats_url:
                with urllib.request.urlopen(stats_url, timeout=3.0) as resp:
                    live = json.loads(resp.read().decode("utf-8"))
                print("  Live Worker Stats :")
                for w in live.get("workers", []):
                    print(
                        f"    - {w['worker_id']:22s} | {w['base_url']:22s} | "
                        f"healthy={str(w['healthy']):5s} | in_flight={w['in_flight']:2d} | "
                        f"reqs={w['total_requests']:5d} | avg_lat={w['avg_latency_sec']:.2f}s"
                    )
        except Exception as exc:
            print(f"  Server State: Could not query live stats: {exc}")


def cmd_stop(args: argparse.Namespace):
    import shutil
    for stype in _target_server_types(args):
        state_dir, job_name, _, _ = resolve_server_defaults(args, stype)
        jobs = get_existing_server_jobs(job_name)
        for jid, st, _ in jobs:
            print(f"[Stop:{stype}] Cancelling Slurm job {jid} ({st})...")
            subprocess.run(["scancel", jid], check=False)

        for p in [state_dir / "active_server.json", state_dir / "lb.lock"]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)
        workers_dir = state_dir / "workers"
        if workers_dir.is_dir():
            for p in workers_dir.glob("worker_*.json"):
                p.unlink(missing_ok=True)
        print(f"[Stop:{stype}] Cleanly stopped '{job_name}' and removed active endpoint files.")


def cmd_start_one(args: argparse.Namespace, server_type: str):
    import shutil
    state_dir, job_name, num_gpus_str, lb_port_start = resolve_server_defaults(args, server_type)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(parents=True, exist_ok=True)
    (state_dir / "workers").mkdir(parents=True, exist_ok=True)

    existing = get_existing_server_jobs(job_name)
    if existing and not args.dry_run:
        if args.force_restart:
            for jid, _, _ in existing:
                subprocess.run(["scancel", jid], check=False)
            shutil.rmtree(state_dir / "lb.lock", ignore_errors=True)
            (state_dir / "active_server.json").unlink(missing_ok=True)
            time.sleep(2)
        else:
            print(f"[Start:{server_type}] Server job '{job_name}' is already in Slurm queue: {existing}")
            if args.wait:
                wait_for_server_ready(
                    state_dir,
                    server_type,
                    job_name,
                    startup_timeout_sec=args.wait_timeout,
                    min_workers=args.min_ready_workers,
                )
            return

    probed_nodes = probe_cluster_free_gpus(args.gpu_type, args.partition)
    total_free_probed = sum(n["free"] for n in probed_nodes)
    default_fallback_gpus = 6 if server_type == "vlm" else 4
    if num_gpus_str.lower() == "auto":
        total_gpus = max(1, min(args.max_auto_gpus, total_free_probed or default_fallback_gpus))
    else:
        total_gpus = int(num_gpus_str)

    if args.gpus_per_job and args.gpus_per_job > 0 and args.gpus_per_job < total_gpus:
        shard_sizes = []
        rem = total_gpus
        while rem > 0:
            take = min(rem, args.gpus_per_job)
            shard_sizes.append(take)
            rem -= take
    else:
        shard_sizes = [total_gpus]

    for s_idx, s_gpus in enumerate(shard_sizes):
        sbatch_content = build_server_sbatch_script(
            args, server_type, state_dir, job_name, s_gpus, lb_port_start, shard_idx=s_idx
        )
        suffix = f"_shard{s_idx}" if len(shard_sizes) > 1 else ""
        sbatch_file = state_dir / f"run_{server_type}_server{suffix}.sbatch"
        sbatch_file.write_text(sbatch_content, encoding="utf-8")

        if args.dry_run:
            print("=" * 78)
            print(
                f"[Dry-Run:{server_type}] Generated {sbatch_file} "
                f"(Shard {s_idx + 1}/{len(shard_sizes)}: {s_gpus} GPUs, Total: {total_gpus} GPUs, Type: {args.gpu_type})"
            )
            print("=" * 78)
            print(sbatch_content)
            continue

        try:
            out = subprocess.check_output(["sbatch", str(sbatch_file)], text=True, stderr=subprocess.STDOUT)
            job_id = out.strip().rsplit(" ", maxsplit=1)[-1]
            print(
                f"[Start:{server_type}] Submitted {server_type} server job "
                f"(Shard {s_idx + 1}/{len(shard_sizes)}, {s_gpus} GPUs): JobID={job_id}"
            )
        except subprocess.CalledProcessError as exc:
            if args.alloc_mode in ("auto", "gpus-per-task"):
                print(
                    f"[Start:{server_type}] Note: `sbatch` rejected `--gpus-per-task` ({exc.output.strip()}). "
                    "Automatically falling back to `hetjob` / `gres-per-node` allocation..."
                )
                orig_mode = args.alloc_mode
                args.alloc_mode = "hetjob" if probed_nodes else "gres-per-node"
                sbatch_content = build_server_sbatch_script(
                    args, server_type, state_dir, job_name, s_gpus, lb_port_start, shard_idx=s_idx
                )
                args.alloc_mode = orig_mode
                sbatch_file.write_text(sbatch_content, encoding="utf-8")
                out = subprocess.check_output(["sbatch", str(sbatch_file)], text=True)
                job_id = out.strip().rsplit(" ", maxsplit=1)[-1]
                print(
                    f"[Start:{server_type}] Submitted {server_type} server job via fallback "
                    f"(Shard {s_idx + 1}/{len(shard_sizes)}, {s_gpus} GPUs): JobID={job_id}"
                )
            else:
                raise

    if args.wait and not args.dry_run:
        wait_for_server_ready(
            state_dir,
            server_type,
            job_name,
            startup_timeout_sec=args.wait_timeout,
            min_workers=args.min_ready_workers,
        )


def cmd_start(args: argparse.Namespace):
    for stype in _target_server_types(args):
        cmd_start_one(args, stype)


def wait_for_server_ready(
    state_dir: Path,
    server_type: str,
    job_name: str,
    startup_timeout_sec: int = 900,
    min_workers: int = 1,
) -> dict:
    active_file = state_dir / "active_server.json"
    print(
        f"[Wait:{server_type}] Waiting for '{job_name}' (queue wait: unlimited while PENDING; "
        f"model startup timeout once RUNNING: {startup_timeout_sec}s; min_workers={min_workers})..."
    )
    queue_start_ts = time.time()
    running_start_ts = None
    last_log_ts = 0.0

    while True:
        jobs = get_existing_server_jobs(job_name)
        any_running = any(st == "RUNNING" for _, st, _ in jobs)

        if any_running and running_start_ts is None:
            running_start_ts = time.time()
            print(f"[Wait:{server_type}] Job entered RUNNING state! Waiting for model initialization...")

        if active_file.exists():
            try:
                info = json.loads(active_file.read_text(encoding="utf-8"))
                health_url = info.get("health_url")
                if health_url:
                    with urllib.request.urlopen(health_url, timeout=3.0) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    healthy = int(data.get("healthy_workers", 0))
                    total = int(data.get("total_workers", 0))
                    if healthy >= min_workers:
                        ep_key = "vlm_server_url" if server_type == "vlm" else "depth_ttc_server_url"
                        print(
                            f"[Wait:{server_type}] Server READY! Endpoint: {info.get(ep_key)} "
                            f"({healthy}/{total} GPU workers online)"
                        )
                        return info
            except Exception:
                pass

        now = time.time()
        if running_start_ts is not None and (now - running_start_ts) >= startup_timeout_sec:
            raise TimeoutError(
                f"[{server_type}] Server job is RUNNING but did not pass /health within {startup_timeout_sec}s. "
                f"Check {state_dir}/logs/"
            )

        if not jobs and not active_file.exists() and (now - queue_start_ts) > 30:
            raise RuntimeError(
                f"[{server_type}] No server jobs named '{job_name}' found in squeue. "
                f"Check {state_dir}/logs/ for startup errors."
            )

        if now - last_log_ts >= 30:
            last_log_ts = now
            q_elapsed = int(now - queue_start_ts)
            if not any_running:
                print(f"[Wait:{server_type}] Still PENDING in Slurm queue ({q_elapsed}s elapsed): {jobs}")
            else:
                r_elapsed = int(now - running_start_ts)
                print(f"[Wait:{server_type}] RUNNING ({r_elapsed}s), warming up model weights on GPU...")

        time.sleep(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Multi-GPU / Cross-Node Alpamayo VLM & Depth-TTC Slurm Server Manager"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=("start", "status", "stop", "_run_lb"),
        help="Action to perform: start (default), status, or stop",
    )
    parser.add_argument(
        "--server-type",
        choices=("vlm", "depth_ttc", "all"),
        default="vlm",
        help="Server type to manage: 'vlm' (Alpamayo), 'depth_ttc' (Depth-Anything TTC), or 'all' (default: vlm)",
    )
    parser.add_argument(
        "--num-gpus",
        default=None,
        help="Total GPUs to request across the server pool (e.g. 6, 10, 4, or 'auto'; defaults: 10 for vlm, 4 for depth_ttc)",
    )
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=0,
        help="Split total GPUs into smaller elastic Slurm jobs of G GPUs each (e.g. --num-gpus 8 --gpus-per-job 2 "
             "submits 4 jobs of 2 GPUs so you don't wait for 8 GPUs at once! Default 0 = 1 monolithic job)",
    )
    parser.add_argument(
        "--max-auto-gpus",
        type=int,
        default=10,
        help="Upper bound when --num-gpus=auto is used (default: 10)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=0,
        help="Optional fixed GPUs per node (0 = flexible cross-node packing)",
    )
    parser.add_argument(
        "--alloc-mode",
        choices=("auto", "single-node-gres", "gpus-per-task", "hetjob", "gres-per-node"),
        default="auto",
        help="Slurm multi-GPU allocation strategy (default: auto)",
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help=f"Slurm GRES GPU type, or 'any' for any available GPU (default: {DEFAULT_GPU_TYPE})",
    )
    parser.add_argument("--partition", default="devq", help="Slurm partition (default: devq)")
    parser.add_argument("--qos", default="normal", help="Slurm QoS (default: normal)")
    parser.add_argument("--time-limit", default="2-00:00:00", help="Slurm walltime limit (default: 2-00:00:00)")
    parser.add_argument("--cpus-per-gpu", type=int, default=4, help="CPUs allocated per GPU worker (default: 4)")
    parser.add_argument("--mem-per-gpu-gb", type=int, default=16, help="RAM (GB) allocated per GPU worker (default: 16)")
    parser.add_argument("--conda-env", default="fail2drive", help="Conda env for the job (default: fail2drive)")
    parser.add_argument("--job-name", default="", help="Override Slurm job name")
    parser.add_argument("--state-dir", default="", help="Override state/log directory")
    parser.add_argument("--lb-port-start", type=int, default=None, help="Starting port range for Load Balancer")
    parser.add_argument("--lb-port", type=int, default=41234, help="Internal flag used by _run_lb")
    parser.add_argument("--wait", action="store_true", help="Wait until at least 1 server job is RUNNING and /health is ready")
    parser.add_argument("--wait-timeout", type=int, default=900, help="Model startup timeout in seconds AFTER job enters RUNNING (default: 900)")
    parser.add_argument("--min-ready-workers", type=int, default=1, help="Minimum healthy GPU workers before --wait returns")
    parser.add_argument("--force-restart", action="store_true", help="Cancel existing server job and start a fresh one")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated sbatch script without submitting")
    parser.add_argument("--verbose", action="store_true", help="Verbose cluster node output in status")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "_run_lb":
        run_load_balancer(args.state_dir, args.server_type, args.lb_port)
    elif args.action == "status":
        cmd_status(args)
    elif args.action == "stop":
        cmd_stop(args)
    else:
        cmd_start(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Launch the shared model servers (Alpamayo-VLM / depth_ttc) as ONE (or a few) long-lived Slurm job(s).

Each job gets N GPUs on one node and starts ONE server process per GPU (CUDA_VISIBLE_DEVICES=<gpu>,
--device cuda:0), bound to 0.0.0.0 so CARLA workers on other nodes can reach it. When a server passes
/health it registers itself in  <registry-dir>/<kind>/<jobid>_gpu<i>.json ; workers discover servers there.

Run it inside the same conda env you use for run_all_evals.py (sbatch inherits it).

  python submit_model_servers.py --kind vlm --num-gpus 6                # one job, 6 GPUs
  python submit_model_servers.py --kind vlm --num-gpus 6 --gpus-per-job 2   # 3 jobs x 2 GPUs (easier to schedule)
  python submit_model_servers.py --kind vlm --status
  python submit_model_servers.py --kind vlm --stop
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import f2d_pool_common as C

TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=@@JOBNAME@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --qos=@@QOS@@
#SBATCH -o @@LOGDIR@@/%x_%j.out
#SBATCH -e @@LOGDIR@@/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@@CPUS@@
#SBATCH --mem=@@MEM@@
#SBATCH --time=@@TIME@@
#SBATCH --gres=@@GRES@@

echo "HPC Node: $(hostname)   JOB: $SLURM_JOB_ID"
@@ENV_CMD@@
@@ENV_CHECK@@
nvidia-smi -L

KIND=@@KIND@@
REG=@@REG@@/$KIND
SAVE=@@SAVE@@/$SLURM_JOB_ID
BASE_PORT=@@BASE_PORT@@
mkdir -p "$REG" "$SAVE"
HOST=$(hostname)

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 @@LASTIDX@@)}"
echo "GPUs in this job: ${GPUS[*]}"

free_port() {
python - "$1" <<'PY'
import socket, sys
p = int(sys.argv[1])
while True:
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", p)); s.close(); print(p); break
    except OSError:
        s.close(); p += 1
PY
}

PIDS=(); PORTS=(); ENTRIES=(); STATE=()
cleanup() {
    for f in "${ENTRIES[@]}"; do rm -f "$f"; done
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]}
    PORT=$(free_port $((BASE_PORT + (SLURM_JOB_ID % 100) * 20 + i)))
    mkdir -p "$SAVE/gpu$i"
@@LAUNCH@@
    PIDS[$i]=$!; PORTS[$i]=$PORT; ENTRIES[$i]="$REG/${SLURM_JOB_ID}_gpu${i}.json"; STATE[$i]=starting
    echo "started server $i on GPU $g -> $HOST:$PORT (pid ${PIDS[$i]})"
done

# ---- wait until every server answers /health, then register it
deadline=$((SECONDS + @@READY_TIMEOUT@@))
while [ $SECONDS -lt $deadline ]; do
    pending=0
    for i in "${!GPUS[@]}"; do
        [ "${STATE[$i]}" = starting ] || continue
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            STATE[$i]=failed; echo "server $i died during startup:"; tail -50 "$SAVE/gpu$i/server.log"; continue
        fi
        if curl -sf "http://127.0.0.1:${PORTS[$i]}/health" >/dev/null; then
            printf '{"kind":"%s","host":"%s","port":%s,"base":"http://%s:%s","job_id":"%s","gpu":"%s"}\n' \
                "$KIND" "$HOST" "${PORTS[$i]}" "$HOST" "${PORTS[$i]}" "$SLURM_JOB_ID" "${GPUS[$i]}" > "${ENTRIES[$i]}.tmp"
            mv "${ENTRIES[$i]}.tmp" "${ENTRIES[$i]}"
            STATE[$i]=ready; echo "server $i ready: $HOST:${PORTS[$i]}"
        else
            pending=$((pending + 1))
        fi
    done
    [ $pending -eq 0 ] && break
    sleep 5
done

# ---- keep the job alive; unregister servers that die
while true; do
    alive=0
    for i in "${!GPUS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then alive=$((alive + 1)); else rm -f "${ENTRIES[$i]}"; fi
    done
    if [ $alive -eq 0 ]; then echo "all servers exited"; exit 1; fi
    sleep 15
done
"""

LAUNCH = {
    "vlm": r"""    CUDA_VISIBLE_DEVICES="$g" "$ALPAMAYO_PY" -u "$F2D/tools/alpamayo_vlm_server.py" \
        --host 0.0.0.0 --port "$PORT" --device cuda:0 \
        --model "$ALPAMAYO_MODEL" --alpamayo-version "$ALPAMAYO_VERSION" \
        --max-new-tokens "$ALPAMAYO_MAX_NEW_TOKENS" \
        --input-save-dir "$SAVE/gpu$i/inputs" \
        --raw-log-path "$SAVE/gpu$i/raw_responses.txt" \
        >"$SAVE/gpu$i/server.log" 2>&1 &""",
    "depth_ttc": r"""    DEPTH_TTC_DEBUG_DIR="$SAVE/gpu$i/debug" CUDA_VISIBLE_DEVICES="$g" python -u "$F2D/tools/depth_ttc_server.py" \
        --host 0.0.0.0 --port "$PORT" --device cuda:0 \
        --depth-anything-root "$DEPTH_ROOT" --checkpoint "$DEPTH_CHECKPOINT" \
        >"$SAVE/gpu$i/server.log" 2>&1 &""",
}
ENV_CHECK = {
    "vlm": ': "${ALPAMAYO_PY:?ALPAMAYO_PY not set by env cmd}" "${F2D:?}" "${ALPAMAYO_MODEL:?}" "${ALPAMAYO_VERSION:?}" "${ALPAMAYO_MAX_NEW_TOKENS:?}"',
    "depth_ttc": ': "${F2D:?}" "${DEPTH_ROOT:?}" "${DEPTH_CHECKPOINT:?}"',
}


def squeue_names():
    try:
        out = subprocess.check_output(["squeue", "--me", "--noheader", "--format", "%A|%j"], text=True)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] squeue failed: {exc}")
        return {}
    res = {}
    for line in out.splitlines():
        jid, _, name = line.partition("|")
        res[name.strip()] = jid.strip()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=sorted(C.SERVER_PATH), required=True)
    ap.add_argument("--num-gpus", type=int, default=6, help="total GPUs for this server pool (NOT hard-coded)")
    ap.add_argument("--gpus-per-job", type=int, default=None,
                    help="GPUs per Slurm job (default: all in ONE job). Use 2-4 if no node has that many GPUs.")
    ap.add_argument("--gres-type", default="nvidia_geforce_rtx_4090", help="'' -> plain gpu:N")
    ap.add_argument("--partition", default="devq")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--time", default="24:00:00", help="QoS normal allows up to 2 days")
    ap.add_argument("--cpus-per-gpu", type=int, default=8)
    ap.add_argument("--mem-per-gpu-gb", type=int, default=16)
    ap.add_argument("--base-port", type=int, default=41000)
    ap.add_argument("--ready-timeout", type=int, default=1800, help="seconds to wait for model load")
    ap.add_argument("--env-var-cmd", default=None,
                    help='default: "source $PWD/f2d_env_config.sh && load_f2d_env common <kind>"')
    ap.add_argument("--registry-dir", default="server_registry")
    ap.add_argument("--job-prefix", default="F2D_SRV_")
    ap.add_argument("--log-dir", default="logs/model_servers")
    ap.add_argument("--dry-run", action="store_true", help="print the sbatch script(s), submit nothing")
    ap.add_argument("--force", action="store_true", help="submit even if jobs with this prefix already exist")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()

    reg = Path(a.registry_dir).resolve()
    name_prefix = f"{a.job_prefix}{a.kind}_"

    if a.status:
        jobs = {n: j for n, j in squeue_names().items() if n.startswith(name_prefix)}
        print(f"slurm jobs: {jobs or 'none'}")
        d = reg / a.kind
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        print(f"registry entries: {len(files)}")
        for f in files:
            try:
                import json
                base = json.loads(f.read_text())["base"]
            except Exception:
                print(f"  {f.name}: unreadable"); continue
            print(f"  {f.name}: {base}  health={'OK' if C.http_ok(base + '/health') else 'DOWN'}")
        return

    if a.stop:
        jobs = {n: j for n, j in squeue_names().items() if n.startswith(name_prefix)}
        for n, j in jobs.items():
            print(f"scancel {j} ({n})")
            subprocess.run(["scancel", j])
        d = reg / a.kind
        for f in (d.glob("*.json") if d.is_dir() else []):
            f.unlink()
        return

    per_job = a.gpus_per_job or a.num_gpus
    sizes = []
    left = a.num_gpus
    while left > 0:
        sizes.append(min(per_job, left))
        left -= sizes[-1]

    existing = {n: j for n, j in squeue_names().items() if n.startswith(name_prefix)} if not a.dry_run else {}
    if existing and not a.force:
        sys.exit(f"[abort] server jobs already queued/running: {existing}. Use --stop first or --force.")

    env_cmd = a.env_var_cmd or f"source $PWD/f2d_env_config.sh && load_f2d_env common {a.kind}"
    log_dir = Path(a.log_dir).resolve()
    run_dir = Path("pool_state/run").resolve()
    if not a.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (reg / a.kind).mkdir(parents=True, exist_ok=True)

    for idx, n in enumerate(sizes):
        gres = f"gpu:{a.gres_type}:{n}" if a.gres_type else f"gpu:{n}"
        script = TEMPLATE
        for k, v in {
            "JOBNAME": f"{name_prefix}{idx}", "PARTITION": a.partition, "QOS": a.qos, "LOGDIR": str(log_dir),
            "CPUS": str(a.cpus_per_gpu * n), "MEM": f"{a.mem_per_gpu_gb * n}G", "TIME": a.time, "GRES": gres,
            "ENV_CMD": env_cmd, "ENV_CHECK": ENV_CHECK[a.kind], "KIND": a.kind, "REG": str(reg),
            "SAVE": str(log_dir / "saved"), "BASE_PORT": str(a.base_port), "LASTIDX": str(n - 1),
            "LAUNCH": LAUNCH[a.kind], "READY_TIMEOUT": str(a.ready_timeout),
        }.items():
            script = script.replace(f"@@{k}@@", v)
        if a.dry_run:
            print(f"# ===== job {idx + 1}/{len(sizes)} ({n} GPUs) =====\n{script}")
            continue
        path = run_dir / f"server_{a.kind}_{idx}.sh"
        path.write_text(script)
        out = subprocess.check_output(["sbatch", str(path)], text=True).strip()
        print(f"[{idx + 1}/{len(sizes)}] {out}   ({n} x GPU, script: {path})")
    if not a.dry_run:
        print(f"Watch: python submit_model_servers.py --kind {a.kind} --status   (servers register after model load)")


if __name__ == "__main__":
    main()
